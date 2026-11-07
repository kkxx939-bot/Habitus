"""判断的确定性派生、落盘与回执。"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from habitus.behavior.fusion import (
    JUDGEMENT_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION,
    BehaviorFusionConfig,
    BehaviorFusionReceipt,
    BehaviorFusionReceiptStore,
    BehaviorJudgementStore,
    build_fusion_receipt,
    derive_judgements,
    judgement_payload,
    receipt_identity,
    segment_identity,
)
from habitus.behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from habitus.behavior.observation import BehaviorObservation, BehaviorObservationConfig
from habitus.foundation.integrity import canonical_json, canonicalize
from tests.lock_store import ProcessLocalLockStore
from tests.unit.behavior.judgement_builders import SUBJECT, batch_from, judgement, unreadable, wire

TZ8 = timezone(timedelta(hours=8))
# 东八区凌晨：UTC 落在前一天，专门用来考验本地日历日的保留。
MIDNIGHT = datetime(2026, 8, 14, 0, 30, tzinfo=TZ8)
JUDGED_AT = datetime(2026, 8, 14, 1, 15, tzinfo=TZ8)
OBSERVATION_CONFIG = BehaviorObservationConfig()
SOURCE_REFS = ("observation-delivery:abc",)
# 版本由写判断的那条 lane 给；这里用一组测试用的标签。
LANE_VERSION = "behavior_test_lane_v1"
PROMPT_VERSION = "behavior_test_rule_v1"


def fragment(
    offset: int,
    semantics: str,
    *,
    delay_ms: int = 800,
    participants: list[str] | None = None,
) -> BehaviorObservation:
    at = MIDNIGHT + timedelta(seconds=offset)
    return BehaviorObservation.create(
        observer_id="home-a/hall",
        occurred_at=at,
        available_at=at + timedelta(milliseconds=delay_ms),
        modality="physical",
        semantics=semantics,
        participants=participants or [SUBJECT],
        knowledge_state="observed",
        confidence=0.9,
        evidence_refs=[f"cam:{offset}"],
        config=OBSERVATION_CONFIG,
    )


FRAGMENTS = [
    fragment(0, "人走到水池边"),
    # 中途一条延迟 30 秒：它比最后一条片段更晚可用，因此取 max 与"取最后一条"在这里可区分。
    fragment(4, "手伸向水龙头", delay_ms=30_000),
    fragment(8, "水流出"),
    fragment(12, "打肥皂"),
    fragment(16, "人打了个哈欠"),
    fragment(20, "画面模糊"),
]


def body() -> dict[str, Any]:
    return wire(
        [
            judgement(
                1,
                behavior="洗手",
                goal="清洁双手",
                basis=["走到水池边打开水龙头", "打肥皂搓手"],
            ),
            judgement(2, behavior="打哈欠"),
            unreadable(3),
        ],
        [[(1, 1)], [(1, 1)], [(1, 1)], [(1, 2)], [(2, None)], [(3, None)]],
    )


def derived(raw: dict[str, Any] | None = None, *, judged_at: datetime = JUDGED_AT):
    batch = batch_from(raw or body())
    return derive_judgements(
        batch,
        FRAGMENTS,
        source_refs=SOURCE_REFS,
        judged_at=judged_at,
        fusion_version=LANE_VERSION,
        prompt_version=PROMPT_VERSION,
    )


# --- 两个时间 ---------------------------------------------------------------------------


def test_evidence_ready_at_takes_the_maximum_of_its_evidence() -> None:
    """一条判断要等它全部证据到齐才成立；取 min 会把还不存在的信息标成已可用。"""

    washing = derived()[0]
    assert washing.evidence_ready_at == max(item.available_at for item in FRAGMENTS[:4])
    # 决定它的是中途那条迟到 30 秒的片段，不是最后一条——取 max 与取末条在这里分得开。
    assert washing.evidence_ready_at == FRAGMENTS[1].available_at
    assert washing.evidence_ready_at > FRAGMENTS[3].available_at


def test_judged_at_is_recorded_because_it_cannot_be_derived() -> None:
    """证据齐备时刻可以从 covers 算出来，判断时刻不行——不记就永远丢了。"""

    washing = derived()[0]
    assert washing.judged_at == JUDGED_AT.astimezone(UTC)
    assert washing.judged_at > washing.evidence_ready_at


def test_behavior_time_keeps_its_local_offset() -> None:
    """行为的时间事实是实际发生的本地时刻；折 UTC 会把东八区凌晨的行为掉到前一天。"""

    washing = derived()[0]
    assert FRAGMENTS[0].occurred_at.isoformat() == "2026-08-13T16:30:00+00:00"
    assert washing.started_at.isoformat() == "2026-08-14T00:30:00+08:00"
    assert washing.started_at.date().isoformat() == "2026-08-14"
    assert washing.last_observed_at.isoformat() == "2026-08-14T00:30:12+08:00"


def test_canonicalize_would_have_moved_the_behavior_to_the_previous_day() -> None:
    """这条是坑本身：``canonicalize`` 折 UTC，所以判断的序列化不能走它。"""

    assert canonicalize(derived()[0].started_at) == "2026-08-13T16:30:00.000000Z"


# --- 身份 -------------------------------------------------------------------------------


def test_identity_changes_with_the_semantics_it_carries() -> None:
    raw = body()
    raw["judgements"][0]["goal"] = "另一个目标"
    assert derived(raw)[0].judgement_id != derived()[0].judgement_id


def test_identity_changes_with_the_moment_it_was_judged() -> None:
    """同一批观测在不同时刻的判断是两条判断，不是一条——它们都为真，只是知道的时间不同。"""

    later = derived(judged_at=JUDGED_AT + timedelta(hours=1))
    assert later[0].judgement_id != derived()[0].judgement_id


# --- 系统绑定的事实 ---------------------------------------------------------------------


def test_fragment_numbers_become_durable_observation_identities() -> None:
    washing = derived()[0]
    assert washing.observation_ids == tuple(item.observation_id for item in FRAGMENTS[:4])
    assert washing.basis[1].observation_ids == (FRAGMENTS[3].observation_id,)
    encoded = json.dumps(judgement_payload(washing), ensure_ascii=False)
    assert "fragment_no" not in encoded


def test_system_bound_fields_are_not_taken_from_the_model() -> None:
    washing = derived()[0]
    assert washing.fusion_version == LANE_VERSION  # 哪条 lane 写的，由那条 lane 给
    assert washing.prompt_version == PROMPT_VERSION
    assert washing.source_refs == SOURCE_REFS


def test_source_refs_must_be_supplied_by_the_caller() -> None:
    """来源身份无法从片段推出，属于系统在模型输出之外绑定的事实。"""

    batch = batch_from(body())
    with pytest.raises(BehaviorFusionError):
        derive_judgements(
            batch,
            FRAGMENTS,
            source_refs=[],
            judged_at=JUDGED_AT,
            fusion_version=LANE_VERSION,
            prompt_version=PROMPT_VERSION,
        )


def test_judged_at_must_be_timezone_aware() -> None:
    batch = batch_from(body())
    with pytest.raises(BehaviorFusionError, match="timezone-aware"):
        derive_judgements(
            batch,
            FRAGMENTS,
            source_refs=SOURCE_REFS,
            judged_at=datetime(2026, 8, 14, 1, 15),
            fusion_version=LANE_VERSION,
            prompt_version=PROMPT_VERSION,
        )


# --- 判断存储 ---------------------------------------------------------------------------


def test_judgements_round_trip_through_the_store(tmp_path) -> None:
    store = BehaviorJudgementStore(tmp_path)
    items = derived()
    for item in items:
        store.put(item)
    stored = store.read(items[0].judgement_id)
    assert stored is not None
    assert stored["started_at"] == "2026-08-14T00:30:00.000000+08:00"
    assert stored["judged_at"].endswith("Z")
    assert len(store.list()) == 3


def test_rewriting_the_same_judgement_is_idempotent(tmp_path) -> None:
    """内容身份意味着重放落盘就是写同一个文件名同样的字节。

    注意这条幂等**由底层的内容寻址提供**，不经过 ``put_payload`` 自己的撞车分支——那条分支只在
    同身份不同内容时才走得到，由 ``test_a_forged_record_under_an_existing_identity_is_refused``
    覆盖。两者都需要，不要把前者当成后者的证据。
    """

    store = BehaviorJudgementStore(tmp_path)
    item = derived()[0]
    store.put(item)
    store.put(item)
    assert len(store.list()) == 1


def test_a_forged_record_under_an_existing_identity_is_refused(tmp_path) -> None:
    store = BehaviorJudgementStore(tmp_path)
    item = derived()[0]
    store.put(item)
    forged = dict(judgement_payload(item))
    forged["summary"] = "被改写的摘要"
    with pytest.raises(BehaviorFusionError, match="collides with different stored content"):
        store.put_payload(forged)


def test_enumeration_survives_foreign_files(tmp_path) -> None:
    """.DS_Store 之类的偶发污染不该永久瘫痪枚举。"""

    store = BehaviorJudgementStore(tmp_path)
    store.put(derived()[0])
    (store.judgement_root / ".DS_Store").write_bytes(b"\x00")
    assert len(store.list()) == 1


def test_a_judgement_over_its_file_bound_is_refused(tmp_path) -> None:
    store = BehaviorJudgementStore(tmp_path, config=BehaviorFusionConfig(max_judgement_file_bytes=64))
    with pytest.raises(BehaviorFusionLimitError):
        store.put(derived()[0])


# --- 回执 -------------------------------------------------------------------------------


def receipt() -> BehaviorFusionReceipt:
    return build_fusion_receipt(
        derived(),
        FRAGMENTS,
        source_refs=SOURCE_REFS,
        prompt_version=PROMPT_VERSION,
        validation_attempts=1,
        primary_subject=SUBJECT,
    )


def test_the_receipt_records_disposition_without_copying_content() -> None:
    """判断自己有耐久记录，所以回执回到它本来该是的样子——一份处置清单。"""

    record = receipt()
    # 主体的判断与没读懂的观测段进 judgement_ids（后者归约层要物化成 gap）；旁人的不进。
    assert record.judgement_ids == tuple(
        item.judgement_id for item in derived() if not item.is_readable or SUBJECT in item.subjects
    )
    assert record.unreadable_observation_ids == (FRAGMENTS[5].observation_id,)
    encoded = json.dumps(record.to_dict(), ensure_ascii=False)
    assert "洗手" not in encoded


def test_receipt_identity_separates_prompt_and_implementation_versions() -> None:
    digest = segment_identity(FRAGMENTS)
    base = receipt_identity(digest, "impl_v1", "prompt_v1")
    assert base != receipt_identity(digest, "impl_v2", "prompt_v1")
    assert base != receipt_identity(digest, "impl_v1", "prompt_v2")


def test_segment_identity_ignores_presentation_order() -> None:
    assert segment_identity(FRAGMENTS) == segment_identity(list(reversed(FRAGMENTS)))
    assert segment_identity(FRAGMENTS) != segment_identity(FRAGMENTS[:3])


def test_a_tampered_receipt_is_caught_by_its_digest() -> None:
    payload = receipt().to_dict()
    payload["unreadable_observation_ids"] = []
    with pytest.raises(BehaviorFusionError, match="record_digest does not match"):
        BehaviorFusionReceipt.from_dict(payload)


def test_a_receipt_cannot_report_observations_outside_its_segment() -> None:
    payload = receipt().to_dict()
    payload["unreadable_observation_ids"] = [fragment(99, "别处的观测").observation_id]
    with pytest.raises(BehaviorFusionError, match="outside its segment"):
        BehaviorFusionReceipt.from_dict(payload)


def test_receipts_round_trip_through_the_store(tmp_path) -> None:
    store = BehaviorFusionReceiptStore(tmp_path, lock_store=ProcessLocalLockStore())
    record = receipt()
    assert store.put(record).to_dict() == record.to_dict()
    assert store.list() == (store.read(record.receipt_id),)


def test_a_corrupt_receipt_file_fails_loudly(tmp_path) -> None:
    store = BehaviorFusionReceiptStore(tmp_path, lock_store=ProcessLocalLockStore())
    record = store.put(receipt())
    (path,) = store.receipt_root.glob(f"*_{record.receipt_id}.json")
    payload = record.to_dict()
    payload["prompt_version"] = "tampered"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with pytest.raises(BehaviorFusionError):
        store.read(record.receipt_id)


@pytest.mark.parametrize("bad", [True, "1", 1.0, None])
def test_a_receipt_file_with_non_integer_validation_attempts_is_refused(tmp_path, bad: object) -> None:
    """W16：回执文件里 ``validation_attempts`` 类型错（含 JSON 的 true）在解码器就被拒。"""

    store = BehaviorFusionReceiptStore(tmp_path, lock_store=ProcessLocalLockStore())
    record = store.put(receipt())
    (path,) = store.receipt_root.glob(f"*_{record.receipt_id}.json")
    payload = record.to_dict()
    payload["validation_attempts"] = bad
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with pytest.raises(BehaviorFusionError, match="record is corrupt") as caught:
        store.read(record.receipt_id)
    assert "validation_attempts must be an integer" in str(caught.value.__cause__)


@pytest.mark.parametrize("bad", [1, None, ["a" * 64]])
def test_a_judgement_payload_without_text_identity_is_refused(tmp_path, bad: object) -> None:
    """W16：``put_payload`` 也收重试堆里读回的记录；``judgement_id`` 不是文本在落盘前就被拒。"""

    store = BehaviorJudgementStore(tmp_path)
    payload = {**judgement_payload(derived()[0]), "judgement_id": bad}
    with pytest.raises(BehaviorFusionError, match="judgement_id must be lowercase SHA-256 text"):
        store.put_payload(payload)
    assert store.list() == ()


def test_a_receipt_carries_only_the_slimmed_keys() -> None:
    """旁人、无归属两份清单两条 lane 都恒空，回执里不再有这两个键。"""

    payload = receipt().to_dict()
    assert payload["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert "out_of_scope_observation_ids" not in payload and "unowned_observation_ids" not in payload
    with pytest.raises(BehaviorFusionError, match="schema is invalid"):
        BehaviorFusionReceipt.from_dict({**payload, "unowned_observation_ids": []})


def test_the_receipt_filename_carries_its_judgement_moment(tmp_path) -> None:
    store = BehaviorFusionReceiptStore(tmp_path, lock_store=ProcessLocalLockStore())
    record = store.put(receipt())
    (path,) = store.receipt_root.iterdir()
    assert path.name == f"{record.judged_at.astimezone(UTC):%Y%m%dT%H%M%S%fZ}_{record.receipt_id}.json"


def test_a_rerun_at_another_moment_reuses_the_first_receipt(tmp_path) -> None:
    """同一片段同版本重跑、判定时刻不同：文件名不同，撞车仍按 receipt_id 找到先到的那份。"""

    store = BehaviorFusionReceiptStore(tmp_path, lock_store=ProcessLocalLockStore())
    first = store.put(receipt())
    later = build_fusion_receipt(
        derived(judged_at=JUDGED_AT + timedelta(hours=20)),
        FRAGMENTS,
        source_refs=SOURCE_REFS,
        prompt_version=PROMPT_VERSION,
        validation_attempts=2,
        primary_subject=SUBJECT,
    )
    assert later.receipt_id == first.receipt_id and later.judged_at != first.judged_at
    assert store.put(later) == first
    assert len(list(store.receipt_root.iterdir())) == 1


def test_concurrent_puts_of_one_receipt_id_leave_a_single_receipt(tmp_path, monkeypatch) -> None:
    """两个写者同时为同一 ``receipt_id`` 以不同时刻落回执：盘上只留一份，两边拿到的是同一份。

    "先找、再创建"之间放一道栅栏：没有按 ``receipt_id`` 的锁时，两个写者都会在栅栏处汇合、都没找到、各建一份；
    有锁时后一个写者等在锁外，栅栏等不到它就放行，它进来时已经能找到先到的那份。
    """

    store = BehaviorFusionReceiptStore(tmp_path, lock_store=ProcessLocalLockStore())
    first = receipt()
    second = build_fusion_receipt(
        derived(judged_at=JUDGED_AT + timedelta(hours=20)),
        FRAGMENTS,
        source_refs=SOURCE_REFS,
        prompt_version=PROMPT_VERSION,
        validation_attempts=2,
        primary_subject=SUBJECT,
    )
    assert first.receipt_id == second.receipt_id and first.judged_at != second.judged_at
    barrier = threading.Barrier(2)
    entries = BehaviorFusionReceiptStore._entries

    def rendezvous(self: BehaviorFusionReceiptStore) -> Any:
        found = entries(self)
        try:
            barrier.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        return found

    monkeypatch.setattr(BehaviorFusionReceiptStore, "_entries", rendezvous)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result() for future in [pool.submit(store.put, first), pool.submit(store.put, second)]]
    monkeypatch.undo()

    assert len(list(store.receipt_root.iterdir())) == 1
    assert results[0] == results[1] == store.read(first.receipt_id)


def test_expiry_reads_only_the_filename_moment(tmp_path) -> None:
    """过期只看文件名里的时刻，不解码正文：正文读不懂（例如旧版本）的回执照样按时刻过期。"""

    store = BehaviorFusionReceiptStore(tmp_path, lock_store=ProcessLocalLockStore())
    record = store.put(receipt())
    (path,) = store.receipt_root.iterdir()
    path.write_text('{"schema_version": "behavior_fusion_receipt_v4"}\n', encoding="utf-8")
    assert store.expire(record.judged_at) == 0
    assert store.expire(record.judged_at + timedelta(microseconds=1)) == 1
    assert list(store.receipt_root.iterdir()) == []


def test_a_malformed_payload_is_refused_before_it_poisons_the_store(tmp_path) -> None:
    """写路径必须与读路径一样严。

    一份缺键的记录写得进、读不回：``list()`` 会在排序键上抛裸 KeyError（连错误谱系都不在），
    而本层没有删除接口——枚举就此永久报废。观测存储与回执存储各自做了这道防护，判断存储不能少。
    """

    store = BehaviorJudgementStore(tmp_path)
    with pytest.raises(BehaviorFusionError, match="shape is invalid"):
        store.put_payload({"judgement_id": "a" * 64})

    partial = dict(judgement_payload(derived()[0]))
    del partial["evidence_ready_at"]
    with pytest.raises(BehaviorFusionError, match="missing=\\['evidence_ready_at'\\]"):
        store.put_payload(partial)

    extra = dict(judgement_payload(derived()[0]))
    extra["smuggled"] = 1
    with pytest.raises(BehaviorFusionError, match="unknown=\\['smuggled'\\]"):
        store.put_payload(extra)

    assert store.list() == ()


def test_judgements_no_longer_carry_relations_and_the_old_shape_is_refused(tmp_path) -> None:
    """判断之间的关系不再产出，键也删了；带 ``relations`` 的旧格式（v1）按严格读取拒绝。"""

    store = BehaviorJudgementStore(tmp_path)
    payload = judgement_payload(derived()[0])
    assert "relations" not in payload
    store.put_payload(payload)
    (stored,) = store.list()
    assert stored["schema_version"] == JUDGEMENT_SCHEMA_VERSION and "relations" not in stored
    with pytest.raises(BehaviorFusionError, match="unknown=\\['relations'\\]"):
        store.put_payload({**payload, "judgement_id": "f" * 64, "relations": []})
    with pytest.raises(BehaviorFusionError, match="schema is invalid"):
        store.put_payload({**payload, "judgement_id": "e" * 64, "schema_version": "behavior_judgement_v1"})


def test_identical_replays_are_absorbed_below_the_store_not_by_it(tmp_path) -> None:
    """把幂等的真正来源写清楚，免得把"底层保证的"误当成"本层验证过的"。

    ``atomic_create_bytes`` 对**字节完全相同**的重复写入返回 False 而不抛冲突，所以
    ``put_payload`` 的撞车分支在重放场景下根本进不去——判断按内容身份命名，重放就是写同一个
    文件名同样的字节。本层自己的撞车逻辑由伪造记录那条测试覆盖（同身份、不同内容）。
    """

    from habitus.infrastructure.store.filesystem import atomic_create_bytes

    path = tmp_path / "probe.json"
    assert atomic_create_bytes(path, b"same\n", artifact_root=tmp_path) is True
    assert atomic_create_bytes(path, b"same\n", artifact_root=tmp_path) is False


def test_the_receipt_separates_other_peoples_behaviour_from_unreadable_frames() -> None:
    """路人多不等于上游在退化——两个信号混进同一个比例就都失真了。

    个人版只跟踪主要主体，但别人的帧不能被静默丢掉：它们的观测身份留在回执里，
    比例上升说明场景里出现了别人（来客人、保姆在干活），是场景复杂度而非质量问题。
    """

    guest = "客人C"
    fragments = [
        fragment(0, "主体走到水池边"),
        fragment(4, "主体在洗手"),
        fragment(8, "主体关水龙头"),
        BehaviorObservation.create(
            observer_id="home-a/hall",
            occurred_at=MIDNIGHT + timedelta(seconds=12),
            available_at=MIDNIGHT + timedelta(seconds=13),
            modality="physical",
            semantics="客人从门口走过",
            participants=[SUBJECT, guest],
            knowledge_state="observed",
            confidence=0.9,
            evidence_refs=["cam:12"],
            config=OBSERVATION_CONFIG,
        ),
        fragment(16, "画面模糊"),
    ]
    raw = wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗双手"]),
            judgement(2, behavior="走过", subjects=[guest]),
            unreadable(3),
        ],
        [[(1, 1)], [(1, 1)], [(1, 1)], [(2, None)], [(3, None)]],
    )
    batch = batch_from(raw)
    items = derive_judgements(
        batch,
        fragments,
        source_refs=SOURCE_REFS,
        judged_at=JUDGED_AT,
        fusion_version=LANE_VERSION,
        prompt_version=PROMPT_VERSION,
    )
    record = build_fusion_receipt(
        items,
        fragments,
        source_refs=SOURCE_REFS,
        prompt_version=PROMPT_VERSION,
        validation_attempts=1,
        primary_subject=SUBJECT,
    )
    # 主体那条 + 没读懂那条进 judgement_ids；旁人（客人）那条不进。
    assert len(record.judgement_ids) == 2
    assert record.unreadable_observation_ids == (fragments[4].observation_id,)


def test_a_segment_with_no_behaviour_of_the_subject_still_yields_a_receipt() -> None:
    """一段观测里全是别人的行为，"本段没有主体的行为"本身是有效结论，不该无法落回执。"""

    guest = "客人C"
    fragments = [
        BehaviorObservation.create(
            observer_id="home-a/hall",
            occurred_at=MIDNIGHT + timedelta(seconds=index * 4),
            available_at=MIDNIGHT + timedelta(seconds=index * 4 + 1),
            modality="physical",
            semantics=f"客人在客厅活动{index}",
            participants=[SUBJECT, guest],
            knowledge_state="observed",
            confidence=0.9,
            evidence_refs=[f"cam:{index}"],
            config=OBSERVATION_CONFIG,
        )
        for index in range(3)
    ]
    raw = wire(
        [judgement(1, behavior="走动", subjects=[guest])],
        [[(1, None)]] * 3,
    )
    batch = batch_from(raw)
    items = derive_judgements(
        batch,
        fragments,
        source_refs=SOURCE_REFS,
        judged_at=JUDGED_AT,
        fusion_version=LANE_VERSION,
        prompt_version=PROMPT_VERSION,
    )
    record = build_fusion_receipt(
        items,
        fragments,
        source_refs=SOURCE_REFS,
        prompt_version=PROMPT_VERSION,
        validation_attempts=1,
        primary_subject=SUBJECT,
    )
    assert record.judgement_ids == ()


def test_a_frame_also_claimed_by_a_readable_judgement_is_not_unreadable() -> None:
    """一帧同时被可读判断与没读懂判断认领，它已经被读懂了，不计入 unreadable。"""

    guest = "客人C"
    shared = BehaviorObservation.create(
        observer_id="home-a/hall",
        occurred_at=MIDNIGHT + timedelta(seconds=12),
        available_at=MIDNIGHT + timedelta(seconds=13),
        modality="physical",
        semantics="客人走过，画面另一半模糊",
        participants=[SUBJECT, guest],
        knowledge_state="observed",
        confidence=0.9,
        evidence_refs=["cam:12"],
        config=OBSERVATION_CONFIG,
    )
    fragments = [fragment(0, "主体在洗手"), shared]
    raw = wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗双手"]),
            judgement(2, behavior="走过", subjects=[guest]),
            unreadable(3),
        ],
        [[(1, 1)], [(2, None), (3, None)]],  # 同一帧：旁人可读判断 + 没读懂判断都认领
    )
    batch = batch_from(raw)
    items = derive_judgements(
        batch,
        fragments,
        source_refs=SOURCE_REFS,
        judged_at=JUDGED_AT,
        fusion_version=LANE_VERSION,
        prompt_version=PROMPT_VERSION,
    )
    record = build_fusion_receipt(
        items,
        fragments,
        source_refs=SOURCE_REFS,
        prompt_version=PROMPT_VERSION,
        validation_attempts=1,
        primary_subject=SUBJECT,
    )
    assert record.unreadable_observation_ids == ()
