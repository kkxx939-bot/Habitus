"""判断的确定性派生、落盘与回执。"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from behavior.fusion import (
    FUSION_PROMPT_VERSION,
    FUSION_VERSION,
    BehaviorFusionConfig,
    BehaviorFusionReceipt,
    BehaviorFusionReceiptStore,
    BehaviorJudgementStore,
    JudgementRelation,
    assemble_judgement_batch,
    build_fusion_receipt,
    derive_judgements,
    judgement_payload,
    receipt_identity,
    segment_identity,
    validate_judgement_batch,
    without_unresolvable_relations,
)
from behavior.fusion.errors import BehaviorFusionError, BehaviorFusionLimitError
from behavior.fusion.schema import JUDGEMENT_FUSION_JSON_SCHEMA
from behavior.fusion.store import _JUDGEMENT_KEYS
from behavior.observation import BehaviorObservation, BehaviorObservationConfig
from foundation.integrity import canonical_digest, canonical_json, canonicalize
from tests.unit.behavior.fusion_wire import SUBJECT, judgement, unreadable, wire

TZ8 = timezone(timedelta(hours=8))
# 东八区凌晨：UTC 落在前一天，专门用来考验本地日历日的保留。
MIDNIGHT = datetime(2026, 8, 14, 0, 30, tzinfo=TZ8)
JUDGED_AT = datetime(2026, 8, 14, 1, 15, tzinfo=TZ8)
OBSERVATION_CONFIG = BehaviorObservationConfig()
SOURCE_REFS = ("observation-delivery:abc",)
# 从提示词模块取，而不是写死字面量：这里要证的是"版本由系统写入、不来自模型返回的线格式"，
# 写死只会让每次改提示词都顺带来改一次测试。
PROMPT_VERSION = FUSION_PROMPT_VERSION


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
        modality="vision",
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
            judgement(2, behavior="打哈欠", relations=[("concurrent_with", 1)]),
            unreadable(3),
        ],
        [[(1, 1)], [(1, 1)], [(1, 1)], [(1, 2)], [(2, None)], [(3, None)]],
    )


def derived(raw: dict[str, Any] | None = None, *, judged_at: datetime = JUDGED_AT):
    batch = assemble_judgement_batch(raw or body(), fragment_count=len(FRAGMENTS))
    validate_judgement_batch(batch, FRAGMENTS)
    return derive_judgements(batch, FRAGMENTS, source_refs=SOURCE_REFS, judged_at=judged_at)


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
    assert washing.judged_at == JUDGED_AT.astimezone(timezone.utc)
    assert washing.judged_at > washing.evidence_ready_at
    assert washing.fusion_lag_seconds == pytest.approx(
        (JUDGED_AT - FRAGMENTS[1].available_at).total_seconds()
    )


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


def test_identity_excludes_relations_so_mutual_links_do_not_deadlock() -> None:
    """并行是互指的；关系若进身份就成了循环依赖，谁也算不出来。"""

    items = derived()
    washing, yawn = items[0], items[1]
    assert yawn.relations == ((JudgementRelation.CONCURRENT_WITH.value, washing.judgement_id),)
    assert washing.relations == ((JudgementRelation.CONCURRENT_WITH.value, yawn.judgement_id),)
    assert len({item.judgement_id for item in items}) == 3


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
    assert washing.fusion_version == FUSION_VERSION
    assert washing.prompt_version == PROMPT_VERSION
    assert washing.source_refs == SOURCE_REFS


def test_source_refs_must_be_supplied_by_the_caller() -> None:
    """来源身份无法从片段推出，属于系统在模型输出之外绑定的事实。"""

    batch = assemble_judgement_batch(body(), fragment_count=len(FRAGMENTS))
    with pytest.raises(BehaviorFusionError):
        derive_judgements(batch, FRAGMENTS, source_refs=[], judged_at=JUDGED_AT)


def test_judged_at_must_be_timezone_aware() -> None:
    batch = assemble_judgement_batch(body(), fragment_count=len(FRAGMENTS))
    with pytest.raises(BehaviorFusionError, match="timezone-aware"):
        derive_judgements(
            batch, FRAGMENTS, source_refs=SOURCE_REFS, judged_at=datetime(2026, 8, 14, 1, 15)
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
    # 主体的判断与没读懂的观测段进 judgement_ids（后者归约层要物化成 gap）；旁人的只留观测身份。
    assert record.judgement_ids == tuple(
        item.judgement_id
        for item in derived()
        if not item.is_readable or SUBJECT in item.subjects
    )
    assert record.unreadable_observation_ids == (FRAGMENTS[5].observation_id,)
    assert record.unreadable_ratio == pytest.approx(1 / 6)
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
    store = BehaviorFusionReceiptStore(tmp_path)
    record = receipt()
    assert store.put(record).to_dict() == record.to_dict()
    assert store.list() == (store.read(record.receipt_id),)


def test_a_corrupt_receipt_file_fails_loudly(tmp_path) -> None:
    store = BehaviorFusionReceiptStore(tmp_path)
    record = store.put(receipt())
    path = store.receipt_root / f"{record.receipt_id}.json"
    payload = record.to_dict()
    payload["prompt_version"] = "tampered"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with pytest.raises(BehaviorFusionError):
        store.read(record.receipt_id)


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


def test_identical_replays_are_absorbed_below_the_store_not_by_it(tmp_path) -> None:
    """把幂等的真正来源写清楚，免得把"底层保证的"误当成"本层验证过的"。

    ``atomic_create_bytes`` 对**字节完全相同**的重复写入返回 False 而不抛冲突，所以
    ``put_payload`` 的撞车分支在重放场景下根本进不去——判断按内容身份命名，重放就是写同一个
    文件名同样的字节。本层自己的撞车逻辑由伪造记录那条测试覆盖（同身份、不同内容）。
    """

    from infrastructure.store.filesystem import atomic_create_bytes

    path = tmp_path / "probe.json"
    assert atomic_create_bytes(path, b"same\n", artifact_root=tmp_path) is True
    assert atomic_create_bytes(path, b"same\n", artifact_root=tmp_path) is False


def test_the_receipt_uses_the_same_unreadable_criterion_as_the_batch() -> None:
    """两处口径不一致时，落盘的那一份会永久偏高且改不了——而它正是告警依据。"""

    from behavior.fusion import unreadable_ratio

    raw = wire(
        [
            judgement(1, behavior="洗手", goal="清洁双手", basis=["冲洗"]),
            unreadable(2),
        ],
        [[(1, 1)], [(1, 1)], [(1, 1)], [(1, 1)], [(1, 1), (2, None)], [(1, 1)]],
    )
    batch = assemble_judgement_batch(raw, fragment_count=len(FRAGMENTS))
    validate_judgement_batch(batch, FRAGMENTS)
    items = derive_judgements(
        batch, FRAGMENTS, source_refs=SOURCE_REFS, judged_at=JUDGED_AT
    )
    record = build_fusion_receipt(
        items,
        FRAGMENTS,
        source_refs=SOURCE_REFS,
        prompt_version=PROMPT_VERSION,
        validation_attempts=1,
        primary_subject=SUBJECT,
    )
    # 第 5 帧同时被「洗手」和一条"没读懂"认领——它已经被读懂了，两处都不该把它计入。
    assert unreadable_ratio(batch, len(FRAGMENTS)) == record.unreadable_ratio
    assert record.unreadable_observation_ids == ()


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
            modality="vision",
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
    batch = assemble_judgement_batch(raw, fragment_count=len(fragments))
    validate_judgement_batch(batch, fragments)
    items = derive_judgements(
        batch, fragments, source_refs=SOURCE_REFS, judged_at=JUDGED_AT
    )
    record = build_fusion_receipt(
        items,
        fragments,
        source_refs=SOURCE_REFS,
        prompt_version=PROMPT_VERSION,
        validation_attempts=1,
        primary_subject=SUBJECT,
    )
    # 主体那条 + 没读懂那条进 judgement_ids；旁人（客人）那条只留观测身份。
    assert len(record.judgement_ids) == 2
    assert record.out_of_scope_observation_ids == (fragments[3].observation_id,)
    assert record.unreadable_observation_ids == (fragments[4].observation_id,)
    assert record.out_of_scope_ratio == pytest.approx(0.2)
    assert record.unreadable_ratio == pytest.approx(0.2)


def test_a_segment_with_no_behaviour_of_the_subject_still_yields_a_receipt() -> None:
    """一段观测里全是别人的行为，"本段没有主体的行为"本身是有效结论，不该无法落回执。"""

    guest = "客人C"
    fragments = [
        BehaviorObservation.create(
            observer_id="home-a/hall",
            occurred_at=MIDNIGHT + timedelta(seconds=index * 4),
            available_at=MIDNIGHT + timedelta(seconds=index * 4 + 1),
            modality="vision",
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
    batch = assemble_judgement_batch(raw, fragment_count=len(fragments))
    items = derive_judgements(
        batch, fragments, source_refs=SOURCE_REFS, judged_at=JUDGED_AT
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
    assert record.out_of_scope_ratio == 1.0


def test_relations_pointing_at_judgements_that_will_not_be_stored_are_pruned() -> None:
    """个人版只落主体的判断；指向被分流掉的旁人判断的关系必须剪掉。

    并行是**相互**的：提示词要求模型只在一边声明，装配层自动补上另一边。于是主体那条判断上会
    出现一条指向旁人判断的关系，而旁人那条随后被分流。留着它，不可变存储里就永久躺着一个指向
    不存在记录的 ``target_id``，而回执只记观测身份、不记被丢掉的判断身份，事后无从还原。

    剪枝不改身份：``judgement_id`` 是不含 relations 的内容摘要。
    """

    fragments = (
        fragment(0, "主人在餐桌前吃饭", participants=[SUBJECT]),
        fragment(5, "主人夹菜", participants=[SUBJECT]),
        fragment(10, "客人拿起电话通话", participants=["客人B"]),
    )
    raw = wire(
        [
            judgement(1, behavior="吃饭", goal="吃完这顿饭", summary="在餐桌前吃饭", basis=["进食"]),
            judgement(
                2,
                behavior="打电话",
                goal="通话",
                summary="客人在打电话",
                subjects=["客人B"],
                basis=["拿起电话"],
                relations=[("concurrent_with", 1)],
            ),
        ],
        [[(1, 1)], [(1, 1)], [(2, 1)]],
    )
    batch = assemble_judgement_batch(raw, fragment_count=len(fragments))
    derived = derive_judgements(batch, fragments, source_refs=("d1",), judged_at=JUDGED_AT)

    owner = next(item for item in derived if SUBJECT in item.subjects)
    # 对称闭包确实给主体那条补出了关系，而它的目标不会落盘。
    assert [kind for kind, _ in owner.relations] == ["concurrent_with"]
    visible = {item.judgement_id for item in derived if SUBJECT in item.subjects}
    assert all(target not in visible for _, target in owner.relations)

    pruned = without_unresolvable_relations(owner, visible)

    assert pruned.relations == ()
    assert pruned.judgement_id == owner.judgement_id


def test_pruning_keeps_relations_whose_target_survives() -> None:
    """只剪不可解析的那些；指向落盘判断或上下文判断的关系原样保留。"""

    fragments = (fragment(0, "洗手", participants=[SUBJECT]),)
    raw = wire([judgement(1, behavior="洗手", goal="清洁双手", summary="洗手", basis=["冲手"])], [[(1, 1)]])
    batch = assemble_judgement_batch(raw, fragment_count=1)
    derived = derive_judgements(batch, fragments, source_refs=("d1",), judged_at=JUDGED_AT)
    only = derived[0]

    assert without_unresolvable_relations(only, {only.judgement_id}) is only


def test_context_is_ordered_by_real_time_not_by_the_local_offset_string(tmp_path) -> None:
    """上下文按**时刻**排序，不能按 ``started_at`` 的字符串排。

    ``started_at`` 刻意保留本地偏移（时间事实是实际发生的本地时刻，折 UTC 会把东八区凌晨的
    行为掉到前一天），所以它的字符串序不是时间序。出行跨时区、或者一次 DST 切换带来的 1 小时偏移变化，
    就足以让两条判断的先后颠倒——而这个顺序直接决定模型看到的 C1..Cn 编号。
    """

    store = BehaviorJudgementStore(tmp_path)
    template = {key: None for key in _JUDGEMENT_KEYS}

    def record(identity: str, started_at: str, ready_at: str) -> dict[str, object]:
        payload = dict(template)
        payload.update(
            {
                "schema_version": "behavior_judgement_v1",
                "judgement_id": identity,
                "judged_at": ready_at,
                "evidence_ready_at": ready_at,
                "started_at": started_at,
                "last_observed_at": started_at,
                "observation_ids": [f"observation-{identity[:4]}"],
                "source_refs": ["delivery"],
                "subjects": [SUBJECT],
                "behavior": "行为",
                "goal": None,
                "summary": "摘要",
                "basis": [],
                "status": "completed",
                "status_basis": "observed",
                "relations": [],
                "fusion_version": "v",
                "prompt_version": "p",
            }
        )
        return payload

    # 东八区的这条真实更早（UTC 16:30），但字符串以 "2026-08-15" 开头，字符串序会把它排到后面。
    store.put_payload(record("b" * 64, "2026-08-15T00:30:00.000000+08:00", "2026-08-15T00:00:00.000000Z"))
    store.put_payload(record("a" * 64, "2026-08-14T20:00:00.000000-05:00", "2026-08-15T02:00:00.000000Z"))

    context = store.recent_before(
        datetime(2026, 8, 15, 3, tzinfo=timezone.utc), limit=8, lookback_seconds=86_400
    )

    moments = [
        datetime.fromisoformat(str(item["started_at"])).astimezone(timezone.utc)
        for item in context
    ]
    assert moments == sorted(moments)
    assert str(context[0]["started_at"]).endswith("+08:00")


def test_a_long_running_judgement_stays_inside_the_lookback_window() -> None:
    """回看下界按 ``last_observed_at``，不按 ``started_at``：做饭、看电影这类超过 lookback 的长行为，
    其后续段必须仍能看到自己的前半截，否则跨窗口 ``continues`` 断链（审计 NEW-4 复现）。"""

    import tempfile

    from behavior.fusion.store import BehaviorJudgementStore

    with tempfile.TemporaryDirectory(dir="/Users/gulf/.claude/jobs") as raw_root:
        store = BehaviorJudgementStore(raw_root)
        template = {key: None for key in _JUDGEMENT_KEYS}

        def record(identity: str, started: str, last: str) -> dict[str, object]:
            payload = dict(template)
            payload.update(
                {
                    "schema_version": "behavior_judgement_v1",
                    "judgement_id": identity,
                    "judged_at": last,
                    "evidence_ready_at": last,
                    "started_at": started,
                    "last_observed_at": last,
                    "observation_ids": [f"observation-{identity[:4]}"],
                    "source_refs": ["delivery"],
                    "subjects": [SUBJECT],
                    "behavior": "行为",
                    "goal": None,
                    "summary": "摘要",
                    "basis": [],
                    "status": "ongoing",
                    "status_basis": "observation_lost",
                    "relations": [],
                    "fusion_version": "v",
                    "prompt_version": "p",
                }
            )
            return payload

        cutoff = datetime(2026, 8, 15, 3, tzinfo=timezone.utc)
        # 做饭：两小时前开始（早于 1 小时回看下界），但最后观测就在本段之前一分钟
        store.put_payload(
            record("c" * 64, "2026-08-15T01:00:00.000000Z", "2026-08-15T02:59:00.000000Z")
        )
        # 洗手：完全落在窗口内
        store.put_payload(
            record("d" * 64, "2026-08-15T02:50:00.000000Z", "2026-08-15T02:55:00.000000Z")
        )
        # 昨天那条：最后观测也早于下界，应当被排除
        store.put_payload(
            record("e" * 64, "2026-08-14T01:00:00.000000Z", "2026-08-14T02:00:00.000000Z")
        )

        context = store.recent_before(cutoff, limit=8, lookback_seconds=3_600)

        assert [str(item["judgement_id"])[0] for item in context] == ["c", "d"]


def test_the_fusion_version_covers_the_schema_not_just_the_prompt() -> None:
    """改 schema 的字段描述必须改变版本号——否则两批语义不同的判断在数据上分辨不出来。

    schema 不只是形状声明：字段描述里承载着硬契约（goal 非 null 时 basis 必须至少写一条、
    interrupted 与 abandoned 的区别、关系目标的合法性），实测改这些描述会实打实地改变模型行为。
    版本一度只覆盖实现与提示词，于是改完描述、模型行为变了，版本却可以一字不动。

    用摘要而不是手写版本号，是为了**不可能忘记**。这条测试守的就是那个自动性。
    """

    fingerprint = canonical_digest(JUDGEMENT_FUSION_JSON_SCHEMA)[:12]
    assert f"schema{fingerprint}" in FUSION_VERSION
    assert FUSION_PROMPT_VERSION in FUSION_VERSION

    # 只改一个字段描述里的一个字，指纹就必须变。
    altered = deepcopy(JUDGEMENT_FUSION_JSON_SCHEMA)
    properties = altered["properties"]["judgements"]["items"]["properties"]
    properties["basis"]["description"] = properties["basis"]["description"] + "。"
    assert canonical_digest(altered)[:12] != fingerprint


def test_a_frame_shared_by_a_guest_and_an_unreadable_judgement_stays_out_of_scope() -> None:
    """tracked 只算主体可读判断的观测：v3 把没读懂判断并入 in_scope 后，这条口径不许跟着漂。"""

    guest = "客人C"
    shared = BehaviorObservation.create(
        observer_id="home-a/hall",
        occurred_at=MIDNIGHT + timedelta(seconds=12),
        available_at=MIDNIGHT + timedelta(seconds=13),
        modality="vision",
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
    batch = assemble_judgement_batch(raw, fragment_count=len(fragments))
    validate_judgement_batch(batch, fragments)
    items = derive_judgements(batch, fragments, source_refs=SOURCE_REFS, judged_at=JUDGED_AT)
    record = build_fusion_receipt(
        items,
        fragments,
        source_refs=SOURCE_REFS,
        prompt_version=PROMPT_VERSION,
        validation_attempts=1,
        primary_subject=SUBJECT,
    )
    # 这一帧被旁人读懂了、且不属于主体——必须留在 out_of_scope；同时它已被读懂，不计 unreadable。
    assert record.out_of_scope_observation_ids == (shared.observation_id,)
    assert record.unreadable_observation_ids == ()
