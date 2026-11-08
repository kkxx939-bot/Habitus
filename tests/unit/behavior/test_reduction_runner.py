"""归约 runner 端到端：真实存储进、真实行为树出。

模型触点（白天归类）用按名字配对的确定性假模型：判断原话与某个类名逐字相同就归那一类，否则进待定。
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta

import pytest

from habitus.behavior.fusion.receipt import build_fusion_receipt
from habitus.behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from habitus.behavior.fusion.store import BehaviorJudgementStore
from habitus.behavior.model import BehaviorKind
from habitus.behavior.observation import (
    BehaviorObservationBatch,
    BehaviorObservationEnvelope,
    BehaviorObservationStore,
)
from habitus.behavior.reduction import BehaviorReductionLedger, BehaviorReductionRunner  # noqa: F401
from habitus.behavior.reduction.failed import FailedChains
from habitus.behavior.reduction.refresh import RefreshRetries
from habitus.behavior.retry import RetryBacklogStore, RetryKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.tree.dirty_days import DirtyDayLog
from habitus.behavior.uri import BehaviorURI
from habitus.foundation.integrity import canonical_digest
from habitus.foundation.lifecycle import BackoffPolicy
from habitus.infrastructure.lifecycle import RetryLedger
from habitus.infrastructure.store.contracts import PathLock
from tests.lock_store import ProcessLocalLockStore
from tests.unit.behavior.kinds_fixtures import NameMatchingProvider, kind_components
from tests.unit.behavior.reduction_fixtures import (
    OBSERVATION_CONFIG,
    OBSERVER,
    SUBJECT,
    at,
    judgement_record,
    observation,
    record_id,
)
from tests.unit.kind_ids import kind_id

OBS_A = observation(18, "人走到水池边打开水龙头")
OBS_B = observation(40, "人打肥皂搓手")
OBS_C = observation(62, "人冲水擦干")
OBS_BLUR = observation(120, "画面模糊")

# 版本标签由写判断的那条 lane 给；这里用测试用的。
FUSION_PROMPT_VERSION = "behavior_test_rule_v1"


class Harness:
    """一套真实存储 + 可拨时钟的归约现场。"""

    def __init__(
        self,
        tmp_path,
        known_kinds: tuple[str, ...] = ("洗手",),
        semantic_refresher=None,
        seal_on_arrival: bool = False,
    ) -> None:
        self.now = at(7200)  # 默认拨到 lookback(1h) 之外
        self.observations = BehaviorObservationStore(tmp_path / "observations")
        self.judgements = BehaviorJudgementStore(tmp_path / "judgements")
        self.lock_store = ProcessLocalLockStore()
        self.receipts = BehaviorFusionReceiptStore(tmp_path / "receipts", lock_store=self.lock_store)
        self.tree = BehaviorTree(tmp_path / "tree")
        # 词表的文件就在树根，与树共存；known_kinds 写成第 1 版（会话 lane，按顺序编号 s-k0001…）
        self.provider = NameMatchingProvider()
        self.kinds, self.vocabulary_jobs, self.kind_store = kind_components(
            self.tree, self.lock_store, names=known_kinds, provider=self.provider
        )
        self.ledger = BehaviorReductionLedger(tmp_path / "reduction")
        self.runner = BehaviorReductionRunner(
            judgements=self.judgements,
            observations=self.observations,
            receipts=self.receipts,
            tree=self.tree,
            lock_store=self.lock_store,
            kinds=self.kinds,
            vocabulary_jobs=self.vocabulary_jobs,
            ledger=self.ledger,
            semantic_refresher=semantic_refresher,
            refresh_retries=RefreshRetries(
                RetryLedger(
                    tmp_path / "lifecycle",
                    owner="behavior-semantic-test",
                    policy=BackoffPolicy(base_seconds=60.0, cap_seconds=3600.0),
                    max_attempts=3,
                    clock=lambda: self.now,
                )
            ),
            failed_chains=FailedChains(
                self.judgements,
                ledger=self.ledger,
                clock=lambda: self.now,
                backlog=RetryBacklogStore(tmp_path, max_items=100),
            ),
            clock=lambda: self.now,
            dirty_days=DirtyDayLog(tmp_path / "reduction"),
            seal_on_arrival=seal_on_arrival,
        )

    def deliver(self, *observations_batch, seed: str = "delivery", fused: bool = True) -> str:
        """投递一批观测；``fused=True`` 时同时写一份覆盖它们的融合回执。

        封口前沿的判据是"未被任何回执覆盖的观测"——测试直接绕过融合层播种判断时，必须
        把观测标成已融合，否则它们会被当成待融合积压、把封口视界永远压住。
        """

        envelope = BehaviorObservationEnvelope.create(
            observer_id=OBSERVER,
            protocol="habitus_behavior_observation_v1",
            batch=BehaviorObservationBatch(observer_id=OBSERVER, observations=tuple(observations_batch)),
            delivery_id=canonical_digest(seed),
            recorded_at=at(600),
            config=OBSERVATION_CONFIG,
        )
        self.observations.put(envelope)
        if fused:
            stored = self.receipts.put(
                build_fusion_receipt(
                    (),
                    tuple(observations_batch),
                    source_refs=(envelope.source_id,),
                    prompt_version=FUSION_PROMPT_VERSION,
                    validation_attempts=1,
                    primary_subject=SUBJECT,
                    judged_at=at(600),
                )
            )
            # 覆盖信息由索引回答（回执只服务作业期幂等），融合 runner 在落回执的同一步写它。
            self.runner.coverage.record(stored)
        return envelope.source_id


def seed_wash(harness: Harness, source: str) -> None:
    harness.judgements.put_payload(
        judgement_record(
            "wash",
            behavior="洗手",
            started_at=at(18),
            last_observed_at=at(62),
            evidence_ready_at=at(64),
            observation_ids=(OBS_A.observation_id, OBS_B.observation_id, OBS_C.observation_id),
            source_refs=(source,),
            goal="s-k0002",
            summary="在水池边洗手；冲水擦干结束",
            status="completed",
            status_basis="observed",
            basis=(
                ("打开水龙头打肥皂搓手", (OBS_A.observation_id, OBS_B.observation_id)),
                ("冲水擦干", (OBS_C.observation_id,)),
            ),
        )
    )


def test_full_pipeline_reduces_a_chain_and_a_gap_into_the_tree(tmp_path) -> None:
    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C, OBS_BLUR)
    seed_wash(harness, source)
    harness.judgements.put_payload(
        judgement_record(
            "blur",
            behavior=None,
            started_at=at(120),
            last_observed_at=at(120),
            evidence_ready_at=at(122),
            observation_ids=(OBS_BLUR.observation_id,),
            source_refs=(source,),
        )
    )

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    assert report.published_gaps == 1
    occurrences = harness.tree.list_addresses(BehaviorKind.OCCURRENCE)
    # 改树改到的日子记进脏日日志，按天缓存的序列据此重读（26 号清单 6.7）
    assert {address.occurred_on for address in occurrences} <= harness.runner.dirty_days.after(0)[1]
    gaps = harness.tree.list_addresses(BehaviorKind.GAP)
    assert [address.name for address in occurrences] == ["洗手"]
    assert [address.name for address in gaps] == ["没读懂"]

    document = harness.tree.read(occurrences[0])
    assert document.fields["started_at"] == at(18).isoformat(timespec="microseconds")
    assert document.fields["started_at"].endswith("+08:00")
    assert document.fields["kind_token"] == "s-k0001"  # known_kinds 的第一类 = 洗手
    assert document.fields["status"] == "completed"
    assert document.fields["summary"] == "在水池边洗手；冲水擦干结束"
    assert document.fields["onset_available_at"] == at(64).isoformat(timespec="microseconds")
    ledger_by_uri = {entry.uri: entry for entry in harness.ledger.load()}
    assert document.fields["chain_digest"] == ledger_by_uri[str(BehaviorURI.from_address(occurrences[0]))].chain_digest

    gap_document = harness.tree.read(gaps[0])
    assert gap_document.fields["started_at"] == gap_document.fields["ended_at"]

    # 消费账本闭环：全部判断都已入账
    consumed = harness.ledger.consumed_judgement_ids()
    assert {record_id("wash"), record_id("blur")} <= consumed


def test_a_round_reads_the_consumption_ledger_once_after_expiry(tmp_path, monkeypatch) -> None:
    """26 号清单 6.3：过期自己读一遍，之后拼链与物化共用一份——不再各读一遍全量。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C, OBS_BLUR)
    seed_wash(harness, source)
    original = harness.ledger.load
    calls: list[str] = []

    def counted():  # type: ignore[no-untyped-def]
        calls.append("load")
        return original()

    monkeypatch.setattr(harness.ledger, "load", counted)
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1
    assert len(calls) == 2  # 过期一遍 + 这一轮共用的一遍


def test_the_reminded_flag_is_asked_from_the_injected_port_before_staging(tmp_path) -> None:
    """预测层接上之后（第 0 步）：物化 payload 时按（类, 开始时刻）问一次端口，答案盖在 occurrence 上。"""

    harness = Harness(tmp_path)
    asked: list[tuple[str, datetime]] = []

    def lookup(kind_token: str, started_at: datetime) -> bool:
        asked.append((kind_token, started_at))
        return True

    harness.runner.attach_reminder_lookup(lookup)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1
    (address,) = harness.tree.list_addresses(BehaviorKind.OCCURRENCE)
    assert harness.tree.read(address).fields["reminded"] is True
    assert asked and asked[0] == ("s-k0001", at(18)) and all(item == asked[0] for item in asked)


def test_second_sweep_is_a_no_op(tmp_path) -> None:
    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    asyncio.run(harness.runner.run_once())

    report = asyncio.run(harness.runner.run_once())

    assert report.published_documents == 0
    assert report.replayed_documents == 0
    assert len(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)) == 1


def test_chains_inside_the_window_stay_pending(tmp_path) -> None:
    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    harness.now = at(600)  # 判断 evidence(64s) 仍在 1h 窗口内

    report = asyncio.run(harness.runner.run_once())

    assert report.published_documents == 0
    assert report.chains_pending == 1
    assert harness.tree.list_addresses(BehaviorKind.OCCURRENCE) == ()


@pytest.mark.parametrize(("seal_on_arrival", "published"), [(True, 1), (False, 0)])
def test_a_session_record_lands_as_soon_as_it_arrives_while_a_physical_one_waits_out_the_window(
    tmp_path, seal_on_arrival, published
) -> None:  # type: ignore[no-untyped-def]
    """裁定 31：会话 lane 的记录一送到就封口落树；物理 lane 仍等满 lookback。定稿日两支都仍看 lookback。"""

    harness = Harness(tmp_path, seal_on_arrival=seal_on_arrival)
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    harness.now = at(64 + 60)  # 证据齐备之后 1 分钟

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == published
    assert len(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)) == published
    end_of_day = at(16_200)  # 16 日本地日终（at(0) 是 19:30 CST）
    harness.now = end_of_day + timedelta(minutes=1)
    asyncio.run(harness.runner.run_once())
    assert harness.runner.closed_days() == ()  # 视界还没过那天的日终
    harness.now = end_of_day + timedelta(seconds=3_600 + 60)
    asyncio.run(harness.runner.run_once())
    assert harness.runner.closed_days() == (date(2026, 8, 16),)


def test_an_unfused_observation_blocks_sealing_despite_the_wall_clock(tmp_path) -> None:
    """停机补算：墙钟早已出窗，但还有观测没被任何融合回执覆盖——它将来会切成
    cutoff 是历史时刻的段、仍可能引用旧判断，不许封口。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    backlog = observation(300, "人在客厅走动")
    harness.deliver(backlog, seed="backlog", fused=False)  # 已交付、未融合
    harness.now = at(7200)

    report = asyncio.run(harness.runner.run_once())

    # 前沿 cutoff = 302s（backlog 的 available_at），视界 = 302s - 3600s < 判断 evidence
    assert report.published_documents == 0
    assert report.chains_pending == 1

    # 该观测融合完成（回执覆盖）后，同一墙钟下立即可封。
    harness.runner.coverage.record(
        harness.receipts.put(
            build_fusion_receipt(
                (),
                (backlog,),
                source_refs=("f" * 64,),
                prompt_version=FUSION_PROMPT_VERSION,
                validation_attempts=1,
                primary_subject=SUBJECT,
                judged_at=at(700),
            )
        )
    )
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1


def test_crash_after_stage_replays_byte_identically(tmp_path) -> None:
    """死规则⑤：stage 之后崩溃，重放逐字节落盘——不重算、不撞车。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)

    original_publish = harness.runner._publish_checkpoint
    harness.runner._publish_checkpoint = lambda checkpoint, guard: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("crash after stage")
    )
    with pytest.raises(RuntimeError, match="crash after stage"):
        asyncio.run(harness.runner.run_once())
    assert harness.tree.list_addresses(BehaviorKind.OCCURRENCE) == ()  # 没落盘，检查点在

    harness.runner._publish_checkpoint = original_publish  # type: ignore[method-assign]
    report = asyncio.run(harness.runner.run_once())

    assert report.replayed_documents == 1
    assert len(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)) == 1
    document = harness.tree.read(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)[0])
    assert document.fields["summary"] == "在水池边洗手；冲水擦干结束"


def test_same_address_collision_gets_a_deterministic_suffix(tmp_path) -> None:
    """写入层保命阀（死规则②）：融合层漏网的同址双链，后序加序号后缀、原始名照存。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B)
    for seed, obs in (("first", OBS_A), ("second", OBS_B)):
        harness.judgements.put_payload(
            judgement_record(
                seed,
                behavior="洗手",
                started_at=at(18),
                last_observed_at=at(40),
                evidence_ready_at=at(42 if seed == "first" else 44),
                observation_ids=(obs.observation_id,),
                source_refs=(source,),
                summary=f"{seed} 洗手",
            )
        )

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 2
    names = sorted(address.name for address in harness.tree.list_addresses(BehaviorKind.OCCURRENCE))
    assert names == ["洗手", "洗手-2"]
    suffixed = next(
        harness.tree.read(address)
        for address in harness.tree.list_addresses(BehaviorKind.OCCURRENCE)
        if address.name == "洗手-2"
    )
    assert suffixed.fields["original_name"] == "洗手"
    assert suffixed.fields["summary"] == "second 洗手"  # 后序 = evidence_ready 更晚的那条


def test_same_second_gaps_merge_and_cross_batch_collision_consumes_by_reference(
    tmp_path,
) -> None:
    harness = Harness(tmp_path, known_kinds=())
    blur_b = observation(120, "另一路模糊画面")
    source = harness.deliver(OBS_BLUR, blur_b)
    for seed, obs in (("blur", OBS_BLUR), ("blur-b", blur_b)):
        harness.judgements.put_payload(
            judgement_record(
                seed,
                behavior=None,
                started_at=at(120),
                last_observed_at=at(120 if seed == "blur" else 150),
                evidence_ready_at=at(152),
                observation_ids=(obs.observation_id,),
                source_refs=(source,),
            )
        )

    report = asyncio.run(harness.runner.run_once())

    # 批内同秒空白机械合并成一个节点：终点取最大、溯源并集。
    assert report.published_gaps == 1
    gap = harness.tree.read(harness.tree.list_addresses(BehaviorKind.GAP)[0])
    assert gap.fields["ended_at"] == at(150).isoformat(timespec="microseconds")
    gap_entry = next(entry for entry in harness.ledger.load() if entry.kind == "gap")
    assert gap.fields["chain_digest"] == gap_entry.chain_digest
    assert set(gap_entry.judgement_ids) == {record_id("blur"), record_id("blur-b")}

    # 跨批撞已落盘空白：记账指向既有节点，不重复、不丢账、不卡队列。
    blur_c = observation(121, "第三路模糊画面")
    source_c = harness.deliver(blur_c, seed="delivery-c")
    harness.judgements.put_payload(
        judgement_record(
            "blur-c",
            behavior=None,
            started_at=at(120),
            last_observed_at=at(121),
            evidence_ready_at=at(600),
            observation_ids=(blur_c.observation_id,),
            source_refs=(source_c,),
        )
    )
    harness.now = at(10_000)
    second = asyncio.run(harness.runner.run_once())

    assert second.published_gaps == 0
    assert any("by reference" in note for note in second.dropped_edges)
    assert record_id("blur-c") in harness.ledger.consumed_judgement_ids()
    assert len(harness.tree.list_addresses(BehaviorKind.GAP)) == 1


def test_a_corrupt_record_is_quarantined_without_stalling_the_sweep(tmp_path) -> None:
    """单条坏记录隔离降级：留信号、不许瘫痪整轮归约；原记录从存储撤下，不再每轮重报。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    harness.judgements.put_payload(
        {"judgement_id": "f" * 64, "behavior": 123, "started_at": "not-a-time"}
        | {
            k: v
            for k, v in judgement_record(
                "corrupt-base",
                behavior="占位",
                started_at=at(1),
                last_observed_at=at(2),
                evidence_ready_at=at(3),
                observation_ids=(OBS_A.observation_id,),
                source_refs=(source,),
                summary="x",
            ).items()
            if k not in {"judgement_id", "behavior", "started_at"}
        }
    )

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1  # 好链照常落
    assert any("quarantined" in note for note in report.dropped_edges)
    # 读不出的原记录从存储撤下（2.4.3）：读不出时间、没法重试，不进每晚重试的堆（裁定 67 第 10 条）；下一轮不再重报
    assert harness.judgements.read("f" * 64) is None
    assert harness.runner.failed_chains.backlog.count() == 0
    later = asyncio.run(harness.runner.run_once())
    assert not any("quarantined" in note for note in later.dropped_edges)


def test_a_chain_with_an_unaddressable_name_is_isolated_and_becomes_an_unreadable_gap(tmp_path) -> None:
    """绕过融合守卫写入的坏行为名：按链隔离，不吞整轮、不写检查点炸弹；原判断进每晚重试的堆、换成一条「没读懂」判断，
    之后归约成空白，不再每轮重报（26 号清单 2.4.2；裁定 67）。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    harness.judgements.put_payload(
        judgement_record(
            "bad-name",
            behavior=" 前导空格",
            started_at=at(300),
            last_observed_at=at(310),
            evidence_ready_at=at(312),
            observation_ids=(OBS_C.observation_id,),
            source_refs=(source,),
            summary="名字带前导空格",
        )
    )

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    assert any("set aside" in note for note in report.dropped_edges)
    assert harness.judgements.read(record_id("bad-name")) is None
    (item,) = harness.runner.failed_chains.backlog.snapshot().items
    assert item.kind is RetryKind.CHAIN
    assert [record["judgement_id"] for record in item.payload["records"]] == [record_id("bad-name")]
    later = asyncio.run(harness.runner.run_once())
    assert later.published_gaps == 1  # 顶替它的「没读懂」判断落成空白
    assert not any("bad-name" in note or "set aside" in note for note in later.dropped_edges)

    # 当晚重试：原判断放回存储再归约一次，同一份代码照样失败——撤下就完，不再造第二条「没读懂」、不再进堆
    harness.runner.failed_chains.backlog.discard(item.item_id for item in [item])
    for record in item.payload["records"]:
        harness.judgements.put_payload(record)
    retried = asyncio.run(harness.runner.run_once())
    assert any("failed again on retry" in note for note in retried.dropped_edges)
    assert retried.published_gaps == 0
    assert harness.judgements.read(record_id("bad-name")) is None
    assert harness.runner.failed_chains.backlog.count() == 0


def test_an_invalid_payload_is_dropped_before_the_checkpoint(tmp_path) -> None:
    """stage 末端干跑校验：会在 publish 期炸的文档不进检查点，降级成带信号的跳过。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C, OBS_BLUR)
    seed_wash(harness, source)
    # 终点早于起点的没读懂段：解析层不查跨字段，schema 校验会拒——必须在 stage 前被拦下。
    harness.judgements.put_payload(
        judgement_record(
            "backwards-gap",
            behavior=None,
            started_at=at(120),
            last_observed_at=at(90),
            evidence_ready_at=at(122),
            observation_ids=(OBS_BLUR.observation_id,),
            source_refs=(source,),
        )
    )

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    assert report.published_gaps == 0
    assert any("fails validation before stage" in note for note in report.dropped_edges)
    assert harness.tree.list_addresses(BehaviorKind.GAP) == ()


def test_the_checkpoint_byte_bound_fails_before_stage(tmp_path, monkeypatch) -> None:
    """检查点写入与重放共用同一上界：超限在 stage 前失败，可重试，不留读不回的检查点。"""

    import habitus.behavior.reduction.checkpoint as checkpoint_module

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    monkeypatch.setattr(checkpoint_module, "_MAX_CHECKPOINT_BYTES", 64)
    with pytest.raises(Exception, match="byte bound"):
        asyncio.run(harness.runner.run_once())
    assert harness.tree.list_addresses(BehaviorKind.OCCURRENCE) == ()
    assert not (tmp_path / "reduction" / "staged.json").exists()

    monkeypatch.setattr(checkpoint_module, "_MAX_CHECKPOINT_BYTES", 67_108_864)
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1


def test_concurrent_sweeps_are_serialised_by_the_root_lock(tmp_path) -> None:
    """单写入方是机械保证：同一 behavior-root 的第二个归约进程拿不到锁即失败。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    lock = PathLock(harness.runner.lock_store)
    with lock.acquire(harness.runner._sweep_lock_key, ttl_seconds=30):
        with pytest.raises(TimeoutError):
            asyncio.run(harness.runner.run_once())
    report = asyncio.run(harness.runner.run_once())  # 锁释放后照常
    assert report.published_occurrences == 1


def test_sweep_refreshes_day_summaries_after_publishing(tmp_path) -> None:
    """归约落盘后在同一把 sweep 锁内刷新 L0/L1；树没变的下一轮零模型调用。"""

    from habitus.behavior.model import BehaviorDirectory, BehaviorLevel
    from habitus.behavior.semantic import BehaviorSemanticRefresher
    from tests.unit.behavior.test_semantic_layers import ScriptedOverviewGenerator

    generator = ScriptedOverviewGenerator()
    tree = BehaviorTree(tmp_path / "tree")
    harness = Harness(
        tmp_path,
        semantic_refresher=BehaviorSemanticRefresher(tree, generator),
    )
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    assert not any("semantic refresh failed" in note for note in report.dropped_edges)
    day_dir = BehaviorDirectory.occurrences(2026, 8, 16)
    assert harness.tree.layer_exists(day_dir, BehaviorLevel.OVERVIEW)
    assert harness.tree.layer_exists(BehaviorDirectory.occurrences(2026), BehaviorLevel.ABSTRACT)
    calls = len(generator.snapshots)

    second = asyncio.run(harness.runner.run_once())
    assert second.published_documents == 0
    assert len(generator.snapshots) == calls  # 树没变——digest 短路，零模型调用


def test_a_day_is_closed_only_after_its_chains_seal_and_its_local_day_ends(tmp_path) -> None:
    """定稿日是派生层（语义树、预测层的结算）处理历史的前提：树变了的日子先记为开放，链都封口且封口视界
    过了那天的本地结束才记为定稿；之后的补发不改变这个事实。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    assert harness.runner.closed_days() == ()  # 21:30 CST：那天还没结束

    harness.now = at(6 * 3600)  # 次日 01:30 CST，封口视界 00:30 已过 16 日本地结束
    asyncio.run(harness.runner.run_once())

    assert harness.runner.closed_days() == (date(2026, 8, 16),)
    closure = json.loads((tmp_path / "reduction" / "day_closure.json").read_text())
    assert closure["open"] == [] and list(closure["closed"]) == ["2026-08-16"]

    # 定稿后再来一条同一天的链：照常落树，定稿事实不变
    later = harness.deliver(OBS_A, OBS_B, OBS_C, seed="late")
    seed_wash(harness, later)
    asyncio.run(harness.runner.run_once())
    assert harness.runner.closed_days() == (date(2026, 8, 16),)


def test_failed_semantic_refresh_is_retried_from_the_pending_set(tmp_path) -> None:
    """刷新失败降级为带日期的信号；那一天留在待刷新集合里，下轮补刷；检查点不被扣住。"""

    from habitus.behavior.semantic import BehaviorSemanticRefresher
    from tests.unit.behavior.test_semantic_layers import ScriptedOverviewGenerator

    class FlakyGenerator(ScriptedOverviewGenerator):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        async def generate(self, snapshot):
            if self.fail:
                raise RuntimeError("llm down")
            return await super().generate(snapshot)

    generator = FlakyGenerator()
    tree = BehaviorTree(tmp_path / "tree")
    harness = Harness(tmp_path, semantic_refresher=BehaviorSemanticRefresher(tree, generator))
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1  # 归约不被刷新失败阻塞
    assert any("semantic refresh failed for [2026-08-16]" in n for n in report.dropped_edges)
    # 检查点发布完即清（不再被刷新失败扣住——那会让 merge 之后的重放撞出永久冲突）；
    # 待刷新的日子记在独立的小文件里，下轮逐日重试。
    assert not (tmp_path / "reduction" / "staged.json").exists()
    import json as _json

    assert _json.loads((tmp_path / "reduction" / "refresh_pending.json").read_text()) == ["2026-08-16"]

    generator.fail = False
    waiting = asyncio.run(harness.runner.run_once())  # 退避期内：这一天留在集合里，这一轮不试
    assert not any("semantic refresh failed" in n for n in waiting.dropped_edges)
    assert _json.loads((tmp_path / "reduction" / "refresh_pending.json").read_text()) == ["2026-08-16"]
    harness.now = harness.now + timedelta(seconds=61)
    second = asyncio.run(harness.runner.run_once())

    assert second.replayed_documents == 0  # 没有检查点可重放；补刷来自待刷新集合
    from habitus.behavior.model import BehaviorDirectory, BehaviorLevel

    assert harness.tree.layer_exists(BehaviorDirectory.occurrences(2026, 8, 16), BehaviorLevel.OVERVIEW)
    assert not (tmp_path / "reduction" / "refresh_pending.json").exists()


def test_an_oversized_overview_is_given_up_after_one_model_call(tmp_path) -> None:
    """超限是确定性失败：第一次就放弃、移出待刷新集合，之后几轮不再调模型（2.4.1）。"""

    from habitus.behavior.semantic import BehaviorOverviewLimitError, BehaviorSemanticRefresher
    from tests.unit.behavior.test_semantic_layers import ScriptedOverviewGenerator

    class TooLong(ScriptedOverviewGenerator):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate(self, snapshot):
            self.calls += 1
            raise BehaviorOverviewLimitError("rendered behavior overview exceeds its configured character bound")

    generator = TooLong()
    harness = Harness(
        tmp_path, semantic_refresher=BehaviorSemanticRefresher(BehaviorTree(tmp_path / "tree"), generator)
    )
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))

    first = asyncio.run(harness.runner.run_once())
    assert any("semantic refresh given up for [2026-08-16]" in n for n in first.dropped_edges)
    assert not (tmp_path / "reduction" / "refresh_pending.json").exists()
    calls = generator.calls
    for _ in range(3):
        harness.now = harness.now + timedelta(hours=2)
        asyncio.run(harness.runner.run_once())
    assert generator.calls == calls


def test_run_once_works_with_the_production_sqlite_lock_store(tmp_path) -> None:
    """接缝测试：生产锁实现（SQLite）下整轮 sweep 必须能跑通——租约互斥 + 短围栏，
    不许出现"sweep 事务压住 writer 文档锁"的自死锁（评审在旧实现下实测过每轮必败）。"""

    from habitus.infrastructure.store.sqlite.lock_store import SQLiteLockStore

    lock_store = SQLiteLockStore(tmp_path / "locks.sqlite3")
    harness = Harness(tmp_path)
    # 同一棵树上的词表已由 Harness 写好第 1 版；这里只换锁实现（词表的迁移与归约必须共用同一把锁库）。
    kinds, jobs, _ = kind_components(harness.tree, lock_store, names=(), provider=harness.provider)
    harness.runner = BehaviorReductionRunner(
        judgements=harness.judgements,
        observations=harness.observations,
        receipts=harness.receipts,
        tree=harness.tree,
        lock_store=lock_store,
        kinds=kinds,
        vocabulary_jobs=jobs,
        ledger=harness.ledger,
        failed_chains=harness.runner.failed_chains,
        clock=lambda: harness.now,
    )
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    assert len(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)) == 1


def test_tree_replace_only_allows_the_kind_token_to_change(tmp_path) -> None:
    """树是 add-only 的；replace 是唯一改写通道，且只为 kind_token 重打而开。"""

    from dataclasses import replace as dc_replace

    from habitus.behavior import BehaviorTree, BehaviorTreeConflictError
    from habitus.behavior.tree.writer import BehaviorDocumentWriter
    from tests.unit.behavior.tree_payloads import local, occurrence_payload

    tree = BehaviorTree(tmp_path / "tree")
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    published = writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    bumped = dc_replace(published.metadata, revision=2)
    # 改 kind_token：允许
    ok = tree.document_codec.build(published.kind, {**published.fields, "kind_token": "s-k0002"}, metadata=bumped)
    tree.replace(ok)
    assert tree.read(published.address).fields["kind_token"] == "s-k0002"
    # 改别的字段：拒绝
    bad = tree.document_codec.build(
        published.kind,
        {**published.fields, "kind_token": "s-k0002", "summary": "改了正文"},
        metadata=dc_replace(bumped, revision=3),
    )
    with pytest.raises(BehaviorTreeConflictError, match="only change kind_token"):
        tree.replace(bad)
    # 修订号不连续：拒绝
    stale = tree.document_codec.build(
        published.kind,
        {**published.fields, "kind_token": kind_id("洗手")},
        metadata=published.metadata,
    )
    with pytest.raises(BehaviorTreeConflictError, match="revision"):
        tree.replace(stale)


def test_writer_restamp_is_idempotent_and_bumps_revision(tmp_path) -> None:
    from habitus.behavior import BehaviorTree
    from habitus.behavior.tree.writer import BehaviorDocumentWriter
    from tests.unit.behavior.tree_payloads import local, occurrence_payload

    tree = BehaviorTree(tmp_path / "tree")
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    published = writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    restamped = writer.restamp_kind_token(published.address, "s-k0002")
    assert restamped.fields["kind_token"] == "s-k0002" and restamped.metadata.revision == 2
    assert restamped.fields["name"] == published.fields["name"]
    again = writer.restamp_kind_token(published.address, "s-k0002")  # 同值：不动
    assert again.metadata.revision == 2
    assert tree.read(published.address) == restamped


def test_vocabulary_jobs_refuse_while_a_checkpoint_is_pending(tmp_path) -> None:
    """词表整理活不许压在悬挂的检查点上（重打 token 后重放会撞永久冲突）。"""

    import json as _json

    from habitus.behavior.reduction import BehaviorReductionBusyError, BehaviorReductionError

    harness = Harness(tmp_path)
    (tmp_path / "reduction").mkdir(exist_ok=True)
    staged = tmp_path / "reduction" / "staged.json"
    staged.write_text(_json.dumps({"staged_at": harness.now.isoformat(), "documents": []}), encoding="utf-8")
    with pytest.raises(BehaviorReductionBusyError, match="checkpoint is pending"):
        asyncio.run(harness.runner.grow_vocabulary())
    with pytest.raises(BehaviorReductionBusyError, match="checkpoint is pending"):
        asyncio.run(harness.runner.revise_vocabulary())
    # 损坏的检查点：三条路都堵死是死锁，要给出"手动删除"的指引而不是"先跑 sweep"
    staged.write_text("garbage", encoding="utf-8")
    with pytest.raises(BehaviorReductionError, match="delete it manually"):
        asyncio.run(harness.runner.revise_vocabulary())


def test_an_address_already_on_the_tree_is_disambiguated_without_a_ledger_entry(tmp_path) -> None:
    """账本按窗口过期、树不会：树上已有同址文档时照样消歧成 -2，而不是 publish 硬冲突卡死检查点。"""

    from habitus.behavior.tree.writer import BehaviorDocumentWriter
    from tests.unit.behavior.tree_payloads import local, occurrence_payload

    harness = Harness(tmp_path)
    writer = BehaviorDocumentWriter(harness.tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    writer.publish(
        BehaviorKind.OCCURRENCE,
        occurrence_payload(name="洗手", occurred_on=at(18).date(), started_at=at(18), last_observed_at=at(40)),
    )
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1
    names = sorted(document.fields["name"] for document in harness.tree.iter_documents(BehaviorKind.OCCURRENCE))
    assert names == ["洗手", "洗手-2"]
    assert not (tmp_path / "reduction" / "staged.json").exists()
    # 消歧记录 = 已知重复：词表整理与事件序列都不计（同一口径）


def test_coverage_survives_the_window_while_its_delivery_is_still_stored(tmp_path) -> None:
    """覆盖记录以观测释放为过期前提：交付还在，"已融合"就还答得出来——不重入队、不拖前沿；
    释放后下一轮才过期。下游停机再久也不丢数据（不再按送达时间把观测当作已覆盖）。"""

    from datetime import timedelta

    harness = Harness(tmp_path)
    orphan = harness.deliver(OBS_A, seed="orphan", fused=True)  # 有回执、无判断
    later = harness.now + timedelta(days=10)
    ids = {OBS_A.observation_id}
    harness.runner.coverage.expire(later, retain=frozenset(ids))
    assert ids <= harness.runner.coverage.covered_observation_ids(harness.now)  # 交付在，覆盖留
    harness.observations.discard(orphan)
    harness.runner.coverage.expire(later, retain=frozenset())
    assert not (ids & harness.runner.coverage.covered_observation_ids(harness.now))  # 释放后才过期


def test_an_old_unfused_observation_still_pins_the_frontier(tmp_path) -> None:
    """从未融合的观测无论多旧都是待融合：它照常拖住封口前沿、照常入队——不按送达时间丢弃。"""

    from tests.unit.behavior.reduction_fixtures import observation

    harness = Harness(tmp_path)
    old = observation(-10 * 86_400, "十天前的旧观测")
    harness.deliver(old, seed="old", fused=False)
    assert harness.runner._frontier_cutoff() == old.available_at


def test_covered_deliveries_without_judgements_are_released(tmp_path) -> None:
    """全段旁人/无归属、或重复投递的交付：有回执覆盖、无判断引用 → 一轮 sweep 后释放，不等 7 天重入队。"""

    harness = Harness(tmp_path)
    orphan = harness.deliver(OBS_A, seed="orphan", fused=True)  # 回执覆盖但没有任何判断
    assert harness.observations.read(orphan) is not None
    asyncio.run(harness.runner.run_once())
    assert harness.observations.read(orphan) is None


def test_reduction_recovers_after_downtime_longer_than_the_coverage_window(tmp_path) -> None:
    """归约停机超过覆盖窗口后照常开工：覆盖记录留得下也**读得回**，前沿不被钉死、交付照常释放。

    读取侧曾同样按窗口过滤，于是"留得下、读不回"形成死锁——停机 > 7 天后 published 0 / pending 1、
    封口视界永久钉在过去、已融合的交付被重新入队重跑模型（审计 NEW-1，无崩溃即可复现）。
    """

    from datetime import timedelta

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    harness.now = harness.now + timedelta(days=8)  # 归约停机八天后首次开跑

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1 and report.chains_pending == 0
    assert harness.runner._frontier_cutoff() is None
    assert harness.observations.list() == ()  # 交付照常释放，不会被重新入队


def test_a_published_delivery_is_released_and_never_re_enqueued_after_the_window(tmp_path) -> None:
    """发布之后再跨过窗口：交付已释放、覆盖记录随之过期，入队扫描不会把它当成未融合重跑。"""

    from datetime import timedelta

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    assert asyncio.run(harness.runner.run_once()).published_occurrences == 1
    assert harness.observations.list() == ()

    harness.now = harness.now + timedelta(days=8)
    again = asyncio.run(harness.runner.run_once())

    assert again.published_occurrences == 0 and again.chains_pending == 0
    assert harness.runner._frontier_cutoff() is None


# ── 词表：白天归类的三种结局落到树与待定池 ───────────────────────────────────────────


def test_an_unknown_behaviour_is_published_as_pending_and_enters_the_pool_once(tmp_path) -> None:
    harness = Harness(tmp_path, known_kinds=("擦桌子",))
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    report = asyncio.run(harness.runner.run_once())
    document = harness.tree.read(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)[0])
    assert document.fields["kind_token"] == "s-待定"
    pool = harness.kind_store.read_pending().pool
    (entry,) = pool.entries.values()
    assert entry.occurrence == str(BehaviorURI.from_address(document.address)) and entry.proposed == "洗手"
    assert "在水池边洗手" in entry.content and "打开水龙头打肥皂搓手" in entry.content  # 整条链的内容
    assert any(note.startswith("kind_pending_added 1") for note in report.kind_signals)
    assert harness.kinds.record_pending(list(pool.entries.values())).added == 0  # 重放同一批不多算


def test_a_non_event_is_published_with_the_marker_and_a_signal_for_fusion(tmp_path) -> None:
    harness = Harness(tmp_path)
    harness.provider.not_events = frozenset({"洗手"})
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    report = asyncio.run(harness.runner.run_once())
    document = harness.tree.read(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)[0])
    assert document.fields["kind_token"] == "s-非事件"
    assert any(note.startswith("kind_not_event") for note in report.kind_signals)
    assert harness.kind_store.read_pending().pool.entries == {}


def test_the_first_sweep_on_an_empty_vocabulary_writes_no_classes(tmp_path) -> None:
    """词表从空开始（裁定 29：没有预置清单）：第一次归约不写任何类；记录照样交白天归类判是不是一件事、给提议名，进待定池，
    类等每晚新增按复现从池里长出来。"""

    harness = Harness(tmp_path, known_kinds=())
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    asyncio.run(harness.runner.run_once())
    assert harness.kind_store.read().version == 0 and not (tmp_path / "tree" / "kinds.md").exists()
    assert harness.provider.calls == 1 and "还没有类" in harness.provider.prompts[0]
    (entry,) = harness.kind_store.read_pending().pool.entries.values()
    assert entry.proposed == "洗手"
    # 每晚新增与定期拆改都不在 sweep 里跑（夜里那条线上，裁定 94 第 8、9 条）：只问了那一次白天归类
    assert harness.kind_store.read_jobs().revision_at is None


def test_a_replayed_checkpoint_is_cleared_before_the_vocabulary_migration_resumes(tmp_path) -> None:
    """重放落树之后立刻清检查点：之后续做的词表迁移可能重打这些条目，检查点留着就会让下一轮重放永久冲突。"""

    harness = Harness(tmp_path)
    (tmp_path / "reduction").mkdir(exist_ok=True)
    checkpoint = tmp_path / "reduction" / "staged.json"
    checkpoint.write_text(json.dumps({"staged_at": harness.now.isoformat(), "documents": []}), encoding="utf-8")
    seen: list[bool] = []
    original = harness.vocabulary_jobs.migrator.resume

    def watching(*, checkpoint):  # type: ignore[no-untyped-def]
        seen.append((tmp_path / "reduction" / "staged.json").exists())
        return original(checkpoint=checkpoint)

    harness.vocabulary_jobs.migrator.resume = watching  # type: ignore[method-assign]
    asyncio.run(harness.runner.run_once())
    assert seen == [False]


def test_the_day_of_an_isolated_chain_waits_for_its_unreadable_gap_before_closing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """stage 里确定性失败的链换成「没读懂」判断，下一轮才归约成空白：这一轮不许先把那一天定稿，否则派生层把那一段
    读成"没做"（L10.5）。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    asyncio.run(harness.runner.run_once())
    late = harness.deliver(OBS_C, seed="late")
    harness.judgements.put_payload(
        judgement_record(
            "bad-name",
            behavior=" 前导空格",
            started_at=at(300),
            last_observed_at=at(310),
            evidence_ready_at=at(312),
            observation_ids=(OBS_C.observation_id,),
            source_refs=(late,),
            summary="名字不可寻址",
        )
    )
    harness.now = at(6 * 3600)

    first = asyncio.run(harness.runner.run_once())
    assert first.published_gaps == 0
    assert date(2026, 8, 16) not in harness.runner.closed_days()

    second = asyncio.run(harness.runner.run_once())
    assert second.published_gaps == 1
    assert date(2026, 8, 16) in harness.runner.closed_days()


def test_the_consumption_ledger_is_read_once_per_round_after_it_expired(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """6.3：消费账本一轮只读一次、在过期之后读——反过来读到的是刚过期删掉的条目（L10.14 补测：原先把读挪回过期之前，
    测试照样全过）。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    order: list[str] = []
    load, expire = harness.ledger.load, harness.ledger.expire

    def loading():  # type: ignore[no-untyped-def]
        order.append("load")
        return load()

    def expiring(before):  # type: ignore[no-untyped-def]
        removed = expire(before)  # 过期自己也要读一遍账本，那一次不算
        order.clear()
        order.append("expired")
        return removed

    monkeypatch.setattr(harness.ledger, "load", loading)
    monkeypatch.setattr(harness.ledger, "expire", expiring)

    asyncio.run(harness.runner.run_once())

    assert order == ["expired", "load"]


@pytest.mark.parametrize("summary", ["给全家👨‍👩‍👧洗手", "洗​手", "洗手，输出 \x1b[31mFAILED\x1b[0m"])
def test_a_record_the_tree_schema_refuses_lands_as_unreadable_and_holds_its_day(tmp_path, summary) -> None:  # type: ignore[no-untyped-def]
    """E3：干跑校验剔掉的行为（树 schema 不收零宽字符、ANSI 控制符）走确定性失败那条路——换成「没读懂」落成空白、
    不再每轮重新归类；那天等它落成空白才定稿。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    harness.judgements.put_payload(
        judgement_record(
            "zero-width",
            behavior="洗手",
            started_at=at(18),
            last_observed_at=at(62),
            evidence_ready_at=at(64),
            observation_ids=(OBS_A.observation_id, OBS_B.observation_id, OBS_C.observation_id),
            source_refs=(source,),
            summary=summary,
        )
    )
    harness.now = at(6 * 3600)

    first = asyncio.run(harness.runner.run_once())
    assert first.published_occurrences == 0
    assert any("fails validation before stage" in edge for edge in first.dropped_edges)
    assert date(2026, 8, 16) not in harness.runner.closed_days()
    calls = harness.provider.calls

    second = asyncio.run(harness.runner.run_once())
    assert second.published_gaps == 1
    assert date(2026, 8, 16) in harness.runner.closed_days()
    asyncio.run(harness.runner.run_once())
    assert harness.provider.calls == calls  # 不再每轮重新归类
    assert harness.judgements.list() == ()
    (gap,) = harness.tree.list_addresses(BehaviorKind.GAP)
    assert gap.started_at == at(18)
    assert harness.tree.list_addresses(BehaviorKind.OCCURRENCE) == ()


def test_a_held_sweep_lock_makes_the_sweep_and_the_vocabulary_jobs_step_aside(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """E20：``PathLock.acquire`` 是上下文管理器、真正取锁在进入时——锁被占时 sweep 与词表整理活都抛
    ``BehaviorReductionBusyError``（让路），不是裸 ``TimeoutError``（Worker 会把裸的记成失败）。"""

    from habitus.behavior.reduction import BehaviorReductionBusyError

    harness = Harness(tmp_path)
    lock = PathLock(harness.runner.lock_store)
    with lock.acquire(harness.runner._sweep_lock_key, ttl_seconds=30):
        with pytest.raises(BehaviorReductionBusyError):
            asyncio.run(harness.runner.run_once())
        with pytest.raises(BehaviorReductionBusyError):
            asyncio.run(harness.runner.grow_vocabulary())
        with pytest.raises(BehaviorReductionBusyError):
            asyncio.run(harness.runner.revise_vocabulary())
    asyncio.run(harness.runner.run_once())  # 锁释放后照常


def test_a_sweep_lock_held_by_another_process_on_the_sqlite_store_is_lock_busy(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """E20 的生产锁实现：另一份 SQLite 锁库实例（另一个进程）占着 sweep 锁时，这一轮让路。"""

    from habitus.behavior.reduction import BehaviorReductionBusyError
    from habitus.infrastructure.store.sqlite.lock_store import SQLiteLockStore

    harness = Harness(tmp_path)
    kinds, jobs, _ = kind_components(
        harness.tree, SQLiteLockStore(tmp_path / "locks.sqlite3"), names=(), provider=harness.provider
    )
    runner = BehaviorReductionRunner(
        judgements=harness.judgements,
        observations=harness.observations,
        receipts=harness.receipts,
        tree=harness.tree,
        lock_store=SQLiteLockStore(tmp_path / "locks.sqlite3"),
        kinds=kinds,
        vocabulary_jobs=jobs,
        ledger=harness.ledger,
        failed_chains=harness.runner.failed_chains,
        clock=lambda: harness.now,
    )
    other = PathLock(SQLiteLockStore(tmp_path / "locks.sqlite3"))
    with other.acquire(runner._sweep_lock_key, ttl_seconds=30):
        with pytest.raises(BehaviorReductionBusyError):
            asyncio.run(runner.run_once())
    asyncio.run(runner.run_once())


def test_a_document_over_its_encoded_byte_bound_is_truncated_in_the_dry_run_and_leaves_no_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """E16：干跑与写盘同一道检查（含编码后字节）。600 步、每步 300 字的链编码后超出上限：当轮截断 basis 中段落树，
    不留检查点，词表整理活不被"检查点未清"挡住。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    record = judgement_record(
        "huge",
        behavior="洗手",
        started_at=at(18),
        last_observed_at=at(62),
        evidence_ready_at=at(64),
        observation_ids=(OBS_A.observation_id,),
        source_refs=(source,),
        summary="很长的洗手",
    )
    record["basis"] = [
        {"semantics": f"第{i}步" + "搓" * 300, "observation_ids": [OBS_A.observation_id]} for i in range(600)
    ]
    harness.judgements.put_payload(record)

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1, report.dropped_edges
    assert any("truncated: basis 600 steps -> 25" in note for note in report.dropped_edges)
    assert not (tmp_path / "reduction" / "staged.json").exists()
    asyncio.run(harness.runner.grow_vocabulary())  # 没有挂着的检查点挡路


def test_a_document_over_its_byte_bound_that_cannot_be_truncated_becomes_unreadable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """截不动的（步骤本来就少）在干跑里剔除、走确定性失败那条路：换成「没读懂」，不进检查点、不每轮重试。"""

    from habitus.behavior.document import BehaviorDocumentConfig

    harness = Harness(tmp_path)
    # 洗手那篇编码后约 1.85 KB，放不下；「没读懂」的空白放得下
    harness.tree.document_config = BehaviorDocumentConfig(max_encoded_bytes=1_500)
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))

    first = asyncio.run(harness.runner.run_once())
    second = asyncio.run(harness.runner.run_once())

    assert first.published_occurrences == 0
    assert any("fails validation before stage" in note for note in first.dropped_edges)
    assert not (tmp_path / "reduction" / "staged.json").exists()
    assert second.published_gaps == 1  # 下一轮那一段落成「没读懂」
    assert harness.tree.list_addresses(BehaviorKind.OCCURRENCE) == ()


def test_days_a_resumed_migration_changed_are_recorded_by_the_sweep(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """P9：sweep 续做的迁移已经落树——迁移改过的日子进脏日日志与待刷新集合，常驻的按天读侧下一次读就拿到新类。每晚新增与
    定期拆改已不在 sweep 里（裁定 94 第 8、9 条），sweep 只续做线上没做完的迁移。"""

    from habitus.behavior.kinds.changes import ChangeReason, Move, VersionRecord
    from habitus.series.reader import SeriesReader

    harness = Harness(tmp_path, known_kinds=("洗手", "擦干"))
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    asyncio.run(harness.runner.run_once())
    (address,) = harness.tree.list_addresses(BehaviorKind.OCCURRENCE)
    uri = str(BehaviorURI.from_address(address))
    cutoff = address.occurred_on + timedelta(days=1)
    resident = SeriesReader(harness.tree, (harness.runner.dirty_days,))
    assert [item.kind_token for item in resident.read(cutoff=cutoff).records] == ["s-k0001"]
    jobs = harness.vocabulary_jobs
    record = VersionRecord(
        harness.kind_store.read().version + 1, harness.now, ChangeReason.NIGHTLY, (), (Move(uri, "s-k0001", "s-k0002"),)
    )

    jobs.migrator.plan(record)  # 计划已写、还没执行：这一轮先续做它
    report = asyncio.run(harness.runner.run_once())
    assert any(signal.startswith("kind_migration_resumed") for signal in report.kind_signals)

    assert harness.tree.read(address).fields["kind_token"] == "s-k0002"
    assert address.occurred_on in harness.runner.dirty_days.after(0)[1]
    assert [item.kind_token for item in resident.read(cutoff=cutoff).records] == ["s-k0002"]


# ── 变异测试补的探针（段 4 E19）：每条钉住一处原来没有测试守着的行为 ──────────────────────


class _NoLease:
    """直接调写待刷新集合时用的租约替身。"""

    def fenced(self):  # type: ignore[no-untyped-def]
        import contextlib

        return contextlib.nullcontext()

    def checkpoint(self) -> None:
        return None


def test_an_unfused_delivery_that_no_judgement_references_is_not_released(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """R18：没被回执覆盖（还没融合）的交付，即使没有判断引用它，收尾也不许释放——那是删原料。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C, fused=False)
    harness.now = at(10 * 3600)
    asyncio.run(harness.runner.run_once())
    assert harness.observations.read(source) is not None


def test_a_judgement_with_a_blank_name_is_isolated_in_its_first_round(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """R25：名字全是空白的可读判断给不出归类的内容：第一轮就隔离（换成「没读懂」），不是每轮跳过。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    harness.judgements.put_payload(
        judgement_record(
            "blank",
            behavior="   ",
            started_at=at(300),
            last_observed_at=at(310),
            evidence_ready_at=at(312),
            observation_ids=(OBS_C.observation_id,),
            source_refs=(source,),
            summary="名字全是空白",
        )
    )
    report = asyncio.run(harness.runner.run_once())
    assert any("isolated" in note for note in report.dropped_edges), report.dropped_edges
    assert harness.judgements.read(record_id("blank")) is None


def _given_up_refresher(tmp_path):  # type: ignore[no-untyped-def]
    from habitus.behavior.semantic import BehaviorOverviewLimitError, BehaviorSemanticRefresher
    from tests.unit.behavior.test_semantic_layers import ScriptedOverviewGenerator

    class TooLong(ScriptedOverviewGenerator):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate(self, snapshot):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise BehaviorOverviewLimitError("too long")

    generator = TooLong()
    return generator, BehaviorSemanticRefresher(BehaviorTree(tmp_path / "tree"), generator)


def test_a_day_whose_overview_was_given_up_gets_another_chance_once_its_content_changes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """R27：某天的概览超限放弃之后，那一天又落了新记录：重试账清掉、重新给一次机会。"""

    generator, refresher = _given_up_refresher(tmp_path)
    harness = Harness(tmp_path, semantic_refresher=refresher)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    asyncio.run(harness.runner.run_once())
    assert generator.calls == 1
    harness.judgements.put_payload(
        judgement_record(
            "second",
            behavior="洗手",
            started_at=at(900),
            last_observed_at=at(910),
            evidence_ready_at=at(912),
            observation_ids=(OBS_C.observation_id,),
            source_refs=(source,),
            summary="又洗了一次手",
        )
    )
    harness.now = harness.now + timedelta(hours=2)
    asyncio.run(harness.runner.run_once())
    assert generator.calls == 2


def test_a_day_whose_overview_was_given_up_does_not_stay_pending(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """R28：放弃了的那一天（内容没变）又被别处标成待刷新：不调模型、也不留在待刷新集合里。"""

    _generator, refresher = _given_up_refresher(tmp_path)
    harness = Harness(tmp_path, semantic_refresher=refresher)
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    asyncio.run(harness.runner.run_once())
    harness.runner.refresh_pending.write({date(2026, 8, 16)}, _NoLease())
    harness.now = harness.now + timedelta(hours=2)
    asyncio.run(harness.runner.run_once())
    assert not (tmp_path / "reduction" / "refresh_pending.json").exists()


def test_duplicates_disambiguated_by_a_suffix_are_not_revision_members(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """R51：撞车加了后缀的重复记录（``original_name`` 非空）不算进定期拆改的成员。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B)
    for seed, item in (("first", OBS_A), ("second", OBS_B)):
        harness.judgements.put_payload(
            judgement_record(
                seed,
                behavior="洗手",
                started_at=at(18),
                last_observed_at=at(40),
                evidence_ready_at=at(42 if seed == "first" else 44),
                observation_ids=(item.observation_id,),
                source_refs=(source,),
                summary=f"{seed} 洗手",
            )
        )
    asyncio.run(harness.runner.run_once())
    assert len(list(harness.vocabulary_jobs._occurrences(lambda: None))) == 1


def test_the_publish_loop_renews_the_sweep_lease_as_it_writes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """R62：发布检查点时逐段续约（每 10 篇一次）：慢文件系统上一段慢写就能把租约耗光，发布循环里不能不续。"""

    harness = Harness(tmp_path)
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    publish = harness.runner._publish_documents
    renewals: list[int] = []

    class Counting:
        def __init__(self, guard) -> None:  # type: ignore[no-untyped-def]
            self._guard = guard

        def checkpoint(self) -> None:
            renewals.append(1)
            self._guard.checkpoint()

        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(self._guard, name)

    harness.runner._publish_documents = lambda checkpoint, guard: publish(checkpoint, Counting(guard))  # type: ignore[method-assign]

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    assert len(renewals) == 1  # 第 0 篇之前续一次


# ── 裁定 94 第 3 条：错误放一边，不挡定稿 ────────────────────────────────────────────────


def test_a_classification_the_model_cannot_reach_sets_the_chain_aside_as_an_error_and_lets_the_day_close(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """归类时模型调不通：不再"下轮再归"拖住那天——链换成「错误」判断、进夜里待处理的堆（原因 ``UNAVAILABLE``，夜里放回再归一次），
    下一轮归约成「错误」空白，那天照常定稿。"""

    from habitus.behavior.retry import RetryCause
    from habitus.model_client import ModelQuotaError

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)

    async def quota(_request):  # type: ignore[no-untyped-def]
        raise ModelQuotaError("quota exhausted")

    monkeypatch.setattr(harness.provider, "complete_async", quota)
    harness.now = at(6 * 3600)

    first = asyncio.run(harness.runner.run_once())
    assert first.published_occurrences == 0
    assert any("classification unavailable" in note for note in first.dropped_edges)
    assert any(signal.startswith("kind_classify_failed unavailable ModelQuotaError") for signal in first.kind_signals)
    assert first.unavailable == 1 and first.bugs == ()  # 夜里那一支的归约据 ``unavailable`` 停线（裁定 94 第 8 条）
    (item,) = harness.runner.failed_chains.backlog.snapshot().items
    assert (item.kind, item.cause) == (RetryKind.CHAIN, RetryCause.UNAVAILABLE) and item.cause.retried

    second = asyncio.run(harness.runner.run_once())
    assert second.published_gaps == 1
    day = date(2026, 8, 16)
    (gap,) = harness.tree.list_day_addresses(BehaviorKind.GAP, day)
    assert gap.name == "错误"
    assert day in harness.runner.closed_days()


@pytest.mark.parametrize(
    ("error", "cause", "reason"),
    [
        # 这一份触发内容安全：只这几条记「错误」、不重试，夜里按"内容安全"单列
        ("ModelContentSafetyError", "refused", "classification refused"),
        # 输入超长：材料上限与模型窗口没对上，是我们自己处理不了的，不重试
        ("ModelInputTooLargeError", "deterministic", "classification too_large"),
    ],
)
def test_a_classification_refused_or_too_large_is_set_aside_without_asking_again(
    tmp_path, monkeypatch, error: str, cause: str, reason: str
) -> None:  # type: ignore[no-untyped-def]
    """裁定 94 第 8 条：归约归类按错误分类分开——内容安全与超长不重问（同一份输入再问也一样），换成「错误」、按各自的原因进堆，
    不落成「待定」、不按"模型调不通"停线。"""

    import habitus.model_client as model_client
    from habitus.behavior.retry import RetryCause

    harness = Harness(tmp_path)
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))

    async def refuse(_request):  # type: ignore[no-untyped-def]
        raise getattr(model_client, error)("scripted")

    monkeypatch.setattr(harness.provider, "complete_async", refuse)
    harness.now = at(6 * 3600)

    report = asyncio.run(harness.runner.run_once())
    # 归类只问了一次、不重问（另一次是同一轮到点的定期拆改）
    classify = [signal for signal in report.kind_signals if signal.startswith("kind_classify_")]
    assert len(classify) == 1 and classify[0].startswith(f"kind_classify_{reason.split()[-1]} ")
    assert report.published_occurrences == 0 and report.unavailable == 0 and report.bugs == ()
    assert any(reason in note for note in report.dropped_edges)
    (item,) = harness.runner.failed_chains.backlog.snapshot().items
    assert item.cause is RetryCause(cause) and not item.cause.retried
    assert harness.kind_store.read_pending().pool.entries == {}


def test_a_bug_in_classification_is_set_aside_and_reported_on_the_spot(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """裁定 94 第 8 条：归类抛了模型之外的异常（我们自己的 bug）——这批链换成「错误」、不白试（``DETERMINISTIC``），
    报告带上 ``bugs`` 让常驻节拍当场挂到健康面；不按"模型调不通"停线。"""

    from habitus.behavior.retry import RetryCause

    harness = Harness(tmp_path)
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))

    async def broken(_records, *, checkpoint):  # type: ignore[no-untyped-def]
        raise KeyError("verdicts")

    monkeypatch.setattr(harness.runner.kinds, "classify", broken)
    harness.now = at(6 * 3600)

    report = asyncio.run(harness.runner.run_once())
    assert report.unavailable == 0 and len(report.bugs) == 1 and "KeyError" in report.bugs[0]
    (item,) = harness.runner.failed_chains.backlog.snapshot().items
    assert item.cause is RetryCause.DETERMINISTIC


def test_an_unreadable_observation_delivery_is_isolated_and_does_not_stop_the_round(tmp_path) -> None:
    """数据坏了（存储里一份交付读不出来）：隔离坏的那一份、报出，这一轮照常归约，不再整轮失败把那天拖住（裁定 94 错误分类）。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash(harness, source)
    broken = harness.observations.envelope_root / f"{'f' * 64}.json"
    broken.write_bytes(b"{not json")

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    assert any("unreadable; isolated" in note for note in report.dropped_edges)
    assert not broken.exists() and (harness.observations.isolated_root / broken.name).read_bytes() == b"{not json"


def test_a_seal_horizon_pinned_by_one_observation_is_reported_without_a_time_limit(tmp_path) -> None:
    """裁定 94 第 8 条：一份交付里没被回执覆盖的观测、比它晚的观测已有被覆盖的——报出"封口视界被这份观测钉住"（哪份、从何时起），
    不加时限、不隔离；比所有已覆盖观测都晚、还没覆盖的（还在判）不报。"""

    harness = Harness(tmp_path)
    stuck = harness.deliver(OBS_A, seed="stuck", fused=False)
    harness.deliver(OBS_B, OBS_C, seed="later")
    report = asyncio.run(harness.runner.run_once())
    (pinned,) = harness.runner.pinned
    assert pinned.source_id == stuck and pinned.since == OBS_A.available_at
    assert any(
        note.startswith(f"seal horizon pinned by observation delivery {stuck[:12]}") for note in report.dropped_edges
    )

    fusing = Harness(tmp_path / "fusing")
    fusing.deliver(OBS_A, seed="covered")
    fusing.deliver(OBS_B, seed="still-fusing", fused=False)
    asyncio.run(fusing.runner.run_once())
    assert fusing.runner.pinned == ()


def test_two_deliveries_stuck_at_different_moments_are_both_reported(tmp_path) -> None:
    """review-94b 低 5：两份交付各卡在不同时刻（A 卡住、B 覆盖了、C 卡住、更晚的 D 覆盖了）——两份都报，按钉住的时刻排；
    原来"只有一个卡住时刻才算钉住"的判据两份都漏报。最晚那份还没覆盖的（还在判）照旧不报。"""

    obs_d = observation(84, "人离开水池")
    obs_e = observation(100, "人走回书桌")
    harness = Harness(tmp_path)
    first = harness.deliver(OBS_A, seed="stuck-1", fused=False)
    harness.deliver(OBS_B, seed="covered-1")
    second = harness.deliver(OBS_C, seed="stuck-2", fused=False)
    harness.deliver(obs_d, seed="covered-2")
    harness.deliver(obs_e, seed="still-fusing", fused=False)
    report = asyncio.run(harness.runner.run_once())

    assert [(item.source_id, item.since) for item in harness.runner.pinned] == [
        (first, OBS_A.available_at),
        (second, OBS_C.available_at),
    ]
    noted = [note for note in report.dropped_edges if note.startswith("seal horizon pinned by observation delivery")]
    assert len(noted) == 2 and first[:12] in noted[0] and second[:12] in noted[1]


def test_the_pinned_seal_horizon_reaches_the_health_surface(tmp_path) -> None:
    from types import SimpleNamespace

    from habitus.behavior.kinds.ids import Lane
    from habitus.runtime.health import RuntimeHealthService, RuntimeHealthStatus

    harness = Harness(tmp_path)
    stuck = harness.deliver(OBS_A, seed="stuck", fused=False)
    harness.deliver(OBS_B, OBS_C, seed="later")
    asyncio.run(harness.runner.run_once())
    service = object.__new__(RuntimeHealthService)
    worker = SimpleNamespace(last_error=None, running=True)
    service.components = SimpleNamespace(  # type: ignore[assignment]
        behavior=[SimpleNamespace(lane=Lane.SESSION, reduction_worker=worker, reduction_runner=harness.runner)]
    )
    checks = service._behavior_checks("running")
    (pinned,) = [check for check in checks if check.name == "behavior_seal_horizon_session"]
    assert pinned.status is RuntimeHealthStatus.DEGRADED and stuck in pinned.detail


# ── 工程化 S3：拆出的协作者里原先没有测试守着的三处 ─────────────────────────────────────


def test_a_replayed_checkpoint_lands_the_documents_it_staged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """重放落树的是检查点里定格的那一份：原料判断已经不在、时钟也往后拨了，树上仍是 stage 那一刻的文档。"""

    harness = Harness(tmp_path)
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    original_publish = harness.runner._publish_checkpoint
    harness.runner._publish_checkpoint = lambda checkpoint, guard: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("crash after stage")
    )
    with pytest.raises(RuntimeError, match="crash after stage"):
        asyncio.run(harness.runner.run_once())
    staged_at = harness.now
    harness.judgements.discard(record_id("wash"))  # 原料没了：只有检查点里那一份能落树
    harness.now = staged_at + timedelta(hours=1)
    harness.runner._publish_checkpoint = original_publish  # type: ignore[method-assign]

    report = asyncio.run(harness.runner.run_once())

    assert report.replayed_documents == 1
    (address,) = harness.tree.list_addresses(BehaviorKind.OCCURRENCE)
    assert harness.tree.read(address).metadata.created_at == staged_at


def test_a_vocabulary_job_refreshes_the_days_it_changed(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """词表整理活改过的日子：记进待刷新集合，同一把锁内刷新语义面，刷成了就从集合里清掉。"""

    from habitus.behavior.reduction.kinds_jobs import JobsOutcome
    from habitus.behavior.semantic import BehaviorSemanticRefresher
    from tests.unit.behavior.test_semantic_layers import ScriptedOverviewGenerator

    harness = Harness(
        tmp_path,
        semantic_refresher=BehaviorSemanticRefresher(BehaviorTree(tmp_path / "tree"), ScriptedOverviewGenerator()),
    )
    seed_wash(harness, harness.deliver(OBS_A, OBS_B, OBS_C))
    asyncio.run(harness.runner.run_once())
    day = date(2026, 8, 16)

    async def grown(now, *, checkpoint):  # type: ignore[no-untyped-def]
        return JobsOutcome(frozenset({day}), 1, ("grown",))

    monkeypatch.setattr(harness.vocabulary_jobs, "grow", grown)
    report = asyncio.run(harness.runner.grow_vocabulary())

    assert (report.days, report.model_calls, report.signals) == ((day,), 1, ("grown",))
    assert not (tmp_path / "reduction" / "refresh_pending.json").exists()


def test_a_chain_whose_delivery_is_gone_still_lands_and_says_so(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """交付已不在存储里：没有 basis 的链照常落树（判断本体自带 summary），留一条信号是唯一的可见痕迹（审计 NEW-8）。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    harness.judgements.put_payload(
        judgement_record(
            "wash",
            behavior="洗手",
            started_at=at(18),
            last_observed_at=at(62),
            evidence_ready_at=at(64),
            observation_ids=(OBS_A.observation_id, OBS_B.observation_id, OBS_C.observation_id),
            source_refs=(source,),
            summary="在水池边洗手",
        )
    )
    harness.observations.discard(source)

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    assert f"source_delivery_missing {source}" in report.dropped_edges
