"""归约 runner 端到端：真实存储进、真实行为树出。

模型触点（kinds 归一）用预置词表压到零调用——词表命中走确定性快路径，stub 客户端被调用即
测试失败，这本身就是"写入层唯一 LLM 触点"边界的守卫。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import pytest

from behavior.fusion import FUSION_PROMPT_VERSION
from behavior.fusion.receipt import build_fusion_receipt
from behavior.fusion.receipt_store import BehaviorFusionReceiptStore
from behavior.fusion.store import BehaviorJudgementStore
from behavior.kinds.model import BehaviorKindRegistry
from behavior.kinds.resolver import BehaviorKindResolver
from behavior.kinds.store import BehaviorKindStore
from behavior.model import BehaviorKind
from behavior.observation import (
    BehaviorObservationBatch,
    BehaviorObservationEnvelope,
    BehaviorObservationStore,
)
from behavior.reduction import BehaviorReductionLedger, BehaviorReductionRunner  # noqa: F401
from behavior.tree import BehaviorTree
from foundation.integrity import canonical_digest
from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from ModelClient import (
    ChatClient,
    ChatModelConfig,
    ChatRequest,
    PreparedChatRequest,
    ProviderCapabilities,
    ProviderConfig,
    StructuredChatClient,
)
from tests.unit.behavior.reduction_fixtures import (
    OBSERVATION_CONFIG,
    OBSERVER,
    SUBJECT,
    at,
    judgement_record,
    observation,
    record_id,
)

OBS_A = observation(18, "人走到水池边打开水龙头")
OBS_B = observation(40, "人打肥皂搓手")
OBS_C = observation(62, "人冲水擦干")
OBS_BLUR = observation(120, "画面模糊")


class _ForbiddenProvider:
    """kinds 词表命中快路径后模型绝不应被调用；被调用即失败。"""

    provider_name = "fake"
    model = "fake-1"
    is_remote = False
    capabilities = ProviderCapabilities(
        async_completion=True,
        streaming=False,
        tools=False,
        structured_output_mode="json_schema",
        reasoning=False,
    )

    def prepare(self, request: ChatRequest, *, stream: bool) -> PreparedChatRequest:
        return PreparedChatRequest(
            request=request,
            body=b"{}",
            model_visible_body=b"{}",
            reserved_output_tokens=1_000,
            stream=stream,
        )

    async def complete_async(self, request: PreparedChatRequest) -> Any:
        raise AssertionError("reduction must not call the model for a known kind name")

    def complete(self, request: PreparedChatRequest) -> Any:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: PreparedChatRequest) -> Iterator[Any]:  # pragma: no cover
        raise NotImplementedError

    def stream_async(self, request: PreparedChatRequest) -> AsyncIterator[Any]:  # pragma: no cover
        raise NotImplementedError

    def health_check(self) -> Mapping[str, object]:  # pragma: no cover
        return {}

    async def aclose(self) -> None:  # pragma: no cover
        return None


def build_resolver() -> BehaviorKindResolver:
    model_config = ChatModelConfig(
        route=ProviderConfig(
            provider="fake",
            adapter="openai_compatible_chat",
            model="fake-1",
            base_url="https://example.invalid",
            credential_ref="FAKE_KEY",
        ),
        context_window_tokens=128_000,
        max_output_tokens=8_000,
        structured_output_mode="json_schema",
    )
    client = StructuredChatClient(ChatClient(model_config, _ForbiddenProvider()), validation_retries=1)
    return BehaviorKindResolver(client)


class Harness:
    """一套真实存储 + 可拨时钟的归约现场。"""

    def __init__(
        self,
        tmp_path,
        known_kinds: tuple[str, ...] = ("洗手",),
        semantic_refresher=None,
    ) -> None:
        self.now = at(7200)  # 默认拨到 lookback(1h) 之外
        self.observations = BehaviorObservationStore(tmp_path / "observations")
        self.judgements = BehaviorJudgementStore(tmp_path / "judgements")
        self.receipts = BehaviorFusionReceiptStore(tmp_path / "receipts")
        self.tree = BehaviorTree(tmp_path / "tree")
        # kinds 已并入地址空间：词表就在树根（behavior://kinds.md），与树共存
        self.kind_store = BehaviorKindStore(tmp_path / "tree")
        if known_kinds:
            registry = BehaviorKindRegistry({name: () for name in known_kinds})
            self.kind_store.replace(registry, expected_revision=0, timestamp=self.now)
        self.ledger = BehaviorReductionLedger(tmp_path / "reduction")
        self.runner = BehaviorReductionRunner(
            judgements=self.judgements,
            observations=self.observations,
            receipts=self.receipts,
            tree=self.tree,
            lock_store=ProcessLocalLockStore(),
            kind_store=self.kind_store,
            kind_resolver=build_resolver(),
            ledger=self.ledger,
            semantic_refresher=semantic_refresher,
            clock=lambda: self.now,
        )

    def deliver(self, *observations_batch, seed: str = "delivery", fused: bool = True) -> str:
        """投递一批观测；``fused=True`` 时同时写一份覆盖它们的融合回执。

        封口前沿的判据是"未被任何回执覆盖的观测"——测试直接绕过融合层播种判断时，必须
        把观测标成已融合，否则它们会被当成待融合积压、把封口视界永远压住。
        """

        envelope = BehaviorObservationEnvelope.create(
            observer_id=OBSERVER,
            protocol="habitus_behavior_observation_v1",
            batch=BehaviorObservationBatch(
                observer_id=OBSERVER, observations=tuple(observations_batch)
            ),
            delivery_id=canonical_digest(seed),
            recorded_at=at(600),
            config=OBSERVATION_CONFIG,
        )
        self.observations.put(envelope)
        if fused:
            self.receipts.put(
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
        return envelope.source_id


def seed_wash_chain(harness: Harness, source: str) -> None:
    harness.judgements.put_payload(
        judgement_record(
            "head",
            behavior="洗手",
            started_at=at(18),
            last_observed_at=at(40),
            evidence_ready_at=at(42),
            observation_ids=(OBS_A.observation_id, OBS_B.observation_id),
            source_refs=(source,),
            goal="清洁双手",
            summary="在水池边洗手",
            status="ongoing",
            status_basis="observation_lost",
            basis=(("打开水龙头打肥皂搓手", (OBS_A.observation_id, OBS_B.observation_id)),),
        )
    )
    harness.judgements.put_payload(
        judgement_record(
            "tail",
            behavior="继续洗手",
            started_at=at(62),
            last_observed_at=at(62),
            evidence_ready_at=at(64),
            observation_ids=(OBS_C.observation_id,),
            source_refs=(source,),
            goal="清洁双手",
            summary="冲水擦干结束",
            status="completed",
            status_basis="observed",
            basis=(("冲水擦干", (OBS_C.observation_id,)),),
            relations=(("continues", record_id("head")),),
        )
    )


def test_full_pipeline_reduces_a_chain_and_a_gap_into_the_tree(tmp_path) -> None:
    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C, OBS_BLUR)
    seed_wash_chain(harness, source)
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
    gaps = harness.tree.list_addresses(BehaviorKind.GAP)
    assert [address.name for address in occurrences] == ["洗手"]
    assert [address.name for address in gaps] == ["没读懂"]

    document = harness.tree.read(occurrences[0])
    assert document.fields["started_at"] == at(18).isoformat(timespec="microseconds")
    assert document.fields["started_at"].endswith("+08:00")
    assert document.fields["kind_token"] == "洗手"
    assert document.fields["status"] == "completed"
    assert document.fields["summary"] == "在水池边洗手；冲水擦干结束"
    assert document.fields["onset_available_at"] == at(42).isoformat(timespec="microseconds")
    assert set(document.fields["judgement_ids"]) == {record_id("head"), record_id("tail")}

    gap_document = harness.tree.read(gaps[0])
    assert gap_document.fields["started_at"] == gap_document.fields["ended_at"]

    # 消费账本闭环：全部判断都已入账
    consumed = harness.ledger.consumed_judgement_ids()
    assert {record_id("head"), record_id("tail"), record_id("blur")} <= consumed


def test_second_sweep_is_a_no_op(tmp_path) -> None:
    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    asyncio.run(harness.runner.run_once())

    report = asyncio.run(harness.runner.run_once())

    assert report.published_documents == 0
    assert report.replayed_documents == 0
    assert len(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)) == 1


def test_chains_inside_the_window_stay_pending(tmp_path) -> None:
    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    harness.now = at(600)  # 链尾 evidence(64s) 仍在 1h 窗口内

    report = asyncio.run(harness.runner.run_once())

    assert report.published_documents == 0
    assert report.chains_pending == 1
    assert harness.tree.list_addresses(BehaviorKind.OCCURRENCE) == ()


def test_an_unfused_observation_blocks_sealing_despite_the_wall_clock(tmp_path) -> None:
    """停机补算：墙钟早已出窗，但还有观测没被任何融合回执覆盖——它将来会切成
    cutoff 是历史时刻的段、仍可能引用旧判断，不许封口。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    backlog = observation(300, "人在客厅走动")
    harness.deliver(backlog, seed="backlog", fused=False)  # 已交付、未融合
    harness.now = at(7200)

    report = asyncio.run(harness.runner.run_once())

    # 前沿 cutoff = 302s（backlog 的 available_at），视界 = 302s - 3600s < 链尾 evidence
    assert report.published_documents == 0
    assert report.chains_pending == 1

    # 该观测融合完成（回执覆盖）后，同一墙钟下立即可封。
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
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1


def test_crash_after_stage_replays_byte_identically(tmp_path) -> None:
    """死规则⑤：stage 之后崩溃，重放逐字节落盘——不重算、不撞车。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)

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


def test_late_chain_links_to_an_already_consumed_target_via_the_ledger(tmp_path) -> None:
    harness = Harness(tmp_path, known_kinds=("洗手", "擦桌子"))
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    asyncio.run(harness.runner.run_once())

    harness.judgements.put_payload(
        judgement_record(
            "wipe",
            behavior="擦桌子",
            started_at=at(200),
            last_observed_at=at(230),
            evidence_ready_at=at(232),
            observation_ids=(OBS_C.observation_id,),
            source_refs=(source,),
            summary="洗完手顺手擦了桌子",
            relations=(("results_from", record_id("tail")),),
        )
    )
    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    wipe = next(
        harness.tree.read(address)
        for address in harness.tree.list_addresses(BehaviorKind.OCCURRENCE)
        if address.name == "擦桌子"
    )
    assert len(wipe.links) == 1
    assert wipe.links[0].link_type.value == "results_from"
    assert "洗手" in str(wipe.links[0].to_uri)


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
    assert set(gap.fields["judgement_ids"]) == {record_id("blur"), record_id("blur-b")}

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


def test_supersedes_view_lands_and_history_is_consumed(tmp_path) -> None:
    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B)
    harness.judgements.put_payload(
        judgement_record(
            "vague",
            behavior="在水池边忙碌",
            started_at=at(18),
            last_observed_at=at(40),
            evidence_ready_at=at(42),
            observation_ids=(OBS_A.observation_id,),
            source_refs=(source,),
            summary="看不清在做什么",
            status="ongoing",
            status_basis="observation_lost",
        )
    )
    harness.judgements.put_payload(
        judgement_record(
            "clear",
            behavior="洗手",
            started_at=at(18),
            last_observed_at=at(40),
            evidence_ready_at=at(60),
            observation_ids=(OBS_A.observation_id, OBS_B.observation_id),
            source_refs=(source,),
            summary="看清了，是在洗手",
            relations=(("supersedes", record_id("vague")),),
        )
    )

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    addresses = harness.tree.list_addresses(BehaviorKind.OCCURRENCE)
    assert [address.name for address in addresses] == ["洗手"]  # 树存修正后视图
    document = harness.tree.read(addresses[0])
    assert set(document.fields["judgement_ids"]) == {record_id("vague"), record_id("clear")}
    consumed = harness.ledger.consumed_judgement_ids()
    assert record_id("vague") in consumed  # 全史随链消费




def test_a_link_to_a_published_gap_is_dropped_not_published(tmp_path) -> None:
    """指向已落盘 gap 节点的跨批链接必须作废——放行会在检查点之后被 writer 硬拒、永久卡死。"""

    harness = Harness(tmp_path, known_kinds=("擦桌子",))
    source = harness.deliver(OBS_C, OBS_BLUR)
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
    asyncio.run(harness.runner.run_once())  # gap 落盘并记账

    harness.judgements.put_payload(
        judgement_record(
            "wipe",
            behavior="擦桌子",
            started_at=at(200),
            last_observed_at=at(230),
            evidence_ready_at=at(232),
            observation_ids=(OBS_C.observation_id,),
            source_refs=(source,),
            summary="擦了桌子",
            relations=(("results_from", record_id("blur")),),
        )
    )
    harness.now = at(10_000)
    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    wipe = harness.tree.read(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)[0])
    assert wipe.links == ()
    assert any("published gap node" in note for note in report.dropped_edges)


def test_a_corrupt_record_is_quarantined_without_stalling_the_sweep(tmp_path) -> None:
    """单条坏记录隔离降级：留信号、不消费、不许瘫痪整轮归约。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
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


def test_a_chain_with_an_unaddressable_name_is_skipped_with_a_signal(tmp_path) -> None:
    """绕过融合守卫写入的坏行为名：按链隔离跳过，不吞整轮、不写检查点炸弹。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
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
    assert any("skipped" in note for note in report.dropped_edges)


def test_an_invalid_payload_is_dropped_before_the_checkpoint(tmp_path) -> None:
    """stage 末端干跑校验：会在 publish 期炸的文档不进检查点，降级成带信号的跳过。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C, OBS_BLUR)
    seed_wash_chain(harness, source)
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

    import behavior.reduction.runner as runner_module

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    monkeypatch.setattr(runner_module, "_MAX_CHECKPOINT_BYTES", 64)
    with pytest.raises(Exception, match="byte bound"):
        asyncio.run(harness.runner.run_once())
    assert harness.tree.list_addresses(BehaviorKind.OCCURRENCE) == ()
    assert not (tmp_path / "reduction" / "staged.json").exists()

    monkeypatch.setattr(runner_module, "_MAX_CHECKPOINT_BYTES", 67_108_864)
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1


def test_concurrent_sweeps_are_serialised_by_the_root_lock(tmp_path) -> None:
    """单写入方是机械保证：同一 behavior-root 的第二个归约进程拿不到锁即失败。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    lock = PathLock(harness.runner.lock_store)
    with lock.acquire(harness.runner._sweep_lock_key, ttl_seconds=30):
        with pytest.raises(TimeoutError):
            asyncio.run(harness.runner.run_once())
    report = asyncio.run(harness.runner.run_once())  # 锁释放后照常
    assert report.published_occurrences == 1


def test_sweep_refreshes_day_summaries_after_publishing(tmp_path) -> None:
    """归约落盘后在同一把 sweep 锁内刷新 L0/L1；树没变的下一轮零模型调用。"""

    from behavior.model import BehaviorDirectory, BehaviorLevel
    from behavior.semantic import BehaviorSemanticRefresher
    from tests.unit.behavior.test_semantic_layers import ScriptedOverviewGenerator

    generator = ScriptedOverviewGenerator()
    tree = BehaviorTree(tmp_path / "tree")
    harness = Harness(
        tmp_path,
        semantic_refresher=BehaviorSemanticRefresher(tree, generator),
    )
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)

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


def test_a_link_target_skipped_after_naming_defers_the_source(tmp_path) -> None:
    """目标链命名成功、payload 物化失败被跳过：源链必须一并推迟——评审实测过反面：
    源链带着悬空 URI 进检查点，重放永败、整条归约卡死。"""

    harness = Harness(tmp_path, known_kinds=("洗手", "擦桌子"))
    source = harness.deliver(OBS_A, OBS_C)
    # 目标链：basis 引用一条不在任何投递里的观测 → 物化期失败被跳过
    harness.judgements.put_payload(
        judgement_record(
            "broken-target",
            behavior="洗手",
            started_at=at(18),
            last_observed_at=at(40),
            evidence_ready_at=at(42),
            observation_ids=(OBS_A.observation_id, "f" * 64),
            source_refs=(source,),
            goal="清洁双手",
            summary="basis 观测已不在存储",
            basis=(("引用了幽灵观测", ("f" * 64,)),),
        )
    )
    harness.judgements.put_payload(
        judgement_record(
            "linker",
            behavior="擦桌子",
            started_at=at(200),
            last_observed_at=at(230),
            evidence_ready_at=at(232),
            observation_ids=(OBS_C.observation_id,),
            source_refs=(source,),
            summary="想链接洗手",
            relations=(("results_from", record_id("broken-target")),),
        )
    )

    report = asyncio.run(harness.runner.run_once())

    assert report.published_documents == 0  # 谁都没落，也没有卡死的检查点
    assert any("skipped" in note for note in report.dropped_edges)
    assert any("deferred: its link target was skipped" in note for note in report.dropped_edges)
    assert report.chains_pending == 2
    assert not (tmp_path / "reduction" / "staged.json").exists()
    # 再跑一轮：同样干净失败，不退化成永久撞车
    second = asyncio.run(harness.runner.run_once())
    assert second.published_documents == 0


def test_failed_semantic_refresh_leaves_the_checkpoint_for_healing(tmp_path) -> None:
    """刷新失败降级为带日期的信号；检查点保留到刷新成功——崩溃/故障后的摘要缺口靠重放自愈。"""

    from behavior.semantic import BehaviorSemanticRefresher
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
    seed_wash_chain(harness, source)

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1  # 归约不被刷新失败阻塞
    assert any("semantic refresh failed for [2026-08-16]" in n for n in report.dropped_edges)
    assert (tmp_path / "reduction" / "staged.json").exists()  # 检查点留作自愈依据

    generator.fail = False
    second = asyncio.run(harness.runner.run_once())

    assert second.replayed_documents == 1  # 幂等重放 + 补刷
    from behavior.model import BehaviorDirectory, BehaviorLevel

    assert harness.tree.layer_exists(
        BehaviorDirectory.occurrences(2026, 8, 16), BehaviorLevel.OVERVIEW
    )
    assert not (tmp_path / "reduction" / "staged.json").exists()


def test_run_once_works_with_the_production_sqlite_lock_store(tmp_path) -> None:
    """接缝测试：生产锁实现（SQLite）下整轮 sweep 必须能跑通——租约互斥 + 短围栏，
    不许出现"sweep 事务压住 writer 文档锁"的自死锁（评审在旧实现下实测过每轮必败）。"""

    from infrastructure.store.sqlite.lock_store import SQLiteLockStore

    lock_store = SQLiteLockStore(tmp_path / "locks.sqlite3")
    harness = Harness(tmp_path)
    harness.runner = BehaviorReductionRunner(
        judgements=harness.judgements,
        observations=harness.observations,
        receipts=harness.receipts,
        tree=harness.tree,
        lock_store=lock_store,
        kind_store=harness.kind_store,
        kind_resolver=build_resolver(),
        ledger=harness.ledger,
        clock=lambda: harness.now,
    )
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)

    report = asyncio.run(harness.runner.run_once())

    assert report.published_occurrences == 1
    assert len(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)) == 1
