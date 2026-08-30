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
from behavior.uri import BehaviorURI
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
    ledger_by_uri = {entry.uri: entry for entry in harness.ledger.load()}
    assert document.fields["chain_digest"] == ledger_by_uri[str(BehaviorURI.from_address(occurrences[0]))].chain_digest

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
    # 第一轮发布后，这份交付的观测全部已融合且无判断再引用——按"消费即释放"它已被删除。
    assert harness.observations.read(source) is None
    assert harness.judgements.list() == ()

    # 晚到的链带着自己的观测与交付（一条观测只会被融合一次，不会被后来的判断复用）。
    wipe_obs = observation(200, "人在擦桌子")
    source_wipe = harness.deliver(wipe_obs, seed="delivery-wipe")
    harness.judgements.put_payload(
        judgement_record(
            "wipe",
            behavior="擦桌子",
            started_at=at(200),
            last_observed_at=at(230),
            evidence_ready_at=at(232),
            observation_ids=(wipe_obs.observation_id,),
            source_refs=(source_wipe,),
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
    entry = next(item for item in harness.ledger.load() if item.chain_digest == document.fields["chain_digest"])
    assert set(entry.judgement_ids) == {record_id("vague"), record_id("clear")}
    consumed = harness.ledger.consumed_judgement_ids()
    assert record_id("vague") in consumed  # 全史随链消费




def test_a_link_to_a_published_gap_is_dropped_not_published(tmp_path) -> None:
    """指向已落盘 gap 节点的跨批链接必须作废——放行会在检查点之后被 writer 硬拒、永久卡死。"""

    harness = Harness(tmp_path, known_kinds=("擦桌子",))
    source = harness.deliver(OBS_BLUR)
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
    asyncio.run(harness.runner.run_once())  # gap 落盘并记账；这份交付随之释放

    source_wipe = harness.deliver(OBS_C, seed="delivery-wipe")
    harness.judgements.put_payload(
        judgement_record(
            "wipe",
            behavior="擦桌子",
            started_at=at(200),
            last_observed_at=at(230),
            evidence_ready_at=at(232),
            observation_ids=(OBS_C.observation_id,),
            source_refs=(source_wipe,),
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


def test_failed_semantic_refresh_is_retried_from_the_pending_set(tmp_path) -> None:
    """刷新失败降级为带日期的信号；那一天留在待刷新集合里，下轮补刷；检查点不被扣住。"""

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
    # 检查点发布完即清（不再被刷新失败扣住——那会让 merge 之后的重放撞出永久冲突）；
    # 待刷新的日子记在独立的小文件里，下轮逐日重试。
    assert not (tmp_path / "reduction" / "staged.json").exists()
    import json as _json

    assert _json.loads((tmp_path / "reduction" / "refresh_pending.json").read_text()) == ["2026-08-16"]

    generator.fail = False
    second = asyncio.run(harness.runner.run_once())

    assert second.replayed_documents == 0  # 没有检查点可重放；补刷来自待刷新集合
    from behavior.model import BehaviorDirectory, BehaviorLevel

    assert harness.tree.layer_exists(
        BehaviorDirectory.occurrences(2026, 8, 16), BehaviorLevel.OVERVIEW
    )
    assert not (tmp_path / "reduction" / "refresh_pending.json").exists()


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


class _FullRegistryResolver(BehaviorKindResolver):
    """词表已满：未知名字降级为原始名作 token 并留信号（与 resolver 的撞顶路径同形）。"""

    async def resolve_batches(self, requests, registry, *, vectors=None):  # type: ignore[override]
        from behavior.kinds.resolver import BehaviorKindBatchResolution

        tokens = {item.name: registry.token_for(item.name) or item.name for item in requests}
        signals = tuple(
            f"kind_registry_full {item.name!r} kept as its own token"
            for item in requests
            if registry.token_for(item.name) is None
        )
        yield BehaviorKindBatchResolution(tokens, registry, vectors, (), 0, signals)


def test_a_full_kind_registry_degrades_to_the_raw_name_instead_of_failing_the_sweep(tmp_path) -> None:
    harness = Harness(tmp_path, known_kinds=())
    harness.runner.kind_resolver = _FullRegistryResolver(build_resolver().client)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1
    document = harness.tree.read(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)[0])
    assert document.fields["kind_token"] == "洗手"  # 原始名暂作 token，事后可重打
    assert any("kind_registry_full" in note for note in report.kind_signals)


def test_reduction_records_hits_by_behaviour_day_and_expires_stale_kinds(tmp_path) -> None:
    """命中账按行为日记；到期按数据时钟（本轮最新行为日）删，树上文档不动（BHV-KINDS-002）。"""

    from datetime import timedelta

    from behavior.kinds import BehaviorKindEntry

    harness = Harness(tmp_path, known_kinds=())
    behaviour_day = at(18).date()
    stale = BehaviorKindEntry(token="卷账单", label="卷账单").with_hit(behaviour_day - timedelta(days=40))
    recent = BehaviorKindEntry(token="打球", label="打球").with_hit(behaviour_day - timedelta(days=10))
    harness.kind_store.replace(
        BehaviorKindRegistry({"卷账单": stale, "打球": recent, "洗手": ()}),
        expected_revision=0,
        timestamp=harness.now,
    )
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1
    registry = harness.kind_store.read().registry
    assert registry.entry_of("洗手").hit_days == (behaviour_day,)
    assert "卷账单" not in registry.tokens  # 40 天没再命中 > 基础期 30 天
    assert "打球" in registry.tokens  # 10 天前命中过，还在存活期内
    assert any("kind_expired '卷账单'" in note for note in report.kind_signals)
    assert not any(note.startswith("kind_") for note in report.dropped_edges)  # 词表信号单独列


def test_kind_hits_are_recorded_at_publish_and_replay_is_idempotent(tmp_path) -> None:
    """命中账在发布时记（与树上 occurrence 一一对应）；重放同一检查点不重复记。"""

    class _KeepCheckpoint(type(Harness(tmp_path / "probe").runner)):  # type: ignore[misc]
        def _clear_checkpoint(self, guard):  # noqa: ANN001 - 与父类签名一致
            return None  # 假装"语义刷新前崩溃"，让下一轮重放同一检查点

    harness = Harness(tmp_path)
    harness.runner.__class__ = _KeepCheckpoint
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    first = asyncio.run(harness.runner.run_once())
    assert first.published_occurrences == 1
    entry = harness.kind_store.read().registry.entry_of("洗手")
    assert entry.hit_days == (at(18).date(),) and entry.hit_count == 1
    second = asyncio.run(harness.runner.run_once())  # 重放检查点：树、账本、命中账都不变
    assert second.replayed_documents >= 1 and second.published_occurrences == 0
    entry = harness.kind_store.read().registry.entry_of("洗手")
    assert entry.hit_days == (at(18).date(),) and entry.hit_count == 1


def test_reduction_persists_kind_vectors_and_recovers_a_corrupt_sidecar(tmp_path) -> None:
    """旁册随本轮写回；损坏时按空重建并留信号，不阻塞归约（BHV-KINDS-002）。"""

    from behavior.kinds import BehaviorKindVectorStore
    from tests.unit.behavior.test_kinds import FakeEmbedder

    harness = Harness(tmp_path)
    embedder = FakeEmbedder()
    harness.runner.kind_resolver = BehaviorKindResolver(build_resolver().client, embedder=embedder)
    harness.runner.kind_vectors = BehaviorKindVectorStore(
        tmp_path / "tree", model="fake-embed", dimension=FakeEmbedder.DIMENSION
    )
    (tmp_path / "tree").mkdir(exist_ok=True)
    (tmp_path / "tree" / "kinds.vectors.json").write_text("garbage", encoding="utf-8")
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1
    assert any("kind_vectors_unreadable" in note for note in report.kind_signals)
    assert harness.runner.kind_vectors.read().has("洗手")  # 已重写：词表里的 token 补了向量


def test_reduction_without_chains_does_not_expire_anything(tmp_path) -> None:
    from datetime import timedelta

    from behavior.kinds import BehaviorKindEntry

    harness = Harness(tmp_path, known_kinds=())
    stale = BehaviorKindEntry(token="卷账单", label="卷账单").with_hit(at(18).date() - timedelta(days=400))
    harness.kind_store.replace(BehaviorKindRegistry({"卷账单": stale}), expected_revision=0, timestamp=harness.now)
    report = asyncio.run(harness.runner.run_once())
    assert report.published_documents == 0
    assert "卷账单" in harness.kind_store.read().registry.tokens  # 没有数据时钟就不判过期


def test_tree_replace_only_allows_the_kind_token_to_change(tmp_path) -> None:
    """树是 add-only 的；replace 是唯一改写通道，且只为 kind_token 重打而开。"""

    from dataclasses import replace as dc_replace

    from behavior import BehaviorDocumentWriter, BehaviorTree, BehaviorTreeConflictError
    from tests.unit.behavior.tree_payloads import local, occurrence_payload

    tree = BehaviorTree(tmp_path / "tree")
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    published = writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    bumped = dc_replace(published.metadata, revision=2)
    # 改 kind_token：允许
    ok = tree.document_codec.build(
        published.kind, {**published.fields, "kind_token": "清洁双手"}, metadata=bumped, links=published.links
    )
    tree.replace(ok)
    assert tree.read(published.address).fields["kind_token"] == "清洁双手"
    # 改别的字段：拒绝
    bad = tree.document_codec.build(
        published.kind, {**published.fields, "kind_token": "清洁双手", "summary": "改了正文"}, metadata=dc_replace(bumped, revision=3), links=()
    )
    with pytest.raises(BehaviorTreeConflictError, match="only change kind_token"):
        tree.replace(bad)
    # 修订号不连续：拒绝
    stale = tree.document_codec.build(
        published.kind, {**published.fields, "kind_token": "洗手"}, metadata=published.metadata, links=published.links
    )
    with pytest.raises(BehaviorTreeConflictError, match="revision"):
        tree.replace(stale)


def test_writer_restamp_is_idempotent_and_bumps_revision(tmp_path) -> None:
    from behavior import BehaviorDocumentWriter, BehaviorTree
    from tests.unit.behavior.tree_payloads import local, occurrence_payload

    tree = BehaviorTree(tmp_path / "tree")
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    published = writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())
    restamped = writer.restamp_kind_token(published.address, "清洁双手")
    assert restamped.fields["kind_token"] == "清洁双手" and restamped.metadata.revision == 2
    assert restamped.fields["name"] == published.fields["name"] and restamped.links == published.links
    again = writer.restamp_kind_token(published.address, "清洁双手")  # 同值：不动
    assert again.metadata.revision == 2
    assert tree.read(published.address) == restamped


def test_merge_kinds_folds_the_vocabulary_and_restamps_the_tree(tmp_path) -> None:
    """合并道的落地动作：词表 merged + 树上旧 token 全部重打；预测源随后读到同一个 token。"""

    from behavior.kinds import BehaviorKindEntry
    from prediction.source import read as read_snapshot

    harness = Harness(tmp_path, known_kinds=())
    harness.kind_store.replace(
        BehaviorKindRegistry(
            {
                "洗手": BehaviorKindEntry(token="洗手", label="洗手"),
                "清洁双手": BehaviorKindEntry(token="清洁双手", label="清洁双手"),
            }
        ),
        expected_revision=0,
        timestamp=harness.now,
    )
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)  # 链头名「洗手」→ token 洗手
    assert asyncio.run(harness.runner.run_once()).published_occurrences == 1
    report = asyncio.run(harness.runner.merge_kinds("洗手", "清洁双手"))
    assert report.restamped == 1 and report.days == (at(18).date(),)
    assert any("kind_merged" in note for note in report.signals)
    registry = harness.kind_store.read().registry
    assert registry.tokens == ("清洁双手",) and registry.token_for("洗手") == "清洁双手"
    assert registry.entry_of("清洁双手").hit_days == (at(18).date(),)  # 账并过去了
    document = harness.tree.read(harness.tree.list_addresses(BehaviorKind.OCCURRENCE)[0])
    assert document.fields["kind_token"] == "清洁双手" and document.fields["name"] == "洗手"
    assert document.metadata.revision == 2
    assert {item.action for item in read_snapshot(harness.tree).actions} == {"清洁双手"}
    # 重跑幂等：词表里已无 source、树上已无旧 token
    again = asyncio.run(harness.runner.merge_kinds("洗手", "清洁双手"))
    assert again.restamped == 0


# ── 生命周期闭合（三方审计根因 1–4）────────────────────────────────────────────────


def test_kind_hits_survive_a_crash_between_ledger_append_and_hit_recording(tmp_path) -> None:
    """账本已落、命中未记时崩溃：重放后命中仍记且只记一次（幂等键是检查点，不是账本条目）。"""

    class _CrashOnce(type(Harness(tmp_path / "probe").runner)):  # type: ignore[misc]
        crashed = False

        def _record_kind_hits(self, hits, staged_at, now):  # noqa: ANN001
            if not type(self).crashed:
                type(self).crashed = True
                raise RuntimeError("crash after ledger, before hits")
            return super()._record_kind_hits(hits, staged_at, now)

    harness = Harness(tmp_path)
    harness.runner.__class__ = _CrashOnce
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    with pytest.raises(RuntimeError, match="crash after ledger"):
        asyncio.run(harness.runner.run_once())
    assert (tmp_path / "reduction" / "staged.json").exists()  # 检查点还在，下轮重放
    report = asyncio.run(harness.runner.run_once())
    assert report.replayed_documents == 1
    entry = harness.kind_store.read().registry.entry_of("洗手")
    assert entry.hit_days == (at(18).date(),) and entry.hit_count == 1
    assert asyncio.run(harness.runner.run_once()).replayed_documents == 0
    assert harness.kind_store.read().registry.entry_of("洗手").hit_count == 1  # 再跑不重记


def test_merge_and_rebuild_refuse_while_a_checkpoint_is_pending(tmp_path) -> None:
    """运维动作不许压在悬挂的检查点上（重打 token 后重放会撞永久冲突）。"""

    import json as _json

    from behavior.reduction import BehaviorReductionBusyError, BehaviorReductionError

    harness = Harness(tmp_path)
    (tmp_path / "reduction").mkdir(exist_ok=True)
    staged = tmp_path / "reduction" / "staged.json"
    staged.write_text(_json.dumps({"staged_at": harness.now.isoformat(), "documents": []}), encoding="utf-8")
    with pytest.raises(BehaviorReductionBusyError, match="checkpoint is pending"):
        asyncio.run(harness.runner.merge_kinds("洗手", "清洁双手"))
    with pytest.raises(BehaviorReductionBusyError, match="checkpoint is pending"):
        asyncio.run(harness.runner.rebuild_kinds())
    # 损坏的检查点：三条路都堵死是死锁，要给出"手动删除"的指引而不是"先跑 sweep"
    staged.write_text("garbage", encoding="utf-8")
    with pytest.raises(BehaviorReductionError, match="delete it manually"):
        asyncio.run(harness.runner.merge_kinds("洗手", "清洁双手"))


def test_an_address_already_on_the_tree_is_disambiguated_without_a_ledger_entry(tmp_path) -> None:
    """账本按窗口过期、树不会：树上已有同址文档时照样消歧成 -2，而不是 publish 硬冲突卡死检查点。"""

    from behavior import BehaviorDocumentWriter
    from tests.unit.behavior.tree_payloads import local, occurrence_payload

    harness = Harness(tmp_path)
    writer = BehaviorDocumentWriter(harness.tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    writer.publish(
        BehaviorKind.OCCURRENCE,
        occurrence_payload(name="洗手", occurred_on=at(18).date(), started_at=at(18), last_observed_at=at(40)),
    )
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    report = asyncio.run(harness.runner.run_once())
    assert report.published_occurrences == 1
    names = sorted(document.fields["name"] for document in harness.tree.iter_documents(BehaviorKind.OCCURRENCE))
    assert names == ["洗手", "洗手-2"]
    assert not (tmp_path / "reduction" / "staged.json").exists()
    # 消歧记录 = 已知重复：命中账不计（与 rebuild、预测树同一口径）
    assert harness.kind_store.read().registry.entry_of("洗手").hit_count == 0


def test_kind_hits_use_the_checkpoint_content_as_the_idempotence_key(tmp_path) -> None:
    """冻结时钟下两轮 sweep 的 staged_at 相同：命中账仍按检查点内容各记一次。"""

    harness = Harness(tmp_path)
    source = harness.deliver(OBS_A, OBS_B, OBS_C)
    seed_wash_chain(harness, source)
    assert asyncio.run(harness.runner.run_once()).published_occurrences == 1
    obs_d, obs_e = observation(70, "第二次走到水池边"), observation(80, "再次搓手")
    second = harness.deliver(obs_d, obs_e, seed="second")
    harness.judgements.put_payload(
        judgement_record(
            "second-head",
            behavior="洗手",
            started_at=at(70),
            last_observed_at=at(80),
            evidence_ready_at=at(82),
            observation_ids=(obs_d.observation_id, obs_e.observation_id),
            source_refs=(second,),
            goal="清洁双手",
            summary="第二次洗手",
            status="completed",
            status_basis="observed",
            basis=(),
        )
    )
    assert asyncio.run(harness.runner.run_once()).published_occurrences == 1
    assert harness.kind_store.read().registry.entry_of("洗手").hit_count == 2


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


def test_data_clock_is_clamped_to_the_wall_clock(tmp_path) -> None:
    """一条 2099 的坏时间戳不能把词表删空：过期时钟钳在墙钟当日之内。"""

    from datetime import date, timedelta

    from behavior.kinds import BehaviorKindEntry

    harness = Harness(tmp_path, known_kinds=())
    recent = BehaviorKindEntry(token="打球", label="打球").with_hit(harness.now.date() - timedelta(days=10))
    harness.kind_store.replace(BehaviorKindRegistry({"打球": recent, "洗手": ()}), expected_revision=0, timestamp=harness.now)
    harness.runner._kind_signals = []
    harness.runner._record_kind_hits([("洗手", date(2099, 1, 1))], "cp-1", harness.now)
    registry = harness.kind_store.read().registry
    assert "打球" in registry.tokens  # 若按 2099 判，它早该被删
    assert registry.entry_of("洗手").hit_days == (date(2099, 1, 1),)
