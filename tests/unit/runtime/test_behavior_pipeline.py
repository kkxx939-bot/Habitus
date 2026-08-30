"""行为管线的组合根接线：可选启用、配置注入的一致性、投递正门与端到端链路。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from behavior.fusion.config import FUSION_CONTEXT_LOOKBACK_SECONDS
from behavior.model import BehaviorKind
from behavior.observation import (
    BehaviorObservation,
    BehaviorObservationBatch,
    BehaviorObservationConfig,
    BehaviorObservationEnvelope,
)
from foundation.integrity import canonical_digest
from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from infrastructure.vector import VectorStoreFactory
from ModelClient import ModelResponse, ProviderCapabilities, ProviderFactory
from Runtime import build_runtime
from Runtime.runtime import RuntimeStateError
from tests.integration.test_runtime_assembly import (
    REPOSITORY_ROOT,
    FakeEmbeddingProvider,
    FakeRerankProvider,
    FakeVectorBackend,
    prepare_chat_request,
    runtime_config,
    runtime_dependencies,
)

SUBJECT = "家庭成员A"
CST = timezone(timedelta(hours=8))
OBSERVATION_CONFIG = BehaviorObservationConfig()


def behavior_enabled_config(tmp_path: Path):
    """在既有 fake 路由配置之上启用行为侧（窗口取唯一出处默认值 1 小时）。"""

    raw = yaml.safe_load((REPOSITORY_ROOT / "Config" / "example.yaml").read_text(encoding="utf-8"))
    raw["storage"]["root"] = str(tmp_path / "data")
    raw["models"]["chat"]["route"].update(provider="fake", adapter="fake_chat", credential_ref="")
    raw["models"]["embedding"]["route"].update(
        provider="fake", adapter="fake_embedding", credential_ref=""
    )
    raw["models"]["rerank"]["route"].update(
        provider="fake", adapter="fake_rerank", credential_ref=""
    )
    raw["memory"]["vector_store"]["route"].update(
        provider="fake", adapter="fake_vector", credential_ref=""
    )
    raw["conversation"]["summary_vector_store"]["route"].update(
        provider="fake", adapter="fake_vector", credential_ref=""
    )
    raw["behavior"] = {"primary_subject": SUBJECT}
    from Config import HabitusConfig

    return HabitusConfig.from_mapping(raw)


class ScriptedChatProvider:
    """按脚本回放结构化输出；行为端到端里第一答融合判断，之后的（语义层）拿到垃圾。"""

    capabilities = ProviderCapabilities()
    is_remote = False
    prepare = staticmethod(prepare_chat_request)

    def __init__(self, provider_name: str, model: str, bodies: list[dict]) -> None:
        self.provider_name = provider_name
        self.model = model
        self.bodies = bodies
        self.calls = 0

    async def complete_async(self, request) -> ModelResponse:
        body = self.bodies[min(self.calls, len(self.bodies) - 1)]
        self.calls += 1
        return ModelResponse(
            content=json.dumps(body, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
        )

    def complete(self, request) -> ModelResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request):  # pragma: no cover
        raise NotImplementedError

    def stream_async(self, request):  # pragma: no cover
        raise NotImplementedError

    def health_check(self):  # pragma: no cover
        return {}

    async def aclose(self) -> None:  # pragma: no cover
        return None


def scripted_dependencies(bodies: list[dict]):
    providers = ProviderFactory()
    providers.register_adapter(
        "chat",
        "fake_chat",
        lambda context: ScriptedChatProvider(context.route.provider, context.route.model, bodies),
    )
    providers.register_adapter(
        "embedding",
        "fake_embedding",
        lambda context: FakeEmbeddingProvider(
            context.route.provider, context.route.model, context.config.dimension
        ),
    )
    providers.register_adapter(
        "rerank",
        "fake_rerank",
        lambda context: FakeRerankProvider(context.route.provider, context.route.model),
    )
    vectors = VectorStoreFactory()
    vectors.register_adapter(
        "fake_vector",
        lambda context: FakeVectorBackend(context.config.provider, context.config.collection),
        requires_cross_process_publication_fencing=False,
    )
    return providers, vectors


def observation_envelope() -> BehaviorObservationEnvelope:
    base = datetime.now(tz=CST) - timedelta(hours=2)
    observations = tuple(
        BehaviorObservation.create(
            observer_id="home-a/hall",
            occurred_at=base + timedelta(seconds=offset),
            available_at=base + timedelta(seconds=offset + 1),
            modality="vision",
            semantics=semantics,
            participants=[SUBJECT],
            knowledge_state="observed",
            confidence=0.9,
            evidence_refs=[f"cam:{offset}"],
            config=OBSERVATION_CONFIG,
        )
        for offset, semantics in ((0, "人走到水池边"), (4, "人在洗手"))
    )
    return BehaviorObservationEnvelope.create(
        observer_id="home-a/hall",
        protocol="habitus_behavior_observation_v1",
        batch=BehaviorObservationBatch(observer_id="home-a/hall", observations=observations),
        delivery_id=canonical_digest("delivery-e2e"),
        recorded_at=datetime.now(tz=CST),
        config=OBSERVATION_CONFIG,
    )


FUSION_BODY = {
    "judgements": [
        {
            "judgement_no": 1,
            "subjects": [SUBJECT],
            "behavior": "洗手",
            "goal": "清洁双手",
            "summary": "在水池边洗了手",
            "basis": [{"basis_no": 1, "semantics": "走到水池边冲洗双手"}],
            "status": "completed",
            "status_basis": "observed",
            "relations": [],
        }
    ],
    "frames": [
        {"no": 1, "assignments": [{"judgement_no": 1, "basis_no": 1}]},
        {"no": 2, "assignments": [{"judgement_no": 1, "basis_no": 1}]},
    ],
}


def test_behavior_stays_dark_until_a_subject_is_configured(tmp_path: Path) -> None:
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        runtime_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    assert runtime.components.behavior is None
    runtime.initialize()
    with pytest.raises(RuntimeStateError, match="behavior pipeline is not configured"):
        asyncio.run(runtime.deliver_behavior_observations(observation_envelope()))


def test_enabled_wiring_shares_one_lookback_and_one_chat_client(tmp_path: Path) -> None:
    """接线一致性：融合与归约拿到**同一份**窗口数值；模型客户端与 memory 共用同一个实例。"""

    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    behavior = runtime.components.behavior
    assert behavior is not None
    assert behavior.fusion_runner.context_lookback_seconds == FUSION_CONTEXT_LOOKBACK_SECONDS
    assert (
        behavior.reduction_runner.context_lookback_seconds
        == behavior.fusion_runner.context_lookback_seconds
    )
    assert behavior.fusion_runner.primary_subject == SUBJECT
    assert (
        behavior.fusion_runner.fuser.client
        is runtime.components.models.structured_chat
    )
    assert behavior.kind_store.path == behavior.tree.root / "kinds.md"
    # 词表向量旁册随 embedder 组装、与词表同根；身份键只从配置取
    vectors = behavior.reduction_runner.kind_vectors
    assert vectors is not None and vectors.path == behavior.tree.root / "kinds.vectors.json"
    assert vectors.dimension == runtime.config.models.embedding.dimension
    assert behavior.reduction_runner.kind_resolver.embedder is runtime.components.models.embedder
    # 配置层不复制窗口默认值：BehaviorConfig 缺省为 None，由组合根从唯一出处解析
    from Config.behavior import BehaviorConfig

    assert BehaviorConfig().context_lookback_seconds is None
    assert FUSION_CONTEXT_LOOKBACK_SECONDS == 3_600.0


def test_delivery_to_tree_end_to_end_through_the_runtime(tmp_path: Path) -> None:
    """全链路接缝：投递正门 → 入队 → 融合（脚本模型）→ 判断/回执 → 归约 sweep → 行为树。

    语义层的模型响应故意给垃圾：刷新按设计降级成信号、不阻塞归约——树上必须已有 occurrence。
    """

    providers, vectors = scripted_dependencies([FUSION_BODY])
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    behavior = runtime.components.behavior
    assert behavior is not None

    source_id = asyncio.run(runtime.deliver_behavior_observations(observation_envelope()))
    assert len(source_id) == 64

    worked = asyncio.run(behavior.fusion_worker.run_once())
    assert worked is True
    assert len(behavior.judgements.list()) == 1  # 判断落库
    assert len(behavior.receipts.list()) == 1  # 回执覆盖观测 → 封口前沿放行
    assert behavior.jobs.oldest_uncommitted() is None  # 作业提交即清
    observation_ids = {
        item.observation_id for envelope in behavior.observations.list() for item in envelope.batch.observations
    }
    assert observation_ids <= behavior.fusion_runner.coverage.covered_observation_ids(datetime.now(tz=CST))

    report = asyncio.run(behavior.reduction_runner.run_once())

    assert report.published_occurrences == 1
    addresses = behavior.tree.list_addresses(BehaviorKind.OCCURRENCE)
    assert [address.name for address in addresses] == ["洗手"]
    document = behavior.tree.read(addresses[0])
    assert document.fields["kind_token"] == "洗手"
    assert document.fields["status"] == "completed"
    # 原料消费即释放：判断与交付在发布到树后即删，真正的数据只在树上；覆盖索引仍记得这批观测
    # 已融合（上游补发时靠它去重）。
    assert behavior.judgements.list() == ()
    assert behavior.observations.list() == ()
    assert observation_ids <= behavior.fusion_runner.coverage.covered_observation_ids(datetime.now(tz=CST))
    # 语义层被垃圾响应打断：降级为信号，不阻塞归约
    assert any("semantic refresh failed" in note for note in report.dropped_edges)


def test_runtime_start_and_stop_manage_behavior_workers(tmp_path: Path) -> None:
    providers, vectors = scripted_dependencies([FUSION_BODY])
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    async def scenario() -> None:
        await runtime.start()
        behavior = runtime.components.behavior
        assert behavior is not None
        assert behavior.fusion_worker.running and behavior.reduction_worker.running
        await runtime.stop()
        assert not behavior.fusion_worker.running
        assert not behavior.reduction_worker.running

    asyncio.run(scenario())


def test_storage_layout_follows_each_stores_root_contract(tmp_path: Path) -> None:
    """全部存储传同一个行为根：传子目录会目录二次嵌套 + 作业锁键漂移（评审实测的口吃布局）。"""

    providers, vectors = scripted_dependencies([FUSION_BODY])
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    behavior = runtime.components.behavior
    assert behavior is not None
    asyncio.run(runtime.deliver_behavior_observations(observation_envelope()))
    asyncio.run(behavior.fusion_worker.run_once())

    root = runtime.config.behavior_root
    assert (root / "observations" / "envelopes").is_dir()
    assert (root / "fusion" / "judgements").is_dir()
    assert (root / "fusion" / "receipts").is_dir()
    assert (root / "fusion" / "jobs").is_dir()
    assert not (root / "observations" / "observations").exists()  # 口吃布局不许回来
    assert not (root / "fusion" / "jobs" / "fusion").exists()


def test_close_stops_behavior_workers_like_stop_does(tmp_path: Path) -> None:
    """``async with`` 的退出走 close 不走 stop——close 漏停行为循环会让它带着已关闭的
    模型客户端继续空转（评审实测泄漏）。"""

    providers, vectors = scripted_dependencies([FUSION_BODY])
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    async def scenario() -> None:
        await runtime.start()
        behavior = runtime.components.behavior
        assert behavior is not None
        assert behavior.fusion_worker.running
        await runtime.close()
        assert not behavior.fusion_worker.running
        assert not behavior.reduction_worker.running

    asyncio.run(scenario())


def test_resident_loop_consumes_a_delivery_end_to_end(tmp_path: Path) -> None:
    """常驻循环真跑：投递唤醒 → 循环自己完成融合——wake/clear 的竞态只存在于循环里，
    手动 run_once 测不到。"""

    providers, vectors = scripted_dependencies([FUSION_BODY])
    config = behavior_enabled_config(tmp_path)
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    async def scenario() -> None:
        await runtime.start()
        behavior = runtime.components.behavior
        assert behavior is not None
        envelope = observation_envelope()
        first_id = await runtime.deliver_behavior_observations(envelope)
        for _ in range(200):  # 最多等 10s
            if len(behavior.judgements.list()) == 1:
                break
            await asyncio.sleep(0.05)
        assert len(behavior.judgements.list()) == 1
        assert behavior.fusion_worker.last_error is None

        # 幂等重投：同身份同内容 → 同交付、不产生第二次融合（同身份异内容会 fail-closed，
        # 那是另一条已被存储层测试钉死的路径）
        second_id = await runtime.deliver_behavior_observations(envelope)
        await asyncio.sleep(0.3)
        assert second_id == first_id
        assert len(behavior.judgements.list()) == 1
        await runtime.close()

    asyncio.run(scenario())


def test_component_group_rejects_foreign_instances_and_split_windows(tmp_path: Path) -> None:
    """行为组件组的结构性契约：外来存储实例、融合与归约的窗口分叉，都在构造期被拒。"""

    from dataclasses import replace

    from behavior.fusion.store import BehaviorJudgementStore

    providers, vectors = scripted_dependencies([FUSION_BODY])
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    behavior = runtime.components.behavior
    assert behavior is not None
    foreign = BehaviorJudgementStore(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="share one"):
        replace(behavior, judgements=foreign)


def test_fusion_worker_renews_the_lease_during_slow_execution() -> None:
    """慢模型调用期间租约必须被持续续期——不续约的下场是租约被接管、白烧模型调用（评审推演）。"""

    class _JobsStub:
        class config:
            lease_ttl_seconds = 3  # interval = 1s

        def __init__(self) -> None:
            self.renewals = 0

        def renew(self, lease) -> None:
            self.renewals += 1

    class _RunnerStub:
        def __init__(self) -> None:
            self.jobs = _JobsStub()

        def claim(self, worker_id):
            return object()

        async def execute(self, lease):
            await asyncio.sleep(2.6)
            # 与真实 ``BehaviorFusionRunResult`` 同形：worker 执行完会读降级记录做计数。
            from types import SimpleNamespace

            return SimpleNamespace(degradations=(), fused=False, job=SimpleNamespace(job_id="stub"))

    class _EnqueuerStub:
        def enqueue_ready(self) -> None:
            return None

    from Runtime.behavior import BehaviorFusionWorker

    worker = BehaviorFusionWorker(
        _EnqueuerStub(),  # type: ignore[arg-type]
        _RunnerStub(),  # type: ignore[arg-type]
        poll_interval_seconds=1.0,
        shutdown_timeout_seconds=5.0,
    )
    asyncio.run(worker.run_once())
    assert worker.runner.jobs.renewals >= 2  # type: ignore[attr-defined]


def test_reduction_worker_treats_lock_busy_as_a_skip_not_a_failure() -> None:
    from behavior.reduction import BehaviorReductionBusyError
    from Runtime.behavior import BehaviorReductionWorker

    class _BusyRunner:
        def __init__(self) -> None:
            self.calls = 0

        async def run_once(self):
            self.calls += 1
            raise BehaviorReductionBusyError("lock busy")

    worker = BehaviorReductionWorker(
        _BusyRunner(),  # type: ignore[arg-type]
        interval_seconds=1.0,
        shutdown_timeout_seconds=5.0,
    )

    async def scenario() -> None:
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()

    asyncio.run(scenario())
    assert worker.runner.calls >= 1  # type: ignore[attr-defined]
    assert worker.last_error is None  # 让路不是故障


def test_kind_merge_and_rebuild_are_exposed_through_the_access_layer(tmp_path) -> None:
    """词表运维动作从接入层暴露（与观测投递同一正门形态），不让调用方伸手进归约 runner。"""

    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    rebuilt = asyncio.run(runtime.rebuild_behavior_kinds())
    assert rebuilt.occurrences == 0 and rebuilt.kinds == 0
    merged = asyncio.run(runtime.merge_behavior_kinds("洗手", "清洁双手"))
    assert merged.restamped == 0 and merged.days == ()
    # 行为侧未启用时同样明确拒绝
    dark = build_runtime(
        runtime_config(tmp_path / "dark"), providers=providers, vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    dark.initialize()
    with pytest.raises(RuntimeStateError, match="behavior pipeline is not configured"):
        asyncio.run(dark.rebuild_behavior_kinds())
