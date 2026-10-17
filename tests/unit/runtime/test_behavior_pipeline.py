"""行为管线的组合根接线：可选启用、配置注入的一致性、物理 lane 的正门与端到端链路。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from habitus.behavior.fusion.config import FUSION_CONTEXT_LOOKBACK_SECONDS
from habitus.behavior.fusion.lanes.physical import PhysicalDeliveryError, PhysicalLane, UnitDisposition
from habitus.behavior.fusion.lanes.session import SessionLane
from habitus.behavior.model import BehaviorKind
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.infrastructure.vector import VectorStoreFactory
from habitus.model_client import ModelResponse, ProviderCapabilities, ProviderFactory
from habitus.runtime import build_runtime
from habitus.runtime.runtime import RuntimeStateError
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


def behavior_enabled_config(tmp_path: Path, **behavior: object):
    """在既有 fake 路由配置之上启用行为侧（窗口取唯一出处默认值 1 小时）；``behavior`` 覆盖行为侧其余字段。"""

    raw = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    raw["storage"]["root"] = str(tmp_path / "data")
    raw["models"]["chat"]["route"].update(provider="fake", adapter="fake_chat", credential_ref="")
    raw["models"]["embedding"]["route"].update(provider="fake", adapter="fake_embedding", credential_ref="")
    raw["models"]["rerank"]["route"].update(provider="fake", adapter="fake_rerank", credential_ref="")
    raw["memory"]["vector_store"]["route"].update(provider="fake", adapter="fake_vector", credential_ref="")
    raw["conversation"]["summary_vector_store"]["route"].update(
        provider="fake", adapter="fake_vector", credential_ref=""
    )
    raw["behavior"] = {"primary_subject": SUBJECT, **behavior}
    from habitus.config import HabitusConfig

    return HabitusConfig.from_mapping(raw)


class ScriptedChatProvider:
    """按脚本回放结构化输出；行为端到端里第一答词表的白天归类，之后的（语义层）拿到垃圾。"""

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
        lambda context: FakeEmbeddingProvider(context.route.provider, context.route.model, context.config.dimension),
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


def physical_delivery() -> dict:
    """手表两小时前的一份交付：洗了一次手，随后有一段在充电。时刻都早于封口视界，送到就能封口。"""

    base = (datetime.now(tz=CST) - timedelta(hours=3)).replace(microsecond=0)

    def at(seconds: int) -> str:
        return (base + timedelta(seconds=seconds)).isoformat()

    return {
        "source": "physical/s12",
        "device": "watch-01",
        "delivery_id": "delivery-e2e",
        "sent_at": at(1_900),
        "segments": [{"id": "seg-1", "behavior": "洗手", "started_at": at(60), "ended_at": at(98)}],
        "coverage": [
            {"started_at": at(0), "ended_at": at(600), "state": "observing"},
            {"started_at": at(600), "ended_at": at(1_800), "state": "charging"},
        ],
    }


CLASSIFY_BODY = {"items": [{"record": "R1", "reason": "在水池边洗手", "choice": "都不是", "proposed": "洗手"}]}


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
        asyncio.run(runtime.deliver_physical(physical_delivery()))


def test_each_lane_gets_its_own_whole_pipeline_and_they_share_only_the_chat_client(tmp_path: Path) -> None:
    """两条 lane 各是一整套（存储、融合、归约、行为树、词表），互不共用；模型客户端与 memory 共用同一个实例。"""

    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    branches = runtime.components.behavior
    assert branches is not None
    from habitus.behavior.reduction import DEFAULT_SWEEP_LOCK_TTL_SECONDS

    root = runtime.config.behavior_root
    assert [(branch.lane.value, branch.tree.root) for branch in branches] == [
        ("session", (root / "session" / "tree").resolve()),
        ("physical", (root / "physical" / "tree").resolve()),
    ]
    assert isinstance(branches.session.fusion, SessionLane)
    assert isinstance(branches.physical.fusion, PhysicalLane)
    assert branches.session.fusion.describer.client is runtime.components.models.structured_chat
    for behavior in branches:
        assert behavior.reduction_runner.context_lookback_seconds == FUSION_CONTEXT_LOOKBACK_SECONDS
        assert behavior.fusion.recorder.primary_subject == SUBJECT
        assert behavior.fusion.recorder.coverage is behavior.coverage
        assert behavior.kind_store.catalog_path == behavior.tree.root / "kinds.md"
        # 这一支归的是它自己那条 lane 的类
        assert behavior.reduction_runner.kinds.lane is behavior.lane
        # 词表两块（白天归类、整理活）共用这一支的那一份词表存储与组合根的那一个结构化客户端
        assert behavior.reduction_runner.kinds.store is behavior.kind_store
        assert behavior.reduction_runner.vocabulary_jobs.store is behavior.kind_store
        assert behavior.reduction_runner.kinds.classifier.caller.client is runtime.components.models.structured_chat
        # 所有用户都装定期拆改（裁定 17），锚点自检与白天归类共用同一个归类器
        assert (
            behavior.reduction_runner.vocabulary_jobs.reviser.anchors.classifier
            is behavior.reduction_runner.kinds.classifier
        )
        # sweep 锁 TTL 走 Config 边界（周尺度回填要调大它，不该改代码）；留空取领域默认
        assert behavior.reduction_runner.sweep_lock_ttl_seconds == DEFAULT_SWEEP_LOCK_TTL_SECONDS
    # 两支不共用任何一样东西：存储、词表、归约、Worker 都各是各的
    for name in ("observations", "judgements", "receipts", "coverage", "tree", "kind_store", "reduction_runner"):
        assert getattr(branches.session, name) is not getattr(branches.physical, name)
    assert branches.session.reduction_worker is not branches.physical.reduction_worker
    # 配置层不复制窗口默认值：BehaviorConfig 缺省为 None，由组合根从唯一出处解析
    from habitus.config.behavior import BehaviorConfig

    assert BehaviorConfig().context_lookback_seconds is None
    assert FUSION_CONTEXT_LOOKBACK_SECONDS == 3_600.0


def test_a_physical_delivery_reaches_the_tree_end_to_end_through_the_runtime(tmp_path: Path) -> None:
    """全链路接缝：物理 lane 的正门 → 一段一条判断（不调模型）→ 归约 sweep → 行为树上的行为与「未观测」空白。

    语义层的模型响应故意给垃圾：刷新按设计降级成信号、不阻塞归约——树上必须已有 occurrence。
    """

    # 唯一的模型调用是词表的白天归类：词表从空开始（裁定 29），这一条判成一件事、提议名「洗手」进待定池。
    # 之后的语义层调用也会拿到它——对语义层是垃圾，按设计降级成信号。
    providers, vectors = scripted_dependencies([CLASSIFY_BODY])
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    branches = runtime.components.behavior
    assert branches is not None
    behavior = branches.physical
    payload = physical_delivery()

    outcomes = asyncio.run(runtime.deliver_physical(payload))
    assert [item.disposition for item in outcomes] == [UnitDisposition.RECORDED, UnitDisposition.RECORDED]
    assert len(behavior.judgements.list()) == 2  # 一段行为、一段没在看
    assert len(behavior.receipts.list()) == 2
    # 幂等重投：同一段再送一遍不重复写；不合约定的整份拒收
    again = asyncio.run(runtime.deliver_physical({**payload, "delivery_id": "delivery-e2e-resend"}))
    assert {item.disposition for item in again} == {UnitDisposition.ALREADY_RECORDED}
    assert len(behavior.judgements.list()) == 2
    with pytest.raises(PhysicalDeliveryError, match="outside the contract"):
        asyncio.run(runtime.deliver_physical({**payload, "heart_rate": 92}))

    report = asyncio.run(behavior.reduction_runner.run_once())

    assert (report.published_occurrences, report.published_gaps) == (1, 1)
    addresses = behavior.tree.list_addresses(BehaviorKind.OCCURRENCE)
    assert [address.name for address in addresses] == ["洗手"]
    document = behavior.tree.read(addresses[0])
    assert document.fields["kind_token"] == "p-待定"
    (pending,) = behavior.kind_store.read_pending().pool.entries.values()
    assert pending.proposed == "洗手"
    # 手表在充电的那一段是"没在看"，不是"看了没读懂"
    (gap,) = behavior.tree.list_addresses(BehaviorKind.GAP)
    assert behavior.tree.read(gap).fields["gap_kind"] == "未观测"
    # 原料消费即释放：判断与凭据在发布到树后即删，真正的数据只在树上；收过哪些段另有一份小账记着，
    # 之后再送来照样认得出。
    assert behavior.judgements.list() == ()
    assert behavior.observations.list() == ()
    late = asyncio.run(runtime.deliver_physical({**payload, "delivery_id": "delivery-e2e-late"}))
    assert {item.disposition for item in late} == {UnitDisposition.ALREADY_RECORDED}
    # 语义层被垃圾响应打断：降级为信号，不阻塞归约
    assert any("semantic refresh failed" in note for note in report.dropped_edges)
    # 会话那一支什么都没收到：手表的行为与空白只在物理这一支上
    session = branches.session
    assert session.judgements.list() == () and session.receipts.list() == ()
    assert session.tree.list_addresses(BehaviorKind.OCCURRENCE) == ()
    assert session.tree.list_addresses(BehaviorKind.GAP) == ()
    assert asyncio.run(session.reduction_runner.run_once()).published_gaps == 0


def test_runtime_start_and_stop_manage_behavior_workers(tmp_path: Path) -> None:
    providers, vectors = scripted_dependencies([CLASSIFY_BODY])
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    async def scenario() -> None:
        await runtime.start()
        branches = runtime.components.behavior
        assert branches is not None
        assert all(branch.reduction_worker.running for branch in branches)
        await runtime.stop()
        assert not any(branch.reduction_worker.running for branch in branches)

    asyncio.run(scenario())


def test_storage_layout_follows_each_stores_root_contract(tmp_path: Path) -> None:
    """全部存储传同一个行为根：传子目录会目录二次嵌套（评审实测的口吃布局）。"""

    providers, vectors = scripted_dependencies([CLASSIFY_BODY])
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    assert runtime.components.behavior is not None
    asyncio.run(runtime.deliver_physical(physical_delivery()))

    root = runtime.config.behavior_root / "physical"
    assert (runtime.config.behavior_root / "session" / "tree").is_dir()  # 另一支有自己的树，里面没有这份交付
    assert not (runtime.config.behavior_root / "session" / "observations").exists()
    assert (root / "observations" / "envelopes").is_dir()
    assert (root / "fusion" / "judgements").is_dir()
    assert (root / "fusion" / "receipts").is_dir()
    assert (root / "fusion" / "coverage").is_dir()
    assert (root / "fusion" / "physical" / "segments").is_dir()
    assert not (root / "observations" / "observations").exists()  # 口吃布局不许回来
    assert not (root / "fusion" / "fusion").exists()


def test_close_stops_behavior_workers_like_stop_does(tmp_path: Path) -> None:
    """``async with`` 的退出走 close 不走 stop——close 漏停行为循环会让它带着已关闭的
    模型客户端继续空转（评审实测泄漏）。"""

    providers, vectors = scripted_dependencies([CLASSIFY_BODY])
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    async def scenario() -> None:
        await runtime.start()
        branches = runtime.components.behavior
        assert branches is not None
        assert all(branch.reduction_worker.running for branch in branches)
        await runtime.close()
        assert not any(branch.reduction_worker.running for branch in branches)

    asyncio.run(scenario())


def test_component_group_rejects_foreign_instances(tmp_path: Path) -> None:
    """行为组件组的结构性契约：外来的存储实例在构造期被拒。"""

    from dataclasses import replace

    from habitus.behavior.fusion.store import BehaviorJudgementStore

    providers, vectors = scripted_dependencies([CLASSIFY_BODY])
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    branches = runtime.components.behavior
    assert branches is not None
    foreign = BehaviorJudgementStore(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="share one"):
        replace(branches.session, judgements=foreign)
    # 两支不许是同一套：把物理那一支换成会话那一支的部件，构造期就拒
    with pytest.raises(TypeError, match="physical lane's components"):
        replace(branches, physical=branches.session)


def test_reduction_worker_treats_lock_busy_as_a_skip_not_a_failure() -> None:
    from habitus.behavior.reduction import BehaviorReductionBusyError
    from habitus.runtime.behavior import BehaviorReductionWorker

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


def test_vocabulary_jobs_are_exposed_through_the_access_layer(tmp_path) -> None:
    """词表运维动作从接入层暴露（与物理 lane 的交付同一正门形态），不让调用方伸手进归约 runner。"""

    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        behavior_enabled_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    grown = asyncio.run(runtime.run_vocabulary_job("grow", lane="session"))
    assert grown.model_calls == 0  # 待定池是空的
    assert runtime.components.behavior is not None
    assert all(branch.kind_store.read().version == 0 for branch in runtime.components.behavior)
    revised = asyncio.run(runtime.run_vocabulary_job("revise", lane="physical"))
    assert (
        revised.model_calls == 0 and revised.days == ()
    )  # 词表从空开始（裁定 29）：两条 lane 都还没有类，没什么可拆改
    with pytest.raises(ValueError, match="unknown vocabulary job"):
        asyncio.run(runtime.run_vocabulary_job("bootstrap", lane="session"))
    # 行为侧未启用时同样明确拒绝
    dark = build_runtime(
        runtime_config(tmp_path / "dark"),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    dark.initialize()
    with pytest.raises(RuntimeStateError, match="behavior pipeline is not configured"):
        asyncio.run(dark.run_vocabulary_job("grow", lane="session"))
