"""顶层 Runtime 对所有领域组件的唯一组装与初始化主链测试。"""

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from habitus.behavior.tree import BehaviorTree
from habitus.config import HabitusConfig
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.infrastructure.vector import VectorStoreFactory
from habitus.integrations.agent import AgentMemoryGateway
from habitus.integrations.http import RuntimeHTTPHandlers
from habitus.integrations.sdk import AgentMemoryHooks, PreparedAgentTurn
from habitus.memory.conversation import ConversationAddress
from habitus.memory.model import MemoryKind
from habitus.memory.retrieval import MemoryRetrievalSufficiency, SQLiteMemoryRecallLifecycleStore
from habitus.memory.uri import MemoryURI
from habitus.model_client import (
    EmbeddingVector,
    ModelResponse,
    ProviderCapabilities,
    ProviderFactory,
)
from habitus.pre.conversation import ConversationAdapterRegistry
from habitus.runtime import (
    LifecycleWorkerState,
    MemoryWorkerState,
    RuntimeState,
    RuntimeStateError,
    build_runtime,
)
from tests.helpers import BASE_TIME
from tests.model_helpers import prepare_chat_request

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class FakeChatProvider:
    capabilities = ProviderCapabilities()
    is_remote = False

    def __init__(self, provider_name: str, model: str) -> None:
        self.provider_name = provider_name
        self.model = model
        self.closed = False

    prepare = staticmethod(prepare_chat_request)

    def complete(self, _request):
        return ModelResponse("{}", self.model, self.provider_name)

    async def complete_async(self, request):
        return self.complete(request)

    def stream(self, _request):
        return iter(())

    async def stream_async(self, _request):
        if False:
            yield None

    def health_check(self):
        return {"ok": True}

    async def aclose(self):
        self.closed = True


class FakeEmbeddingProvider:
    is_remote = False

    def __init__(self, provider_name: str, model: str, dimension: int) -> None:
        self.provider_name = provider_name
        self.model = model
        self.dimension = dimension
        self.closed = False

    async def embed(self, _text: str, *, is_query: bool) -> EmbeddingVector:
        return EmbeddingVector((1.0,) + (0.0,) * (self.dimension - 1))

    async def aclose(self):
        self.closed = True


class FakeRerankProvider:
    is_remote = False

    def __init__(self, provider_name: str, model: str) -> None:
        self.provider_name = provider_name
        self.model = model
        self.closed = False

    async def rerank(self, _query: str, documents) -> tuple[float, ...]:
        return tuple(1.0 for _ in documents)

    async def aclose(self):
        self.closed = True


class FakeVectorBackend:
    adapter_name = "fake_vector"
    requires_cross_process_publication_fencing = False
    max_records = 100_000
    max_search_hits = 10_000

    def __init__(self, provider_name: str, collection: str) -> None:
        self.provider_name = provider_name
        self.collection = collection

    async def initialize(self):
        return None

    async def read_metadata(self, _names):
        return {}

    async def write_metadata(self, _values, *, dimension):
        return None

    async def ensure_schema(self, _dimension, *, published_dimension):
        return None

    async def read(self, _identities):
        return ()

    async def delete_all(self):
        return None

    async def upsert(self, _records):
        return None

    async def delete(self, _identities):
        return None

    async def validate_records(self, _records, *, replacing):
        return None

    async def wait_visible(self, _upserts, _deletes, *, complete):
        return None

    async def search(self, _query_vector, *, filters, limit):
        return ()

    async def scan(self, *, filters, limit):
        return ()

    async def close(self):
        return None


def runtime_dependencies() -> tuple[ProviderFactory, VectorStoreFactory]:
    providers = ProviderFactory()
    providers.register_adapter(
        "chat",
        "fake_chat",
        lambda context: FakeChatProvider(context.route.provider, context.route.model),
    )
    providers.register_adapter(
        "embedding",
        "fake_embedding",
        lambda context: FakeEmbeddingProvider(
            context.route.provider,
            context.route.model,
            context.config.dimension,
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
        lambda context: FakeVectorBackend(
            context.config.provider,
            context.config.collection,
        ),
        requires_cross_process_publication_fencing=False,
    )
    return providers, vectors


def runtime_config(tmp_path: Path, *, with_credentials: bool = False) -> HabitusConfig:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["models"]["chat"]["route"].update(
        provider="fake",
        adapter="fake_chat",
        credential_ref="deepseek" if with_credentials else "",
    )
    payload["models"]["embedding"]["route"].update(
        provider="fake",
        adapter="fake_embedding",
        credential_ref="ark" if with_credentials else "",
    )
    payload["models"]["rerank"]["route"].update(
        provider="fake",
        adapter="fake_rerank",
        credential_ref="dashscope" if with_credentials else "",
    )
    for name in ("vector_store",):
        payload["memory"][name]["route"].update(
            provider="fake",
            adapter="fake_vector",
            credential_ref="vikingdb" if with_credentials else "",
        )
    payload["conversation"]["summary_vector_store"]["route"].update(
        provider="fake",
        adapter="fake_vector",
        credential_ref="vikingdb" if with_credentials else "",
    )
    if with_credentials:
        payload["credentials"]["deepseek"]["api_key"] = "deepseek-secret"
        payload["credentials"]["ark"]["api_key"] = "ark-secret"
        payload["credentials"]["dashscope"]["api_key"] = "dashscope-secret"
        payload["credentials"]["vikingdb"]["access_key"] = "viking-access"
        payload["credentials"]["vikingdb"]["secret_key"] = "viking-secret"
    return HabitusConfig.from_mapping(payload)


def test_assembly_resolves_each_provider_and_database_credential_by_reference(tmp_path: Path) -> None:
    model_credentials: dict[str, str] = {}
    vector_credentials: list[dict[str, str]] = []
    providers = ProviderFactory()

    def chat_builder(context):
        model_credentials["chat"] = context.api_key
        return FakeChatProvider(context.route.provider, context.route.model)

    def embedding_builder(context):
        model_credentials["embedding"] = context.api_key
        return FakeEmbeddingProvider(
            context.route.provider,
            context.route.model,
            context.config.dimension,
        )

    def rerank_builder(context):
        model_credentials["rerank"] = context.api_key
        return FakeRerankProvider(context.route.provider, context.route.model)

    providers.register_adapter("chat", "fake_chat", chat_builder)
    providers.register_adapter("embedding", "fake_embedding", embedding_builder)
    providers.register_adapter("rerank", "fake_rerank", rerank_builder)
    vectors = VectorStoreFactory()

    def vector_builder(context):
        vector_credentials.append(dict(context.credentials))
        return FakeVectorBackend(context.config.provider, context.config.collection)

    vectors.register_adapter(
        "fake_vector",
        vector_builder,
        requires_cross_process_publication_fencing=False,
    )

    build_runtime(
        runtime_config(tmp_path, with_credentials=True),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    assert model_credentials == {
        "chat": "deepseek-secret",
        "embedding": "ark-secret",
        "rerank": "dashscope-secret",
    }
    assert vector_credentials == [
        {"access_key": "viking-access", "secret_key": "viking-secret"},
        {"access_key": "viking-access", "secret_key": "viking-secret"},
    ]


def test_assembly_wires_one_shared_chain_without_touching_storage(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()

    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    components = runtime.components
    assert runtime.state is RuntimeState.CREATED
    assert not config.storage_root.exists()
    assert components.workflow.enqueuer.conversations is components.conversation.journal
    assert components.workflow.runner.executor.editor is components.memory.editor
    assert components.memory.search.semantic_search.index is components.memory.vector_index
    assert components.memory.search.summary_search is components.conversation.summary_vector_index
    assert components.memory.search.recall_lifecycle.config is config.memory.recall_lifecycle
    recall_store = components.memory.search.recall_lifecycle.store
    assert isinstance(recall_store, SQLiteMemoryRecallLifecycleStore)
    assert recall_store.path == config.workflow_root / "memory_recall_lifecycle.sqlite3"
    assert not recall_store.initialized
    assert not recall_store.path.exists()
    assert components.workflow.worker.runner is components.workflow.runner


def test_assembly_accepts_an_external_harness_protocol_registry(tmp_path: Path) -> None:
    class FutureHarnessAdapter:
        protocol = "future_harness"

        def adapt(self, _payload, _context):
            raise AssertionError("protocol listing must not invoke adaptation")

    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()
    adapters = ConversationAdapterRegistry()
    adapters.register(FutureHarnessAdapter())

    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        conversation_adapters=adapters,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    assert runtime.conversation_protocols() == ("future_harness",)


def test_default_vector_publication_fences_use_dedicated_lock_databases(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()

    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
    )

    components = runtime.components
    memory_store = components.memory.vector_index.store
    summary_store = components.conversation.summary_vector_index.store
    transaction_lock = components.infrastructure.path_lock.lock_store
    memory_lock = memory_store._path_lock.lock_store
    summary_lock = summary_store._path_lock.lock_store

    assert transaction_lock.path == config.workflow_root / "locks.sqlite3"
    assert memory_lock.path == config.workflow_root / "memory_vector_locks.sqlite3"
    assert summary_lock.path == config.workflow_root / "summary_vector_locks.sqlite3"
    assert len({transaction_lock.path, memory_lock.path, summary_lock.path}) == 3
    assert not config.storage_root.exists()


def test_initialize_is_idempotent_and_creates_only_local_durable_roots(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    first = runtime.initialize()
    second = runtime.initialize()

    assert first is second
    assert first.recovered_transaction_ids == ()
    assert first.memory_root == config.memory_root
    assert runtime.state is RuntimeState.READY
    assert config.memory_root.is_dir()
    assert config.workflow_root.is_dir()
    recall_store = runtime.components.memory.search.recall_lifecycle.store
    assert isinstance(recall_store, SQLiteMemoryRecallLifecycleStore)
    assert recall_store.initialized
    assert runtime.components.conversation.summary_use.initialized
    assert runtime.components.memory.lifecycle.operation_store.pending() == ()
    assert recall_store.path.exists()
    assert runtime.components.conversation.summary_use.path.exists()
    assert not config.conversation_root.exists()


def test_disabled_recall_lifecycle_ignores_bad_store_and_skips_batch_constraint(tmp_path: Path) -> None:
    base = runtime_config(tmp_path)
    lifecycle_config = replace(
        base.memory.recall_lifecycle,
        enabled=False,
        max_batch_size=1,
    )
    config = replace(
        base,
        memory=replace(base.memory, recall_lifecycle=lifecycle_config),
    )
    path = config.workflow_root / "memory_recall_lifecycle.sqlite3"
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE memory_recall_lifecycle (uri TEXT PRIMARY KEY)")

    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()

    assert runtime.state is RuntimeState.READY
    recall_store = runtime.components.memory.search.recall_lifecycle.store
    assert isinstance(recall_store, SQLiteMemoryRecallLifecycleStore)
    assert not recall_store.initialized


def test_runtime_start_stop_restart_and_close_coordinate_both_workers(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path),
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )

        await runtime.start()
        await runtime.start()
        assert runtime.state is RuntimeState.RUNNING
        assert runtime.components.workflow.worker.state is MemoryWorkerState.RUNNING
        assert runtime.components.workflow.lifecycle_worker.state is LifecycleWorkerState.RUNNING
        deep_health = await runtime.health(deep=True)
        assert {
            "chat_model",
            "memory_index_consistency",
            "summary_index_consistency",
        }.issubset({check.name for check in deep_health.checks})
        with pytest.raises(RuntimeStateError, match="worker stopped"):
            await runtime.run_next()

        await runtime.stop()
        assert runtime.state is RuntimeState.READY
        assert runtime.components.workflow.worker.state is MemoryWorkerState.STOPPED
        assert runtime.components.workflow.lifecycle_worker.state is LifecycleWorkerState.STOPPED

        await runtime.start()
        assert runtime.state is RuntimeState.RUNNING
        await runtime.close()
        await runtime.close()
        assert runtime.state is RuntimeState.CLOSED
        assert runtime.components.models.chat.provider.closed
        assert runtime.components.models.embedder.provider.closed
        with pytest.raises(RuntimeStateError, match="closed runtime"):
            await runtime.start()

    asyncio.run(scenario())


def test_runtime_public_conversation_interface_returns_pending_consistency_handle(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path),
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )
        runtime.initialize()
        address = ConversationAddress("conversation-public", date(2026, 7, 28))
        assert await runtime.conversation_cursor(address) == 0
        payload = {
            "messages": [
                {"role": "user", "content": "记住我喜欢简洁回答"},
                {"role": "assistant", "content": "好的"},
            ]
        }
        result = await runtime.append_protocol_conversation(
            address,
            protocol="openai_chat_completions",
            payload=payload,
            start_sequence=0,
            occurred_at=BASE_TIME,
            after_turn=False,
            delivery_id="a" * 64,
        )
        assert result.ingest.jobs == ()
        assert result.effective_after_turn is False
        assert result.next_sequence == 2
        assert await runtime.conversation_cursor(address) == 2
        replayed = await runtime.append_protocol_conversation(
            address,
            protocol="openai_chat_completions",
            payload=payload,
            start_sequence=0,
            occurred_at=BASE_TIME,
            after_turn=False,
            delivery_id="a" * 64,
        )
        # Source 重放正式返回首次耐久 Memory Output，而不是从当前 Journal 猜测新结果。
        assert replayed.ingest.append.status.value == "created"
        assert replayed.next_sequence == 2
        second_payload = {
            "messages": [
                {"role": "user", "content": "第二轮"},
                {"role": "assistant", "content": "继续保持简洁"},
            ]
        }
        second = await runtime.append_protocol_conversation(
            address,
            protocol="openai_chat_completions",
            payload=second_payload,
            start_sequence=2,
            occurred_at=BASE_TIME,
            after_turn=False,
        )
        assert second.next_sequence == 4
        stale_replay = await runtime.append_protocol_conversation(
            address,
            protocol="openai_chat_completions",
            payload=payload,
            start_sequence=0,
            occurred_at=BASE_TIME,
            after_turn=False,
        )
        assert stale_replay.ingest.append.status.value == "unchanged"
        assert stale_replay.next_sequence == 4
        assert await runtime.conversation_cursor(address) == 4
        flushed = await runtime.flush_conversation(address)
        assert len(flushed.jobs) == 1
        consistency = await runtime.memory_consistency(flushed.jobs[0])
        assert consistency.state.value == "pending"
        assert await runtime.read_live_conversation(address) is None
        assert len(await runtime.list_conversation_history(address)) == 1
        assert runtime.conversation_protocols() == (
            "anthropic_messages",
            "claude_code",
            "codex_rollout",
            "openai_chat_completions",
            "openai_responses",
            "openclaw",
            "opencode",
        )
        report = await runtime.health()
        assert report.status.value in {"healthy", "degraded"}
        assert "habitus_" in runtime.prometheus_metrics()
        http = RuntimeHTTPHandlers(runtime)
        capabilities = http.capabilities()
        assert capabilities["api_version"] == "1.0"
        assert capabilities["protocols"] == list(runtime.conversation_protocols())
        assert "remember_idempotency_v1" in capabilities["features"]
        status, readiness = await http.readiness()
        assert status in {200, 503}
        assert readiness["status"] in {"healthy", "degraded", "unhealthy"}

        hooks = AgentMemoryHooks(AgentMemoryGateway(runtime))
        hook_session = hooks.new_session(
            "conversation-hook-runtime",
            date(2026, 7, 28),
            "openai_chat_completions",
        )
        prepared = hooks.prepare_after_turn(
            hook_session,
            payload,
            occurred_at=BASE_TIME,
            after_turn=False,
        )
        hook_result = await hooks.after_turn(prepared)
        assert hook_result.session.next_sequence == 2
        hook_closed = await hooks.on_session_close(hook_result.session)
        assert len(hook_closed.flush.jobs) == 1

        retry_gateway = AgentMemoryGateway(runtime)
        attempts = 0

        async def remember_then_lose_response(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            remembered = await retry_gateway.remember(*args, **kwargs)
            if attempts == 1:
                raise TimeoutError("simulated lost response")
            return remembered

        retry_hooks = AgentMemoryHooks(
            SimpleNamespace(
                remember=remember_then_lose_response,
                recall=retry_gateway.recall,
                flush=retry_gateway.flush,
                cursor=retry_gateway.cursor,
            )
        )
        retry_session = retry_hooks.new_session(
            "conversation-hook-retry",
            date(2026, 7, 28),
            "openai_chat_completions",
        )
        retry_prepared = retry_hooks.prepare_after_turn(
            retry_session,
            payload,
            occurred_at=BASE_TIME,
            after_turn=False,
        )
        with pytest.raises(TimeoutError, match="simulated lost response"):
            await retry_hooks.after_turn(retry_prepared)
        retry_address = ConversationAddress("conversation-hook-retry", date(2026, 7, 28))
        assert await runtime.conversation_cursor(retry_address) == 2
        restored_prepared = PreparedAgentTurn.from_dict(retry_prepared.to_dict())
        retry_result = await retry_hooks.after_turn(restored_prepared)
        assert retry_result.session.next_sequence == 2
        retry_live = await runtime.read_live_conversation(retry_address)
        assert retry_live is not None and len(retry_live.messages) == 2
        retry_closed = await retry_hooks.on_session_close(retry_result.session)
        assert len(retry_closed.flush.jobs) == 1
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_memory_search_facades_use_the_real_search_service_and_lifecycle_gate(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path),
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )

        with pytest.raises(RuntimeStateError, match="initialized runtime"):
            await runtime.find_memory("回答偏好")
        with pytest.raises(RuntimeStateError, match="initialized runtime"):
            await runtime.search_memory("之前如何决定")

        runtime.initialize()
        direct = await runtime.find_memory(
            "  回答偏好  ",
            target_uris="memory://preferences",
            limit=2,
            kinds=(MemoryKind.PREFERENCE,),
        )
        assert direct.query == "回答偏好"
        assert direct.target_roots == (MemoryURI.parse("memory://preferences"),)
        assert direct.kinds == (MemoryKind.PREFERENCE,)
        assert direct.memories == ()
        assert direct.retrieval_assessment is None
        assert not direct.summary_fallback_attempted

        contextual = await runtime.search_memory("之前如何决定")
        assert contextual.memories == ()
        assert contextual.retrieval_assessment is not None
        assert contextual.retrieval_assessment.decision is MemoryRetrievalSufficiency.INSUFFICIENT
        assert contextual.summary_fallback_attempted
        assert contextual.summary_fallbacks == ()
        await runtime.close()

    asyncio.run(scenario())


def _fully_enabled_runtime(tmp_path: Path, *, lifecycle: dict | None = None):  # type: ignore[no-untyped-def]
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    if lifecycle:
        payload["lifecycle"] = {**payload.get("lifecycle", {}), **lifecycle}
    payload["storage"]["root"] = str(tmp_path / "data")
    for route, adapter in (
        (payload["models"]["chat"]["route"], "fake_chat"),
        (payload["models"]["embedding"]["route"], "fake_embedding"),
        (payload["models"]["rerank"]["route"], "fake_rerank"),
        (payload["memory"]["vector_store"]["route"], "fake_vector"),
        (payload["conversation"]["summary_vector_store"]["route"], "fake_vector"),
    ):
        route.update(provider="fake", adapter=adapter, credential_ref="")
    payload["behavior"] = {"primary_subject": "家庭成员A"}
    payload["scene"] = {"enabled": True}
    payload["foresight"] = {"enabled": True}
    config = HabitusConfig.from_mapping(payload)
    providers, vectors = runtime_dependencies()
    return build_runtime(
        config, providers=providers, vector_stores=vectors, path_lock=PathLock(ProcessLocalLockStore())
    )


def test_the_whole_derived_chain_assembles_with_every_layer_switched_on(tmp_path: Path) -> None:
    """四组全开走一遍组合根。

    这条路径此前**零覆盖**：语义层与预测层的接线测试都直接调各自的 build，从不经过
    ``build_runtime``，于是一处恒不成立的实例同一性校验让生产配置起不来，而 634 条相关测试全绿。
    """

    runtime = _fully_enabled_runtime(tmp_path)
    branches = runtime.components.behavior
    worker = runtime.components.nightly
    assert branches is not None and worker is not None
    # 整晚那一趟与预测层（与结算）现在只接会话那一支；预测树删了（裁定 59），整晚自己一个常驻节拍。
    behavior = branches.session
    assert worker.at == runtime.config.scene.nightly_time and worker.zone == runtime.config.locale.zone()
    assert worker.last_run.path == runtime.config.scene_root / "nightly" / "last_run.json"
    # 每晚开头的重试接着两支各自的重试堆（裁定 67）；会话一支连同它的会话 lane
    retry = worker.nightly.retry
    assert retry is not None
    from habitus.behavior.retry import RetryKind
    from habitus.behavior.retry.handlers import ChainHandler, TurnHandler

    assert [(item.name, item.backlog, sorted(item.handlers)) for item in retry.branches] == [
        ("session", branches.session.retry_backlog, [RetryKind.CHAIN, RetryKind.TURN]),
        ("physical", branches.physical.retry_backlog, [RetryKind.CHAIN]),
    ]
    session_handlers = retry.branches[0].handlers
    assert isinstance(session_handlers[RetryKind.TURN], TurnHandler)
    assert session_handlers[RetryKind.TURN].lane is branches.session.fusion
    assert isinstance(session_handlers[RetryKind.CHAIN], ChainHandler)
    assert session_handlers[RetryKind.CHAIN].failed is branches.session.reduction_runner.failed_chains
    assert branches.session.fusion.tree is branches.session.tree  # 补救时认"已经落树"（裁定 68 第 5 条）
    # 时窗末当场重读接着会话一支的重试堆（裁定 68 第 1 条）
    assert runtime.components.foresight is not None and runtime.components.foresight.worker is not None
    assert runtime.components.foresight.worker.tick.reread is not None
    assert worker.shutdown_timeout_seconds == runtime.config.scene.nightly_shutdown_timeout_seconds
    foresight = runtime.components.foresight
    assert foresight is not None
    assert foresight.read_side.behavior_tree is behavior.tree
    # 账本在 storage.root/foresight/<支>；结算认的是归约的定稿日。
    assert foresight.ledger.root == runtime.config.foresight_root / "session"
    assert foresight.settlement.ledger is foresight.ledger
    assert foresight.settlement.closed_days == behavior.reduction_runner.closed_days
    # 机会表与账本同住这一支的目录（``opportunities/<lane>/<晚>.json``）
    assert foresight.opportunities.root == foresight.ledger.root
    # 白天那一拍的常驻 worker 接上了：读会话那一支的机会表、同一本账本，未封口的判断经那座桥读（不读预测树，裁定 59）
    from habitus.runtime.foresight_worker import ForesightWorker
    from habitus.runtime.unsealed import UnsealedFromJudgements

    assert isinstance(foresight.worker, ForesightWorker)
    tick = foresight.worker.tick
    assert tick.ledger is foresight.ledger and tick.opportunities is foresight.opportunities
    # 提醒出口（方案 32）：递送那一拍按它的活性问不问模型；会话源的行为侧消费者外面包着它，同一份会话先过出口再进 lane
    from habitus.conversation import ConversationSourceConsumer
    from habitus.runtime.foresight_outlet import ReminderOutlet
    from habitus.runtime.foresight_outlet_consumer import OutletObservingConsumer

    assert isinstance(foresight.outlet, ReminderOutlet) and foresight.outlet.ledger is foresight.ledger
    assert foresight.outlet.store.root == foresight.ledger.root / "outlet"
    assert tick.delivery.outlet is foresight.outlet
    consumer = runtime.components.conversation.behavior_session_consumer
    assert isinstance(consumer, OutletObservingConsumer) and consumer.outlet is foresight.outlet
    assert consumer.inner.lane is branches.session_lane
    assert (
        runtime.components.conversation.source_delivery.consumers[ConversationSourceConsumer.BEHAVIOR_SESSION]
        is consumer
    )
    # 整晚那一趟：结算第一步，语义树读同一棵行为树、机会表写进同一本账本旁边（语义树第 11 步）；"今天"按主体所在地算
    from habitus.runtime.nightly import Nightly

    nightly = worker.nightly
    assert isinstance(nightly, Nightly)
    assert nightly.behavior_tree is behavior.tree and nightly.zone == runtime.config.locale.zone()
    # 当地日历只建一份：语义树、预测层的读侧共用，整晚开头重读的就是它（2026-10-18 四方评审）
    assert nightly.scene.calendar is foresight.read_side.calendar and nightly.refresh_calendar is not None
    assert nightly.foresight.calendar is foresight.read_side.calendar
    assert nightly.scene.behavior_tree is behavior.tree and nightly.scene.relations.root == runtime.config.scene_root
    assert nightly.foresight is not None and nightly.foresight.store is foresight.opportunities
    assert nightly.closed_days == behavior.reduction_runner.closed_days
    assert nightly.settlement is foresight.settlement
    # 单元历史：夜里那一拍重标、白天的单元读同一份（裁定 55 新版第 2 条）
    assert nightly.foresight.history is foresight.history and foresight.history.store.root == foresight.ledger.root
    assert tick.delivery.units.history is foresight.history.store
    assert not hasattr(tick, "prediction") and tick.read_side is foresight.read_side
    assert isinstance(tick.unsealed, UnsealedFromJudgements) and tick.unsealed.judgements is behavior.judgements
    assert tick.delivery is not None and tick.delivery.ledger is foresight.ledger
    assert foresight.worker.interval_seconds == runtime.config.foresight.tick_interval_seconds
    from habitus.runtime.foresight import ForesightRuntimeComponents

    other = BehaviorTree(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="assembled behaviour tree"):
        ForesightRuntimeComponents(
            read_side=replace(foresight.read_side, behavior_tree=other),
            ledger=foresight.ledger,
            settlement=foresight.settlement,
            opportunities=foresight.opportunities,
            history=foresight.history,
        )


def test_the_runtime_starts_and_stops_the_foresight_worker(tmp_path: Path) -> None:
    """Runtime 起停带着预测层的常驻循环：起了在跑（第一拍没有机会表、只报原因，不算失败），停了就停。"""

    import asyncio

    runtime = _fully_enabled_runtime(tmp_path)
    foresight = runtime.components.foresight
    assert foresight is not None and foresight.worker is not None

    async def scenario() -> None:
        await runtime.start()
        await asyncio.sleep(0.05)
        assert foresight.worker is not None and foresight.worker.running and foresight.worker.last_error is None
        await runtime.stop()
        assert not foresight.worker.running

    asyncio.run(scenario())


def test_an_enabled_foresight_layer_that_is_not_attached_to_the_nightly_batch_is_refused(tmp_path: Path) -> None:
    """接线漏了的话，进程照常启动、health 全绿、结算与新语义树的各拍一夜都不会跑，而配置里还开着——无处可查。"""

    runtime = _fully_enabled_runtime(tmp_path)
    components = runtime.components
    assert components.nightly is not None
    components.nightly.nightly.settlement = None  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="first step of the nightly run"):
        replace(components)
    with pytest.raises(ValueError, match="nightly run must be enabled"):
        replace(components, nightly=None)
    components.nightly.nightly.settlement = runtime.components.foresight.settlement  # type: ignore[attr-defined, union-attr]
    components.nightly.nightly.foresight = None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="opportunity table the foresight layer reads"):
        replace(components)


def test_the_runtime_starts_and_stops_the_nightly_worker_and_reports_how_old_the_opportunity_table_is(
    tmp_path: Path,
) -> None:
    """整晚那一趟自己一个常驻节拍，Runtime 起停它；健康面看它活着，接了预测层时再看机会表最晚一晚有多旧（清单 5.1、5.3）。"""

    import asyncio
    from datetime import UTC, datetime, timedelta

    from habitus.foresight.opportunities import OpportunityTable

    runtime = _fully_enabled_runtime(tmp_path)
    worker = runtime.components.nightly
    foresight = runtime.components.foresight
    assert worker is not None and foresight is not None

    def check(report, name):  # type: ignore[no-untyped-def]
        return next(item for item in report.checks if item.name == name)

    async def scenario() -> None:
        report = await runtime.health()
        assert check(report, "opportunities_session").detail == "never_written"  # 第一夜还没到：不算故障
        await runtime.start()
        await asyncio.sleep(0.05)
        assert worker.running
        report = await runtime.health()
        assert check(report, "nightly_worker").status.value == "healthy"
        await runtime.stop()
        assert not worker.running

    asyncio.run(scenario())
    zone = runtime.config.locale.zone()
    # 放假数据有年份读不了：健康面报出来（非关键），进程照常
    worker.nightly.scene.calendar.problems = ("2027.json: 写坏了",)  # type: ignore[attr-defined]
    calendar_check = check(asyncio.run(runtime.health()), "calendar")
    assert (
        calendar_check.status.value == "degraded"
        and "2027.json" in calendar_check.detail
        and not calendar_check.critical
    )
    worker.nightly.scene.calendar.problems = ()  # type: ignore[attr-defined]
    today = datetime.now(UTC).astimezone(zone).date()
    stale = today - timedelta(days=3)  # 连着两个周期（一天一趟）没出新的一晚
    foresight.opportunities.write(OpportunityTable("session", stale, stale, 15, 3, (), ()))
    report = asyncio.run(runtime.health())
    assert check(report, "opportunities_session").status.value == "degraded"
    foresight.opportunities.write(OpportunityTable("session", today, today, 15, 3, (), ()))
    report = asyncio.run(runtime.health())
    item = check(report, "opportunities_session")
    assert item.status.value == "healthy" and item.detail == f"night:{today}" and not item.critical
    # 节拍短于半天：一天最多出一晚新的表，今天那一晚到晚上也不算旧（2026-10-18 四方评审）
    worker.interval_seconds = 3600.0
    assert check(asyncio.run(runtime.health()), "opportunities_session").status.value == "healthy"


def test_every_derived_store_keeps_the_configured_number_of_fallback_generations(tmp_path: Path) -> None:
    """6.12：关系表、机会表、单元历史快照都留"当前一代 + ``lifecycle.fallback_generations`` 代"（L10.14 补测：默认值碰巧
    与各存储自己的默认相等，接线断了测试也看不出来——这里配一个不同于默认的数）。"""

    runtime = _fully_enabled_runtime(tmp_path, lifecycle={"fallback_generations": 4})
    foresight = runtime.components.foresight
    scene = runtime.components.nightly.nightly.scene  # type: ignore[union-attr]
    assert foresight is not None
    assert scene.relations.fallback_generations == 4
    assert foresight.opportunities._files.retained_nights == 5
    assert foresight.history.store._files.retained_nights == 5
    # 6.5：语义树还要读的词表版本经组合根交给行为侧参与者，截断不会越过它们
    (behavior,) = [
        item for item in runtime.components.workflow.lifecycle_worker._participants if item.name == "behavior"
    ]
    assert behavior.derived_versions is not None
    assert scene.concepts.synced_version() in behavior.derived_versions()
