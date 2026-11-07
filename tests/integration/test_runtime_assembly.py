"""顶层 Runtime 对所有领域组件的唯一组装与初始化主链测试。"""

import asyncio
import os
import sqlite3
import sys
from contextlib import closing
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest
import yaml

from habitus.behavior.tree import BehaviorTree
from habitus.config import HabitusConfig
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector import VectorStoreFactory
from habitus.integrations.http_api import create_http_app
from habitus.integrations.http_api.handlers import RuntimeHTTPHandlers
from habitus.integrations.local_service import ServiceInstanceLock
from habitus.memory.browse import MemoryBrowseNotFoundError
from habitus.memory.conversation import ConversationAddress
from habitus.memory.model import MemoryKind
from habitus.memory.recall import (
    SQLiteMemoryRecallLifecycleStore,
)
from habitus.memory.retrieval import (
    MemoryRetrievalSufficiency,
)
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
from tests.lock_store import ProcessLocalLockStore
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
    )
    return providers, vectors


def runtime_config(tmp_path: Path, *, with_credentials: bool = False) -> HabitusConfig:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["models"]["chat"]["route"].update(
        provider="fake",
        adapter="fake_chat",
        credential_ref="chat" if with_credentials else "",
    )
    payload["models"]["embedding"]["route"].update(
        provider="fake",
        adapter="fake_embedding",
        credential_ref="embedding" if with_credentials else "",
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
            credential_ref="vector_service" if with_credentials else "",
        )
    payload["conversation"]["summary_vector_store"]["route"].update(
        provider="fake",
        adapter="fake_vector",
        credential_ref="vector_service" if with_credentials else "",
    )
    if with_credentials:
        payload["credentials"]["chat"]["api_key"] = "chat-secret"
        payload["credentials"]["embedding"]["api_key"] = "embedding-secret"
        payload["credentials"]["dashscope"]["api_key"] = "dashscope-secret"
        payload["credentials"]["vector_service"] = {"access_key": "vector-access", "secret_key": "vector-secret"}
    return HabitusConfig.from_mapping(payload)


def test_assembly_resolves_each_provider_and_database_credential_by_reference(tmp_path: Path) -> None:
    model_credentials: dict[str, str] = {}
    vector_credentials: list[dict[str, str]] = []
    vector_roots: list[Path | None] = []
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
        vector_roots.append(context.data_root)
        return FakeVectorBackend(context.config.provider, context.config.collection)

    vectors.register_adapter(
        "fake_vector",
        vector_builder,
    )

    config = runtime_config(tmp_path, with_credentials=True)
    build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    assert model_credentials == {
        "chat": "chat-secret",
        "embedding": "embedding-secret",
        "rerank": "dashscope-secret",
    }
    assert vector_credentials == [
        {"access_key": "vector-access", "secret_key": "vector-secret"},
        {"access_key": "vector-access", "secret_key": "vector-secret"},
    ]
    # 两个 Collection 都拿到存储根下同一个 vector/ 目录，各自按 Collection 名落文件
    assert vector_roots == [config.storage_root / "vector"] * 2 == [tmp_path / "data" / "vector"] * 2


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


def test_document_prefix_shrinks_the_segmentation_unit_budget(tmp_path: Path) -> None:
    """切段打分把每个单元截到"max_input_chars 减去文档前缀"：不让出前缀，拼上后超长，打分静默退回结构边界（W7）。"""

    config = runtime_config(tmp_path)
    embedding = replace(
        config.models.embedding,
        max_input_chars=config.models.embedding.max_input_chars + 64,
        document_prefix="passage: ",
    )
    config = replace(config, models=replace(config.models, embedding=embedding))
    providers, vectors = runtime_dependencies()

    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    assert runtime.components.conversation.boundary_scorer.max_unit_chars == embedding.max_input_chars - len(
        "passage: "
    )


def test_memory_workflow_state_is_injected_from_the_configured_workflow_root(tmp_path: Path) -> None:
    """记忆侧树外状态由组装注入 ``config.workflow_root``；默认配置下路径与"记忆根的兄弟 workflow 目录"逐字相同。"""

    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )

    lifecycle = runtime.components.memory.lifecycle
    sibling = lifecycle.tree.root.parent / "workflow"
    assert config.workflow_root == sibling
    journal = lifecycle.committer.transaction.journal
    assert journal.root == sibling / "transactions"
    assert lifecycle.tree.visibility_journal is journal
    assert lifecycle.recovery_store.root == sibling / "lifecycle" / "l2_recovery"
    assert lifecycle.operation_store.root == sibling / "lifecycle" / "l2_operations"
    assert lifecycle.scan_store.path == sibling / "lifecycle" / "l2_scan_state.json"
    assert not config.storage_root.exists()


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
            "dsh",
            "openai_chat_completions",
            "openai_responses",
            "openclaw",
            "opencode",
            "pi",
            "zcode",
        )
        # 记忆树的公开浏览经同一棵树（方案 36 第 2 步）：初始化后根下只有固定分支、还没有文档
        root = await runtime.list_memory()
        assert str(root.uri) == "memory://" and root.documents == ()
        assert "memory://preferences" in {str(item) for item in root.directories}
        with pytest.raises(MemoryBrowseNotFoundError):
            await runtime.read_memory("memory://profile.md")
        assert (await runtime.reminders(date(2026, 7, 28))).reminders == ()  # 还没有提醒
        assert (await runtime.reminders()).day == datetime.now(runtime.config.locale.zone()).date()
        report = await runtime.health()
        assert report.status.value in {"healthy", "degraded"}
        assert "habitus_" in runtime.prometheus_metrics()
        http = RuntimeHTTPHandlers(runtime)
        capabilities = http.capabilities()
        assert capabilities.api_version == "1.0"
        assert capabilities.protocols == list(runtime.conversation_protocols())
        assert "remember_idempotency_v1" in capabilities.features
        # 预测层没开：取口恒为空，不公布提醒出口（裁定 80 第 1 条）
        assert not runtime.reminders_enabled() and "reminders_v1" not in capabilities.features
        status, readiness = await http.readiness()
        assert status in {200, 503}
        assert readiness.status in {"healthy", "degraded", "unhealthy"}

        # 插件经 HTTP 走的同一条链：写一轮、会话关闭时 flush 封段
        transport = httpx.ASGITransport(app=create_http_app(runtime))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
            turn = {
                "conversation_id": "conversation-hook-runtime",
                "started_on": "2026-07-28",
                "protocol": "openai_chat_completions",
                "payload": payload,
                "start_sequence": 0,
                "occurred_at": BASE_TIME.isoformat(),
                "after_turn": False,
            }
            remembered = await client.post("/api/v1/memory/remember", json=turn)
            assert remembered.status_code == 200, remembered.text
            assert remembered.json()["result"]["next_sequence"] == 2
            closed = await client.post(
                "/api/v1/memory/flush",
                json={"conversation_id": "conversation-hook-runtime", "started_on": "2026-07-28"},
            )
            assert len(closed.json()["result"]["jobs"]) == 1

            # 第一次写入的响应丢了，插件原样重投同一轮（同一 start_sequence）：服务端不重复追加
            retry_turn = {**turn, "conversation_id": "conversation-hook-retry"}
            assert (await client.post("/api/v1/memory/remember", json=retry_turn)).status_code == 200
            retry_address = ConversationAddress("conversation-hook-retry", date(2026, 7, 28))
            assert await runtime.conversation_cursor(retry_address) == 2
            replayed_turn = await client.post("/api/v1/memory/remember", json=retry_turn)
            assert replayed_turn.status_code == 200, replayed_turn.text
            assert replayed_turn.json()["result"]["next_sequence"] == 2
            retry_live = await runtime.read_live_conversation(retry_address)
            assert retry_live is not None and len(retry_live.messages) == 2
            retry_closed = await client.post(
                "/api/v1/memory/flush",
                json={"conversation_id": "conversation-hook-retry", "started_on": "2026-07-28"},
            )
            assert len(retry_closed.json()["result"]["jobs"]) == 1
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
    assert worker.last_run.path == runtime.config.scene_root / "nightly" / "last_run.json"
    # 每晚开头的重试接着两支各自的重试堆（裁定 67）；会话一支连同它的会话 lane
    retry = worker.nightly.retry
    assert retry is not None
    from habitus.behavior.fusion.lanes.session import TurnHandler
    from habitus.behavior.reduction.failed import ChainHandler
    from habitus.behavior.retry import RetryKind

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
    assert runtime.reminders_enabled()
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
    # 当天的日子预期是夜里那条线的最后一步（裁定 94 第 1 条），与白天那一拍读同一种未封口的桥、写同一本账；线上那一份碰到模型
    # 整个用不了就抛给线（裁定 94 第 8 条），白天那一份留到下一拍
    assert nightly.days is foresight.days and foresight.days is not None
    assert foresight.days.ledger is foresight.ledger and foresight.days.opportunities is foresight.opportunities
    in_line = foresight.days.forming.unsealed
    assert isinstance(in_line, UnsealedFromJudgements) and in_line.in_line
    assert in_line.judgements is behavior.judgements and not tick.forming.unsealed.in_line  # type: ignore[attr-defined]
    # 线自己的记录只有一份：常驻节拍写，结算与白天那一拍读同一个"这一晚跑完了没有"（裁定 94 第 1、9 条）
    assert foresight.settlement.finished == worker.last_run.finished
    assert tick.line_finished == worker.last_run.finished
    # 单元历史：夜里那一拍重标、白天的单元读同一份（裁定 55 新版第 2 条）
    assert nightly.foresight.history is foresight.history and foresight.history.store.root == foresight.ledger.root
    assert tick.delivery.units.history is foresight.history.store
    assert not hasattr(tick, "prediction") and tick.read_side is foresight.read_side
    unsealed = tick.forming.unsealed
    assert isinstance(unsealed, UnsealedFromJudgements) and unsealed.judgements is behavior.judgements
    assert tick.delivery is not None and tick.delivery.ledger is foresight.ledger
    assert foresight.worker.interval_seconds == runtime.config.foresight.tick_interval_seconds
    # 「离上次够久」的时窗往后接几天用的是递送那同一条线（裁定 89 第 1 条）：配置里的 1 − 可接受的误提醒率
    assert tick.forming.tolerance == tick.delivery.tolerance == foresight.days.forming.tolerance
    assert tick.forming.tolerance.line == pytest.approx(1.0 - runtime.config.foresight.false_alarm_rate)
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
    stale = today - timedelta(days=1)  # 昨天已定稿、今天这一晚的表还没出来：落后于"最后定稿日 + 1"（裁定 68 第 4 条）
    worker.nightly.closed_days = lambda: (today - timedelta(days=1),)  # type: ignore[attr-defined]
    foresight.opportunities.write(OpportunityTable("session", stale, stale, 15, 3, (), ()))
    report = asyncio.run(runtime.health())
    assert check(report, "opportunities_session").status.value == "degraded"
    foresight.opportunities.write(OpportunityTable("session", today, today, 15, 3, (), ()))
    report = asyncio.run(runtime.health())
    item = check(report, "opportunities_session")
    assert item.status.value == "healthy" and item.detail == f"night:{today}" and not item.critical


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


def test_a_host_transcript_with_parallel_tools_whose_times_go_backwards_is_remembered_over_http(tmp_path: Path) -> None:
    """ZCode / OpenCode 并行工具的时刻倒退（调用 2 早于结果 1 开始、结果 2 早于结果 1 结束）：宿主自己合法的转写，
    本地服务收下（不是不可重试的 400、插件不会挡住这个会话），按转写顺序存、时刻不降。"""

    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path), providers=providers, vector_stores=vectors, path_lock=PathLock(ProcessLocalLockStore())
        )
        runtime.initialize()
        start = 1_790_000_000_000

        def tool(call_id: str, begin: int, end: int) -> dict:
            return {
                "type": "tool", "callID": call_id, "tool": "WebSearch",
                "state": {"status": "completed", "input": {"q": call_id}, "output": call_id, "time": {"start": start + begin, "end": start + end}},
            }

        records = [
            {"info": {"role": "user", "time": {"created": start}}, "parts": [{"type": "text", "text": "查一下", "time": {"end": start}}]},
            {"info": {"role": "assistant", "time": {"created": start + 50}}, "parts": [
                tool("a", 100, 5000), tool("b", 200, 4000), {"type": "text", "text": "好了", "time": {"end": start + 5200}},
            ]},
        ]
        transport = httpx.ASGITransport(app=create_http_app(runtime))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
            remembered = await client.post(
                "/api/v1/memory/remember",
                json={
                    "conversation_id": "zcode-parallel-tools",
                    "started_on": "2026-10-01",
                    "protocol": "zcode",
                    "payload": {"records": records},
                    "start_sequence": 0,
                    "occurred_at": BASE_TIME.isoformat(),
                    "after_turn": True,
                },
            )
        assert remembered.status_code == 200, remembered.text
        live = await runtime.read_live_conversation(ConversationAddress("zcode-parallel-tools", date(2026, 10, 1)))
        assert live is not None
        assert [(item.role.value, item.tool_call_id) for item in live.messages] == [
            ("prompt", None), ("tool_call", "a"), ("tool_result", "a"), ("tool_call", "b"), ("tool_result", "b"), ("completion", None)
        ]
        assert all(left.occurred_at <= right.occurred_at for left, right in zip(live.messages, live.messages[1:], strict=False))
        await runtime.close()

    asyncio.run(scenario())


def test_rounds_a_host_ended_with_a_quota_error_are_remembered_and_do_not_hold_up_the_next_one(tmp_path: Path) -> None:
    """真实 ZCode 形状（裁定 76 第 2 条）：两次工具之后 429、"继续"之后直接 429（这一轮只有他那句话），报错只在
    ``info.error``；两轮都照出一条答复封口，后面正常的一轮照常进记忆，没有一批被挡在前一批后面。"""

    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path), providers=providers, vector_stores=vectors, path_lock=PathLock(ProcessLocalLockStore())
        )
        runtime.initialize()
        start = 1_791_467_400_000
        busy = {
            "name": "StartPlanBusyAutoRetryExhaustedError",
            "data": {"message": "Start Plan is busy and automatic model stream recovery reached the maximum retry count.",
                     "code": "model_rate_limited"},
        }

        def user(at: int, text: str) -> dict:
            return {"info": {"role": "user", "time": {"created": start + at}}, "parts": [{"type": "text", "text": text}]}

        def bash(call_id: str, at: int) -> dict:
            return {"type": "tool", "callID": call_id, "tool": "Bash",
                    "state": {"status": "completed", "input": {"command": "ls"}, "output": "a.txt", "time": {"start": start + at, "end": start + at + 10}}}

        batches = [
            [user(0, "看下目录"),
             {"info": {"role": "assistant", "time": {"created": start + 5}}, "parts": [{"type": "step-start"}, bash("b1", 10), bash("b2", 30)]},
             {"info": {"role": "assistant", "time": {"created": start + 60, "completed": start + 170}, "error": busy}, "parts": [{"type": "step-start"}]}],
            [user(1_000, "继续"),
             {"info": {"role": "assistant", "time": {"created": start + 1_010, "completed": start + 1_100}, "error": busy}, "parts": [{"type": "step-start"}]}],
            [user(2_000, "再试一次"),
             {"info": {"role": "assistant", "time": {"created": start + 2_010}}, "parts": [{"type": "text", "text": "好了"}]}],
        ]
        sequence = 0
        transport = httpx.ASGITransport(app=create_http_app(runtime))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
            for records in batches:
                remembered = await client.post(
                    "/api/v1/memory/remember",
                    json={
                        "conversation_id": "zcode-quota",
                        "started_on": "2026-10-01",
                        "protocol": "zcode",
                        "payload": {"records": records},
                        "start_sequence": sequence,
                        "occurred_at": BASE_TIME.isoformat(),
                        "after_turn": True,
                    },
                )
                assert remembered.status_code == 200, remembered.text
                sequence = await runtime.conversation_cursor(ConversationAddress("zcode-quota", date(2026, 10, 1)))
        assert sequence == 10  # 6 + 2 + 2：每一批都进了 journal，一批也没被挡住
        await runtime.close()

    asyncio.run(scenario())


def test_a_daily_reminder_is_not_held_back_as_a_return_after_absence_by_the_settlement_lag(tmp_path: Path) -> None:
    """G3：天天做、已在提醒档的事，第 T 天白天账上的事实只到 T−2（D 日那条 D+2 04:00 才结），T−1 那条已形成还没结算。
    经 ``build_runtime`` 组装的递送那一拍，时窗开时记「等时窗末」，不记「久别回来」。"""

    from datetime import timedelta

    from habitus.foresight.edges import PastMoment
    from habitus.foresight.history import Moment, UnitHistory
    from habitus.foresight.ledger import AccountFact, AccountLevel, Expectation, UnitLevel, account_key
    from habitus.foresight.opportunities import DayWatch, OpportunityTable
    from habitus.runtime.foresight_delivery import JudgeMaterials

    runtime = _fully_enabled_runtime(tmp_path)
    foresight = runtime.components.foresight
    assert foresight is not None and foresight.worker is not None
    delivery = foresight.worker.tick.delivery
    assert delivery is not None
    ledger = foresight.ledger
    zone = runtime.config.locale.zone()
    kind = "s-k0002"
    account = account_key(kind, "日子", "rel")
    today = datetime(2026, 10, 31, 9, 0, tzinfo=zone)

    def formed(offset: int) -> Expectation:
        at = today - timedelta(days=offset)
        return Expectation(
            expectation_id=f"e-{offset}",
            kind_token=kind,
            day=at.date(),
            formed_at=at,
            source="日子",
            trigger="rel",
            trigger_instance=f"behavior://e-{offset}.md",
            window=(at + timedelta(minutes=10), at + timedelta(minutes=40)),
            deadline=datetime.combine(at.date() + timedelta(days=1), datetime.min.time(), tzinfo=zone),
            basis=(),
            opportunities=8,
            hits=8,
            observed_from=at,
        )

    settled = range(30, 1, -1)  # T−30 … T−2：结过、那天结完了
    for offset in settled:
        item = formed(offset)
        ledger.record_expectation(item)
        ledger.append_fact(AccountFact(account, item.formed_at, True, 1.0, "准时", item.expectation_id))
        ledger.mark_final(item.day, at=item.formed_at + timedelta(days=2))
    ledger.record_expectation(formed(1))  # T−1：已形成，最晚时限那天还没定稿，没结算
    ledger.record_expectation(formed(0))
    ledger.record_level(AccountLevel(account, today - timedelta(days=31), 1))
    ledger.record_unit_level(UnitLevel(kind, today - timedelta(days=31), 1))
    foresight.history.store.write(
        UnitHistory(
            "session",
            (today - timedelta(days=1)).date(),
            {
                kind: tuple(
                    Moment(PastMoment(formed(offset).formed_at, frozenset({account}), True), (f"{kind}-{offset}",))
                    for offset in settled
                )
            },
        )
    )
    watch = tuple(DayWatch((today - timedelta(days=offset)).date(), 1.0) for offset in range(31, 0, -1))
    table = OpportunityTable("session", (today - timedelta(days=1)).date(), None, 15, 3, (), (), watch=watch)

    report = asyncio.run(delivery.run_once(today + timedelta(minutes=15), JudgeMaterials(table, lambda uris: {})))
    assert report.by_reason == {"等时窗末": 1} and report.by_level == {1: 1}


def _extreme_base(tmp_path: Path, shape: str) -> Path:
    if shape == "chinese-space-user":
        return tmp_path / "Users" / "张三 李四"
    base = tmp_path / "Users" / "张三 李四"
    while len(str(base)) < 230:
        base = base / ("深层-" + "d" * 30)
    return base


def _real_storage_config(base: Path) -> HabitusConfig:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(base / "data")
    for capability in ("chat", "embedding", "rerank"):
        payload["models"][capability]["route"].update(provider="fake", adapter=f"fake_{capability}", credential_ref="")
    return HabitusConfig.from_mapping(payload)


def _all_paths(root: Path) -> list[str]:
    """列出根下全部路径；Windows 上用 ``\\\\?\\`` 扩展路径枚举，测试自己不先撞上 MAX_PATH。"""

    walk_root = str(root)
    if sys.platform == "win32":
        walk_root = "\\\\?\\" + walk_root
    found: list[str] = []
    for directory, names, files in os.walk(walk_root):
        found.extend(os.path.join(directory, name).removeprefix("\\\\?\\") for name in (*names, *files))
    return found


@pytest.mark.parametrize("with_service_lock", [True, False], ids=["serve-order", "runtime-only"])
@pytest.mark.parametrize("shape", ["chinese-space-user", "beyond-max-path"])
def test_runtime_remembers_a_turn_under_extreme_storage_roots(tmp_path: Path, shape: str, with_service_lock: bool) -> None:
    """阶段 3 极限用例：用户名含中文与空格、存储根深到其下文件超过 Windows 260 字符时，照 ``serve`` 的顺序先取实例锁、
    再初始化 Runtime、打开两个 local_sqlite 向量集合，收录一轮并读回。``runtime-only`` 一组不取实例锁，
    让 Windows 上实例锁那一层若先断，Runtime 建根与 SQLite 这两层仍各自被测到。

    全用真实存储：实例锁、SQLite 锁库、local_sqlite 向量库。windows-latest 上关掉了系统长路径支持，这条用例在那里
    如实反映不经耐久写入层的部分能否越过 MAX_PATH（"需要用户定的事"第 2 条等它的结果）。"""

    base = _extreme_base(tmp_path, shape)
    config = _real_storage_config(base)

    async def scenario() -> None:
        providers, _vectors = runtime_dependencies()
        lock = ServiceInstanceLock(config.storage_root / "service" / "http.lock")
        if with_service_lock:
            lock.acquire()
        runtime = build_runtime(config, providers=providers)
        try:
            runtime.initialize()
            await runtime.components.memory.vector_index.ensure_ready()
            await runtime.components.conversation.summary_vector_index.ensure_ready()
            await _remember_one_turn(runtime)
            live = await runtime.read_live_conversation(ConversationAddress("极限-路径-会话", date(2026, 10, 1)))
            assert live is not None
            assert [item.role.value for item in live.messages] == ["prompt", "completion"]
        finally:
            await runtime.close()
            lock.release()  # 没取到时是空操作

    asyncio.run(scenario())
    paths = _all_paths(base / "data")
    vector_files = [path for path in paths if os.path.dirname(path).endswith("vector") and ".sqlite" in path]
    assert vector_files, paths
    if shape == "beyond-max-path":
        assert max(len(path) for path in vector_files) > 260
        assert max(len(path) for path in paths) > 260


async def _remember_one_turn(runtime: object) -> None:
    start = 1_790_000_000_000
    records = [
        {"info": {"role": "user", "time": {"created": start}}, "parts": [{"type": "text", "text": "记住我叫张三", "time": {"end": start}}]},
        {"info": {"role": "assistant", "time": {"created": start + 50}}, "parts": [{"type": "text", "text": "好的", "time": {"end": start + 60}}]},
    ]
    transport = httpx.ASGITransport(app=create_http_app(runtime))  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
        remembered = await client.post(
            "/api/v1/memory/remember",
            json={
                "conversation_id": "极限-路径-会话",
                "started_on": "2026-10-01",
                "protocol": "zcode",
                "payload": {"records": records},
                "start_sequence": 0,
                "occurred_at": BASE_TIME.isoformat(),
                "after_turn": True,
            },
        )
    assert remembered.status_code == 200, remembered.text
