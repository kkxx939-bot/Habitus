"""从完整 ConversationSegment 到 Job/Receipt/双向量 checkpoint 的主链验收测试。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from habitus.config import HabitusConfig
from habitus.foundation.observability import ObservationEvent
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector import VectorStoreBusyError, VectorStoreFactory, VectorStoreMatch, VectorStoreState
from habitus.memory.conversation import ConversationAddress
from habitus.memory.editor import MemoryTransactionJournalState
from habitus.memory.workflow import (
    MemoryChangeReceiptState,
    MemoryChangeSource,
    MemoryJobExecutionError,
    MemoryJobStatus,
)
from habitus.model_client import (
    EmbeddingVector,
    ModelResponse,
    ModelTransportError,
    ProviderCapabilities,
    ProviderFactory,
)
from habitus.pre.conversation import ConversationBatch
from habitus.runtime import build_runtime
from tests.helpers import BASE_TIME, closed_turn, summary_content, tool_turn
from tests.lock_store import ProcessLocalLockStore
from tests.model_helpers import prepare_chat_request

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def runtime_config(tmp_path: Path, *, max_attempts: int = 3) -> HabitusConfig:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["models"]["chat"]["route"].update(
        provider="fake",
        adapter="fake_chat",
        credential_ref="",
    )
    payload["models"]["embedding"]["route"].update(
        provider="fake",
        adapter="fake_embedding",
        credential_ref="",
    )
    payload["models"]["rerank"]["route"].update(
        provider="fake",
        adapter="fake_rerank",
        credential_ref="",
    )
    payload["memory"]["vector_store"]["route"].update(
        provider="fake",
        adapter="fake_vector",
        credential_ref="",
    )
    payload["conversation"]["summary_vector_store"]["route"].update(
        provider="fake",
        adapter="fake_vector",
        credential_ref="",
    )
    payload["workflow"]["jobs"]["max_attempts"] = max_attempts
    return HabitusConfig.from_mapping(payload)


@dataclass
class DispatchingChatProvider:
    provider_name: str
    model: str
    response_names: list[str] = field(default_factory=list)
    capabilities: ProviderCapabilities = ProviderCapabilities()
    is_remote: bool = False
    preferences: list[dict[str, object]] = field(default_factory=list)
    retrieval_script: list[dict[str, object]] = field(default_factory=list)
    candidate_script: list[Callable[[dict[str, object]], dict[str, object]]] = field(default_factory=list)
    candidate_requests: list[object] = field(default_factory=list)

    prepare = staticmethod(prepare_chat_request)

    def complete(self, prepared):
        request = prepared.request
        assert request.response_format is not None
        response_name = request.response_format.name
        self.response_names.append(response_name)
        summary = summary_content()
        payloads = {
            "conversation_segment_summary_field_operations": {
                "operations": [
                    {"field": "overview", "operation": "update", "content": summary.overview},
                    {
                        "field": "chronology",
                        "operation": "append",
                        "items": list(summary.chronology),
                    },
                    {
                        "field": "ending_state",
                        "operation": "update",
                        "content": summary.ending_state,
                    },
                ]
            },
            "memory_retrieval_decision": {
                "status": "irrelevant",
                "action": "finish",
                "query": None,
                "uri": None,
                "reason": "当前没有相关旧记忆。",
            },
            "memory_candidate_batch": {
                "profile": [],
                "preferences": self.preferences,
                "entities": [],
                "tools": [],
                "events": [],
                "intentions": [],
                "identity_proposals": [],
                "relations": [],
            },
        }
        if response_name == "memory_directory_overview":
            snapshot = json.loads(str(request.messages[-1].content).split("目录快照：", 1)[1])
            payloads[response_name] = {
                "directory_summary": "目录保存有来源支持的长期记忆。",
                "entries": [
                    {"name": item["name"], "kind": item["kind"], "summary": f"{item['name']} 的稳定摘要。"}
                    for item in snapshot["entries"]
                ],
            }
        if response_name == "memory_retrieval_decision" and self.retrieval_script:
            payloads[response_name] = self.retrieval_script.pop(0)
        if response_name == "memory_candidate_batch":
            self.candidate_requests.append(request)
            # 修补轮：原请求之后接着上一次的答复与错误清单；剧本用完时模型照旧答上一次的内容
            repair = len(request.messages) >= 3 and request.messages[-2].role == "assistant"
            facts = request.messages[-3] if repair else request.messages[-1]
            if self.candidate_script:
                visible = json.loads(str(facts.content).split("\n", 1)[1])
                payloads[response_name] = self.candidate_script.pop(0)(visible)
            elif repair:
                return ModelResponse(str(request.messages[-2].content), self.model, self.provider_name)
        payload = payloads[response_name]
        return ModelResponse(json.dumps(payload, ensure_ascii=False), self.model, self.provider_name)

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
        return None


class EmbeddingProvider:
    is_remote = False

    def __init__(self, provider_name: str, model: str, dimension: int) -> None:
        self.provider_name = provider_name
        self.model = model
        self.dimension = dimension

    async def embed(self, _text: str, *, is_query: bool) -> EmbeddingVector:
        return EmbeddingVector((1.0,) + (0.0,) * (self.dimension - 1))

    async def aclose(self):
        return None


class RerankProvider:
    is_remote = False

    def __init__(self, provider_name: str, model: str) -> None:
        self.provider_name = provider_name
        self.model = model

    async def rerank(self, _query: str, documents) -> tuple[float, ...]:
        return tuple(1.0 for _ in documents)

    async def aclose(self):
        return None


class DurableVectorBackend:
    adapter_name = "fake_vector"
    requires_cross_process_publication_fencing = False
    max_records = 100_000
    max_search_hits = 10_000

    def __init__(self, provider_name: str, collection: str) -> None:
        self.provider_name = provider_name
        self.collection = collection
        self.metadata = {}
        self.records = {}
        self.fail_incremental_visibility_once = False
        self.fail_search_when: Callable[[], bool] = lambda: False

    async def initialize(self):
        return None

    async def read_metadata(self, names):
        return {name: self.metadata[name] for name in names if name in self.metadata}

    async def write_metadata(self, values, *, dimension):
        self.metadata.update({name: dict(value) for name, value in values.items()})

    async def ensure_schema(self, _dimension, *, published_dimension):
        return None

    async def read(self, identities):
        return tuple(self.records[identity] for identity in identities if identity in self.records)

    async def delete_all(self):
        self.records.clear()

    async def upsert(self, records):
        self.records.update({record.identity: record for record in records})

    async def delete(self, identities):
        for identity in identities:
            self.records.pop(identity, None)

    async def validate_records(self, _records, *, replacing):
        return None

    async def wait_visible(self, _upserts, _deletes, *, complete):
        if self.fail_incremental_visibility_once and not complete:
            self.fail_incremental_visibility_once = False
            raise TimeoutError("vector visibility did not converge")
        return None

    async def search(self, _query_vector, *, filters, limit):
        if self.fail_search_when():
            raise ConnectionError("vector service unavailable")
        return tuple(
            VectorStoreMatch(record, 1.0) for record in self.records.values() if filters.matches(record.attributes)
        )[:limit]

    async def scan(self, *, filters, limit):
        return tuple(record for record in self.records.values() if filters.matches(record.attributes))[:limit]

    async def close(self):
        return None


def dependencies():
    providers = ProviderFactory()
    chat_instances: list[DispatchingChatProvider] = []
    vector_instances: list[DurableVectorBackend] = []

    def chat_builder(context):
        provider = DispatchingChatProvider(context.route.provider, context.route.model)
        chat_instances.append(provider)
        return provider

    providers.register_adapter("chat", "fake_chat", chat_builder)
    providers.register_adapter(
        "embedding",
        "fake_embedding",
        lambda context: EmbeddingProvider(
            context.route.provider,
            context.route.model,
            context.config.dimension,
        ),
    )
    providers.register_adapter(
        "rerank",
        "fake_rerank",
        lambda context: RerankProvider(context.route.provider, context.route.model),
    )
    vectors = VectorStoreFactory()

    def vector_builder(context):
        backend = DurableVectorBackend(context.config.provider, context.config.collection)
        vector_instances.append(backend)
        return backend

    vectors.register_adapter(
        "fake_vector",
        vector_builder,
        requires_cross_process_publication_fencing=False,
    )
    return providers, vectors, chat_instances, vector_instances


def test_full_job_chain_commits_summary_receipt_indexes_and_job_before_cleaning_journal(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path)
    providers, vectors, chats, _vector_backends = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-1", closed_turn()),
        after_turn=True,
    )
    queued = runtime.components.workflow.enqueuer.flush(address).jobs[0]

    result = asyncio.run(runtime.run_next())

    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert result.job.memory_sequence == queued.memory_sequence == 1
    assert result.change_receipt is not None
    assert result.change_receipt.state is MemoryChangeReceiptState.COMMITTED
    assert result.change_receipt.node_changes == ()
    assert result.summary_generated and result.summary_indexed and result.vector_indexed
    assert result.journal_cleaned
    assert runtime.components.memory.editor.transaction.journal.try_read(queued.transaction_id) is None
    summary = runtime.components.conversation.summaries.store.read(
        address,
        runtime.components.conversation.journal.read_segment(address, queued.segment_id),
    )
    assert summary.source_message_digest == queued.source_segment_digest
    memory_state = asyncio.run(runtime.components.memory.vector_index.store.state())
    summary_state = asyncio.run(runtime.components.conversation.summary_vector_index.store.state())
    assert isinstance(memory_state, VectorStoreState) and memory_state.checkpoint == 1
    assert isinstance(summary_state, VectorStoreState) and summary_state.checkpoint == 1
    assert set(chats[0].response_names) == {
        "conversation_segment_summary_field_operations",
        "memory_retrieval_decision",
        "memory_candidate_batch",
    }


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def test_empty_tree_final_round_still_insufficient_commits_job_and_audits_limit(
    tmp_path: Path,
) -> None:
    """主链端到端实测（Job #1 首次）：空树上两轮搜索都为空，终轮如实答不足，Job 仍要写完。"""

    config = runtime_config(tmp_path, max_attempts=1)
    providers, vectors, chats, _vector_backends = dependencies()
    observer = RecordingObserver()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
        observer=observer,
    )
    runtime.initialize()
    search = {
        "status": "insufficient",
        "action": "memory_search",
        "query": "磁盘清理偏好",
        "uri": None,
        "reason": "旧 L2 快照为空，需要检索。",
    }
    chats[0].retrieval_script.extend(
        [
            search,
            search,
            {
                "status": "insufficient",
                "action": "finish",
                "query": None,
                "uri": None,
                "reason": "完整旧 L2 快照均缺失，语义检索两次均无结果。",
            },
        ]
    )
    chats[0].preferences.append({"page_id": 100, "topic": "回答风格", "content": "- 偏好简洁回答"})
    address = ConversationAddress("conversation-limit", date(2026, 7, 4))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-limit", closed_turn()),
        after_turn=True,
    )
    runtime.components.workflow.enqueuer.flush(address)

    result = asyncio.run(runtime.run_next())

    assert chats[0].retrieval_script == []
    assert chats[0].response_names.count("memory_retrieval_decision") == 3
    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert result.change_receipt is not None
    assert [str(item.uri) for item in result.change_receipt.node_changes] == ["memory://preferences/回答风格.md"]
    plans = [event for event in observer.events if (event.category, event.operation) == ("workflow", "memory_plan")]
    assert len(plans) == 1
    assert plans[0].attributes["retrieval_ended_at_limit"] is True
    assert plans[0].context.memory_sequence == result.job.memory_sequence


def _empty_candidates() -> dict[str, list[object]]:
    return {
        name: []
        for name in (
            "profile",
            "preferences",
            "entities",
            "tools",
            "events",
            "intentions",
            "identity_proposals",
            "relations",
        )
    }


def _job12_first_session(_visible: dict[str, object]) -> dict[str, object]:
    batch = _empty_candidates()
    batch["preferences"] = [{"page_id": 100, "topic": "源码审查", "content": "- 审查项目代码时只看源码。"}]
    batch["entities"] = [
        {"page_id": 101, "category": "软件项目", "name": "Habitus", "summary": "Habitus 是一个长期记忆软件项目。"}
    ]
    batch["events"] = [
        {
            "page_id": 102,
            "event_date": "2026-07-01",
            "event_name": "完成 Habitus memory 源码审查",
            "summary": "2026-07-01，助手完成了对 Habitus memory 子系统的源码审查。",
        }
    ]
    return batch


def _job12_rewrite(visible: dict[str, object]) -> dict[str, object]:
    """对话 029 的形态：改写旧 Event、更新偏好与 Habitus、新建 OpenViking 并加 derived_from。"""

    page_of = {item["uri"]: item["page_id"] for item in visible["existing_page_ids"]}
    habitus = page_of["memory://entities/软件项目/habitus.md"]
    batch = _empty_candidates()
    batch["preferences"] = [
        {
            "page_id": page_of["memory://preferences/源码审查.md"],
            "topic": "源码审查",
            "content": "- 审查项目代码时只看源码。\n- 回答项目来源关系时先查本机源码。",
        }
    ]
    batch["entities"] = [
        {
            "page_id": habitus,
            "category": "软件项目",
            "name": "Habitus",
            "summary": "Habitus 是一个长期记忆软件项目，其 memory 子系统参考 OpenViking 实现。",
        },
        {
            "page_id": 200,
            "category": "软件项目",
            "name": "OpenViking",
            "summary": "OpenViking 是一个会话记忆软件项目。",
        },
    ]
    batch["events"] = [
        {
            "page_id": page_of["memory://events/2026/07/01/完成%20habitus%20memory%20源码审查.md"],
            "event_date": "2026-07-01",
            "event_name": "完成 Habitus memory 源码审查",
            "summary": "2026-07-01，助手完成了 Habitus memory 与 OpenViking 的对比审查。",
            "details": "审查未将注释或 TODO 视为功能证据。",
        }
    ]
    batch["relations"] = [{"action": "add", "from_page_id": habitus, "to_page_id": 200, "link_type": "derived_from"}]
    return batch


def test_candidate_rewriting_an_old_event_keeps_it_and_still_commits_the_rest_of_the_batch(
    tmp_path: Path,
) -> None:
    """主链端到端扩 3 份 Job #12：旧 Event 以只读给模型；模型仍改写它时保留旧的、记 NOOP，同批其余照常写入。"""

    config = runtime_config(tmp_path, max_attempts=1)
    providers, vectors, chats, _vector_backends = dependencies()
    observer = RecordingObserver()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
        observer=observer,
    )
    runtime.initialize()
    chats[0].candidate_script.extend([_job12_first_session, _job12_rewrite])
    for conversation_id, day in (("conversation-review", date(2026, 7, 1)), ("conversation-compare", date(2026, 7, 2))):
        address = ConversationAddress(conversation_id, day)
        runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
            address,
            ConversationBatch(conversation_id, closed_turn(prompt="审查 Habitus memory 源码")),
            after_turn=True,
        )
        runtime.components.workflow.enqueuer.flush(address)
    first = asyncio.run(runtime.run_next())
    assert first.job is not None and first.job.status is MemoryJobStatus.COMMITTED
    memory_root = tmp_path / "data" / "memory"
    event_path = memory_root / "events" / "2026" / "07" / "01" / "完成 habitus memory 源码审查.md"
    event_before = event_path.read_bytes()

    second = asyncio.run(runtime.run_next())

    assert second.job is not None and second.job.status is MemoryJobStatus.COMMITTED
    assert chats[0].candidate_script == []
    assert second.change_receipt is not None
    assert sorted(str(item.uri) for item in second.change_receipt.node_changes) == [
        "memory://entities/软件项目/habitus.md",
        "memory://entities/软件项目/openviking.md",
        "memory://preferences/源码审查.md",
    ]
    assert event_path.read_bytes() == event_before
    habitus_text = (memory_root / "entities" / "软件项目" / "habitus.md").read_text(encoding="utf-8")
    assert "参考 OpenViking" in habitus_text and "derived_from" in habitus_text
    assert "先查本机源码" in (memory_root / "preferences" / "源码审查.md").read_text(encoding="utf-8")
    plans = [event for event in observer.events if (event.category, event.operation) == ("workflow", "memory_plan")]
    assert [(event.context.memory_sequence, event.attributes["read_only_noop_count"]) for event in plans] == [
        (first.job.memory_sequence, 0),
        (second.job.memory_sequence, 1),
    ]

    request = chats[0].candidate_requests[1]
    visible = json.loads(str(request.messages[-1].content).split("\n", 1)[1])
    read_only = {item["uri"]: item["read_only"] for item in visible["old_memory_snapshots"] if item["state"] == "found"}
    assert read_only == {
        "memory://entities/软件项目/habitus.md": False,
        "memory://events/2026/07/01/完成%20habitus%20memory%20源码审查.md": True,
        "memory://preferences/源码审查.md": False,
    }
    assert {item["uri"]: item["read_only"] for item in visible["existing_page_ids"]} == read_only
    system = "".join(str(item.content) for item in request.messages if item.role == "system")
    assert "read_only 为 true 的旧节点" in system and "另记一条新 Event" in system


def test_retrieval_search_failure_still_fails_job_instead_of_ending_at_limit(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path, max_attempts=1)
    providers, vectors, chats, vector_backends = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    search = {
        "status": "insufficient",
        "action": "memory_search",
        "query": "磁盘清理偏好",
        "uri": None,
        "reason": "旧 L2 快照为空，需要检索。",
    }
    chats[0].retrieval_script.extend([search, search])
    memory_collection = runtime.components.memory.vector_index.store.collection
    memory_backend = next(backend for backend in vector_backends if backend.collection == memory_collection)
    address = ConversationAddress("conversation-search-down", date(2026, 7, 5))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-search-down", closed_turn()),
        after_turn=True,
    )
    runtime.components.workflow.enqueuer.flush(address)
    # 首次检索照常；受控 ReAct 第一轮要求追加搜索之后，向量服务不可用。
    memory_backend.fail_search_when = lambda: "memory_retrieval_decision" in chats[0].response_names

    try:
        asyncio.run(runtime.run_next())
    except MemoryJobExecutionError as exc:
        failed = exc.job
    else:
        raise AssertionError("an unavailable vector search must fail the job")

    assert failed is not None and failed.status is MemoryJobStatus.FAILED
    assert failed.last_error is not None and "additional memory semantic search failed" in failed.last_error
    assert "memory_candidate_batch" not in chats[0].response_names


def test_committed_l2_job_resumes_after_vector_timeout_without_replanning(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path, max_attempts=1)
    providers, vectors, chats, vector_backends = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    address = ConversationAddress("conversation-recovery", date(2026, 7, 2))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-recovery", closed_turn()),
        after_turn=True,
    )
    queued = runtime.components.workflow.enqueuer.flush(address).jobs[0]
    memory_collection = runtime.components.memory.vector_index.store.collection
    memory_backend = next(backend for backend in vector_backends if backend.collection == memory_collection)
    memory_backend.fail_incremental_visibility_once = True

    try:
        asyncio.run(runtime.run_next())
    except MemoryJobExecutionError as exc:
        failed = exc.job
    else:
        raise AssertionError("vector visibility timeout must fail the first attempt")

    assert failed is not None and failed.status is MemoryJobStatus.FAILED
    source = MemoryChangeSource.from_job(failed)
    receipt = runtime.components.workflow.receipts.read(source)
    journal = runtime.components.memory.editor.transaction.journal.read(failed.transaction_id)
    assert receipt.state is MemoryChangeReceiptState.COMMITTED
    assert journal.state is MemoryTransactionJournalState.COMMITTED
    assert chats[0].response_names.count("memory_candidate_batch") == 1
    assert asyncio.run(runtime.failed_memory_job()) == failed

    retried = asyncio.run(runtime.retry_failed_memory_job(failed))
    assert retried.reopened_job.status is MemoryJobStatus.QUEUED
    result = asyncio.run(runtime.run_next())

    assert result.recovered
    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert result.change_receipt is not None
    receipt_source = result.change_receipt.source
    assert receipt_source.same_trigger(source)
    trigger = runtime.components.workflow.runner.executor.conversations.read_segment(
        address,
        failed.segment_id,
    )
    editor_segment = (
        None
        if trigger.ends_mid_turn
        else runtime.components.workflow.runner.executor.conversations.read_editor_turn(address, trigger).whole
    )
    assert editor_segment is not None
    assert receipt_source.editor_segment_id == editor_segment.segment_id
    assert receipt_source.editor_segment_digest == editor_segment.digest
    assert chats[0].response_names.count("memory_candidate_batch") == 1
    assert runtime.components.memory.editor.transaction.journal.try_read(queued.transaction_id) is None
    assert asyncio.run(runtime.failed_memory_job()) is None


def test_l2_written_but_committed_mark_lost_keeps_receipt_and_retry_finishes_job(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path, max_attempts=1)
    providers, vectors, chats, _vector_backends = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    chats[0].preferences.append({"page_id": 100, "topic": "回答风格", "content": "- 偏好简洁回答"})
    address = ConversationAddress("conversation-mark", date(2026, 7, 3))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-mark", closed_turn()),
        after_turn=True,
    )
    queued = runtime.components.workflow.enqueuer.flush(address).jobs[0]
    journal = runtime.components.memory.editor.transaction.journal
    mark = journal.mark
    failures = [OSError("simulated COMMITTED mark loss")]

    def lose_first_commit_mark(transaction_id, state, *, timestamp):
        if state is MemoryTransactionJournalState.COMMITTED and failures:
            raise failures.pop()
        return mark(transaction_id, state, timestamp=timestamp)

    journal.mark = lose_first_commit_mark
    try:
        asyncio.run(runtime.run_next())
    except MemoryJobExecutionError as exc:
        failed = exc.job
    else:
        raise AssertionError("lost COMMITTED mark must fail the first attempt")

    assert failed is not None and failed.status is MemoryJobStatus.FAILED
    source = MemoryChangeSource.from_job(failed)
    assert journal.read(failed.transaction_id).state is MemoryTransactionJournalState.PREPARED
    assert runtime.components.workflow.receipts.read(source).state is MemoryChangeReceiptState.PREPARED

    asyncio.run(runtime.retry_failed_memory_job(failed))
    result = asyncio.run(runtime.run_next())

    assert result.recovered
    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert result.change_receipt is not None
    assert result.change_receipt.state is MemoryChangeReceiptState.COMMITTED
    assert [str(item.uri) for item in result.change_receipt.node_changes] == ["memory://preferences/回答风格.md"]
    assert result.vector_indexed and result.journal_cleaned
    assert journal.try_read(queued.transaction_id) is None
    assert chats[0].response_names.count("memory_candidate_batch") == 1


def test_lifecycle_keeps_history_and_summary_sources_until_terminal_archive_retirement(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path)
    providers, vectors, _chats, _vector_backends = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    address = ConversationAddress("conversation-lifecycle", date(2026, 7, 3))
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-lifecycle", closed_turn()),
        after_turn=True,
    )
    queued = runtime.components.workflow.enqueuer.flush(address).jobs[0]
    completed = asyncio.run(runtime.run_next())
    assert completed.change_receipt is not None
    source = completed.change_receipt.source
    maintenance_time = BASE_TIME + timedelta(days=400)

    released = asyncio.run(runtime.maintain_conversation(address, now=maintenance_time))

    assert released.released_history_segment_ids == ()
    assert released.purged_history_segment_ids == ()
    assert released.deleted_memory_job_sequences == ()
    assert released.deleted_memory_receipt_ids == ()
    assert (
        runtime.components.workflow.jobs.try_read_source(
            address,
            queued.segment_id,
            queued.source_segment_digest,
        )
        is not None
    )
    assert runtime.components.workflow.receipts.try_read(source) is not None
    assert (
        runtime.components.conversation.summaries.store.try_read_by_id(
            address,
            queued.segment_id,
        )
        is not None
    )
    history_path = runtime.components.conversation.journal.layout.history_path(
        address,
        queued.segment_id,
    )
    assert history_path.exists()

    purged = asyncio.run(runtime.maintain_conversation(address, now=maintenance_time))

    assert purged.purged_history_segment_ids == ()
    assert purged.released_history_segment_ids == ()
    assert (
        runtime.components.conversation.summaries.store.try_read_by_id(
            address,
            queued.segment_id,
        )
        is not None
    )


_GOOD_PREFERENCE = {"page_id": 150, "topic": "回答风格", "content": "- 偏好简洁回答"}


def _seed_session(_visible: dict[str, object]) -> dict[str, object]:
    """第一场会话种下实体、Event、Intention 与 tool，供下一场会话的候选引用。"""

    batch = _empty_candidates()
    batch["entities"] = [
        {"page_id": 100, "category": "软件项目", "name": "Habitus", "summary": "Habitus 是长期记忆项目。"}
    ]
    batch["events"] = [
        {
            "page_id": 101,
            "event_date": "2026-07-01",
            "event_name": "完成 Habitus 审查",
            "summary": "2026-07-01 完成审查。",
        }
    ]
    batch["intentions"] = [{"page_id": 102, "intent_name": "写周报", "status": "open", "confirmed": True}]
    batch["tools"] = [{"page_id": 103, "tool_name": "workspace.inspect", "purpose": "检查工作区状态。"}]
    return batch


def _existing_page_id(visible: dict[str, object], suffix: str) -> int:
    items = visible["existing_page_ids"]
    assert isinstance(items, list)
    return next(int(item["page_id"]) for item in items if str(item["uri"]).endswith(suffix))


def _with_good_preference(**extra: Callable[[dict[str, object]], list[object]]):
    def build(visible: dict[str, object]) -> dict[str, object]:
        batch = _empty_candidates()
        batch["preferences"] = [dict(_GOOD_PREFERENCE)]
        for name, items in extra.items():
            batch[name] = items(visible)
        return batch

    return build


def _seeded_runtime(tmp_path: Path, *, max_attempts: int = 1):
    config = runtime_config(tmp_path, max_attempts=max_attempts)
    providers, vectors, chats, vector_backends = dependencies()
    observer = RecordingObserver()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
        observer=observer,
    )
    runtime.initialize()
    chats[0].candidate_script.append(_seed_session)
    _enqueue(runtime, "conversation-seed", date(2026, 7, 1), tool_turn())
    seeded = asyncio.run(runtime.run_next())
    assert seeded.job is not None and seeded.job.status is MemoryJobStatus.COMMITTED
    return runtime, chats, vector_backends, observer


def _enqueue(runtime, conversation_id: str, day: date, messages) -> None:
    address = ConversationAddress(conversation_id, day)
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch(conversation_id, messages),
        after_turn=True,
    )
    runtime.components.workflow.enqueuer.flush(address)


def _plan_events(observer: RecordingObserver) -> list[ObservationEvent]:
    return [event for event in observer.events if (event.category, event.operation) == ("workflow", "memory_plan")]


def test_failing_candidate_after_repair_is_dropped_and_the_rest_of_the_batch_and_later_jobs_commit(
    tmp_path: Path,
) -> None:
    """裁定 82：合格偏好 + 一条另记新 Event 却沿用同日同名的候选，修补后仍不合格 → 只丢那一条，Job 照常提交，
    回执记剔除 1 条及原因，另一场会话的下一个 Job 随后提交。"""

    runtime, chats, _backends, observer = _seeded_runtime(tmp_path)
    same_day_same_name = _with_good_preference(
        events=lambda _visible: [
            {
                "page_id": 160,
                "event_date": "2026-07-01",
                "event_name": "完成 Habitus 审查",
                "summary": "2026-07-01 补充对比结论。",
            }
        ]
    )
    chats[0].candidate_script.append(same_day_same_name)
    _enqueue(runtime, "conversation-x", date(2026, 7, 1), closed_turn())
    _enqueue(runtime, "conversation-y", date(2026, 7, 2), closed_turn(prompt="另一场会话：我喜欢中文回答"))
    calls_before = chats[0].response_names.count("memory_candidate_batch")

    second = asyncio.run(runtime.run_next())

    assert second.job is not None and second.job.status is MemoryJobStatus.COMMITTED and second.job.attempts == 1
    assert chats[0].response_names.count("memory_candidate_batch") - calls_before == 2
    memory_root = tmp_path / "data" / "memory"
    assert (memory_root / "preferences" / "回答风格.md").exists()
    assert "补充对比结论" not in next((memory_root / "events").rglob("*.md")).read_text(encoding="utf-8")
    plan = _plan_events(observer)[-1]
    assert plan.context.memory_sequence == second.job.memory_sequence
    assert plan.attributes["source_rejected_count"] == 1
    assert plan.attributes["source_rejected_reasons"] == (
        "events page_id=160: candidate for an existing memory URI must reuse its existing page_id"
    )

    third = asyncio.run(runtime.run_next())

    assert third.job is not None and third.job.status is MemoryJobStatus.COMMITTED
    assert third.job.conversation_id == "conversation-y"


def test_failing_candidate_repaired_by_the_model_is_written_with_the_rest_of_the_batch(tmp_path: Path) -> None:
    """裁定 82 第 1 条：修补轮改对了，两条都写进，模型被问了两次。"""

    runtime, chats, _backends, observer = _seeded_runtime(tmp_path)
    renamed = _with_good_preference(
        events=lambda _visible: [
            {"page_id": 160, "event_date": "2026-07-01", "event_name": "完成 Habitus 审查", "summary": "补充。"}
        ]
    )
    repaired = _with_good_preference(
        events=lambda _visible: [
            {
                "page_id": 160,
                "event_date": "2026-07-01",
                "event_name": "补充 Habitus 对比结论",
                "summary": "2026-07-01 补充对比结论。",
            }
        ]
    )
    chats[0].candidate_script.extend([renamed, repaired])
    _enqueue(runtime, "conversation-x", date(2026, 7, 1), closed_turn())
    calls_before = chats[0].response_names.count("memory_candidate_batch")

    result = asyncio.run(runtime.run_next())

    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert chats[0].response_names.count("memory_candidate_batch") - calls_before == 2
    assert chats[0].candidate_script == []
    assert result.change_receipt is not None
    assert sorted(str(item.uri) for item in result.change_receipt.node_changes) == [
        "memory://events/2026/07/01/补充%20habitus%20对比结论.md",
        "memory://preferences/回答风格.md",
    ]
    assert _plan_events(observer)[-1].attributes["source_rejected_count"] == 0


@pytest.mark.parametrize(
    ("messages", "bad"),
    [
        # 复用旧编号却改了身份字段（实体名加了"项目"）
        (
            closed_turn,
            {
                "entities": lambda visible: [
                    {
                        "page_id": _existing_page_id(visible, "habitus.md"),
                        "category": "软件项目",
                        "name": "Habitus 项目",
                        "summary": "Habitus 是长期记忆项目。",
                    }
                ]
            },
        ),
        # 已有实体用新编号再出一次
        (
            closed_turn,
            {
                "entities": lambda _visible: [
                    {"page_id": 160, "category": "软件项目", "name": "Habitus", "summary": "参考 OpenViking。"}
                ]
            },
        ),
        # tool 名与本段调用名写法不同
        (
            tool_turn,
            {"tools": lambda _visible: [{"page_id": 160, "tool_name": "workspace_inspect", "purpose": "检查工作区。"}]},
        ),
        # 本段新写 failure_recovery，本段却没有"失败后成功"
        (
            tool_turn,
            {
                "tools": lambda visible: [
                    {
                        "page_id": _existing_page_id(visible, "workspace.inspect.md"),
                        "tool_name": "workspace.inspect",
                        "purpose": "检查工作区状态。",
                        "failure_recovery": "失败就重跑。",
                    }
                ]
            },
        ),
        # completed 一个这批没读到的 Intention
        (
            closed_turn,
            {
                "intentions": lambda _visible: [
                    {"page_id": 160, "intent_name": "写月报", "status": "completed", "confirmed": True}
                ]
            },
        ),
    ],
    ids=[
        "renamed-identity",
        "existing-uri-new-page-id",
        "tool-name-spelling",
        "new-failure-recovery",
        "unread-completed",
    ],
)
def test_normal_model_variation_in_source_checks_no_longer_fails_the_job(
    tmp_path: Path,
    messages: Callable[..., tuple[object, ...]],
    bad: dict[str, Callable[[dict[str, object]], list[object]]],
) -> None:
    """测试工程师 B 列的 5 类模型正常波动：修补后仍不过只丢那一条，Job 不再 FAILED。"""

    runtime, chats, _backends, observer = _seeded_runtime(tmp_path)
    chats[0].candidate_script.append(_with_good_preference(**bad))
    _enqueue(runtime, "conversation-x", date(2026, 7, 1), messages(start_sequence=0))

    result = asyncio.run(runtime.run_next())

    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert (tmp_path / "data" / "memory" / "preferences" / "回答风格.md").exists()
    assert _plan_events(observer)[-1].attributes["source_rejected_count"] == 1


class _ShiftingClock:
    """Job 队列的时钟：每次失败后拨到下一次可认领的时刻，不真的等退避。"""

    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now


def _run_through_outage(runtime, *, outages: int, clock: _ShiftingClock, restore: Callable[[], None]):
    jobs = runtime.components.workflow.jobs
    for _ in range(outages):
        try:
            asyncio.run(runtime.run_next())
        except MemoryJobExecutionError as exc:
            assert exc.job is not None and exc.job.status is MemoryJobStatus.QUEUED
            assert exc.job.next_attempt_at is not None
            clock.now = exc.job.next_attempt_at + timedelta(seconds=1)
        else:
            raise AssertionError("an unavailable dependency must not let the job commit")
    waiting = jobs.oldest_uncommitted()
    restore()
    return waiting, asyncio.run(runtime.run_next())


@pytest.mark.parametrize("dependency", ["vector", "embedding"])
def test_vector_store_or_embedding_outage_longer_than_max_attempts_waits_and_commits_after_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    """裁定 82 第 3 条：向量库忙 / 嵌入服务连不上按根因归"等依赖恢复"，不计次数、不 FAILED，恢复后提交。"""

    max_attempts = 2
    config = runtime_config(tmp_path, max_attempts=max_attempts)
    providers, vectors, chats, vector_backends = dependencies()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    clock = _ShiftingClock()
    runtime.components.workflow.jobs.clock = clock
    chats[0].preferences.append(dict(_GOOD_PREFERENCE))
    down = [True]
    if dependency == "vector":
        memory_collection = runtime.components.memory.vector_index.store.collection
        backend = next(item for item in vector_backends if item.collection == memory_collection)
        search = backend.search

        async def busy_search(query_vector, *, filters, limit):
            if down[0]:
                raise VectorStoreBusyError("vikingdb is temporarily unreachable")
            return await search(query_vector, filters=filters, limit=limit)

        backend.search = busy_search
    else:
        embed = EmbeddingProvider.embed

        async def unreachable_embed(self, text: str, *, is_query: bool) -> EmbeddingVector:
            if down[0]:
                raise ModelTransportError("embedding service unreachable")
            return await embed(self, text, is_query=is_query)

        monkeypatch.setattr(EmbeddingProvider, "embed", unreachable_embed)
    _enqueue(runtime, "conversation-outage", date(2026, 7, 1), closed_turn())

    waiting, result = _run_through_outage(
        runtime,
        outages=max_attempts + 2,
        clock=clock,
        restore=lambda: down.__setitem__(0, False),
    )

    assert waiting is not None and waiting.status is MemoryJobStatus.QUEUED
    assert waiting.attempts == 0 and waiting.model_waits == max_attempts + 2
    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert (tmp_path / "data" / "memory" / "preferences" / "回答风格.md").exists()
