"""从完整 ConversationSegment 到 Job/Receipt/双向量 checkpoint 的主链验收测试。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import yaml

from habitus.config import HabitusConfig
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.infrastructure.vector import VectorStoreFactory, VectorStoreMatch, VectorStoreState
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
    ProviderCapabilities,
    ProviderFactory,
)
from habitus.pre.conversation import ConversationBatch
from habitus.runtime import build_runtime
from tests.helpers import BASE_TIME, closed_turn, summary_content
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
                "preferences": [],
                "entities": [],
                "tools": [],
                "events": [],
                "intentions": [],
                "identity_proposals": [],
                "relations": [],
            },
        }
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
    editor_segment = runtime.components.workflow.runner.executor.conversations.read_editor_segment(
        address,
        trigger,
    )
    assert editor_segment is not None
    assert receipt_source.editor_segment_id == editor_segment.segment_id
    assert receipt_source.editor_segment_digest == editor_segment.digest
    assert chats[0].response_names.count("memory_candidate_batch") == 1
    assert runtime.components.memory.editor.transaction.journal.try_read(queued.transaction_id) is None
    assert asyncio.run(runtime.failed_memory_job()) is None


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
