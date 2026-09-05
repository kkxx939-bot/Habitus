"""Conversation Summary 独立远程索引的真相源、身份和陈旧命中测试。"""

import asyncio
import hashlib
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.infrastructure.vector import VectorStoreMatch, VectorStoreState
from habitus.memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationRangeSummaryGenerator,
    ConversationRangeSummaryStore,
    ConversationSummaryCompactor,
    ConversationSummaryStore,
)
from habitus.memory.conversation.indexing import (
    ConversationSummaryIndexError,
    ConversationSummaryVectorIndexConfig,
    PersistentConversationSummaryVectorIndex,
    conversation_summary_embedding_fingerprint,
)
from habitus.memory.conversation.indexing.model import summary_reference
from habitus.model_client import EmbeddingVector
from habitus.pre.conversation import ConversationBatch
from tests.helpers import closed_turn, segment, segment_summary
from tests.unit.retrieval.test_search_service import structured


class Embedder:
    async def embed_query(self, _value):
        return EmbeddingVector((1.0, 0.0))

    async def embed_documents(self, values):
        return tuple(EmbeddingVector((1.0, 0.0)) for _ in values)


class VectorStore:
    def __init__(self) -> None:
        self.current = None
        self.records = {}

    async def initialize(self):
        return None

    async def state(self):
        return self.current

    async def read(self, identities):
        return tuple(self.records[identity] for identity in identities if identity in self.records)

    async def replace_all(self, records, **kwargs):
        self.records = {record.identity: record for record in records}
        self.current = VectorStoreState(
            kwargs["schema_version"],
            kwargs["embedding_fingerprint"],
            kwargs["dimension"],
            kwargs["checkpoint"],
            1,
            len(records),
        )
        return self.current

    async def apply(self, upserts, deletes, **kwargs):
        for record in upserts:
            self.records[record.identity] = record
        for identity in deletes:
            self.records.pop(identity, None)
        self.current = VectorStoreState(
            self.current.schema_version,
            self.current.embedding_fingerprint,
            self.current.dimension,
            kwargs["checkpoint"],
            self.current.generation + 1,
            len(self.records),
        )
        return self.current

    async def search(self, _vector, *, filters, limit):
        return tuple(
            VectorStoreMatch(record, 0.8)
            for record in self.records.values()
            if filters.matches(record.attributes)
        )[:limit]

    async def scan(self, *, filters, limit):
        return tuple(
            record for record in self.records.values() if filters.matches(record.attributes)
        )[:limit]

    async def close(self):
        return None


class RetirementFilter:
    def __init__(self) -> None:
        self.hidden_identities = set()

    def hidden(self, reference) -> bool:
        return reference.identity in self.hidden_identities


def source_chain(tmp_path: Path):
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    journal.append(address, ConversationBatch("conversation-1", closed_turn()))
    source = segment(
        segment_id="000000000000-000000000001",
        messages=closed_turn(),
    )
    segment_store = ConversationSummaryStore(journal.layout)
    segment_store.create(address, source, segment_summary(source))
    range_store = ConversationRangeSummaryStore(journal.layout)
    compactor = ConversationSummaryCompactor(
        journal,
        segment_store,
        range_store,
        ConversationRangeSummaryGenerator(structured([])),
    )
    return address, source, segment_store, compactor


def test_summary_reference_has_independent_non_memory_identity_and_stable_fingerprint(tmp_path: Path) -> None:
    address, source, segment_store, _compactor = source_chain(tmp_path)
    summary = segment_store.read(address, source)
    reference = summary_reference(address, summary)
    first = conversation_summary_embedding_fingerprint(
        provider="p",
        adapter="a",
        model="m",
        base_url="https://embedding.example.com/v1",
        dimension=2,
        input_mode="text",
        extra_body={},
        document_parameters={},
    )
    second = conversation_summary_embedding_fingerprint(
        provider="p",
        adapter="a",
        model="m2",
        base_url="https://embedding.example.com/v1",
        dimension=2,
        input_mode="text",
        extra_body={},
        document_parameters={},
    )
    assert reference.identity.startswith("conversation-summary:")
    assert not reference.identity.startswith("memory://")
    assert reference.summary_id == source.segment_id
    assert first != second


def test_summary_index_rebuilds_active_frontier_and_rejects_stale_remote_hit(tmp_path: Path) -> None:
    address, source, segment_store, compactor = source_chain(tmp_path)
    store = VectorStore()
    index = PersistentConversationSummaryVectorIndex(
        compactor.journal,
        compactor,
        Embedder(),
        store,
        dimension=2,
        embedding_fingerprint="summary-v1",
        config=ConversationSummaryVectorIndexConfig(min_vector_candidates=1),
    )

    state = asyncio.run(index.ensure_ready())
    matches = asyncio.run(index.search("之前如何决定", limit=1))

    assert state.record_count == 1
    assert matches[0].reference.address == address
    assert matches[0].summary.segment_id == source.segment_id
    segment_store.delete_by_id(address, source.segment_id)
    with pytest.raises(ConversationSummaryIndexError, match="no longer part"):
        asyncio.run(index.search("之前如何决定", limit=1))


def test_summary_synchronize_exactly_deletes_manifest_hidden_identity(tmp_path: Path) -> None:
    address, source, segment_store, compactor = source_chain(tmp_path)
    store = VectorStore()
    retirements = RetirementFilter()
    index = PersistentConversationSummaryVectorIndex(
        compactor.journal,
        compactor,
        Embedder(),
        store,
        dimension=2,
        embedding_fingerprint="summary-v1",
        config=ConversationSummaryVectorIndexConfig(min_vector_candidates=1),
        retirement_store=retirements,
    )
    reference = summary_reference(address, segment_store.read(address, source))
    asyncio.run(index.ensure_ready())
    assert tuple(store.records) == (reference.identity,)

    retirements.hidden_identities.add(reference.identity)
    state = asyncio.run(
        index.synchronize(address, removed_references=(reference,))
    )

    assert state.record_count == 0
    assert store.records == {}


def test_summary_consistency_audit_is_read_only_and_reports_stale_record(tmp_path: Path) -> None:
    _address, _source, _segment_store, compactor = source_chain(tmp_path)
    embedder = Embedder()
    store = VectorStore()
    index = PersistentConversationSummaryVectorIndex(
        compactor.journal,
        compactor,
        embedder,
        store,
        dimension=2,
        embedding_fingerprint="summary-v1",
        config=ConversationSummaryVectorIndexConfig(min_vector_candidates=1),
    )
    asyncio.run(index.ensure_ready())
    identity = next(iter(store.records))
    stale_content = store.records[identity].content + "\n陈旧摘要"
    store.records[identity] = replace(
        store.records[identity],
        content=stale_content,
        content_digest=hashlib.sha256(stale_content.encode("utf-8")).hexdigest(),
    )

    report = asyncio.run(index.audit_consistency())

    assert report.stale_identities == (identity,)
