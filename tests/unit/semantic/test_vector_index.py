"""MemoryTree 真相源到远程 VectorStore 的重建、增量和过滤测试。"""

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from infrastructure.vector import VectorStoreMatch, VectorStoreState
from memory.indexing import MemoryVectorIndexError, PersistentMemoryVectorIndex
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryKind, MemoryLevel
from memory.tree import MemoryTree
from memory.uri import MemoryURI
from ModelClient import EmbeddingVector
from tests.helpers import document


class Embedder:
    def __init__(self) -> None:
        self.inputs = []

    async def embed_documents(self, values):
        self.inputs.extend(values)
        return tuple(EmbeddingVector((1.0, 0.0)) for _ in values)


class VectorStore:
    def __init__(self) -> None:
        self.current = None
        self.records = {}
        self.filters = []

    async def initialize(self):
        pass

    async def state(self):
        return self.current

    async def read(self, identities):
        return tuple(self.records[item] for item in identities if item in self.records)

    async def replace_all(self, records, **kwargs):
        self.records = {item.identity: item for item in records}
        self.current = VectorStoreState(
            kwargs["schema_version"], kwargs["embedding_fingerprint"], kwargs["dimension"],
            kwargs["checkpoint"], 1 if self.current is None else self.current.generation + 1, len(records),
        )
        return self.current
    async def apply(self, upserts, deletes, **kwargs):
        for item in upserts:
            self.records[item.identity] = item
        for identity in deletes:
            self.records.pop(identity, None)
        self.current = VectorStoreState(
            self.current.schema_version, self.current.embedding_fingerprint, self.current.dimension,
            kwargs["checkpoint"], self.current.generation + 1, len(self.records),
        )
        return self.current
    async def search(self, query_vector, *, filters, limit):
        self.filters.append(filters)
        return tuple(
            VectorStoreMatch(item, 0.9)
            for item in self.records.values()
            if filters.matches(item.attributes)
        )[:limit]
    async def scan(self, *, filters, limit):
        return tuple(self.records.values())[:limit]

    async def close(self):
        pass


def index(tmp_path: Path):
    tree = MemoryTree(tmp_path / "memory")
    embedder = Embedder()
    store = VectorStore()
    return tree, embedder, store, PersistentMemoryVectorIndex(
        tree, embedder, store, dimension=2, embedding_fingerprint="embed-v1"
    )


def test_missing_remote_index_is_fully_rebuilt_from_tree_and_consistency_is_clean(tmp_path: Path) -> None:
    tree, embedder, store, vector_index = index(tmp_path)
    tree.write(document(MemoryKind.PROFILE))
    tree.write(document(MemoryKind.PREFERENCE))
    state = asyncio.run(vector_index.ensure_ready())
    report = asyncio.run(vector_index.check_consistency())
    assert state.record_count == 2
    assert len(embedder.inputs) == 2
    assert report.ok
    assert report.expected_count == report.indexed_count == 2


def test_read_only_consistency_audit_reports_stale_record_without_rebuilding(tmp_path: Path) -> None:
    tree, embedder, store, vector_index = index(tmp_path)
    tree.write(document(MemoryKind.PROFILE))
    asyncio.run(vector_index.ensure_ready())
    identity = next(iter(store.records))
    stale_content = store.records[identity].content + "\n陈旧内容"
    store.records[identity] = replace(
        store.records[identity],
        content=stale_content,
        content_digest=hashlib.sha256(stale_content.encode("utf-8")).hexdigest(),
    )
    embedded_before = tuple(embedder.inputs)

    report = asyncio.run(vector_index.audit_consistency())

    assert report.stale_identities == (identity,)
    assert tuple(embedder.inputs) == embedded_before


def test_incremental_sync_updates_changed_l2_and_advances_global_job_checkpoint(tmp_path: Path) -> None:
    tree, embedder, store, vector_index = index(tmp_path)
    original = document(MemoryKind.PREFERENCE)
    tree.write(original)
    asyncio.run(vector_index.ensure_ready())
    changed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 改为极简回答"},
        revision=2,
    )
    tree.write(changed)
    state = asyncio.run(
        vector_index.synchronize(
            changed_uris=(MemoryURI.from_address(changed.address),),
            semantic_results=(),
            checkpoint=2,
        )
    )
    assert state.checkpoint == 2
    assert len(embedder.inputs) == 2
    with pytest.raises(MemoryVectorIndexError, match="backwards"):
        asyncio.run(
            vector_index.synchronize(
                changed_uris=(), semantic_results=(), checkpoint=1
            )
        )


def test_l2_search_applies_kind_and_scope_filters_before_top_k(tmp_path: Path) -> None:
    tree, _embedder, store, vector_index = index(tmp_path)
    tree.write(document(MemoryKind.PROFILE))
    tree.write(document(MemoryKind.PREFERENCE))
    asyncio.run(vector_index.ensure_ready())
    matches = asyncio.run(
        vector_index.search(
            EmbeddingVector((1.0, 0.0)),
            roots=(MemoryURI.root(),),
            levels=(MemoryLevel.DETAIL,),
            kinds=(MemoryKind.PREFERENCE,),
            intention_scope=MemoryIntentionRecallScope.ACTIVE,
            limit=5,
        )
    )
    assert tuple(match.uri.to_address().kind for match in matches) == (MemoryKind.PREFERENCE,)
    assert store.filters[-1].one_of["kind"] == ("preference",)


def test_l2_kind_filter_cannot_be_mixed_with_directory_levels(tmp_path: Path) -> None:
    tree, _embedder, _store, vector_index = index(tmp_path)
    tree.write(document(MemoryKind.PROFILE))
    with pytest.raises(ValueError, match="cannot be mixed"):
        asyncio.run(
            vector_index.search(
                EmbeddingVector((1.0, 0.0)),
                roots=(MemoryURI.root(),),
                levels=(MemoryLevel.DETAIL, MemoryLevel.ABSTRACT),
                kinds=(),
                intention_scope=MemoryIntentionRecallScope.ACTIVE,
                limit=5,
            )
        )
