"""远程向量库之上的全量发布、增量提交和故障恢复语义测试。"""

import asyncio
import hashlib

import pytest

from infrastructure.vector import (
    PublishedVectorStore,
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreFilter,
    VectorStoreIntegrityError,
    VectorStoreMatch,
    VectorStoreRecord,
)
from ModelClient import EmbeddingVector


def make_record(identity: str, vector: tuple[float, ...] = (1.0, 0.0)) -> VectorStoreRecord:
    content = f"content:{identity}"
    return VectorStoreRecord(
        identity,
        EmbeddingVector(vector),
        content,
        hashlib.sha256(content.encode()).hexdigest(),
        {"kind": "profile", "revision": 1},
    )


class MemoryBackend:
    adapter_name = "fake"
    provider_name = "fake-provider"
    collection = "memory"
    max_records = 100
    max_search_hits = 20

    def __init__(self) -> None:
        self.metadata = {}
        self.records = {}
        self.calls = []
        self.fail_visibility = False

    async def initialize(self):
        self.calls.append("initialize")

    async def read_metadata(self, names):
        return {name: self.metadata[name] for name in names if name in self.metadata}

    async def write_metadata(self, values, *, dimension):
        self.metadata.update({name: dict(value) for name, value in values.items()})

    async def ensure_schema(self, dimension, *, published_dimension):
        self.calls.append(("schema", dimension, published_dimension))

    async def read(self, identities):
        return tuple(self.records[item] for item in identities if item in self.records)

    async def delete_all(self):
        self.records.clear()

    async def upsert(self, records):
        self.records.update({item.identity: item for item in records})

    async def delete(self, identities):
        for identity in identities:
            self.records.pop(identity, None)

    async def validate_records(self, records, *, replacing):
        self.calls.append(("validate", replacing))

    async def wait_visible(self, upserts, deletes, *, complete):
        if self.fail_visibility:
            raise TimeoutError("index not visible")

    async def search(self, query_vector, *, filters, limit):
        return tuple(VectorStoreMatch(item, 1.0) for item in self.records.values() if filters.matches(item.attributes))[:limit]

    async def scan(self, *, filters, limit):
        return tuple(item for item in self.records.values() if filters.matches(item.attributes))[:limit]

    async def close(self):
        self.calls.append("close")


def test_full_publication_then_incremental_update_advances_generation_and_checkpoint() -> None:
    async def scenario() -> None:
        backend = MemoryBackend()
        store = PublishedVectorStore(backend)
        first = await store.replace_all(
            (make_record("a"), make_record("b")),
            schema_version="memory-v1",
            embedding_fingerprint="embed-v1",
            dimension=2,
            checkpoint=10,
            expected_generation=None,
        )
        assert (first.generation, first.checkpoint, first.record_count) == (1, 10, 2)
        assert (await store.state()).ready

        second = await store.apply(
            (make_record("c"),),
            ("a",),
            checkpoint=11,
            expected_generation=1,
            expected_checkpoint=10,
        )
        assert (second.generation, second.checkpoint, second.record_count) == (2, 11, 2)
        assert tuple(item.identity for item in await store.read(("a", "b", "c"))) == ("b", "c")

    asyncio.run(scenario())


def test_publication_rejects_stale_generation_overlap_dimension_and_backward_checkpoint() -> None:
    async def scenario() -> None:
        store = PublishedVectorStore(MemoryBackend())
        await store.replace_all(
            (make_record("a"),), schema_version="v1", embedding_fingerprint="e1",
            dimension=2, checkpoint=5, expected_generation=None,
        )
        with pytest.raises(VectorStoreConflictError, match="generation"):
            await store.replace_all(
                (), schema_version="v1", embedding_fingerprint="e1",
                dimension=2, checkpoint=6, expected_generation=None,
            )
        with pytest.raises(ValueError, match="upserted and deleted"):
            await store.apply((make_record("a"),), ("a",), checkpoint=6, expected_generation=1, expected_checkpoint=5)
        with pytest.raises(ValueError, match="backwards"):
            await store.apply((), (), checkpoint=4, expected_generation=1, expected_checkpoint=5)
        with pytest.raises(ValueError, match="dimensions"):
            await store.apply((make_record("b", (1.0, 0.0, 0.0)),), (), checkpoint=6, expected_generation=1, expected_checkpoint=5)

    asyncio.run(scenario())


def test_unfinished_incremental_publication_is_not_silently_reused() -> None:
    async def scenario() -> None:
        backend = MemoryBackend()
        store = PublishedVectorStore(backend)
        await store.replace_all(
            (make_record("a"),), schema_version="v1", embedding_fingerprint="e1",
            dimension=2, checkpoint=1, expected_generation=None,
        )
        backend.fail_visibility = True
        with pytest.raises(TimeoutError, match="not visible"):
            await store.apply((make_record("b"),), (), checkpoint=2, expected_generation=1, expected_checkpoint=1)
        backend.fail_visibility = False
        assert not (await store.state()).ready
        with pytest.raises(VectorStoreConflictError, match="unfinished publication"):
            await store.apply((), (), checkpoint=2, expected_generation=1, expected_checkpoint=1)
        with pytest.raises(VectorStoreBusyError):
            await store.search(EmbeddingVector((1.0, 0.0)), filters=VectorStoreFilter({}, {}), limit=1)

    asyncio.run(scenario())


def test_search_is_empty_before_publish_and_validates_dimension_backend_results_and_limit() -> None:
    async def scenario() -> None:
        backend = MemoryBackend()
        store = PublishedVectorStore(backend)
        filters = VectorStoreFilter({}, {})
        assert await store.search(EmbeddingVector((1.0, 0.0)), filters=filters, limit=1) == ()
        await store.replace_all(
            (make_record("a"),), schema_version="v1", embedding_fingerprint="e1",
            dimension=2, checkpoint=1, expected_generation=None,
        )
        with pytest.raises(VectorStoreIntegrityError, match="dimension"):
            await store.search(EmbeddingVector((1.0,)), filters=filters, limit=1)
        with pytest.raises(ValueError, match="outside"):
            await store.search(EmbeddingVector((1.0, 0.0)), filters=filters, limit=21)

    asyncio.run(scenario())


def test_corrupt_publication_metadata_is_rejected_during_initialization() -> None:
    backend = MemoryBackend()
    backend.metadata["claim"] = {"format": "m2bos_vector_publication_v1", "building": False}
    with pytest.raises(VectorStoreIntegrityError, match="ready without state"):
        asyncio.run(PublishedVectorStore(backend).initialize())
