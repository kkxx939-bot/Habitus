"""远程向量库之上的全量发布、增量提交和故障恢复语义测试。"""

import asyncio
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from infrastructure.store.sqlite import SQLiteLockStore
from infrastructure.vector import (
    PublishedVectorStore,
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreFilter,
    VectorStoreIntegrityError,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreUnsupportedTopologyError,
)
from ModelClient import EmbeddingVector

PUBLICATION_LOCK = PathLock(ProcessLocalLockStore())


def published(backend: "MemoryBackend") -> PublishedVectorStore:
    return PublishedVectorStore(backend, path_lock=PUBLICATION_LOCK)


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
    requires_cross_process_publication_fencing = False

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
        self.calls.append(("metadata", dict(values)))
        self.metadata.update({name: dict(value) for name, value in values.items()})

    async def ensure_schema(self, dimension, *, published_dimension):
        self.calls.append(("schema", dimension, published_dimension))

    async def read(self, identities):
        return tuple(self.records[item] for item in identities if item in self.records)

    async def delete_all(self):
        self.calls.append("delete_all")
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
        return tuple(VectorStoreMatch(item, 1.0) for item in self.records.values() if filters.matches(item.attributes))[
            :limit
        ]

    async def scan(self, *, filters, limit):
        return tuple(item for item in self.records.values() if filters.matches(item.attributes))[:limit]

    async def close(self):
        self.calls.append("close")


def test_remote_publication_requires_host_scoped_fencing(tmp_path) -> None:
    class RemoteBackend(MemoryBackend):
        requires_cross_process_publication_fencing = True

    backend = RemoteBackend()
    with pytest.raises(VectorStoreUnsupportedTopologyError, match="host-scoped"):
        PublishedVectorStore(
            backend,
            path_lock=PathLock(ProcessLocalLockStore()),
        )
    assert backend.calls == []

    store = PublishedVectorStore(
        backend,
        path_lock=PathLock(SQLiteLockStore(tmp_path / "publication.sqlite3")),
    )
    assert store.backend is backend
    assert backend.calls == []


def test_full_publication_then_incremental_update_advances_generation_and_checkpoint() -> None:
    async def scenario() -> None:
        backend = MemoryBackend()
        store = published(backend)
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
        store = published(MemoryBackend())
        await store.replace_all(
            (make_record("a"),),
            schema_version="v1",
            embedding_fingerprint="e1",
            dimension=2,
            checkpoint=5,
            expected_generation=None,
        )
        with pytest.raises(VectorStoreConflictError, match="generation"):
            await store.replace_all(
                (),
                schema_version="v1",
                embedding_fingerprint="e1",
                dimension=2,
                checkpoint=6,
                expected_generation=None,
            )
        with pytest.raises(ValueError, match="upserted and deleted"):
            await store.apply((make_record("a"),), ("a",), checkpoint=6, expected_generation=1, expected_checkpoint=5)
        with pytest.raises(ValueError, match="backwards"):
            await store.apply((), (), checkpoint=4, expected_generation=1, expected_checkpoint=5)
        with pytest.raises(ValueError, match="dimensions"):
            await store.apply(
                (make_record("b", (1.0, 0.0, 0.0)),), (), checkpoint=6, expected_generation=1, expected_checkpoint=5
            )

    asyncio.run(scenario())


def test_unfinished_incremental_publication_is_not_silently_reused() -> None:
    async def scenario() -> None:
        backend = MemoryBackend()
        store = published(backend)
        await store.replace_all(
            (make_record("a"),),
            schema_version="v1",
            embedding_fingerprint="e1",
            dimension=2,
            checkpoint=1,
            expected_generation=None,
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


def test_full_replacement_claims_building_before_deleting_records() -> None:
    async def scenario() -> None:
        backend = MemoryBackend()
        store = published(backend)
        await store.replace_all(
            (make_record("a"),),
            schema_version="v1",
            embedding_fingerprint="e1",
            dimension=2,
            checkpoint=1,
            expected_generation=None,
        )
        building_index = next(
            index
            for index, call in enumerate(backend.calls)
            if isinstance(call, tuple) and call[0] == "metadata" and call[1].get("claim", {}).get("building") is True
        )
        assert building_index < backend.calls.index("delete_all")

    asyncio.run(scenario())


def test_building_claim_without_published_state_never_looks_like_empty_index() -> None:
    async def scenario() -> None:
        backend = MemoryBackend()
        backend.metadata["claim"] = {
            "format": "m2bos_vector_publication_v1",
            "building": True,
        }
        store = published(backend)
        with pytest.raises(VectorStoreBusyError):
            await store.search(
                EmbeddingVector((1.0, 0.0)),
                filters=VectorStoreFilter({}, {}),
                limit=1,
            )

    asyncio.run(scenario())


def test_search_is_empty_before_publish_and_validates_dimension_backend_results_and_limit() -> None:
    async def scenario() -> None:
        backend = MemoryBackend()
        store = published(backend)
        filters = VectorStoreFilter({}, {})
        assert await store.search(EmbeddingVector((1.0, 0.0)), filters=filters, limit=1) == ()
        await store.replace_all(
            (make_record("a"),),
            schema_version="v1",
            embedding_fingerprint="e1",
            dimension=2,
            checkpoint=1,
            expected_generation=None,
        )
        with pytest.raises(VectorStoreIntegrityError, match="dimension"):
            await store.search(EmbeddingVector((1.0,)), filters=filters, limit=1)
        with pytest.raises(ValueError, match="outside"):
            await store.search(EmbeddingVector((1.0, 0.0)), filters=filters, limit=21)

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["read", "search", "scan"])
def test_reads_retry_when_publication_changes_during_backend_read(operation: str) -> None:
    class PausingBackend(MemoryBackend):
        def __init__(self) -> None:
            super().__init__()
            self.pause_operation: str | None = None
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def _pause_once(self, selected: str, result):
            if self.pause_operation == selected:
                self.pause_operation = None
                self.entered.set()
                await self.release.wait()
            return result

        async def read(self, identities):
            result = await super().read(identities)
            return await self._pause_once("read", result)

        async def search(self, query_vector, *, filters, limit):
            result = await super().search(query_vector, filters=filters, limit=limit)
            return await self._pause_once("search", result)

        async def scan(self, *, filters, limit):
            result = await super().scan(filters=filters, limit=limit)
            return await self._pause_once("scan", result)

    async def scenario() -> None:
        backend = PausingBackend()
        store = published(backend)
        await store.replace_all(
            (make_record("old"),),
            schema_version="v1",
            embedding_fingerprint="e1",
            dimension=2,
            checkpoint=1,
            expected_generation=None,
        )
        backend.pause_operation = operation
        filters = VectorStoreFilter({}, {})
        if operation == "read":
            read_task = asyncio.create_task(store.read(("old", "new")))
        elif operation == "search":
            read_task = asyncio.create_task(store.search(EmbeddingVector((1.0, 0.0)), filters=filters, limit=10))
        else:
            read_task = asyncio.create_task(store.scan(filters=filters, limit=10))
        await backend.entered.wait()
        await store.replace_all(
            (make_record("new"),),
            schema_version="v1",
            embedding_fingerprint="e1",
            dimension=2,
            checkpoint=2,
            expected_generation=1,
        )
        backend.release.set()
        result = await read_task
        identities = tuple(
            item.record.identity if isinstance(item, VectorStoreMatch) else item.identity for item in result
        )
        assert identities == ("new",)

    asyncio.run(scenario())


def test_corrupt_publication_metadata_is_rejected_during_initialization() -> None:
    backend = MemoryBackend()
    backend.metadata["claim"] = {"format": "m2bos_vector_publication_v1", "building": False}
    with pytest.raises(VectorStoreIntegrityError, match="ready without state"):
        asyncio.run(published(backend).initialize())


def test_shared_backend_serializes_publication_across_store_instances() -> None:
    class ConcurrentBackend(MemoryBackend):
        def __init__(self) -> None:
            super().__init__()
            self.active_mutations = 0
            self.max_active_mutations = 0

        async def upsert(self, records):
            self.active_mutations += 1
            self.max_active_mutations = max(
                self.max_active_mutations,
                self.active_mutations,
            )
            try:
                await asyncio.sleep(0.01)
                await super().upsert(records)
            finally:
                self.active_mutations -= 1

    async def scenario() -> None:
        backend = ConcurrentBackend()
        seed = published(backend)
        await seed.replace_all(
            (make_record("seed"),),
            schema_version="v1",
            embedding_fingerprint="e1",
            dimension=2,
            checkpoint=1,
            expected_generation=None,
        )
        first = published(backend)
        second = published(backend)
        await asyncio.gather(first.initialize(), second.initialize())
        backend.max_active_mutations = 0

        results = await asyncio.gather(
            first.apply(
                (make_record("high"),),
                (),
                checkpoint=3,
                expected_generation=1,
                expected_checkpoint=1,
            ),
            second.apply(
                (make_record("low"),),
                (),
                checkpoint=2,
                expected_generation=1,
                expected_checkpoint=1,
            ),
            return_exceptions=True,
        )
        successes = tuple(item for item in results if not isinstance(item, BaseException))
        conflicts = tuple(item for item in results if isinstance(item, BaseException))
        final = await seed.state()
        assert final is not None

        assert len(successes) == 1
        assert len(conflicts) == 1
        assert isinstance(conflicts[0], VectorStoreConflictError)
        assert final.generation == 2
        assert backend.max_active_mutations == 1

    asyncio.run(scenario())


def test_shared_backend_serializes_publication_across_thread_event_loops() -> None:
    class ConcurrentBackend(MemoryBackend):
        def __init__(self) -> None:
            super().__init__()
            self.active_mutations = 0
            self.max_active_mutations = 0

        async def upsert(self, records):
            self.active_mutations += 1
            self.max_active_mutations = max(
                self.max_active_mutations,
                self.active_mutations,
            )
            try:
                await asyncio.sleep(0.01)
                await super().upsert(records)
            finally:
                self.active_mutations -= 1

    backend = ConcurrentBackend()
    seed = published(backend)
    asyncio.run(
        seed.replace_all(
            (make_record("seed"),),
            schema_version="v1",
            embedding_fingerprint="e1",
            dimension=2,
            checkpoint=1,
            expected_generation=None,
        )
    )
    backend.max_active_mutations = 0
    ready = threading.Barrier(3)

    def mutate(identity: str, checkpoint: int):
        ready.wait()
        store = published(backend)
        try:
            return asyncio.run(
                store.apply(
                    (make_record(identity),),
                    (),
                    checkpoint=checkpoint,
                    expected_generation=1,
                    expected_checkpoint=1,
                )
            )
        except BaseException as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(mutate, "high", 3),
            executor.submit(mutate, "low", 2),
        )
        ready.wait()
        results = tuple(future.result(timeout=5) for future in futures)

    successes = tuple(item for item in results if not isinstance(item, BaseException))
    conflicts = tuple(item for item in results if isinstance(item, BaseException))
    final = asyncio.run(seed.state())

    assert len(successes) == 1
    assert len(conflicts) == 1
    assert isinstance(conflicts[0], VectorStoreConflictError)
    assert final is not None and final.generation == 2
    assert backend.max_active_mutations == 1
