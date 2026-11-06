"""本地 SQLite+numpy 向量后端：经 PublishedVectorStore 的真实发布、过滤、排序、重启读回与故障映射。"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import stat
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector import (
    PublishedVectorStore,
    VectorStoreBusyError,
    VectorStoreConfig,
    VectorStoreConflictError,
    VectorStoreError,
    VectorStoreFactory,
    VectorStoreFilter,
    VectorStoreIntegrityError,
    VectorStoreRecord,
    VectorStoreRequirements,
    VectorStoreRouteConfig,
)
from habitus.infrastructure.vector.adapters import (
    LocalSQLiteVectorBackend,
    LocalSQLiteVectorStoreConfig,
    local_sqlite,
    register_builtin_vector_adapters,
)
from habitus.memory.workflow.failure import MemoryJobFailureKind, memory_job_failure_kind
from habitus.model_client import EmbeddingVector
from tests.lock_store import ProcessLocalLockStore

LOCK = PathLock(ProcessLocalLockStore())
NO_FILTER = VectorStoreFilter(equals={}, one_of={})


def record(
    identity: str,
    vector: tuple[float, ...],
    *,
    directory_key: str = "memory://",
    level: int = 2,
    scope_roots: tuple[str, ...] = ("memory://",),
    content: str | None = None,
) -> VectorStoreRecord:
    text = content or f"content of {identity}"
    return VectorStoreRecord(
        identity=identity,
        vector=EmbeddingVector(vector),
        content=text,
        content_digest=hashlib.sha256(text.encode()).hexdigest(),
        attributes={
            "uri": identity,
            "level": level,
            "directory_key": directory_key,
            "scope_roots": scope_roots,
            "kind": "profile",
            "revision": 1,
        },
    )


def open_store(
    root: Path,
    *,
    collection: str = "memory",
    busy_timeout_seconds: float = 5.0,
) -> PublishedVectorStore:
    config = VectorStoreConfig(
        collection=collection,
        options={"busy_timeout_seconds": busy_timeout_seconds},
    )
    return register_builtin_vector_adapters().create(
        config,
        requirements=VectorStoreRequirements(dimension=3, max_records=100, max_search_hits=50, max_record_chars=1000),
        path_lock=LOCK,
        data_root=root,
    )


def backend_of(store: PublishedVectorStore) -> LocalSQLiteVectorBackend:
    assert isinstance(store.backend, LocalSQLiteVectorBackend)
    return store.backend


async def publish(store: PublishedVectorStore, records: tuple[VectorStoreRecord, ...], *, dimension: int = 3):
    current = await store.state()
    return await store.replace_all(
        records,
        schema_version="memory-v1",
        embedding_fingerprint="embed-v1",
        dimension=dimension,
        checkpoint=1,
        expected_generation=None if current is None else current.generation,
    )


SAMPLE = (
    record("memory://a.md", (1.0, 0.0, 0.0), directory_key="memory://", level=2),
    record("memory://b.md", (0.8, 0.6, 0.0), directory_key="memory://", level=2),
    record("memory://c.md", (0.0, 1.0, 0.0), directory_key="memory://events/", level=2),
    record(
        "memory://events/",
        (0.6, 0.8, 0.0),
        directory_key="memory://events/",
        level=1,
        scope_roots=("memory://", "memory://events/"),
    ),
    record("memory://d.md", (-1.0, 0.0, 0.0), directory_key="memory://", level=0),
)


def test_publish_search_ranks_by_cosine_and_filters_equals_one_of_and_arrays(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = open_store(tmp_path)
        state = await publish(store, SAMPLE)
        assert (state.generation, state.checkpoint, state.record_count) == (1, 1, 5)
        query = EmbeddingVector((1.0, 0.0, 0.0))

        everything = await store.search(query, filters=NO_FILTER, limit=50)
        assert [match.record.identity for match in everything] == [
            "memory://a.md",
            "memory://b.md",
            "memory://events/",
            "memory://c.md",
            "memory://d.md",
        ]
        assert [round(match.score, 5) for match in everything] == [1.0, 0.8, 0.6, 0.0, -1.0]
        assert everything[0].record == SAMPLE[0]

        by_directory = await store.search(
            query, filters=VectorStoreFilter(equals={"directory_key": "memory://events/"}, one_of={}), limit=50
        )
        assert [match.record.identity for match in by_directory] == ["memory://events/", "memory://c.md"]

        by_level = await store.search(query, filters=VectorStoreFilter(equals={}, one_of={"level": (0, 1)}), limit=50)
        assert [match.record.identity for match in by_level] == ["memory://events/", "memory://d.md"]

        by_scope = await store.search(
            query, filters=VectorStoreFilter(equals={}, one_of={"scope_roots": ("memory://events/",)}), limit=50
        )
        assert [match.record.identity for match in by_scope] == ["memory://events/"]

        top_two = await store.search(query, filters=NO_FILTER, limit=2)
        assert [match.record.identity for match in top_two] == ["memory://a.md", "memory://b.md"]
        assert (
            await store.search(
                query, filters=VectorStoreFilter(equals={"directory_key": "memory://none/"}, one_of={}), limit=5
            )
            == ()
        )
        await store.close()

    asyncio.run(scenario())


def test_equal_scores_are_ordered_by_identity_not_by_insertion(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = open_store(tmp_path)
        first = await publish(store, (record("memory://z.md", (0.0, 0.0, 1.0)),))
        await store.apply(
            (record("memory://a.md", (0.0, 0.0, 1.0)), record("memory://m.md", (0.0, 0.0, 1.0))),
            (),
            checkpoint=2,
            expected_generation=first.generation,
            expected_checkpoint=first.checkpoint,
        )
        matches = await store.search(EmbeddingVector((0.0, 0.0, 1.0)), filters=NO_FILTER, limit=3)
        assert [match.record.identity for match in matches] == ["memory://a.md", "memory://m.md", "memory://z.md"]
        await store.close()

    asyncio.run(scenario())


def test_crash_in_the_middle_of_a_dimension_change_leaves_a_rebuildable_file(tmp_path: Path) -> None:
    """维度切换在清空旧记录之后、发布新状态之前崩溃：重开后是"未发布"，能全量重建，不卡在旧尺寸的向量上。"""

    async def scenario() -> None:
        store = open_store(tmp_path)
        await publish(store, SAMPLE)
        backend = backend_of(store)

        async def crash() -> None:
            raise RuntimeError("process killed")

        backend.delete_all = crash  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="killed"):
            await store.replace_all(
                (record("memory://x.md", (1.0, 0.0, 0.0, 0.0)),),
                schema_version="memory-v1",
                embedding_fingerprint="embed-v2",
                dimension=4,
                checkpoint=1,
                expected_generation=1,
            )
        await store.close()

        reopened = open_store(tmp_path)
        rebuilt = await reopened.replace_all(
            (record("memory://x.md", (1.0, 0.0, 0.0, 0.0)),),
            schema_version="memory-v1",
            embedding_fingerprint="embed-v2",
            dimension=4,
            checkpoint=1,
            expected_generation=None,
        )
        assert (rebuilt.dimension, rebuilt.record_count, rebuilt.ready) == (4, 1, True)
        await reopened.close()

    asyncio.run(scenario())


def test_a_record_deleted_by_another_connection_mid_search_is_a_retryable_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        store = open_store(tmp_path)
        await publish(store, SAMPLE)
        real_ranked = local_sqlite._ranked

        def ranked_then_deleted_elsewhere(*args: object, **kwargs: object) -> list[tuple[float, str]]:
            result = real_ranked(*args, **kwargs)  # type: ignore[arg-type]
            other = sqlite3.connect(tmp_path / "memory.sqlite3", isolation_level=None)
            other.execute("DELETE FROM records WHERE identity = ?", (result[0][1],))
            other.close()
            return result

        monkeypatch.setattr(local_sqlite, "_ranked", ranked_then_deleted_elsewhere)
        with pytest.raises(VectorStoreConflictError) as caught:
            await store.search(EmbeddingVector((1.0, 0.0, 0.0)), filters=NO_FILTER, limit=3)
        assert memory_job_failure_kind(caught.value) is MemoryJobFailureKind.RETRYABLE
        await store.close()

    asyncio.run(scenario())


def test_incremental_apply_upserts_overwrites_and_deletes(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = open_store(tmp_path)
        first = await publish(store, SAMPLE)
        replaced = record("memory://a.md", (0.0, 0.0, 1.0), content="rewritten a")
        added = record("memory://e.md", (0.0, 0.6, 0.8))
        second = await store.apply(
            (replaced, added),
            ("memory://b.md", "memory://missing.md"),
            checkpoint=2,
            expected_generation=first.generation,
            expected_checkpoint=first.checkpoint,
        )
        assert (second.generation, second.checkpoint, second.record_count) == (2, 2, 5)
        backend = backend_of(store)
        assert backend._cache.matrix.shape == (5, 3)
        assert sorted(backend._cache.identities) == sorted(
            ["memory://a.md", "memory://c.md", "memory://events/", "memory://d.md", "memory://e.md"]
        )
        assert await store.read(("memory://b.md",)) == ()
        (read_a,) = await store.read(("memory://a.md",))
        assert read_a.content == "rewritten a"
        best = await store.search(EmbeddingVector((0.0, 0.0, 1.0)), filters=NO_FILTER, limit=2)
        assert [match.record.identity for match in best] == ["memory://a.md", "memory://e.md"]
        await store.close()

    asyncio.run(scenario())


def test_read_preserves_requested_order_and_scan_is_sorted_and_bounded(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = open_store(tmp_path)
        await publish(store, SAMPLE)
        read = await store.read(("memory://d.md", "memory://nope.md", "memory://a.md"))
        assert [item.identity for item in read] == ["memory://d.md", "memory://a.md"]
        scanned = await store.scan(filters=VectorStoreFilter(equals={"level": 2}, one_of={}), limit=2)
        assert [item.identity for item in scanned] == ["memory://a.md", "memory://b.md"]
        assert len(await store.scan(filters=NO_FILTER, limit=50)) == 5
        await store.close()

    asyncio.run(scenario())


def test_reopening_the_file_reads_back_exactly_what_was_written(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = open_store(tmp_path)
        written = await publish(store, SAMPLE)
        await store.close()
        assert (tmp_path / "memory.sqlite3").is_file()

        reopened = open_store(tmp_path)
        assert await reopened.state() == written
        records = await reopened.scan(filters=NO_FILTER, limit=50)
        by_identity = {item.identity: item for item in records}
        assert set(by_identity) == {item.identity for item in SAMPLE}
        for original in SAMPLE:
            restored = by_identity[original.identity]
            assert restored.content == original.content
            assert restored.content_digest == original.content_digest
            assert dict(restored.attributes) == dict(original.attributes)
            assert restored.vector.values == pytest.approx(original.vector.values, abs=1e-6)
        matches = await reopened.search(EmbeddingVector((1.0, 0.0, 0.0)), filters=NO_FILTER, limit=50)
        assert [match.record.identity for match in matches][:2] == ["memory://a.md", "memory://b.md"]
        await reopened.close()

    asyncio.run(scenario())


def test_writes_from_another_connection_are_visible_without_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        reader = open_store(tmp_path)
        writer = open_store(tmp_path)
        first = await publish(writer, SAMPLE[:1])
        assert [
            match.record.identity
            for match in await reader.search(EmbeddingVector((1.0, 0.0, 0.0)), filters=NO_FILTER, limit=5)
        ] == ["memory://a.md"]
        await writer.apply(
            (SAMPLE[1],),
            ("memory://a.md",),
            checkpoint=2,
            expected_generation=first.generation,
            expected_checkpoint=first.checkpoint,
        )
        matches = await reader.search(EmbeddingVector((1.0, 0.0, 0.0)), filters=NO_FILTER, limit=5)
        assert [match.record.identity for match in matches] == ["memory://b.md"]
        await reader.close()
        await writer.close()

    asyncio.run(scenario())


def test_dimension_change_resets_records_and_publication(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = open_store(tmp_path)
        first = await publish(store, SAMPLE)
        wider = (record("memory://x.md", (1.0, 0.0, 0.0, 0.0)),)
        second = await store.replace_all(
            wider,
            schema_version="memory-v1",
            embedding_fingerprint="embed-v2",
            dimension=4,
            checkpoint=3,
            expected_generation=first.generation,
        )
        assert (second.generation, second.dimension, second.record_count) == (2, 4, 1)
        assert [item.identity for item in await store.scan(filters=NO_FILTER, limit=50)] == ["memory://x.md"]
        await store.close()
        reopened = open_store(tmp_path)
        assert (await reopened.state()) == second
        await reopened.close()

    asyncio.run(scenario())


def test_busy_file_maps_to_retryable_dependency_wait(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = open_store(tmp_path, busy_timeout_seconds=0.05)
        await publish(store, SAMPLE[:1])
        holder = sqlite3.connect(tmp_path / "memory.sqlite3", isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(VectorStoreBusyError) as caught:
                await backend_of(store).upsert((SAMPLE[1],))
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        assert memory_job_failure_kind(caught.value) is MemoryJobFailureKind.WAITS_FOR_MODEL
        await backend_of(store).upsert((SAMPLE[1],))
        assert len(await backend_of(store).read(("memory://a.md", "memory://b.md"))) == 2
        await store.close()

    asyncio.run(scenario())


def test_a_full_disk_waits_for_the_dependency_and_the_next_publication_recovers(tmp_path: Path) -> None:
    """磁盘满（SQLITE_FULL）与 busy 同样判"等依赖恢复"；空间回来后，未完成的增量发布由全量重建补齐。"""

    async def scenario() -> None:
        store = open_store(tmp_path)
        first = await publish(store, SAMPLE[:1])
        connection = backend_of(store)._connection
        assert connection is not None
        pages = connection.execute("PRAGMA page_count").fetchone()[0]
        connection.execute(f"PRAGMA max_page_count = {pages}")
        large = record("memory://large.md", (0.0, 1.0, 0.0), content="x" * 200_000)
        with pytest.raises(VectorStoreBusyError, match="temporarily unavailable") as caught:
            await store.apply(
                (large,), (), checkpoint=2, expected_generation=first.generation, expected_checkpoint=first.checkpoint
            )
        assert memory_job_failure_kind(caught.value) is MemoryJobFailureKind.WAITS_FOR_MODEL
        assert (await store.state()).ready is False
        connection.execute("PRAGMA max_page_count = 1073741823")
        rebuilt = await store.replace_all(
            (SAMPLE[0], large),
            schema_version="memory-v1",
            embedding_fingerprint="embed-v1",
            dimension=3,
            checkpoint=2,
            expected_generation=first.generation,
        )
        assert (rebuilt.ready, rebuilt.record_count) == (True, 2)
        await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (sqlite3.SQLITE_BUSY, VectorStoreBusyError),
        (sqlite3.SQLITE_LOCKED, VectorStoreBusyError),
        (sqlite3.SQLITE_FULL, VectorStoreBusyError),
        (sqlite3.SQLITE_IOERR | (13 << 8), VectorStoreBusyError),
        (sqlite3.SQLITE_CORRUPT, VectorStoreIntegrityError),
        (sqlite3.SQLITE_NOTADB, VectorStoreIntegrityError),
        (sqlite3.SQLITE_CONSTRAINT, VectorStoreError),
    ],
)
def test_sqlite_error_codes_map_to_the_vector_error_kinds(code: int, expected: type[VectorStoreError]) -> None:
    error = sqlite3.OperationalError("simulated")
    error.sqlite_errorcode = code
    translated = local_sqlite._translated(error)
    assert type(translated) is expected


@pytest.mark.parametrize(
    ("mutation", "operation", "message"),
    [
        ("UPDATE records SET vector = x'00' WHERE identity = 'memory://a.md'", "state", "wrong size"),
        ("UPDATE records SET content = 'tampered' WHERE identity = 'memory://a.md'", "read", "fields are invalid"),
        ("UPDATE records SET attributes_json = '[1]' WHERE identity = 'memory://a.md'", "state", "root must be"),
        (
            "DELETE FROM records; UPDATE schema_info SET value = '7' WHERE name = 'dimension'",
            "state",
            "does not match the published state",
        ),
        ("UPDATE schema_info SET value = 'other' WHERE name = 'format'", "state", "unknown format"),
        ("UPDATE metadata SET value_json = 'nope' WHERE name = 'state'", "state", "invalid JSON"),
        ("ALTER TABLE records DROP COLUMN attributes_json", "state", "unexpected layout"),
    ],
)
def test_tampered_files_fail_as_integrity_errors(tmp_path: Path, mutation: str, operation: str, message: str) -> None:
    async def scenario() -> None:
        store = open_store(tmp_path)
        await publish(store, SAMPLE)
        await store.close()
        connection = sqlite3.connect(tmp_path / "memory.sqlite3", isolation_level=None)
        connection.executescript(mutation)
        connection.close()

        reopened = open_store(tmp_path)
        try:
            with pytest.raises(VectorStoreIntegrityError, match=message):
                if operation == "read":
                    await reopened.read(("memory://a.md",))
                else:
                    await reopened.state()
        finally:
            await reopened.close()

    asyncio.run(scenario())


def test_vector_files_are_private_like_the_other_storage_databases(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = open_store(tmp_path / "vector")
        await publish(store, SAMPLE)
        await store.close()

    asyncio.run(scenario())
    assert stat.S_IMODE((tmp_path / "vector").stat().st_mode) == 0o700
    files = sorted((tmp_path / "vector").iterdir())
    assert "memory.sqlite3" in [path.name for path in files]
    assert {path.name: stat.S_IMODE(path.stat().st_mode) for path in files} == {path.name: 0o600 for path in files}


def test_construction_does_not_touch_the_disk(tmp_path: Path) -> None:
    store = open_store(tmp_path / "vector")
    assert not (tmp_path / "vector").exists()
    asyncio.run(store.close())
    assert not (tmp_path / "vector").exists()


@pytest.mark.parametrize(
    ("route", "options", "data_root", "message"),
    [
        ({}, {"unknown": 1}, True, "unknown fields"),
        ({}, {"busy_timeout_seconds": 0}, True, "busy_timeout_seconds"),
        ({}, {"busy_timeout_seconds": True}, True, "busy_timeout_seconds"),
        ({"base_url": "https://vector.example.com"}, {}, True, "base_url"),
        ({"extra_headers": {"X-Tenant": "a"}}, {}, True, "extra_headers"),
        ({}, {}, False, "data_root"),
    ],
)
def test_builder_rejects_misconfiguration(
    tmp_path: Path, route: dict[str, object], options: dict[str, object], data_root: bool, message: str
) -> None:
    config = VectorStoreConfig(route=VectorStoreRouteConfig(**route), options=options)  # type: ignore[arg-type]
    with pytest.raises((ValueError, VectorStoreError), match=message):
        register_builtin_vector_adapters().create(
            config,
            requirements=VectorStoreRequirements(3, 100, 50, 1000),
            path_lock=LOCK,
            data_root=tmp_path if data_root else None,
        )


def test_builder_rejects_collection_names_that_differ_only_by_case_on_disk(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lowercase"):
        register_builtin_vector_adapters().create(
            VectorStoreConfig(collection="Memory"),
            requirements=VectorStoreRequirements(3, 100, 50, 1000),
            path_lock=LOCK,
            data_root=tmp_path,
        )


def test_builder_rejects_credential_reference(tmp_path: Path) -> None:
    config = VectorStoreConfig(route=VectorStoreRouteConfig(credential_ref="vector"))
    with pytest.raises(ValueError, match="credential_ref"):
        register_builtin_vector_adapters().create(
            config,
            requirements=VectorStoreRequirements(3, 100, 50, 1000),
            credentials={"api_key": "x"},
            path_lock=LOCK,
            data_root=tmp_path,
        )


def test_options_default_and_builtin_registration() -> None:
    assert LocalSQLiteVectorStoreConfig.from_mapping({}).busy_timeout_seconds == 5.0
    assert register_builtin_vector_adapters().registered_adapters() == ("local_sqlite",)
    factory = VectorStoreFactory()
    assert register_builtin_vector_adapters(factory) is factory
