"""旧版本快照、租约、fencing 与路径锁组合测试。"""

from contextlib import contextmanager
from dataclasses import replace

import pytest

from infrastructure.editor.snapshot import (
    SnapshotBatch,
    SnapshotReadConfig,
    SnapshotReader,
    SnapshotReadLimitError,
    SnapshotState,
    VersionedSnapshot,
)
from infrastructure.store.contracts.lock import LockLostError, LockToken
from infrastructure.store.contracts.path_lock import PathLock
from infrastructure.store.locks import ProcessLocalLockStore


def test_snapshot_reader_records_found_missing_digest_and_stable_batch_order() -> None:
    values = {"b": {"revision": 2, "body": "B"}, "a": {"revision": 1, "body": "A"}}
    reader = SnapshotReader(
        load=lambda identity: values[identity] if identity in values else (_ for _ in ()).throw(FileNotFoundError()),
        revision_of=lambda value: value["revision"],
        serialize=lambda value: value["body"].encode(),
    )

    batch = reader.read_many(("b", "missing", "a", "a"))
    assert tuple(item.identity for item in batch.snapshots) == ("a", "b", "missing")
    assert batch.get("a").revision == 1
    assert batch.get("missing").state is SnapshotState.MISSING
    assert batch.total_bytes == 2


def test_snapshot_reader_only_converts_file_not_found_and_enforces_all_resource_bounds() -> None:
    broken = SnapshotReader(
        load=lambda identity: (_ for _ in ()).throw(PermissionError(identity)),
        revision_of=lambda value: 1,
        serialize=lambda value: b"x",
    )
    with pytest.raises(PermissionError):
        broken.read("secret")

    item_limited = SnapshotReader(
        load=lambda identity: {"revision": 1, "body": "xx"},
        revision_of=lambda value: value["revision"],
        serialize=lambda value: value["body"].encode(),
        config=SnapshotReadConfig(max_items=2, max_item_bytes=1, max_total_bytes=2),
    )
    with pytest.raises(SnapshotReadLimitError, match="item"):
        item_limited.read("a")

    count_limited = SnapshotReader(
        load=lambda identity: {"revision": 1},
        revision_of=lambda value: 1,
        serialize=lambda value: b"x",
        config=SnapshotReadConfig(max_items=2, max_item_bytes=1, max_total_bytes=2),
    )
    with pytest.raises(SnapshotReadLimitError, match="item limit"):
        count_limited.read_many(("a", "a", "a"))

    total_limited = SnapshotReader(
        load=lambda identity: {"revision": 1},
        revision_of=lambda value: 1,
        serialize=lambda value: b"x",
        config=SnapshotReadConfig(max_items=3, max_item_bytes=1, max_total_bytes=2),
    )
    with pytest.raises(SnapshotReadLimitError, match="total byte"):
        total_limited.read_many(("a", "b", "c"))


def test_snapshot_models_reject_incoherent_state_and_batch_metadata() -> None:
    missing = VersionedSnapshot.missing("memory://profile")
    with pytest.raises(ValueError, match="missing snapshot"):
        replace(missing, size_bytes=1)
    found = VersionedSnapshot(
        identity="memory://profile",
        state=SnapshotState.FOUND,
        value="body",
        revision=1,
        source_digest="0" * 64,
        size_bytes=4,
    )
    with pytest.raises(ValueError, match="sorted"):
        SnapshotBatch((VersionedSnapshot.missing("z"), found), 4)
    with pytest.raises(ValueError, match="total_bytes"):
        SnapshotBatch((found,), 3)


def test_process_local_lock_excludes_competitors_and_increments_fence_after_release() -> None:
    store = ProcessLocalLockStore()
    first = store.acquire("memory://profile")
    with pytest.raises(TimeoutError, match="already held"):
        store.acquire("memory://profile")
    store.assert_owned(first)
    store.renew(first)
    store.release(first)

    second = store.acquire("memory://profile")
    assert second.fence == first.fence + 1
    with pytest.raises(LockLostError):
        store.assert_owned(first)


def test_path_lock_fences_multiple_guards_and_always_releases_after_body_error() -> None:
    store = ProcessLocalLockStore()
    path_lock = PathLock(store)
    with pytest.raises(RuntimeError, match="body failed"):
        with path_lock.acquire("a") as first, path_lock.acquire("b") as second:
            first.checkpoint()
            with path_lock.fenced((first, second)):
                raise RuntimeError("body failed")

    assert store.locks == {}
    with path_lock.acquire("a") as guard:
        store.assert_owned(guard.token)


def test_path_lock_never_replays_body_when_fenced_exit_fails() -> None:
    class ExitFailureStore(ProcessLocalLockStore):
        @contextmanager
        def fenced(self, tokens, ttl_seconds=30):
            yield
            raise TimeoutError("exit failed")

    calls = 0
    with pytest.raises(TimeoutError, match="exit failed"):
        with PathLock(ExitFailureStore()).acquire("a", wait_timeout_seconds=0.01) as guard:
            with guard.fenced():
                calls += 1
    assert calls == 1


def test_lease_guard_rejects_foreign_or_stale_token() -> None:
    store = ProcessLocalLockStore()
    token = store.acquire("a")
    forged = LockToken("a", "other", token.fence)
    with pytest.raises(LockLostError):
        store.renew(forged)
    store.release(token)
