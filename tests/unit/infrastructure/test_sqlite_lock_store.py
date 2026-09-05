"""SQLite 跨进程锁的耐久 fencing、过期接管和损坏拒绝测试。"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from habitus.infrastructure.store.contracts import LockLostError
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.infrastructure.store.sqlite.lock_store import SQLiteLockStoreConfig


def test_two_lazy_instances_initialize_one_new_database_concurrently(tmp_path) -> None:
    path = tmp_path / "concurrent-initialize.sqlite3"
    stores = (
        SQLiteLockStore(path, owner="worker-a", initialize=False),
        SQLiteLockStore(path, owner="worker-b", initialize=False),
    )
    barrier = Barrier(2)

    def initialize(store: SQLiteLockStore) -> None:
        barrier.wait()
        store.initialize()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(initialize, store) for store in stores)
        for future in futures:
            future.result()

    assert all(store.initialized for store in stores)
    token = stores[0].acquire("shared-key")
    stores[0].release(token)


def test_sqlite_lock_persists_exclusion_across_independent_store_instances(tmp_path: Path) -> None:
    path = tmp_path / "locks.sqlite3"
    first_store = SQLiteLockStore(path, owner="worker-a")
    second_store = SQLiteLockStore(path, owner="worker-b")

    first = first_store.acquire("memory://profile.md", ttl_seconds=30)
    with pytest.raises(TimeoutError, match="already held"):
        second_store.acquire("memory://profile.md", ttl_seconds=30)

    first_store.release(first)
    second = second_store.acquire("memory://profile.md", ttl_seconds=30)
    assert second.fence == first.fence + 1
    with pytest.raises(LockLostError):
        first_store.assert_owned(first)


def test_expired_sqlite_lease_can_be_reclaimed_but_old_token_cannot_renew_or_release_new_owner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "locks.sqlite3"
    store = SQLiteLockStore(path)
    old = store.acquire("workflow:sequence", ttl_seconds=30)
    expired = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE locks SET expires_at = ? WHERE lock_key = ?",
            (expired, old.lock_key),
        )

    current = SQLiteLockStore(path, owner="replacement").acquire(old.lock_key)
    assert current.fence == old.fence + 1
    with pytest.raises(LockLostError):
        store.renew(old)
    store.release(old)
    SQLiteLockStore(path).assert_owned(current)


def test_fenced_transaction_validates_all_tokens_and_rolls_back_lease_extension_on_body_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "locks.sqlite3"
    store = SQLiteLockStore(path)
    first = store.acquire("a")
    second = store.acquire("b")
    forged = type(first)(lock_key=second.lock_key, token="foreign", fence=second.fence)

    with pytest.raises(LockLostError):
        with store.fenced((first, forged)):
            raise AssertionError("无效 token 不得进入临界区")

    with pytest.raises(RuntimeError, match="body failed"):
        with store.fenced((first, second)):
            raise RuntimeError("body failed")
    store.assert_owned(first)
    store.assert_owned(second)


def test_initialize_rejects_legacy_or_tampered_database_layout(tmp_path: Path) -> None:
    path = tmp_path / "locks.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE locks(lock_key TEXT PRIMARY KEY, token TEXT)")

    with pytest.raises(RuntimeError, match="unsupported LockStore layout"):
        SQLiteLockStore(path)


@pytest.mark.parametrize("value", [True, 0, 0.0001, 61, float("inf")])
def test_sqlite_timeout_configuration_rejects_unsafe_values(value: object) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        SQLiteLockStoreConfig(timeout_seconds=value)  # type: ignore[arg-type]
