"""锁行的退出：只在空闲时删；删后重新加锁拿到新 token，旧 token 不会被误认。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import LockLostError, LockStore
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.infrastructure.store.sqlite import SQLiteLockStore


@pytest.fixture(params=["process", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> LockStore:
    if request.param == "process":
        return ProcessLocalLockStore()
    return SQLiteLockStore(tmp_path / "locks.sqlite3", owner="lifecycle-test")


def test_held_lock_is_not_discarded(store: LockStore) -> None:
    token = store.acquire("source:a", ttl_seconds=30)
    assert not store.discard_idle("source:a")
    store.assert_owned(token)


def test_released_lock_is_discarded_and_old_token_stays_invalid(store: LockStore) -> None:
    old = store.acquire("source:a", ttl_seconds=30)
    store.release(old)
    assert store.discard_idle("source:a")
    new = store.acquire("source:a", ttl_seconds=30)
    assert new.token != old.token
    with pytest.raises((LockLostError, TimeoutError)):
        store.assert_owned(old)
    store.assert_owned(new)


def test_expired_lock_is_discarded(store: LockStore) -> None:
    store.acquire("source:a", ttl_seconds=1)
    time.sleep(1.1)
    assert store.discard_idle("source:a")


def test_missing_lock_reports_nothing_removed(store: LockStore) -> None:
    assert not store.discard_idle("never-acquired")
