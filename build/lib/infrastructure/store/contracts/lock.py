"""持久化锁及其隔离令牌协议。"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LockToken:
    lock_key: str
    token: str
    fence: int = 0


@dataclass(frozen=True)
class LockStoreSnapshot:
    """不暴露锁键和令牌的聚合状态。"""

    active_count: int
    hanging_count: int
    max_active_age_seconds: float


class LockLostError(TimeoutError):
    """写入方已经失去此前获得的锁租约。"""


class LockStore(Protocol):
    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken: ...

    def renew(self, token: LockToken, ttl_seconds: int = 30) -> LockToken: ...

    def assert_owned(self, token: LockToken) -> None: ...

    def fenced(
        self,
        tokens: Sequence[LockToken],
        ttl_seconds: int = 30,
    ) -> AbstractContextManager[None]: ...

    def release(self, token: LockToken) -> None: ...

    def observability_snapshot(self, *, warning_seconds: float) -> LockStoreSnapshot: ...


__all__ = ["LockLostError", "LockStore", "LockStoreSnapshot", "LockToken"]
