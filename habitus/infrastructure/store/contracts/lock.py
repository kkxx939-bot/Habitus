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

    def discard_idle(self, lock_key: str) -> bool:
        """删掉这把锁的那一行，只在它空闲（已释放或租约已过期）时删；返回删没删。

        删空闲行是安全的：fence 只和 uuid token 一起比，重新加锁时 fence 从头数也不会误放行。
        锁行原来只增不删（每份会话源信封两行、每个行为树文档一行），这是它们唯一的退出。"""
        ...

    def observability_snapshot(self, *, warning_seconds: float) -> LockStoreSnapshot: ...


__all__ = ["LockLostError", "LockStore", "LockStoreSnapshot", "LockToken"]
