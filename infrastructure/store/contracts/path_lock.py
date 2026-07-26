"""基于锁存储协议协调同一路径上的写操作。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from math import isfinite
from time import monotonic, sleep

from infrastructure.store.contracts.lock import LockStore, LockToken


@dataclass(frozen=True)
class LeaseGuard:
    lock_store: LockStore
    token: LockToken
    ttl_seconds: int
    wait_timeout_seconds: float = 0.0
    retry_delay_seconds: float = 0.01

    def checkpoint(self) -> None:
        deadline = monotonic() + self.wait_timeout_seconds
        while True:
            try:
                self.lock_store.renew(self.token, ttl_seconds=self.ttl_seconds)
                return
            except TimeoutError:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise
                sleep(min(self.retry_delay_seconds, remaining))

    @contextmanager
    def fenced(self) -> Iterator[None]:
        with _enter_fenced(
            self.lock_store,
            (self.token,),
            ttl_seconds=self.ttl_seconds,
            wait_timeout_seconds=self.wait_timeout_seconds,
            retry_delay_seconds=self.retry_delay_seconds,
        ):
            yield


class PathLock:
    def __init__(self, lock_store: LockStore) -> None:
        self.lock_store = lock_store

    @contextmanager
    def acquire(
        self,
        key: str,
        ttl_seconds: int = 30,
        *,
        wait_timeout_seconds: float = 0.0,
        retry_delay_seconds: float = 0.01,
    ) -> Iterator[LeaseGuard]:
        """取得路径租约；默认快速失败，也可显式配置有界竞争等待。"""

        if isinstance(wait_timeout_seconds, bool) or not isinstance(wait_timeout_seconds, int | float):
            raise ValueError("wait_timeout_seconds must be numeric")
        if isinstance(retry_delay_seconds, bool) or not isinstance(retry_delay_seconds, int | float):
            raise ValueError("retry_delay_seconds must be numeric")
        wait_timeout = float(wait_timeout_seconds)
        retry_delay = float(retry_delay_seconds)
        if not isfinite(wait_timeout) or wait_timeout < 0:
            raise ValueError("wait_timeout_seconds must be non-negative")
        if not isfinite(retry_delay) or retry_delay <= 0:
            raise ValueError("retry_delay_seconds must be positive")
        deadline = monotonic() + wait_timeout
        while True:
            try:
                token = self.lock_store.acquire(key, ttl_seconds=ttl_seconds)
                break
            except TimeoutError:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise
                sleep(min(retry_delay, remaining))
        body_error: BaseException | None = None
        try:
            yield LeaseGuard(
                self.lock_store,
                token,
                max(1, ttl_seconds),
                wait_timeout,
                retry_delay,
            )
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                _release(
                    self.lock_store,
                    token,
                    wait_timeout_seconds=wait_timeout,
                    retry_delay_seconds=retry_delay,
                )
            except Exception as exc:
                if body_error is None:
                    raise
                raise body_error from exc

    @contextmanager
    def fenced(self, guards: Sequence[LeaseGuard]) -> Iterator[None]:
        if not guards:
            yield
            return
        if any(guard.lock_store is not self.lock_store for guard in guards):
            raise ValueError("all lease guards must belong to one LockStore")
        ttl_seconds = min(guard.ttl_seconds for guard in guards)
        wait_timeout = min(guard.wait_timeout_seconds for guard in guards)
        retry_delay = min(guard.retry_delay_seconds for guard in guards)
        with _enter_fenced(
            self.lock_store,
            tuple(guard.token for guard in guards),
            ttl_seconds=ttl_seconds,
            wait_timeout_seconds=wait_timeout,
            retry_delay_seconds=retry_delay,
        ):
            yield


@contextmanager
def _enter_fenced(
    lock_store: LockStore,
    tokens: Sequence[LockToken],
    *,
    ttl_seconds: int,
    wait_timeout_seconds: float,
    retry_delay_seconds: float,
) -> Iterator[None]:
    """只重试 fencing 入口；临界区正文和退出异常绝不机械重放。"""

    deadline = monotonic() + wait_timeout_seconds
    while True:
        stack = ExitStack()
        try:
            stack.enter_context(lock_store.fenced(tokens, ttl_seconds=ttl_seconds))
        except TimeoutError:
            stack.close()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise
            sleep(min(retry_delay_seconds, remaining))
            continue
        break
    with stack:
        yield


def _release(
    lock_store: LockStore,
    token: LockToken,
    *,
    wait_timeout_seconds: float,
    retry_delay_seconds: float,
) -> None:
    """只重试幂等租约释放，不改变已经执行过的临界区正文。"""

    deadline = monotonic() + wait_timeout_seconds
    while True:
        try:
            lock_store.release(token)
            return
        except TimeoutError:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise
            sleep(min(retry_delay_seconds, remaining))
