"""Conversation Source Consumer 的跨进程执行栅栏与租约 Heartbeat。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TypeVar

from conversation.source.model import ConversationSourceEnvelope, ConversationSourceError
from conversation.source.receipt import ConversationSourceConsumer
from foundation.integrity import canonical_digest
from infrastructure.store.contracts import LeaseGuard, PathLock

T = TypeVar("T")


class ConversationConsumerLeaseLostError(ConversationSourceError):
    """执行租约 Heartbeat 已失败，当前执行不得再发布 Output 或 Outcome。"""


@dataclass
class ConversationConsumerExecutionLease:
    path_lock: PathLock
    guards: tuple[LeaseGuard, ...]
    heartbeat_error: BaseException | None = None

    def require_alive(self) -> None:
        if self.heartbeat_error is not None:
            raise ConversationConsumerLeaseLostError("conversation consumer execution lease was lost") from self.heartbeat_error

    async def run_fenced(self, callback: Callable[[], T]) -> T:
        """只在短小同步临界区内 fencing，不跨越异步 Consumer 主体。"""

        self.require_alive()

        def invoke() -> T:
            with self.path_lock.fenced(self.guards):
                return callback()

        value = await asyncio.to_thread(invoke)
        self.require_alive()
        return value


class ConversationConsumerExecutionFence:
    """按固定锁顺序持有 Consumer 租约，并在执行期间自动续租。"""

    def __init__(
        self,
        path_lock: PathLock,
        *,
        ttl_seconds: int,
        heartbeat_interval_seconds: float,
        wait_seconds: float,
    ) -> None:
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be PathLock")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        for name, value in (
            ("heartbeat_interval_seconds", heartbeat_interval_seconds),
            ("wait_seconds", wait_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or float(value) < 0:
                raise ValueError(f"{name} must be a non-negative number")
        if heartbeat_interval_seconds <= 0 or heartbeat_interval_seconds > ttl_seconds / 3:
            raise ValueError("heartbeat interval must be positive and at most one third of TTL")
        self.path_lock = path_lock
        self.ttl_seconds = ttl_seconds
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.wait_seconds = float(wait_seconds)

    @asynccontextmanager
    async def acquire(
        self,
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
    ):
        resolved_consumer = ConversationSourceConsumer(consumer)
        keys = self._keys(source, resolved_consumer)
        contexts: list[AbstractContextManager[LeaseGuard]] = []
        guards: list[LeaseGuard] = []
        lease: ConversationConsumerExecutionLease | None = None
        heartbeat: asyncio.Task[None] | None = None
        body_error: BaseException | None = None
        try:
            for key in keys:
                acquired_context = self.path_lock.acquire(
                    key,
                    ttl_seconds=self.ttl_seconds,
                    wait_timeout_seconds=self.wait_seconds,
                )
                guard = await asyncio.to_thread(acquired_context.__enter__)
                contexts.append(acquired_context)
                guards.append(guard)
            lease = ConversationConsumerExecutionLease(self.path_lock, tuple(guards))
            heartbeat = asyncio.create_task(self._heartbeat(lease))
            yield lease
            lease.require_alive()
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
            first_error: BaseException | None = None
            for release_context in reversed(contexts):
                try:
                    await asyncio.to_thread(release_context.__exit__, None, None, None)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            if first_error is not None and body_error is None:
                raise first_error

    async def _heartbeat(self, lease: ConversationConsumerExecutionLease) -> None:
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval_seconds)
                await asyncio.gather(
                    *(asyncio.to_thread(guard.checkpoint) for guard in lease.guards)
                )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            lease.heartbeat_error = exc

    @staticmethod
    def _keys(
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
    ) -> tuple[str, ...]:
        consumer_key = f"conversation-source-consumer:{source.source_id}:{consumer.value}"
        if consumer is not ConversationSourceConsumer.MEMORY:
            return (consumer_key,)
        address_digest = canonical_digest(
            {
                "schema_version": "conversation_memory_order_lock_v1",
                "conversation_id": source.conversation_id,
                "started_on": source.started_on.isoformat(),
            }
        )
        return (f"conversation-memory:{address_digest}", consumer_key)


__all__ = [
    "ConversationConsumerExecutionFence",
    "ConversationConsumerExecutionLease",
    "ConversationConsumerLeaseLostError",
]
