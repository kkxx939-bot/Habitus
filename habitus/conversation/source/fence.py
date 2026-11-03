"""Conversation Source Consumer 的跨进程执行栅栏与租约 Heartbeat。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, TypeVar

from habitus.conversation.source.model import ConversationSourceEnvelope, ConversationSourceError
from habitus.conversation.source.receipt import ConversationSourceConsumer
from habitus.foundation.integrity import canonical_digest
from habitus.infrastructure.store.contracts import LeaseGuard, PathLock

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
            raise ConversationConsumerLeaseLostError(
                "conversation consumer execution lease was lost"
            ) from self.heartbeat_error

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
        *,
        conversation_ordered: bool = False,
    ):
        """按固定顺序取得本次执行需要的锁。

        ``conversation_ordered`` 由调用方从 Consumer 自身声明传入；栅栏不认识
        任何具体 Consumer，只负责按稳定顺序取锁。
        """

        if not isinstance(conversation_ordered, bool):
            raise TypeError("conversation_ordered must be boolean")
        resolved_consumer = ConversationSourceConsumer(consumer)
        keys = self._keys(source, resolved_consumer, conversation_ordered=conversation_ordered)
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
                guard = await _enter_or_release_on_cancel(acquired_context)
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
                await asyncio.gather(*(asyncio.to_thread(guard.checkpoint) for guard in lease.guards))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            lease.heartbeat_error = exc

    @staticmethod
    def _keys(
        source: ConversationSourceEnvelope,
        consumer: ConversationSourceConsumer,
        *,
        conversation_ordered: bool,
    ) -> tuple[str, ...]:
        consumer_key = consumer_lock_key(source.source_id, consumer)
        if not conversation_ordered:
            return (consumer_key,)
        # 同一地址、同一个有序 Consumer 一把顺序锁
        address_digest = canonical_digest(
            {
                "schema_version": "conversation_order_lock_v2",
                "consumer": consumer.value,
                "conversation_id": source.conversation_id,
                "started_on": source.started_on.isoformat(),
            }
        )
        return (f"conversation-ordered:{address_digest}", consumer_key)


async def _enter_or_release_on_cancel(context: AbstractContextManager[LeaseGuard]) -> LeaseGuard:
    """在线程里等锁。线程里的等待停不下来：被取消时等它出结果，拿到了就当场放回去再抛取消，锁不会漏到 TTL。
    最多多等一个锁等待上限（``wait_seconds``）。"""

    entering = asyncio.ensure_future(asyncio.to_thread(context.__enter__))
    try:
        return await asyncio.shield(entering)
    except asyncio.CancelledError:
        await _finish_despite_cancel(entering)
        if not entering.cancelled() and entering.exception() is None:
            await _finish_despite_cancel(asyncio.ensure_future(asyncio.to_thread(context.__exit__, None, None, None)))
        raise


async def _finish_despite_cancel(work: asyncio.Future[Any]) -> None:
    """等一件已经在线程里跑的事做完；中途再被取消也接着等（取消在调用方收尾后照常抛出）。"""

    while not work.done():
        with suppress(asyncio.CancelledError):
            await asyncio.wait((work,))
    if not work.cancelled():
        work.exception()  # 取走结果，不留"异常没人取"的告警


def consumer_lock_key(source_id: str, consumer: ConversationSourceConsumer) -> str:
    """一份源、一个消费者的执行锁键；交付与释放用同一把，释放时持有它就不会和交付并发。"""

    return f"conversation-source-consumer:{source_id}:{ConversationSourceConsumer(consumer).value}"


__all__ = [
    "ConversationConsumerExecutionFence",
    "ConversationConsumerExecutionLease",
    "ConversationConsumerLeaseLostError",
    "consumer_lock_key",
]
