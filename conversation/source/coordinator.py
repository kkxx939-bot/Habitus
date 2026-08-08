"""持久化 Source，并独立启动 Memory 与 Behavior Projection 交付。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from conversation.source.delivery import ConversationConsumerDelivery, ConversationConsumerEnsureResult
from conversation.source.model import ConversationSourceEnvelope
from conversation.source.receipt import ConversationSourceConsumer
from conversation.source.state import ConversationConsumerState
from conversation.source.store import ConversationSourceStore


@dataclass(frozen=True)
class ConversationSourceDispatchHandle:
    """分离两个 Consumer 的等待与状态检查；等待取消不会传播到底层任务。"""

    envelope: ConversationSourceEnvelope
    delivery: ConversationConsumerDelivery
    memory_task: asyncio.Task[ConversationConsumerEnsureResult]
    behavior_projection_task: asyncio.Task[ConversationConsumerEnsureResult]

    async def wait_memory(self) -> object:
        await asyncio.shield(self.memory_task)
        value = self.delivery.restore_terminal(self.envelope, ConversationSourceConsumer.MEMORY)
        if value is None:
            raise RuntimeError("Memory Consumer cannot complete as SKIPPED")
        return value

    async def wait_behavior_projection(self) -> object | None:
        await asyncio.shield(self.behavior_projection_task)
        return self.delivery.restore_terminal(
            self.envelope,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        )

    def inspect_memory(self) -> ConversationConsumerState:
        return self.delivery.inspect(self.envelope, ConversationSourceConsumer.MEMORY)

    def inspect_behavior_projection(self) -> ConversationConsumerState:
        return self.delivery.inspect(
            self.envelope,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        )


@dataclass(frozen=True)
class ConversationSourcePendingDelivery:
    """Runtime 关闭时仍在执行的耐久 Source/Consumer 身份。"""

    source_id: str
    consumer: ConversationSourceConsumer


class ConversationSourceCoordinator:
    """start() 只启动交付；旧 API 可以只等待 Memory。"""

    def __init__(
        self,
        sources: ConversationSourceStore,
        delivery: ConversationConsumerDelivery,
    ) -> None:
        if delivery.sources is not sources:
            raise ValueError("coordinator and delivery must share the Source Store")
        self.sources = sources
        self.delivery = delivery
        self._tasks: set[asyncio.Task[ConversationConsumerEnsureResult]] = set()
        self._task_deliveries: dict[
            asyncio.Task[ConversationConsumerEnsureResult],
            ConversationSourcePendingDelivery,
        ] = {}

    def start(self, envelope: ConversationSourceEnvelope) -> ConversationSourceDispatchHandle:
        durable = self.sources.put(envelope)
        loop = asyncio.get_running_loop()
        memory_task = loop.create_task(
            self.delivery.ensure_outcome(durable, ConversationSourceConsumer.MEMORY)
        )
        behavior_task = loop.create_task(
            self.delivery.ensure_outcome(
                durable,
                ConversationSourceConsumer.BEHAVIOR_PROJECTION,
            )
        )
        self._track(memory_task, durable.source_id, ConversationSourceConsumer.MEMORY)
        self._track(
            behavior_task,
            durable.source_id,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        )
        return ConversationSourceDispatchHandle(
            durable,
            self.delivery,
            memory_task,
            behavior_task,
        )

    async def wait_for_idle(self, timeout_seconds: float | None) -> bool:
        """有界等待但不取消 Consumer；超时后 Recovery 仍可复用耐久状态。"""

        pending = tuple(task for task in self._tasks if not task.done())
        if not pending:
            return True
        _done, unfinished = await asyncio.wait(pending, timeout=timeout_seconds)
        return not unfinished

    @property
    def running_task_count(self) -> int:
        return sum(not task.done() for task in self._tasks)

    @property
    def pending_deliveries(self) -> tuple[ConversationSourcePendingDelivery, ...]:
        """返回仍在运行的稳定 Source/Consumer 身份，不暴露 asyncio Task。"""

        return tuple(
            sorted(
                (
                    delivery
                    for task, delivery in self._task_deliveries.items()
                    if not task.done()
                ),
                key=lambda item: (item.source_id, item.consumer.value),
            )
        )

    def _track(
        self,
        task: asyncio.Task[ConversationConsumerEnsureResult],
        source_id: str,
        consumer: ConversationSourceConsumer,
    ) -> None:
        self._tasks.add(task)
        self._task_deliveries[task] = ConversationSourcePendingDelivery(source_id, consumer)

        def completed(done: asyncio.Task[ConversationConsumerEnsureResult]) -> None:
            self._tasks.discard(done)
            self._task_deliveries.pop(done, None)
            if not done.cancelled():
                # 显式取出异常，避免后台 Behavior 失败产生未检索异常警告。
                done.exception()

        task.add_done_callback(completed)


__all__ = [
    "ConversationSourceCoordinator",
    "ConversationSourceDispatchHandle",
    "ConversationSourcePendingDelivery",
]
