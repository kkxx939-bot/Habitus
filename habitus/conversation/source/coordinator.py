"""持久化 Source，并为每个已注册 Consumer 独立启动交付。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from habitus.conversation.source.delivery import ConversationConsumerDelivery, ConversationConsumerEnsureResult
from habitus.conversation.source.model import ConversationSourceEnvelope
from habitus.conversation.source.receipt import ConversationSourceConsumer
from habitus.conversation.source.state import ConversationConsumerState
from habitus.conversation.source.store import ConversationSourceStore


@dataclass(frozen=True)
class ConversationSourceDispatchHandle:
    """分离各 Consumer 的等待与状态检查；等待取消不会传播到底层任务。

    ``tasks`` 覆盖全部已注册 Consumer，``wait``/``inspect`` 是通用入口；
    Memory 与 Behavior Projection 的具名方法只是调用方常用的便捷门面。
    """

    envelope: ConversationSourceEnvelope
    delivery: ConversationConsumerDelivery
    tasks: Mapping[ConversationSourceConsumer, asyncio.Task[ConversationConsumerEnsureResult]]

    async def wait(self, consumer: ConversationSourceConsumer) -> object | None:
        """等待某个 Consumer 到达终态，并返回它的耐久 Output（SKIPPED 时为 None）。"""

        resolved = ConversationSourceConsumer(consumer)
        await asyncio.shield(self.tasks[resolved])
        return self.delivery.restore_terminal(self.envelope, resolved)

    def inspect(self, consumer: ConversationSourceConsumer) -> ConversationConsumerState:
        return self.delivery.inspect(self.envelope, ConversationSourceConsumer(consumer))

    @property
    def memory_task(self) -> asyncio.Task[ConversationConsumerEnsureResult]:
        return self.tasks[ConversationSourceConsumer.MEMORY]

    @property
    def behavior_projection_task(self) -> asyncio.Task[ConversationConsumerEnsureResult]:
        return self.tasks[ConversationSourceConsumer.BEHAVIOR_PROJECTION]

    async def wait_memory(self) -> object:
        value = await self.wait(ConversationSourceConsumer.MEMORY)
        if value is None:
            raise RuntimeError("Memory Consumer cannot complete as SKIPPED")
        return value

    async def wait_behavior_projection(self) -> object | None:
        return await self.wait(ConversationSourceConsumer.BEHAVIOR_PROJECTION)

    def inspect_memory(self) -> ConversationConsumerState:
        return self.inspect(ConversationSourceConsumer.MEMORY)

    def inspect_behavior_projection(self) -> ConversationConsumerState:
        return self.inspect(ConversationSourceConsumer.BEHAVIOR_PROJECTION)


@dataclass(frozen=True)
class ConversationSourcePendingDelivery:
    """Runtime 关闭时仍在执行的耐久 Source/Consumer 身份。"""

    source_id: str
    consumer: ConversationSourceConsumer

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumer", ConversationSourceConsumer(self.consumer))


class ConversationSourceCoordinator:
    """start() 只启动交付；调用方可以只等待自己关心的 Consumer。"""

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
        tasks: dict[ConversationSourceConsumer, asyncio.Task[ConversationConsumerEnsureResult]] = {}
        # 按枚举顺序遍历注册表，保证启动顺序确定，且新增 Consumer 不必改这里。
        for consumer in ConversationSourceConsumer:
            task = loop.create_task(self.delivery.ensure_outcome(durable, consumer))
            self._track(task, durable.source_id, consumer)
            tasks[consumer] = task
        return ConversationSourceDispatchHandle(durable, self.delivery, MappingProxyType(tasks))

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
                # 取出异常避免"未检索异常"警告；失败本身由调用方 wait/inspect 观察。
                # TODO(conversation-source): 后台 Consumer 的失败当前不落任何记录，
                # 无人 wait 时会静默停摆；可观测方案待与 Behavior 下游一并确定。
                done.exception()

        task.add_done_callback(completed)


__all__ = [
    "ConversationSourceCoordinator",
    "ConversationSourceDispatchHandle",
    "ConversationSourcePendingDelivery",
]
