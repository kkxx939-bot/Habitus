"""会话源的两个生命周期参与者：

- 释放：每轮按地址释放已经可以放的信封、Outcome、输出与锁行；
- 恢复（26 号清单 L9.2）：每轮挑出"现在该试"的没终态的 Source/Consumer，交给协调器去跑，不占住 Worker 这一轮；
  Memory 恢复出任务时唤醒 Memory Worker。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from habitus.conversation.source.coordinator import ConversationSourceCoordinator
from habitus.conversation.source.delivery import ConversationConsumerDelivery
from habitus.conversation.source.receipt import ConversationSourceConsumer
from habitus.conversation.source.recovery import (
    ConversationSourceRecovery,
    ConversationSourceRecoveryEntry,
    ConversationSourceRecoveryResult,
)
from habitus.conversation.source.release import ConversationSourceReleaser
from habitus.infrastructure.lifecycle import CapacityReading, LifecycleContext, ParticipantReport
from habitus.memory.workflow import ConversationMemoryIngestResult


class ConversationSourceParticipant:
    """每轮从地址游标往后看 ``batch_addresses`` 个地址，到头了从头再来：一轮的工作量不随历史地址数变大。游标只在内存里
    （释放是幂等的，重启从头轮一遍不出错）。"""

    name = "conversation-sources"

    def __init__(self, releaser: ConversationSourceReleaser, *, batch_addresses: int) -> None:
        if not isinstance(releaser, ConversationSourceReleaser):
            raise TypeError("releaser must be ConversationSourceReleaser")
        if isinstance(batch_addresses, bool) or not isinstance(batch_addresses, int) or batch_addresses <= 0:
            raise ValueError("batch_addresses must be a positive integer")
        self.releaser = releaser
        self.batch_addresses = batch_addresses
        self._cursor: str | None = None

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        return await asyncio.to_thread(self._maintain_blocking, ctx)

    def _maintain_blocking(self, ctx: LifecycleContext) -> ParticipantReport:
        ledger = self.releaser.ledger
        directories = sorted(ledger.index.address_directories())
        batch = self._next_batch(directories)
        released = 0
        for directory in batch:
            ctx.assert_owned()
            released += self.releaser.release_address(directory, ctx)
        ctx.assert_owned()
        self.releaser.settle_pending()
        readings = (
            CapacityReading("index_addresses", len(directories), ledger.index.max_addresses),
            CapacityReading("pending_sources", ledger.pending.count(), ledger.pending.max_entries),
        )
        return ParticipantReport(released=released, readings=readings)

    def _next_batch(self, directories: list[str]) -> list[str]:
        after = [name for name in directories if self._cursor is None or name > self._cursor]
        batch = (after or directories)[: self.batch_addresses]
        self._cursor = batch[-1] if batch and len(batch) == self.batch_addresses else None
        return batch


def memory_jobs_restored(
    delivery: ConversationConsumerDelivery, results: tuple[ConversationSourceRecoveryResult, ...]
) -> bool:
    """这一批里有没有 Memory 恢复出了要跑的任务（有就该唤醒 Memory Worker）。"""

    for result in results:
        if result.consumer is not ConversationSourceConsumer.MEMORY or result.error is not None:
            continue
        restored = delivery.restore_terminal(result.envelope, ConversationSourceConsumer.MEMORY)
        if isinstance(restored, ConversationMemoryIngestResult) and restored.jobs:
            return True
    return False


class ConversationSourceRecoveryParticipant:
    name = "conversation-source-recovery"

    def __init__(
        self,
        recovery: ConversationSourceRecovery,
        coordinator: ConversationSourceCoordinator,
        *,
        wake_memory: Callable[[], None],
    ) -> None:
        if not isinstance(recovery, ConversationSourceRecovery):
            raise TypeError("recovery must be ConversationSourceRecovery")
        if not isinstance(coordinator, ConversationSourceCoordinator):
            raise TypeError("coordinator must be ConversationSourceCoordinator")
        if coordinator.delivery is not recovery.delivery:
            raise ValueError("source recovery and coordinator must share the delivery")
        self.recovery = recovery
        self.coordinator = coordinator
        self.wake_memory = wake_memory

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        ctx.assert_owned()
        busy = frozenset((item.source_id, item.consumer) for item in self.coordinator.pending_deliveries)
        plan = await asyncio.to_thread(self.recovery.plan, busy=busy)
        ctx.assert_owned()
        if plan.attempt:
            self.coordinator.dispatch(
                ((entry.envelope.source_id, entry.consumer) for entry in plan.attempt), self._recover(plan.attempt)
            )
        given_up = len(self.recovery.retries.given_up())
        readings = (CapacityReading("source_recovery_given_up", given_up, self.recovery.retries.max_entries),)
        signals: list[str] = []
        if plan.broken:
            signals.append(f"{len(plan.broken)} conversation source deliveries are broken and wait for repair")
        if plan.salvaged:
            signals.append(f"{len(plan.salvaged)} broken behavior deliveries were salvaged")
        if plan.corrupt:
            signals.append(f"{len(plan.corrupt)} conversation source envelopes are unreadable and wait for repair")
        return ParticipantReport(readings=readings, signals=tuple(signals))

    async def _recover(self, entries: tuple[ConversationSourceRecoveryEntry, ...]) -> None:
        results = await self.recovery.recover_all(entries)
        if await asyncio.to_thread(memory_jobs_restored, self.recovery.delivery, results):
            self.wake_memory()


__all__ = ["ConversationSourceParticipant", "ConversationSourceRecoveryParticipant", "memory_jobs_restored"]
