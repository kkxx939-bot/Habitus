"""生命周期骨架自己的状态也要有退出（26 号清单 L10.10；裁定 67）：重试账是改造新加的，原先只进不出。

- 重试账：放弃了超过 ``lifecycle.retry.given_up_days`` 的条目回收（键里带口径的，口径一变旧键再也不会被 ``clear``）；
- 每晚重试的堆：自己每晚试完就删（``behavior.retry.stage``），这里只报还剩几项、到上限没有、累计拒收了几项；
- 都报容量，到告警线时健康面看得见。

重试账与重试的堆由各层组合时建好、交到这里；这里只按统一的寿命清，不认识它们装的是什么。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import timedelta

from habitus.behavior.retry import RetryBacklogStore
from habitus.infrastructure.lifecycle import CapacityReading, LifecycleContext, ParticipantReport, RetryLedger


class LifecycleHousekeepingParticipant:
    name = "lifecycle-housekeeping"

    def __init__(
        self,
        retries: Iterable[RetryLedger],
        *,
        keep_days: int,
        backlogs: Iterable[RetryBacklogStore] = (),
    ) -> None:
        self.retries = tuple(retries)
        self.backlogs = tuple(backlogs)
        if any(not isinstance(item, RetryLedger) for item in self.retries):
            raise TypeError("retries must be RetryLedger instances")
        if any(not isinstance(item, RetryBacklogStore) for item in self.backlogs):
            raise TypeError("backlogs must be RetryBacklogStore instances")
        for label, owners in (
            ("retry ledger", [item.owner for item in self.retries]),
            ("retry backlog", [item.root.name for item in self.backlogs]),
        ):
            if len(owners) != len(set(owners)):
                raise ValueError(f"{label} owners must be distinct")
        if isinstance(keep_days, bool) or not isinstance(keep_days, int) or keep_days <= 0:
            raise ValueError("keep_days must be a positive integer")
        self.keep_days = keep_days

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        return await asyncio.to_thread(self._maintain_blocking, ctx)

    def _maintain_blocking(self, ctx: LifecycleContext) -> ParticipantReport:
        before = ctx.now - timedelta(days=self.keep_days)
        released = 0
        readings: list[CapacityReading] = []
        for ledger in self.retries:
            ctx.assert_owned()
            released += ledger.expire_given_up(before)
            readings.append(CapacityReading(f"retry_{ledger.owner}", ledger.size(), ledger.max_entries))
        for backlog in self.backlogs:
            ctx.assert_owned()
            name = backlog.root.name
            readings.append(CapacityReading(f"retry_backlog_{name}", backlog.count(), backlog.max_items))
            # 拒收累计（满了 / 那天的夜批已经跑过）：这些轮只落了「没读懂」、不会再补（裁定 68 第 3、4 条）
            readings.extend(
                CapacityReading(f"retry_backlog_{name}_refused_{reason}", count)
                for reason, count in sorted(backlog.refused().items())
            )
        return ParticipantReport(released=released, readings=tuple(readings))


__all__ = ["LifecycleHousekeepingParticipant"]
