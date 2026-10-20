"""生命周期骨架自己的状态也要有退出（26 号清单 L10.10）：隔离区与重试账是改造新加的，原先只进不出。

- 隔离区：放弃的项过了 ``lifecycle.quarantine_days`` 清掉（没放弃的是重放的唯一来源，不动）；
- 重试账：放弃了超过同一个天数的条目回收（键里带口径的，口径一变旧键再也不会被 ``clear``）；
- 两样都报容量，到告警线时健康面看得见。

隔离区与重试账由各层组合时建好、交到这里；这里只按统一的寿命清，不认识它们装的是什么。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import timedelta

from habitus.infrastructure.lifecycle import (
    CapacityReading,
    LifecycleContext,
    ParticipantReport,
    Quarantine,
    RetryLedger,
)


class LifecycleHousekeepingParticipant:
    name = "lifecycle-housekeeping"

    def __init__(self, quarantines: Iterable[Quarantine], retries: Iterable[RetryLedger], *, keep_days: int) -> None:
        self.quarantines = tuple(quarantines)
        self.retries = tuple(retries)
        if any(not isinstance(item, Quarantine) for item in self.quarantines):
            raise TypeError("quarantines must be Quarantine instances")
        if any(not isinstance(item, RetryLedger) for item in self.retries):
            raise TypeError("retries must be RetryLedger instances")
        for label, owners in (
            ("quarantine", [item.owner for item in self.quarantines]),
            ("retry ledger", [item.owner for item in self.retries]),
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
        for quarantine in self.quarantines:
            ctx.assert_owned()
            released += quarantine.purge(before)
            readings.append(CapacityReading(f"quarantine_{quarantine.owner}", quarantine.count(), quarantine.max_items))
        for ledger in self.retries:
            ctx.assert_owned()
            released += ledger.expire_given_up(before)
            readings.append(CapacityReading(f"retry_{ledger.owner}", ledger.size(), ledger.max_entries))
        return ParticipantReport(released=released, readings=tuple(readings))


__all__ = ["LifecycleHousekeepingParticipant"]
