"""行为侧的生命周期参与者（26 号清单 6.14）：每轮放掉再也用不上的原料与旧记录，报几个量给健康面。

- 物理 lane 的小账：过了补发跨度的整日删（``PhysicalSegmentLedger``）；
- 容量：每一支在用的词数对上限（满了够条件的词等着，裁定 101）。

词表没有变更日志（只有一个在用状态文件），不用截断。改读数的（压实、窗口）在换代里按固定顺序跑，不在这里。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

from habitus.infrastructure.lifecycle import CapacityReading, LifecycleContext, ParticipantReport
from habitus.runtime.behavior import BehaviorBranches, BehaviorRuntimeComponents


class BehaviorParticipant:
    name = "behavior"

    def __init__(
        self,
        branches: BehaviorBranches,
        *,
        backfill_days: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if backfill_days < 1:
            raise ValueError("backfill_days must be a positive integer")
        self.branches = branches
        self.backfill_days = backfill_days
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        return await asyncio.to_thread(self._maintain_blocking, ctx)

    def _maintain_blocking(self, ctx: LifecycleContext) -> ParticipantReport:
        now = self._clock()
        released = 0
        readings: list[CapacityReading] = []
        for branch in (self.branches.session, self.branches.physical):
            ctx.assert_owned()
            released += self._physical(branch, now)
            store = branch.kind_store
            readings.append(
                CapacityReading(f"{branch.lane.value}_kinds_active", len(store.read().active), store.config.active_cap)
            )
        return ParticipantReport(released=released, readings=tuple(readings))

    def _physical(self, branch: BehaviorRuntimeComponents, now: datetime) -> int:
        ledger = branch.physical_ledger
        if ledger is None:
            return 0
        return ledger.expire(now, backfill_days=self.backfill_days)


__all__ = ["BehaviorParticipant"]
