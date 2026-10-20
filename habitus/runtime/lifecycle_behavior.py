"""行为侧的生命周期参与者（26 号清单 6.14）：每轮放掉再也用不上的原料与旧记录，报几个量给健康面。

- 物理 lane 的小账：过了补发跨度的整日删（``PhysicalSegmentLedger``）；
- 词表变更日志：截断到"还要读 ``changes_since(下限)`` 的最老那一版"（``vocabulary_floor``）；下限里语义树那一份由组合根
  按语义树的关系状态与概念同步进度给（behavior 不认识语义树），没有语义树的那一支只看拆改冷却期；
- 容量：每一支的待定池（满了会拒收，裁定 60 第 2 条）。

改读数的（压实、窗口）在夜批里按固定顺序跑，不在这里。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta

from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.infrastructure.lifecycle import CapacityReading, LifecycleContext, ParticipantReport
from habitus.runtime.behavior import BehaviorBranches, BehaviorRuntimeComponents


def vocabulary_floor(store: BehaviorKindStore, *, now: datetime, derived: Iterable[int]) -> int:
    """变更日志能截到哪一版：派生物（语义树各 lane 关系状态记的版本、概念集的同步进度）里最老的，与拆改冷却期内最老那一版的
    前一版，取最老；都没有就是当前版本。"""

    current = store.read().version
    since = now - timedelta(days=store.config.cooldown_days)
    cooling = [record.version for record in store.records() if record.at >= since]
    floors = [*derived, current]
    if cooling:
        floors.append(min(cooling) - 1)
    return max(0, min(floors))


class BehaviorParticipant:
    name = "behavior"

    def __init__(
        self,
        branches: BehaviorBranches,
        *,
        backfill_days: int,
        derived_versions: Callable[[], Iterable[int]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(branches, BehaviorBranches):
            raise TypeError("branches must be BehaviorBranches")
        if isinstance(backfill_days, bool) or not isinstance(backfill_days, int) or backfill_days < 1:
            raise ValueError("backfill_days must be a positive integer")
        self.branches = branches
        self.backfill_days = backfill_days
        #: 会话一支的派生物（语义树）记着的词表版本；没接语义树时为 None。
        self.derived_versions = derived_versions
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        return await asyncio.to_thread(self._maintain_blocking, ctx)

    def _maintain_blocking(self, ctx: LifecycleContext) -> ParticipantReport:
        now = self._clock()
        released = 0
        readings: list[CapacityReading] = []
        for branch, derived in (
            (self.branches.session, self.derived_versions),
            (self.branches.physical, None),
        ):
            ctx.assert_owned()
            released += self._physical(branch, now)
            store = branch.kind_store
            floor = vocabulary_floor(store, now=now, derived=() if derived is None else derived())
            if store.truncate_before(floor):
                released += 1  # 截断是例行的释放，不是要人看的事
            readings.append(
                CapacityReading(
                    f"{branch.lane.value}_kinds_pending",
                    len(store.read_pending().pool.entries),
                    store.config.max_pending,
                )
            )
        return ParticipantReport(released=released, readings=tuple(readings))

    def _physical(self, branch: BehaviorRuntimeComponents, now: datetime) -> int:
        ledger = branch.physical_ledger
        if ledger is None:
            return 0
        return ledger.expire(now, backfill_days=self.backfill_days)


__all__ = ["BehaviorParticipant", "vocabulary_floor"]
