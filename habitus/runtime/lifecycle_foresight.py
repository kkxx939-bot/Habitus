"""预测层的生命周期参与者（26 号清单 6.11、6.14）：账本按天目录的释放。哪些天能放在这里判，删字节在账本门面。

- **漏报**：读者只看最近 30 天（可见面），更早的整日删；
- **索引**：结完、过了那天最晚时限 + 8 天的日子，删它的预期的索引项（之后不会再有触发按同一身份建它）；
- **整天**（预期、递送、回话、结算、追加裁决）：结完、出了诊断够得着的那一段（30 + 30 天）、单元历史已经记下了那一天、
  那一天的每条预期都已并进它那本账的压实段——四条都满足才删（先删索引）；
- **单元的档**：留最新一条与最近 60 天；
- **单元的提醒反馈**：每个行为留最近 ``feedback_kept`` 条（裁定 66），更早的、那天已经整天释放的删掉。不得小于读数窗口 M：
  读数只看最近 M 项，留得比它少就改了读数。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

from habitus.infrastructure.lifecycle import CapacityReading, LifecycleContext, ParticipantReport
from habitus.runtime.foresight_history_store import HistoryStore
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_settlement import DIAGNOSIS_DAYS, DIAGNOSIS_REACH_DAYS

#: 结完之后再等几天放索引：最晚时限最多在形成之后 7 天（第 2–7 天段），再多一天余量。
INDEX_GRACE_DAYS = 8
#: 单元的档留多近（与账的级同口径：诊断够得着的那一段）。
LEVEL_DAYS = DIAGNOSIS_REACH_DAYS


class ForesightParticipant:
    name = "foresight"

    def __init__(
        self,
        ledger: ForesightLedgerStore,
        history: HistoryStore,
        *,
        lane: str,
        feedback_kept: int,
        window: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ledger, ForesightLedgerStore) or not isinstance(history, HistoryStore):
            raise TypeError("ledger must be a ForesightLedgerStore and history a HistoryStore")
        if isinstance(feedback_kept, bool) or not isinstance(feedback_kept, int) or feedback_kept < window:
            raise ValueError("lifecycle.feedback_kept must not be smaller than the reading window")
        self.feedback_kept = feedback_kept
        self.ledger = ledger
        self.history = history
        self.lane = lane
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        #: 这个进程里已经放过索引的日子：之后每轮不再读它的预期去判（索引放过一次就没了）。
        self._index_released: set[date] = set()

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        return await asyncio.to_thread(self._maintain_blocking, ctx)

    def _maintain_blocking(self, ctx: LifecycleContext) -> ParticipantReport:
        now = self._clock()
        today = now.date()
        released = self.ledger.discard_omissions_before(today - timedelta(days=DIAGNOSIS_DAYS))
        through = self.ledger.final_through()
        if through is not None:
            recorded = self.history.recorded_days(self.lane)
            for day in self.ledger.days_with_expectations():
                if day > through:
                    break
                ctx.assert_owned()
                if self._releasable(day, today, recorded):
                    self.ledger.discard_day(day)
                    self._index_released.discard(day)
                    released += 1
                elif day not in self._index_released and self._index_due(day, now):
                    released += self.ledger.discard_index(day)
                    self._index_released.add(day)
        for kind in self.ledger.unit_kinds():
            ctx.assert_owned()
            released += self.ledger.trim_unit_levels(kind, before=now - timedelta(days=LEVEL_DAYS))
            released += self.ledger.trim_feedback(kind, keep=self.feedback_kept)
        readings = (CapacityReading("foresight_open_days", len(self.ledger.open_days()), DIAGNOSIS_REACH_DAYS),)
        return ParticipantReport(released=released, readings=readings)

    def _releasable(self, day: date, today: date, recorded: frozenset[date]) -> bool:
        if day >= today - timedelta(days=DIAGNOSIS_REACH_DAYS) or day not in recorded:
            return False
        for item in self.ledger.expectations_on(day):
            base = self.ledger.compaction_of(item.account)
            if base is None or not base.covers(item.formed_at, item.expectation_id):
                return False
        return True

    def _index_due(self, day: date, now: datetime) -> bool:
        deadlines = [item.deadline for item in self.ledger.expectations_on(day)]
        return bool(deadlines) and max(deadlines) + timedelta(days=INDEX_GRACE_DAYS) < now


__all__ = ["INDEX_GRACE_DAYS", "ForesightParticipant"]
