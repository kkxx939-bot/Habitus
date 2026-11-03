"""当天的日子预期：夜里那条线的最后一步（裁定 94 第 1 条）。

前一天定稿 → 前置处理 → 结算 → 词表（每晚新增、定期拆改）→ 语义树 → 机会表 → 单元 → 这一步，前一步完成即接下一步（``runtime.nightly``）。「离上次够久」一天只建一次，
用的就是这条线刚写的、含昨天的那张机会表：间隔、簇、命中、时窗只有这一个来源。盘上最晚那张表不是今天这一晚的（昨天还没定稿、
或这一晚的表没写成）就不建——拿缺昨天的表建出来的时窗缺昨天，同一个身份全账本只建一次，夜里算对的那条就再也进不了账。

每天都建、从当天零点起算（裁定 94 第 2 条）：这一步看的是零点以来的整段（``looked_since`` = 零点），所以时窗在这条线跑完之前
就已关掉的也照建——做了照常结，没做结过期待决（时窗已过，递送本来不说）；不许只建做了的一侧（裁定 49 第 1 条）。
读材料与落账同白天那一拍（``runtime.foresight_forming``）。词表迁移做到一半时不建。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as clock_time

from habitus.foresight import Batch, UnsealedReader
from habitus.foresight.ladder import Tolerance
from habitus.foundation.observability import NullObserver, ObservationStatus, Observer
from habitus.runtime.foresight_forming import (
    ExpectationForming,
    Formed,
    FormingReadSide,
    held_by_migration,
    observe_foresight,
    tonight_table,
)
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore


@dataclass(frozen=True)
class DayReport:
    at: datetime
    formed: Formed
    #: 为什么没建（建了是 None）。
    skipped: str | None = None


class DayExpectationStage:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        opportunities: OpportunityStore,
        read_side: FormingReadSide,
        unsealed: UnsealedReader,
        *,
        lane: str = "session",
        recent: int,
        tolerance: Tolerance,
        shadow_unestablished: bool = False,
        vocabulary_version: Callable[[], int] | None = None,
        migrating: Callable[[], bool] | None = None,
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.forming = ExpectationForming(
            ledger,
            read_side,
            unsealed,
            recent=recent,
            tolerance=tolerance,
            shadow_unestablished=shadow_unestablished,
            vocabulary_version=vocabulary_version,
        )
        if not isinstance(opportunities, OpportunityStore):
            raise TypeError("opportunities must be an OpportunityStore")
        if not isinstance(lane, str) or not lane:
            raise ValueError("lane must be non-empty text")
        if migrating is not None and not callable(migrating):
            raise TypeError("migrating must be callable or None")
        self.ledger = ledger
        self.opportunities = opportunities
        self.lane = lane
        self.migrating = migrating
        self.observer: Observer = observer if observer is not None else NullObserver()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    async def run_once(self) -> DayReport:
        started = time.monotonic()
        try:
            report = await self._run()
        except Exception as exc:
            observe_foresight(
                self.observer, "days", ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started
            )
            raise
        observe_foresight(
            self.observer,
            "days",
            ObservationStatus.SUCCESS,
            {
                "formed": report.formed.formed,
                "recorded": report.formed.recorded,
                "skipped": report.skipped or "",
                "unreadable": len(self.ledger.take_unreadable()) + len(self.opportunities.take_unreadable()),
            },
            started,
        )
        return report

    async def _run(self) -> DayReport:
        now = self._clock().astimezone(self.forming.read_side.zone)
        held = held_by_migration(self.migrating)
        if held is not None:
            return DayReport(now, Formed(), held)
        table, skipped = tonight_table(self.opportunities, self.lane, now.date())
        if table is None or skipped is not None:
            return DayReport(now, Formed(), skipped)
        recent = await self.forming.read(now)
        # 从零点起看：零点到这一步之间关掉的时窗也算赶上了，没做的那次照样进账（裁定 94 第 2 条）
        midnight = datetime.combine(now.date(), clock_time(0), tzinfo=now.tzinfo)
        return DayReport(now, self.forming.form(recent, table, batch=Batch.DAYS, looked_since=midnight))


__all__ = ["DayExpectationStage", "DayReport"]
