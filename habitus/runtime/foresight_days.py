"""当天的日子预期：换代的必须成的最后几步之一（裁定 94）。

连续定稿到昨天 → 重试收尾 → 结算 → 词表 → 读序列 → 语义树 → 机会表 → 单元历史 → 这一步 → 完成标记（``runtime.generation``）。
这一步用的就是这一代刚写的、含昨天的那张机会表：间隔、簇、命中、时窗只有这一个来源。盘上最晚那张表不是今天这一代的（这一代
的表没写成）就不建，换代据此判今天没有这一代。今天这一代做成之前，预测拍已经用旧表把"表那一代之后一次都没再出现过"的类建上
（``ExpectationForming.days_unchanged``）：那几类旧表与这张表算出同一个身份，这里按身份查重、不再写一份；别的类在这里第一次建。

每天都建、从当天零点起算（裁定 94）：这一步看的是零点以来的整段（``looked_since`` = 零点），所以时窗在换代做完之前
就已关掉的也照建——做了照常结，没做结过期待决（时窗已过，递送本来不说）；不许只建做了的一侧（裁定 49）。
读材料与落账同预测拍（``runtime.foresight_forming``），未封口的读口也是同一个：模型此刻用不了只是这一次没选上词，不抛。
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
    NO_TABLE,
    ExpectationForming,
    Formed,
    FormingReadSide,
    observe_foresight,
)
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_opportunities import OpportunityStore

STALE_TABLE = "今天这一代的机会表还没有"


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
        if not lane:
            raise ValueError("lane must be non-empty text")
        self.ledger = ledger
        self.opportunities = opportunities
        self.lane = lane
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
        table = self.opportunities.latest(self.lane)
        if table is None or table.night != now.date():
            return DayReport(now, Formed(), NO_TABLE if table is None else STALE_TABLE)
        recent = await self.forming.read(now)
        # 从零点起看：零点到这一步之间关掉的时窗也算赶上了，没做的那次照样进账（裁定 94）
        midnight = datetime.combine(now.date(), clock_time(0), tzinfo=now.tzinfo)
        return DayReport(now, self.forming.form(recent, table, batch=Batch.DAYS, looked_since=midnight))


__all__ = ["DayExpectationStage", "DayReport"]
