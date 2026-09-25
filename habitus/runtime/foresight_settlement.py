"""夜批里的结算一拍：把定稿日子上的承诺对答案。

排在预测树重建之后、关联之前（``runtime/assembly.py``）。它只读两样东西：账本里那天还没结算的承诺，
与行为树上那天的行；**不读判断存储、不读预测树**——承诺自带说话那一代的槽宽，所以换参数重建的新一代
不会把旧承诺核对错。

"那天定稿了"用归约自己的事实（``BehaviorReductionRunner.closed_days``：那天的链都已落树、且封口视界已过
那天的本地结束），由组合根注入。**不能用预测树的出处日**：树每轮从行为树全量重建，今天上午的行中午就在
树上，拿它当定稿会在今天还没过完时把今天的承诺结成「落空」，而结算是一次性的。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol, runtime_checkable

from habitus.behavior.tree import BehaviorTree
from habitus.foresight.ledger import settle
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.scene import DayTypeCalendar
from habitus.scene.views import DayIndexCache


@runtime_checkable
class BehaviorReadSide(Protocol):
    """结算要的只有行为树的读口三样；``EvidenceAssembler`` 正好满足它，所以不另建一份缓存参数。"""

    @property
    def behavior_tree(self) -> BehaviorTree: ...

    @property
    def subject(self) -> str: ...

    @property
    def calendar(self) -> DayTypeCalendar: ...


@dataclass(frozen=True)
class SettlementReport:
    """一拍结算了多少：按结果计数，以及哪几天因为还没定稿被留到下一夜。"""

    settled: int
    verified: int
    deviated: int
    missed: int
    pending_days: tuple[date, ...]


class SettlementStage:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        assembler: BehaviorReadSide,
        *,
        closed_days: Callable[[], Iterable[date]],
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if not isinstance(assembler, BehaviorReadSide):
            raise TypeError("assembler must expose the behaviour tree read side")
        if not callable(closed_days):
            raise TypeError("closed_days must be callable")
        self.ledger = ledger
        self.assembler = assembler
        self.closed_days = closed_days
        self.observer: Observer = observer if observer is not None else NullObserver()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def run_once(self) -> SettlementReport:
        started = time.monotonic()
        try:
            report = self._settle_all()
        except Exception as exc:
            self._observe(ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started)
            raise
        self._observe(
            ObservationStatus.SUCCESS,
            {
                "settled": report.settled,
                "verified": report.verified,
                "deviated": report.deviated,
                "missed": report.missed,
                "pending_days": len(report.pending_days),
            },
            started,
        )
        return report

    def _settle_all(self) -> SettlementReport:
        closed = frozenset(self.closed_days())
        cache = DayIndexCache(
            self.assembler.behavior_tree, subject=self.assembler.subject, calendar=self.assembler.calendar
        )
        counts = {"验证": 0, "偏离": 0, "落空": 0}
        pending: list[date] = []
        settled_at = self._clock()
        for day in self.ledger.days_with_claims():
            unsettled = self.ledger.unsettled_claims(day)
            if not unsettled:
                continue
            if day not in closed:
                pending.append(day)
                continue
            rows = cache.day(day).rows
            for claim in unsettled:
                item = settle(claim, rows, settled_at=settled_at)
                self.ledger.record_settlement(item)
                counts[item.outcome] += 1
        return SettlementReport(
            settled=sum(counts.values()),
            verified=counts["验证"],
            deviated=counts["偏离"],
            missed=counts["落空"],
            pending_days=tuple(pending),
        )

    def _observe(self, status: ObservationStatus, attributes: dict[str, str | int | float | bool], started: float) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category="foresight",
                    operation="settlement",
                    status=status,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:  # noqa: BLE001 - 观测不许影响结算
            pass


__all__ = ["BehaviorReadSide", "SettlementReport", "SettlementStage"]
