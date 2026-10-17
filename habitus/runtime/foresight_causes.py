"""给落空找原因的那一拍（裁定 55，方案第九节 6e）：排在机会表那一拍里、语义树之后。

夜批顺序是 预测树重建 → 结算 → 语义树 → 机会表：结算跑的时候这一晚的概念集与时间线还没建，它们只在内存里交给机会表那一拍。
所以结算照旧先结出落空，同一晚稍后这里给"已结、还没找过原因"的落空找原因（``foresight.causes.find_cause``），作为同值改判
挂在那条预测轴事实上（好坏、强度不变，带上原因，不动读数）。一天里的都找过了就标这一天（``mark_explained``），之后不再读。

只找看得见的落空：现在那条预测轴事实是「过期待决」、强度大于 0（作废、观测空白下的中性在结算里已经判过）、还没带原因。
最晚时限那天还不在这一晚的时间线里（时间线只到最后一个已封口的日子）的，等下一晚。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime

from habitus.behavior.uri import BehaviorURI
from habitus.foresight import current_facts
from habitus.foresight.causes import CauseMaterials, find_cause
from habitus.foresight.ladder import PREDICTION
from habitus.foresight.ledger import AccountFact, Expectation
from habitus.foresight.units import UNIT
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.scene import DayTypeCalendar
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations import LaneTimeline, Relation


class CauseStage:
    def __init__(
        self,
        ledger: ForesightLedgerStore,
        *,
        lane: str,
        current: Callable[[str], Sequence[str]],
        calendar: DayTypeCalendar,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if not isinstance(lane, str) or not lane:
            raise ValueError("lane must be non-empty text")
        if not callable(current):
            raise TypeError("current must be callable")
        if not isinstance(calendar, DayTypeCalendar):
            raise TypeError("calendar must implement DayTypeCalendar")
        self.ledger = ledger
        self.lane = lane
        self.current = current
        self.calendar = calendar
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def run(
        self, *, line: LaneTimeline, concepts: ConceptSet, relations: Sequence[Relation], window_slots: int
    ) -> int:
        """这一晚给几次落空找到了原因。``line`` 是语义树这一晚这条 lane 的时间线，``relations`` 是这一晚的关系表（全部状态）。"""

        if line.lane != self.lane:
            return 0
        now = self._clock()
        horizon = date.fromordinal(line.horizon // line.slots_per_day)
        materials = CauseMaterials(line, concepts, tuple(relations), window_slots, self.calendar.describe)
        found = 0
        for day in self.ledger.days_with_expectations():
            if self.ledger.explained(day):
                continue
            settled = {item.expectation_id for item in self.ledger.settlements_on(day)}
            subsumed = {item.expectation_id for item in self.ledger.deliveries_on(day) if item.reason == UNIT}
            waiting = False
            for item in self.ledger.expectations_on(day):
                if item.expectation_id in subsumed:
                    continue  # 单元替它一并决定过：这一刻的原因在依据那条上找一次，不在每个输入上各找一遍
                if item.expectation_id not in settled or item.deadline.date() >= horizon:
                    waiting = True  # 还没结，或最晚时限那天还不在这一晚的时间线里
                    continue
                fact = self._miss(item)
                if fact is None:
                    continue
                cause = find_cause(item, materials, current=self.current, basis_days=_days(item.basis))
                self.ledger.append_fact(replace(fact, written_at=max(now, fact.precedence), cause=cause))
                found += 1
            if not waiting:
                self.ledger.mark_explained(day, at=now)
        return found

    def _miss(self, item: Expectation) -> AccountFact | None:
        """这条预期现在算数的预测轴事实是一次看得见、还没找过原因的落空，就是它。"""

        mine = [fact for fact in self.ledger.facts_of(item.account) if fact.expectation_id == item.expectation_id]
        fact = next((fact for fact in current_facts(mine) if fact.axis == PREDICTION), None)
        if fact is None or fact.kind != "过期待决" or fact.strength <= 0.0 or fact.cause is not None:
            return None
        return fact


def _days(uris: Iterable[str]) -> tuple[date, ...]:
    found = []
    for uri in uris:
        try:
            found.append(BehaviorURI.parse(uri).to_address().occurred_on)
        except (TypeError, ValueError):
            continue
    return tuple(found)


__all__ = ["CauseStage"]
