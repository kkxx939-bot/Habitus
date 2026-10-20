"""日历情境的量法（裁定 59 第 7 条，2026-10-18 定）：按维度取对照，不再拿"前后紧挨着的平常日子"比。

外部条件只是给每一天打的约束标签。检验"周三 → 例会材料"时：

- **实验组**：周几这一维取值是周三的日子；**对照**：同一维上别的取值（周一、二、四……），即这一维攒出来的全部其余取值都对照。
- **其余维度固定**：比周几、几号、月末时把日型固定住（国庆的周三只和国庆那几天比，不拉低"周三"）；比日型时不固定周几
  （周几决定了名义日型，固定它就没东西可比了）。
- **只在同一段时间里比**：周几、日型按同一周分层，几号、月末按同一个月分层，抵掉习惯慢慢变多变少、这段时间的忙闲；
  而且从这件事第一次自然发生那天起才算（之前那几周它还不存在，算进来只会稀释）。

每层一张四格表（单位是一个人在场、结果已知的日子），零假设下实验组里的阳性数服从超几何分布，各层卷起来——与长跨度同一套检验
（``longspan.EpisodeMeasure``），只是表怎么分不同。样本少（层里只有实验组或只有对照）的层不进表，少见的取值因此攒够之前出不了读数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from habitus.conditions import DAY_TYPE_KEY, WEEKDAY_KEY
from habitus.scene.relations.longspan import EpisodeMeasure, _daily_outcome, _Table
from habitus.scene.relations.timeline import LaneTimeline

#: 日型答不出的那天在分层里的记法（只和同样答不出的日子比）。
UNKNOWN = "?"


@dataclass(frozen=True)
class CalendarMeasure(EpisodeMeasure):
    """``treated``：落在这个日历情境里的日子；``strata``：日子 → 它所在的层（同一段时间 + 固定住的其余维度）。"""

    treated: frozenset[date] = frozenset()
    strata: dict[date, tuple[object, ...]] = field(default_factory=dict)

    def _tables(self, since: date | None = None) -> tuple[_Table, ...]:
        since = self.since if since is None else since
        groups: dict[tuple[object, ...], tuple[list[tuple[int, date]], list[int]]] = {}
        for day, outcome in zip(self.days, self.outcomes, strict=True):
            if outcome is None or day not in self.strata:
                continue
            treated, controls = groups.setdefault(self.strata[day], ([], []))
            if day in self.treated:
                if since is None or day >= since:
                    treated.append((outcome, day))
            else:
                controls.append(outcome)
        return tuple(
            _Table(tuple(value for value, _day in treated), tuple(controls), tuple(day for _value, day in treated))
            for _stratum, (treated, controls) in sorted(groups.items(), key=lambda item: str(item[0]))
            if treated and controls
        )


def calendar_measure(line: LaneTimeline, consequent: str, key: str, values: tuple[str, ...]) -> CalendarMeasure:
    """这个日历情境（``key`` 这一维取 ``values``）对 ``consequent`` 的量法。这一维答不出的日子不算（日型缺数据的年份）。"""

    natural = [mark.day for mark in line.outcomes(consequent)]
    first = min(natural) if natural else None
    days = tuple(day for day in line.days() if first is not None and day >= first)
    treated: set[date] = set()
    strata: dict[date, tuple[object, ...]] = {}
    for day in days:
        conditions = line.calendar_conditions.get(day, {})
        value = conditions.get(key)
        if value is None:
            continue
        if value in values:
            treated.add(day)
        strata[day] = _stratum(day, key, conditions)
    outcomes = tuple(_daily_outcome(line, consequent, day) for day in days)
    return CalendarMeasure(days=days, outcomes=outcomes, episodes=(), treated=frozenset(treated), strata=strata)


def _stratum(day: date, key: str, conditions: dict[str, str]) -> tuple[object, ...]:
    if key in (WEEKDAY_KEY, DAY_TYPE_KEY):
        year, week, _weekday = day.isocalendar()
        period: tuple[object, ...] = ("周", year, week)
    else:
        period = ("月", day.year, day.month)
    if key == DAY_TYPE_KEY:
        return period
    return (*period, conditions.get(DAY_TYPE_KEY, UNKNOWN))


__all__ = ["UNKNOWN", "CalendarMeasure", "calendar_measure"]
