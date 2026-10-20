"""观测覆盖：哪些时段我们在看（裁定 59 第 6 条第 3 步）。密度、日内分布、漏看概率共用这一个分母。

口径沿用旧预测树的曝光（"在看"）：从这条 lane 第一条记录那天起、到最后一个已封口的日子，每个日历日都算——一整天没记录也
算在看（那天就是没做）；扣掉上游明确报的空白（"没读懂"与"未观测"两种都扣；被证伪的"没读懂"已在读序列时作废）；按槽、按
比例扣（一个槽被盖住一半，就算半槽在看）。不按周几分桶。

这和语义树的"人在场"（当天有记录的日子里，第一条开始到最后一条结束）不是同一个问题，各用各的名字、不并。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import MappingProxyType

from habitus.series.model import EventSeries, SeriesGap
from habitus.temporal.episodes import OccurrenceDay
from habitus.temporal.slots import slots_per_day

_SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class Coverage:
    """一条 lane 每天每槽在看的比例（0–1）。没有记录的 lane 是空的。"""

    slot_minutes: int
    days: Mapping[date, tuple[float, ...]]

    def __post_init__(self) -> None:
        per_day = slots_per_day(self.slot_minutes)
        frozen = {}
        for day, slots in self.days.items():
            values = tuple(float(value) for value in slots)
            if len(values) != per_day or any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError("coverage holds one share in [0, 1] per slot of the day")
            frozen[day] = values
        object.__setattr__(self, "days", MappingProxyType(dict(sorted(frozen.items()))))

    def day_share(self, day: date) -> float:
        """这一天有多大比例在看；不在覆盖范围里的日子是 0。"""

        slots = self.days.get(day)
        return 0.0 if slots is None else sum(slots) / len(slots)

    def watched_days(self, *, since: date | None = None, excluding: Iterable[date] = ()) -> float:
        """从 ``since``（含）起在看的天数（按比例），扣掉 ``excluding``（这一类的盲日）。"""

        skipped = frozenset(excluding)
        return sum(
            sum(slots) / len(slots)
            for day, slots in self.days.items()
            if (since is None or day >= since) and day not in skipped
        )


def coverage_of(series: EventSeries, lane: str, *, slot_minutes: int) -> Coverage:
    """``lane`` 从第一条记录那天到序列最后一个已封口的日子，每天每槽在看的比例。"""

    first = next((record.day for record in series.records if record.lane == lane), None)
    if first is None or first > series.last_day:
        return Coverage(slot_minutes=slot_minutes, days={})
    by_day = _gaps_by_day(series.gaps)
    days = {}
    day = first
    while day <= series.last_day:
        days[day] = day_coverage(day, by_day.get(day, ()), slot_minutes=slot_minutes)
        day += timedelta(days=1)
    return Coverage(slot_minutes=slot_minutes, days=days)


def day_coverage(day: date, gaps: Sequence[SeriesGap], *, slot_minutes: int) -> tuple[float, ...]:
    """这一天每个槽有多少比例在看：空白之外一律视作在看（上游不报空白时的既定退化假设）。"""

    slots = slots_per_day(slot_minutes)
    width = float(slot_minutes * 60)
    coverage = [1.0] * slots
    for gap in gaps:
        start = _seconds_into_day(gap.started_at, day)
        end = _seconds_into_day(gap.ended_at, day)
        if end <= start:
            continue
        first = max(int(start // width), 0)
        last = min(int((end - 1e-9) // width), slots - 1)
        for index in range(first, last + 1):
            slot_start = index * width
            overlap = min(end, slot_start + width) - max(start, slot_start)
            if overlap > 0:
                coverage[index] = max(coverage[index] - overlap / width, 0.0)
    return tuple(coverage)


def blind_days(days: Sequence[OccurrenceDay]) -> frozenset[date]:
    """这一类只有被提醒之后才做、没有自然发生的那些日子（裁定 54）：不提醒会不会做看不到，对这一类既不算覆盖、也不算命中。"""

    return frozenset(item.day for item in days if item.natural == 0)


def _gaps_by_day(gaps: Iterable[SeriesGap]) -> dict[date, list[SeriesGap]]:
    """按本地日归并；跨日的空白按天切开（否则次日那半段会整段丢掉）。"""

    grouped: dict[date, list[SeriesGap]] = {}
    for gap in gaps:
        day = gap.started_at.date()
        while day <= gap.ended_at.date():
            grouped.setdefault(day, []).append(gap)
            day += timedelta(days=1)
    return grouped


def _seconds_into_day(moment: datetime, day: date) -> float:
    start_of_day = datetime.combine(day, datetime.min.time(), tzinfo=moment.tzinfo)
    return min(max((moment - start_of_day).total_seconds(), 0.0), _SECONDS_PER_DAY)


__all__ = ["Coverage", "blind_days", "coverage_of", "day_coverage"]
