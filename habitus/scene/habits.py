"""习惯卡片上的数（裁定 113）：一个在用节点的复现（``scene.recurrence``）换成几个只做记录的数。

只抄系统已经算好的：第一次、最近一次是哪天，一共多少天，其中多少天只有下级的记录、它自己没有直接记录（复现把这种日子按
"被提醒才做的"交进去，在复现里和真被提醒的分不出来，所以拿序列现数一遍），相邻间隔几段、能用的几段、截尾的几段，以及
``temporal.median_days`` 的常见间隔。只放日期与整数：同一份序列重算得到同一组数，"没变不写"才判得出来。预测不读它们。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from habitus.scene.recurrence import Recurrence
from habitus.series.model import EventSeries
from habitus.temporal import median_days


@dataclass(frozen=True)
class HabitNumbers:
    """一个在用节点此刻的几个数。``usable_intervals`` 是常见间隔用得上的段数（没截尾、没漏看），``median_days`` 没有能用的段时是 None。"""

    first_day: date
    last_day: date
    days: int
    below_only_days: int
    intervals: int
    usable_intervals: int
    censored_intervals: int
    median_days: int | None


def habit_numbers(series: EventSeries, recurrences: Mapping[str, Recurrence]) -> dict[str, HabitNumbers]:
    """每个有复现的节点的数（按词编号）。``series`` 是算出这份复现的那一份序列。"""

    direct: dict[str, set[date]] = {}
    for record in series.records:
        if record.classified:
            direct.setdefault(record.kind_token, set()).add(record.day)
    found: dict[str, HabitNumbers] = {}
    for kind, recurrence in recurrences.items():
        own = direct.get(kind, set())
        median = median_days(recurrence.intervals)
        found[kind] = HabitNumbers(
            first_day=recurrence.days[0].day,
            last_day=recurrence.days[-1].day,
            days=len(recurrence.days),
            below_only_days=sum(1 for item in recurrence.days if item.day not in own),
            intervals=len(recurrence.intervals),
            usable_intervals=sum(1 for item in recurrence.intervals if not item.censored and item.seen > 0.0),
            censored_intervals=sum(1 for item in recurrence.intervals if item.censored),
            # 中位数是某一段的天数（整数），``median_days`` 给的是浮点形式
            median_days=None if median is None else round(median),
        )
    return found


__all__ = ["HabitNumbers", "habit_numbers"]
