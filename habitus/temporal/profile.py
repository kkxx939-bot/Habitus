"""日内分布：这件事做的那天一般几点做（裁定 59 第 6 条第 5 步）。

按"做的那天几点做"统计：只看最近 ``recent`` 个有自然发生的日子（盲日不算；时段跟着他现在的习惯走，裁定 57），每个槽的率 = 这个槽开始做过的日子数 ÷ 这个槽在看的比例之和。
样本少的类要收缩，否则出 0 / 1 极端值：本槽 ← 前后 ``pool_half_width`` 槽的邻域 ← 全天（不按周几分桶，周几交给语义树的日历条件）。
收缩后归一化成占比（加起来是 1）。相邻间隔读它估每个槽的漏看概率（``intervals.day_intervals``）。
频率交给密度，这里只说位置：例会材料是"做的那天多在 17:00 前后"。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from habitus.temporal.coverage import Coverage, blind_days
from habitus.temporal.episodes import Episode, OccurrenceDay
from habitus.temporal.slots import slot_index, slots_per_day


@dataclass(frozen=True)
class DayProfile:
    slot_minutes: int
    shares: tuple[float, ...]
    occurrence_days: int


def day_profile(
    episodes: Sequence[Episode],
    days: Sequence[OccurrenceDay],
    coverage: Coverage,
    *,
    pool_half_width: int,
    shrink_slot_to_pool: float,
    shrink_pool_to_all_day: float,
    recent: int,
) -> DayProfile | None:
    """没有自然发生的日子、或那些日子全没在看，就没有读数（``None``）。"""

    slot_minutes = coverage.slot_minutes
    per_day = slots_per_day(slot_minutes)
    if not 0 < 2 * pool_half_width < per_day:
        raise ValueError("pool_half_width must leave the neighbourhood shorter than a day")
    if recent < 1:
        raise ValueError("recent must be a positive integer")
    hidden = blind_days(days)
    started: dict[date, set[int]] = {}
    for item in episodes:
        if item.natural and item.day not in hidden:
            started.setdefault(item.day, set()).add(slot_index(item.started_at, slot_minutes=slot_minutes))
    started = {day: started[day] for day in sorted(started)[-recent:]}
    top = [0.0] * per_day
    bottom = [0.0] * per_day
    for day, slots in started.items():
        watched = coverage.days.get(day)
        if watched is None:
            continue
        for index in range(per_day):
            bottom[index] += watched[index]
        for index in slots:
            top[index] += 1.0
    if sum(bottom) <= 0.0 or sum(top) <= 0.0:
        return None
    all_day = sum(top) / sum(bottom)
    pooled_top = _circular_sums(top, pool_half_width)
    pooled_bottom = _circular_sums(bottom, pool_half_width)
    rates = []
    for index in range(per_day):
        pooled = _shrink(pooled_top[index], pooled_bottom[index], all_day, shrink_pool_to_all_day)
        rates.append(min(max(_shrink(top[index], bottom[index], pooled, shrink_slot_to_pool), 0.0), 1.0))
    total = sum(rates)
    shares = tuple(rate / total for rate in rates)
    return DayProfile(slot_minutes=slot_minutes, shares=shares, occurrence_days=len(started))


def _circular_sums(values: Sequence[float], half_width: int) -> list[float]:
    size = len(values)
    return [
        sum(values[(index + offset) % size] for offset in range(-half_width, half_width + 1)) for index in range(size)
    ]


def _shrink(numerator: float, denominator: float, prior_rate: float, strength: float) -> float:
    """Beta-二项伪计数收缩；``strength`` 的单位是"等效观测天数"：分母远大于它时几乎是原始率，分母趋零时退回先验。"""

    return (numerator + strength * prior_rate) / (denominator + strength)


__all__ = ["DayProfile", "day_profile"]
