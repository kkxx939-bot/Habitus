"""日内分布与峰：这件事做的那天一般几点做、一天有几个高峰（裁定 59 第 6 条第 5 步；原节律口）。

按"做的那天几点做"统计：只看最近 ``recent`` 个有自然发生的日子（盲日不算；时段跟着他现在的习惯走，裁定 57），每个槽的率 = 这个槽开始做过的日子数 ÷ 这个槽在看的比例之和。
样本少的类要收缩，否则出 0 / 1 极端值：本槽 ← 前后 ``pool_half_width`` 槽的邻域 ← 全天（不按周几分桶，周几交给语义树的日历条件）。
收缩后归一化成占比（加起来是 1），再找峰：连续一段高于均值、且占比之和够得上 ``min_peak_mass`` 的槽是一个峰；跨午夜的
接成一个；一个峰里至少要有 2 个不同的日子（样本少就没有读数——体检三次都在 10 点，不报"10 点的峰"）。
频率交给密度，这里只说位置：例会材料是"做的那天多在 17:00 前后"。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from habitus.temporal.coverage import Coverage, blind_days
from habitus.temporal.episodes import Episode, OccurrenceDay
from habitus.temporal.slots import slot_index, slots_per_day

#: 一个峰里至少要有几个不同的日子（与预测层"样本少没有读数"的最小表达 MIN_HISTORY = 2 同一口径）。
MIN_PEAK_DAYS = 2


@dataclass(frozen=True)
class Peak:
    """一个峰：起止槽（含两端；跨午夜的 ``last_slot`` ≥ 每天槽数，尾巴落在次日）、中心槽、峰内占比、几个不同的日子。"""

    first_slot: int
    last_slot: int
    centre_slot: int
    share: float
    days: int


@dataclass(frozen=True)
class DayProfile:
    slot_minutes: int
    shares: tuple[float, ...]
    peaks: tuple[Peak, ...]
    occurrence_days: int


def day_profile(
    episodes: Sequence[Episode],
    days: Sequence[OccurrenceDay],
    coverage: Coverage,
    *,
    pool_half_width: int,
    shrink_slot_to_pool: float,
    shrink_pool_to_all_day: float,
    min_peak_mass: float,
    recent: int,
) -> DayProfile | None:
    """没有自然发生的日子、或那些日子全没在看，就没有读数（``None``）。"""

    slot_minutes = coverage.slot_minutes
    per_day = slots_per_day(slot_minutes)
    if not 0 < 2 * pool_half_width < per_day:
        raise ValueError("pool_half_width must leave the neighbourhood shorter than a day")
    if isinstance(recent, bool) or not isinstance(recent, int) or recent < 1:
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
    found = []
    for peak in peaks_of(shares, floor=min_peak_mass):
        covered = {slot % per_day for slot in range(peak.first_slot, peak.last_slot + 1)}
        seen = sum(1 for slots in started.values() if slots & covered)
        if seen >= MIN_PEAK_DAYS:
            found.append(Peak(peak.first_slot, peak.last_slot, peak.centre_slot, peak.share, seen))
    return DayProfile(slot_minutes=slot_minutes, shares=shares, peaks=tuple(found), occurrence_days=len(started))


def peaks_of(curve: Sequence[float], *, floor: float) -> tuple[Peak, ...]:
    """连续一段高于均值、且段内之和够得上 ``floor`` 的槽 = 一个峰；钟面是环形的，跨午夜的那段接成一个、归给它开始的那一侧。

    （``days`` 在这里填 0，由调用方按实际日子数补上。）
    """

    size = len(curve)
    if size == 0:
        return ()
    average = sum(curve) / size
    if average <= 0.0:
        return ()
    above = [value > average for value in curve]
    if all(above):
        return (
            (Peak(0, size - 1, max(range(size), key=lambda slot: (curve[slot], -slot)), sum(curve), 0),)
            if sum(curve) >= floor
            else ()
        )
    # 从一个不高于均值的槽之后开始扫一整圈，跨午夜的段自然连起来
    start = next(index for index in range(size) if not above[index])
    found: list[Peak] = []
    run: list[int] = []
    for offset in range(1, size + 1):
        index = (start + offset) % size
        if above[index]:
            run.append(start + offset)
            continue
        if run:
            found.append(_peak(curve, run))
            run = []
    if run:
        found.append(_peak(curve, run))
    return tuple(sorted((peak for peak in found if peak.share >= floor), key=lambda peak: peak.first_slot))


def _peak(curve: Sequence[float], run: list[int]) -> Peak:
    size = len(curve)
    first = run[0] % size
    last = first + len(run) - 1
    centre = max(range(first, last + 1), key=lambda slot: (curve[slot % size], -slot))
    return Peak(first, last, centre, sum(curve[slot % size] for slot in range(first, last + 1)), 0)


def _circular_sums(values: Sequence[float], half_width: int) -> list[float]:
    size = len(values)
    return [
        sum(values[(index + offset) % size] for offset in range(-half_width, half_width + 1)) for index in range(size)
    ]


def _shrink(numerator: float, denominator: float, prior_rate: float, strength: float) -> float:
    """Beta-二项伪计数收缩；``strength`` 的单位是"等效观测天数"：分母远大于它时几乎是原始率，分母趋零时退回先验。"""

    return (numerator + strength * prior_rate) / (denominator + strength)


__all__ = ["MIN_PEAK_DAYS", "DayProfile", "Peak", "day_profile", "peaks_of"]
