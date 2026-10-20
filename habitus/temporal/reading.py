"""一组发生的整份时间读数（裁定 59 第 6 条）：合并成一次一次 → 按天 → 日内分布与峰 → 密度 → 相邻间隔，一次算完。

语义树的节律（``scene.concepts.rhythm``）与复现（``scene.recurrence``）读的是同一份，原来两处各写一遍这五步，改截尾或盲日的
口径时两份会悄悄分叉（2026-10-18 四方评审）。这里只把步骤串起来，每一步的定义仍在各自的模块。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from habitus.series.model import SeriesRecord
from habitus.temporal.coverage import Coverage
from habitus.temporal.density import Density, density_of
from habitus.temporal.episodes import OccurrenceDay, episodes_of, occurrence_days
from habitus.temporal.intervals import DayInterval, day_intervals
from habitus.temporal.profile import DayProfile, day_profile


@dataclass(frozen=True)
class Rulers:
    """算时间读数用的尺子（组合根从 ``config.temporal`` 抄来，经调用方传进来）。"""

    slot_minutes: int
    merge_window_seconds: int
    pool_half_width: int
    shrink_slot_to_pool: float
    shrink_pool_to_all_day: float
    min_peak_mass: float
    #: 日内分布与密度只看这一类最近几个自然发生日（M = ⌈H · log₂(1/α)⌉，组合根从遗忘的半衰与 α 算好；裁定 57）。
    #: 一年前从 19:00 改到 07:00 的跑步，时段跟着最近的走；间隔多重集不受它限，保留全历史。
    recent: int

    def __post_init__(self) -> None:
        if isinstance(self.recent, bool) or not isinstance(self.recent, int) or self.recent < 1:
            raise ValueError("rulers.recent must be a positive integer")


@dataclass(frozen=True)
class KindReading:
    days: tuple[OccurrenceDay, ...]
    profile: DayProfile | None
    density: Density | None
    intervals: tuple[DayInterval, ...]


def kind_reading(records: Sequence[SeriesRecord], coverage: Coverage, rulers: Rulers) -> KindReading:
    """``records`` 是一个类（或汇总概念的几个成员类合在一起）按开始时刻升序的记录，``coverage`` 是它那条 lane 的观测覆盖。"""

    episodes = episodes_of(records, window_seconds=rulers.merge_window_seconds)
    days = occurrence_days(episodes)
    profile = day_profile(
        episodes,
        days,
        coverage,
        pool_half_width=rulers.pool_half_width,
        shrink_slot_to_pool=rulers.shrink_slot_to_pool,
        shrink_pool_to_all_day=rulers.shrink_pool_to_all_day,
        min_peak_mass=rulers.min_peak_mass,
        recent=rulers.recent,
    )
    density = density_of(days, coverage, recent=rulers.recent)
    intervals = day_intervals(
        days, coverage, profile=profile, occurrence_day_share=None if density is None else density.occurrence_day_share
    )
    return KindReading(days=days, profile=profile, density=density, intervals=intervals)


__all__ = ["KindReading", "Rulers", "kind_reading"]
