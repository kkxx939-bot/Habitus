"""读行为记录的时间（裁定 59）：合并、间隔、观测覆盖、密度、日内分布与峰。纯算法、不调模型。

只依赖 ``series.model``（行为记录的纯值）与标准库；语义树与预测层都从这里取，日历等外部条件在 ``habitus.conditions``。
"""

from habitus.temporal.coverage import Coverage, blind_days, coverage_of, day_coverage
from habitus.temporal.density import Density, density_of
from habitus.temporal.episodes import Episode, OccurrenceDay, episodes_of, occurrence_days
from habitus.temporal.intervals import (
    MIN_AT_RISK,
    ClusterReading,
    DayInterval,
    cluster_at,
    common_interval_days,
    day_intervals,
    gap_days,
    median_days,
)
from habitus.temporal.merge import merge_starts
from habitus.temporal.profile import MIN_PEAK_DAYS, DayProfile, Peak, day_profile, peaks_of
from habitus.temporal.reading import KindReading, Rulers, kind_reading
from habitus.temporal.slots import (
    MINUTES_PER_DAY,
    absolute_slot,
    require_aware,
    require_slot_minutes,
    slot_floor,
    slot_index,
    slots_per_day,
)

__all__ = [
    "KindReading",
    "Rulers",
    "kind_reading",
    "MINUTES_PER_DAY",
    "MIN_AT_RISK",
    "MIN_PEAK_DAYS",
    "ClusterReading",
    "Coverage",
    "DayInterval",
    "DayProfile",
    "Density",
    "Episode",
    "OccurrenceDay",
    "Peak",
    "absolute_slot",
    "blind_days",
    "cluster_at",
    "common_interval_days",
    "coverage_of",
    "day_coverage",
    "day_intervals",
    "day_profile",
    "density_of",
    "episodes_of",
    "gap_days",
    "median_days",
    "merge_starts",
    "occurrence_days",
    "peaks_of",
    "require_aware",
    "require_slot_minutes",
    "slot_floor",
    "slot_index",
    "slots_per_day",
]
