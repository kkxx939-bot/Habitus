"""读行为记录时间用的尺子（裁定 59）：槽宽、合并窗口、日内分布的邻域与收缩、峰下限。

语义树与预测层都经 ``habitus.temporal`` 用这些尺子；全系统只此一处，不在别处另配一份。本模块只有纯标量与校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from habitus.config.loader import construct_config

MINUTES_PER_DAY = 24 * 60


@dataclass(frozen=True)
class TemporalConfig:
    """槽宽（分钟）、合并窗口（分钟，按开始时刻的秒数判断）、日内分布的邻域宽度（槽）与两档收缩强度、峰下限。"""

    # 槽宽：提醒精度目标与链内相邻间隔低分位里取小的那个（待定档）。须整除一天。
    slot_minutes: int = 15
    # 同一类相邻两条开始时刻相隔不超过这么多分钟算同一次（扎堆合并，裁定 59 第 6 条第 1 步）。须是槽宽的整数倍：
    # 关系检验"同一条链"那一段按槽数用它。
    merge_window_minutes: int = 45
    # 日内分布收缩的邻域：本槽 ← 前后各这么多槽 ← 全天。同一习惯的日间时间抖动（待定档）。
    pool_half_width: int = 3
    # 两档收缩的先验分量：本槽向邻域、邻域向全天（待定档：留出集上最小化对数损失）。
    shrink_slot_to_pool: float = 5.0
    shrink_pool_to_all_day: float = 5.0
    # 一个峰里至少有这么多比例的发生（归一化之后比；裁定 59 第 6 条第 5 步）。
    min_peak_mass: float = 0.10

    def __post_init__(self) -> None:
        for name in ("slot_minutes", "merge_window_minutes", "pool_half_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"temporal.{name} must be a positive integer")
        if MINUTES_PER_DAY % self.slot_minutes:
            raise ValueError("temporal.slot_minutes must divide a day")
        if self.merge_window_minutes % self.slot_minutes:
            raise ValueError("temporal.merge_window_minutes must be a whole number of slots")
        if self.pool_half_width >= MINUTES_PER_DAY // self.slot_minutes:
            raise ValueError("temporal.pool_half_width must be shorter than a day")
        for name in ("shrink_slot_to_pool", "shrink_pool_to_all_day"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= float(value) <= 1_000.0:
                raise ValueError(f"temporal.{name} must be a number between 0 and 1000")
        if (
            isinstance(self.min_peak_mass, bool)
            or not isinstance(self.min_peak_mass, int | float)
            or not 0.0 < float(self.min_peak_mass) < 1.0
        ):
            raise ValueError("temporal.min_peak_mass must lie in (0, 1)")

    @property
    def merge_window_seconds(self) -> int:
        return self.merge_window_minutes * 60

    @property
    def transition_window_slots(self) -> int:
        """关系检验"同一条链"那一段的槽数，由合并窗口与槽宽换算，不另配。"""

        return self.merge_window_minutes // self.slot_minutes

    @classmethod
    def from_mapping(cls, value: Any) -> TemporalConfig:
        return construct_config(cls, value, "config.temporal")


__all__ = ["TemporalConfig"]
