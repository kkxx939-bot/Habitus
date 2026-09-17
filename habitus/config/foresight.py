"""预测层的运行配置。

本组只有**保护闸**、判断服务的重试与 worker 的停机时限，外加一个开关：预测层的数值参数（窗口宽度、
容差）一律不在这里——窗口就是预测树的池化邻域（``prediction.pool_half_width``），转移窗就是树的
``transition_window_seconds``，两边配两份会让"数字与背景同一批日子"当场失效。判断的节奏也不在这里：
worker 每个槽一拍，槽宽就是树的 ``slot_minutes``。

保护闸是**保护闸**：它们防的是上下文被撑爆与一次查询把整棵树扫穿，不是调质量的旋钮。数值
全部是启动档，要在真实数据上重定；但重定的是数值，不是"要不要有这个闸"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from habitus.config.loader import construct_config

_INT_BOUNDS = (
    ("window_days", 1, 3_650),
    ("max_days_per_layer", 1, 3_650),
    ("judge_transient_retries", 0, 20),
)
_FLOAT_BOUNDS = (
    ("judge_transient_retry_delay_seconds", 0.0, 600.0),
    ("worker_shutdown_timeout_seconds", 1.0, 3_600.0),
)


@dataclass(frozen=True)
class ForesightConfig:
    """预测层是否启用，以及一次装配的几道保护闸。"""

    enabled: bool = False
    # 卡上"上一次"往回看多少天（``scene.views.last_time``）：超过这个跨度的上一次就写"没有"。
    # 它**不是**取历史卡的窗——卡按出处日取，本来就没有日历窗；每层摊开多少天由下面那道闸管。
    window_days: int = 30
    # 四层里每一层最多摊开多少天的历史背景；截掉的更早日子会以 dropped_days 如实报出。
    max_days_per_layer: int = 40
    # 判断服务对传输层瞬态错的有界重试（路由层另有一层，这里的次数要很小）。
    judge_transient_retries: int = 1
    judge_transient_retry_delay_seconds: float = 5.0
    # 每槽一拍的 worker 停机时限。
    worker_shutdown_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("foresight.enabled must be a boolean")
        for name, lower, upper in _INT_BOUNDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"foresight.{name} must be an integer between {lower} and {upper}")
        for name, low, high in _FLOAT_BOUNDS:
            number = getattr(self, name)
            if isinstance(number, bool) or not isinstance(number, int | float) or not low <= float(number) <= high:
                raise ValueError(f"foresight.{name} must be a number between {low:g} and {high:g}")

    @classmethod
    def from_mapping(cls, value: Any) -> ForesightConfig:
        return construct_config(cls, value, "config.foresight")


__all__ = ["ForesightConfig"]
