"""预测层的运行配置。

本组只有**保护闸**和一个开关：预测层的数值参数（窗口宽度、容差）一律不在这里——窗口就是
预测树的池化邻域（``prediction.pool_half_width``），转移窗就是树的 ``transition_window_seconds``，
两边配两份会让"数字与背景同一批日子"当场失效。这里只管"一次装配最多摊开多少东西"。

保护闸是**保护闸**：它们防的是上下文被撑爆与一次查询把整棵树扫穿，不是调质量的旋钮。数值
全部是启动档，要在真实数据上重定；但重定的是数值，不是"要不要有这个闸"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from habitus.config.loader import construct_config

_INT_BOUNDS = (
    ("max_days_per_layer", 1, 3_650),
    ("max_similar_scenes", 1, 200),
    ("max_pack_chars", 1_000, 4_000_000),
)


@dataclass(frozen=True)
class ForesightConfig:
    """预测层是否启用，以及一次装配的几道保护闸。"""

    enabled: bool = False
    # 四层里每一层最多摊开多少天的历史背景；截掉的更早日子会以 dropped_days 如实报出。
    max_days_per_layer: int = 40
    # 相似情景反查最多带回几件事。
    max_similar_scenes: int = 8
    # 渲染后的证据包字符上限；超了按"先砍相似情景、再砍逐条实例、画像不砍"的次序裁。
    max_pack_chars: int = 60_000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("foresight.enabled must be a boolean")
        for name, lower, upper in _INT_BOUNDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"foresight.{name} must be an integer between {lower} and {upper}")

    @classmethod
    def from_mapping(cls, value: Any) -> ForesightConfig:
        return construct_config(cls, value, "config.foresight")


__all__ = ["ForesightConfig"]
