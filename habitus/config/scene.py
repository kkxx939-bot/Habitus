"""语义关联层（情景树）的运行配置。

本模块只有纯标量、不 import ``scene``（与 ``config/behavior.py`` 同一纪律）：领域内的自洽
校验仍在 ``scene`` 各自的 config 值对象里，这里只管取值范围与"有没有给全"。

情景树每封口日调一次模型，从行为树派生；行为侧没开就没有输入（跨域自洽在 ``root.py``）。
数值全部是**启动档**：回看天数与待用前提的过期期限最终由生命周期算法按 needs 边的滞后
分布定，现在先按用户裁定的初值走。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from habitus.config.loader import construct_config

_INT_BOUNDS = (
    ("lookback_days", 1, 90),
    ("pending_expiry_days", 1, 3_650),
    ("max_occurrences_per_call", 1, 5_000),
    ("max_prompt_chars", 1_000, 4_000_000),
    ("retained_generations", 2, 100),
    ("transient_retries", 0, 20),
    ("max_attempts_per_input", 1, 100),
    ("max_model_calls_per_run", 1, 1_000),
)


@dataclass(frozen=True)
class SceneConfig:
    """情景归组是否启用，以及参照边界、调用边界与留代数。"""

    enabled: bool = False
    # 参照边界：先前的事与"上一次"回看几天；待用前提多久未兑现视为过期（用户裁定初值 90 天）。
    lookback_days: int = 7
    pending_expiry_days: int = 90
    # 单次调用边界：DAY1（348 条）实测一次调用可完成；超过即跳过该日并留信号（第一期不切块）。
    max_occurrences_per_call: int = 400
    max_prompt_chars: int = 200_000
    # 每天保留几代情景文档（供回看与对照；下界 2，翻指针后立刻删旧代会撞读侧）。
    retained_generations: int = 3
    # 传输层瞬态错误的有界重试（路由层自己还有一层，这里保持很小）。
    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0
    # 同一输入连续失败几次后封锁那一天（输入变了自动解封）；一轮 sweep 内最多几次归组调用。
    max_attempts_per_input: int = 3
    max_model_calls_per_run: int = 4
    # 预测侧未启用时情景阶段自己的夜批节拍；预测侧启用时情景阶段排在预测重建之前、同一拍。
    refresh_interval_seconds: float = 86_400.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("scene.enabled must be a boolean")
        for name, lower, upper in _INT_BOUNDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"scene.{name} must be an integer between {lower} and {upper}")
        delay = self.transient_retry_delay_seconds
        if isinstance(delay, bool) or not isinstance(delay, int | float) or not 0.0 <= float(delay) <= 600.0:
            raise ValueError("scene.transient_retry_delay_seconds must be between 0 and 600")
        interval = self.refresh_interval_seconds
        if isinstance(interval, bool) or not isinstance(interval, int | float) or not 60.0 <= float(interval) <= 604_800.0:
            raise ValueError("scene.refresh_interval_seconds must be between 60 and 604800")

    @classmethod
    def from_mapping(cls, value: Any) -> SceneConfig:
        return construct_config(cls, value, "config.scene")


__all__ = ["SceneConfig"]
