"""时间预测树夜批的运行配置。

本模块只有纯标量、不 import ``prediction``（与 ``Config/behavior.py`` 同一纪律）：
取值范围与自洽约束的唯一出处仍是 ``prediction/config.py``，由 Runtime 组合根在装配时
构造 ``PredictionTreeConfig`` 并由它校验。这里只管"有没有给全"。

十三个参数**没有 Python 默认值**：它们全部要在我们自己的数据上定档（见
``TODO(PRED-TUNING-001)``），给缺省等于把"没有依据的初值"伪装成"合理缺省"。
预测侧整体可以不启用；一旦 ``enabled`` 为真，十三个必须在 YAML 里逐个写明。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from habitus.config.loader import ConfigError, construct_config

INTEGER_PARAMETERS = ("slot_minutes", "pool_half_width", "published_generations")

TREE_PARAMETERS = (
    "slot_minutes",
    "decay_half_life_days",
    "recent_half_life_days",
    "recurrence_half_life_days",
    "pool_half_width",
    "shrink_slot_to_pool",
    "shrink_pool_to_weekday",
    "shrink_weekday_to_all_day",
    "laplace_epsilon",
    "transition_window_seconds",
    "shrink_edge",
    "recurrence_window_days",
    "rebuild_interval_seconds",
    "published_generations",
)


@dataclass(frozen=True)
class PredictionConfig:
    """夜批是否启用，以及建树用的十三个参数。"""

    enabled: bool = False
    # 纯运维数，不进 TREE_PARAMETERS：它不影响树上的任何一个数字，也就不该逼 YAML 写死。
    worker_shutdown_timeout_seconds: float = 30.0
    slot_minutes: int | None = None
    decay_half_life_days: float | None = None
    recent_half_life_days: float | None = None
    recurrence_half_life_days: float | None = None
    pool_half_width: int | None = None
    shrink_slot_to_pool: float | None = None
    shrink_pool_to_weekday: float | None = None
    shrink_weekday_to_all_day: float | None = None
    laplace_epsilon: float | None = None
    transition_window_seconds: float | None = None
    shrink_edge: float | None = None
    recurrence_window_days: float | None = None
    rebuild_interval_seconds: float | None = None
    published_generations: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("prediction.enabled must be a boolean")
        if (
            isinstance(self.worker_shutdown_timeout_seconds, bool)
            or not isinstance(self.worker_shutdown_timeout_seconds, int | float)
            or not 1.0 <= float(self.worker_shutdown_timeout_seconds) <= 3_600.0
        ):
            raise ValueError(
                "prediction.worker_shutdown_timeout_seconds must be between 1 and 3600"
            )
        for field in fields(self):
            if field.name in {"enabled", "worker_shutdown_timeout_seconds"}:
                continue
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"prediction.{field.name} must be a number")
            # **类型**属于"给没给全"的范畴，归配置层；**取值范围**属于领域，归
            # PredictionTreeConfig。写 slot_minutes: 15.5 应当在加载期就是 ConfigError，
            # 而不是拖到组合根才以领域错误炸出来。
            if field.name in INTEGER_PARAMETERS and not isinstance(value, int):
                raise ConfigError(f"config.prediction.{field.name} must be an integer")
        if not self.enabled:
            return
        missing = [name for name in TREE_PARAMETERS if getattr(self, name) is None]
        if missing:
            raise ConfigError(
                "config.prediction is enabled but leaves these parameters unset: "
                + ", ".join(missing)
            )

    def tree_parameters(self) -> dict[str, Any]:
        """按 ``PredictionTreeConfig`` 的字段名交出十三个值；未启用时调用即错。

        标注成 ``Any`` 而不是 ``float | int``：这十三个值是异构的（槽宽是 int、半衰期是
        float），拿一个联合类型去 ``**`` 展开，类型检查器只会按最宽的那个成员去匹配每一个
        形参。真正的类型与范围校验在 ``PredictionTreeConfig`` 的构造里，那里才是唯一出处。
        """

        if not self.enabled:
            raise ConfigError("config.prediction is disabled and carries no tree parameters")
        return {name: getattr(self, name) for name in TREE_PARAMETERS}

    @classmethod
    def from_mapping(cls, value: object) -> PredictionConfig:
        return construct_config(cls, value, "config.prediction")


__all__ = ["INTEGER_PARAMETERS", "TREE_PARAMETERS", "PredictionConfig"]
