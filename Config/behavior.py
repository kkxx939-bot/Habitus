"""行为管线（观测 → 融合 → 归约 → 行为树）的运行配置。

本模块只有纯标量、不 import ``behavior``：上下文窗口的**默认值**唯一出处仍是
``behavior/fusion/config.py``（此处留 ``None``，由 Runtime 组合根解析）——组合根用同一份
数值喂融合与归约两处，这正是 BHV-FUSION-003 点名要关死的分叉之门。
"""

from __future__ import annotations

from dataclasses import dataclass

from Config.loader import construct_config


@dataclass(frozen=True)
class BehaviorConfig:
    """行为侧的产品事实与运行节奏。

    ``primary_subject`` 是这套系统跟踪谁的**产品事实**（融合分流、单主体边界都以它为准）；
    留空表示行为侧未启用——上游感知（云侧行为 agent）接入前的合法状态，组合根将完全跳过
    行为组件的组装与 Worker 启动。
    """

    primary_subject: str = ""
    context_limit: int | None = None
    context_lookback_seconds: float | None = None
    # 归约 sweep 节奏（用户裁定 5 分钟）与融合队列的空转轮询间隔；纯运维数，不影响正确性。
    # 轮询下限 1 秒：每拍都要做一次观测存储的全量扫描（BHV-LIFECYCLE-001 欠账），更密的节拍
    # 只会放大扫描成本。Worker 优雅停止的等待上界也在这里（运维参数不藏 Python 构造器）。
    reduction_sweep_interval_seconds: float = 300.0
    fusion_poll_interval_seconds: float = 5.0
    worker_shutdown_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.primary_subject, str):
            raise TypeError("behavior.primary_subject must be a string")
        if self.context_limit is not None and (
            isinstance(self.context_limit, bool)
            or not isinstance(self.context_limit, int)
            or not 1 <= self.context_limit <= 64
        ):
            raise ValueError("behavior.context_limit must be between 1 and 64")
        if self.context_lookback_seconds is not None and (
            isinstance(self.context_lookback_seconds, bool)
            or not isinstance(self.context_lookback_seconds, int | float)
            or not 60.0 <= float(self.context_lookback_seconds) <= 86_400.0
        ):
            raise ValueError(
                "behavior.context_lookback_seconds must be between 60 and 86400"
            )
        for name, value, minimum, maximum in (
            ("reduction_sweep_interval_seconds", self.reduction_sweep_interval_seconds, 1.0, 3_600.0),
            ("fusion_poll_interval_seconds", self.fusion_poll_interval_seconds, 1.0, 600.0),
            ("worker_shutdown_timeout_seconds", self.worker_shutdown_timeout_seconds, 1.0, 3_600.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not minimum <= float(value) <= maximum
            ):
                raise ValueError(
                    f"behavior.{name} must be between {minimum:g} and {maximum:g}"
                )

    @property
    def enabled(self) -> bool:
        return bool(self.primary_subject.strip())

    @classmethod
    def from_mapping(cls, value: object) -> BehaviorConfig:
        return construct_config(cls, value, "config.behavior")


__all__ = ["BehaviorConfig"]
