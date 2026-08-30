"""行为管线（观测 → 融合 → 归约 → 行为树）的运行配置。

本模块只有纯标量、不 import ``behavior``：上下文窗口的**默认值**唯一出处仍是
``behavior/fusion/config.py``（此处留 ``None``，由 Runtime 组合根解析）——组合根用同一份
数值喂融合与归约两处，这正是 BHV-FUSION-003 点名要关死的分叉之门。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from Config.loader import construct_config

# (字段, 下界, 上界)：与 behavior/kinds/config.py 的校验一致。
_KINDS_INT_BOUNDS = (
    ("kinds_batch_size", 1, 100),
    ("kinds_vector_candidates", 0, 500),
    ("kinds_frequent_candidates", 0, 500),
    ("kinds_literal_candidates", 0, 500),
    ("kinds_validation_rounds", 0, 10),
    ("kinds_base_days", 1, 3_650),
    ("kinds_gap_multiplier", 1, 100),
    ("kinds_max_kinds", 1, 1_000_000),
    ("kinds_max_aliases_per_kind", 1, 100_000),
    ("kinds_max_encoded_bytes", 1, 256 * 1024 * 1024),
    ("kinds_transient_retries", 0, 20),
)


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
    # 融合覆盖索引 / 回执 / 消费账本的过期窗口（天）：等于上游可能补发多久以前的观测。
    # 上游契约未定前取 7 天。它同时是：重投去重的判据、入队与封口前沿的时间下界（送达早于窗口的
    # 观测不再入队、不拖前沿）、覆盖记录/回执/账本的过期期限。树上的数据不受它影响（撞车消歧的
    # 占用集合以树上既有地址为最终真相，不依赖账本窗口）。
    coverage_window_days: int = 7
    # 一次融合最多接受的片段数。逐帧归属表按片段数线性增长、判断本体按判断数增长；在 8k 输出
    # 预算下 512/160/100 条的段实测都会截断，60 条才稳（BHV-REALDATA-001 第 6 条）。
    max_fragments_per_segment: int = 60
    # 行为类型词表（BHV-KINDS-002）：全部可调字段，留空取 ``behavior/kinds/config.py`` 的默认值
    #（默认值只在领域模块）。字段名 = ``kinds_`` + BehaviorKindConfig 的同名字段，组合根按前缀派生注入。
    kinds_batch_size: int | None = None
    kinds_vector_candidates: int | None = None
    kinds_frequent_candidates: int | None = None
    kinds_literal_candidates: int | None = None
    kinds_validation_rounds: int | None = None
    kinds_base_days: int | None = None
    kinds_gap_multiplier: int | None = None
    kinds_max_kinds: int | None = None
    kinds_max_aliases_per_kind: int | None = None
    kinds_max_encoded_bytes: int | None = None
    kinds_transient_retries: int | None = None
    kinds_transient_retry_delay_seconds: float | None = None

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
        if (
            isinstance(self.coverage_window_days, bool)
            or not isinstance(self.coverage_window_days, int)
            or not 1 <= self.coverage_window_days <= 365
        ):
            raise ValueError("behavior.coverage_window_days must be between 1 and 365")
        if (
            isinstance(self.max_fragments_per_segment, bool)
            or not isinstance(self.max_fragments_per_segment, int)
            or not 1 <= self.max_fragments_per_segment <= 4096
        ):
            raise ValueError("behavior.max_fragments_per_segment must be between 1 and 4096")
        # 下界与 behavior/kinds/config.py 一致：非法值在配置层就拒，不让它拖到组装期以裸异常爆。
        for name, lower, upper in _KINDS_INT_BOUNDS:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"behavior.{name} must be an integer between {lower} and {upper}")
        delay = self.kinds_transient_retry_delay_seconds
        if delay is not None and (
            isinstance(delay, bool) or not isinstance(delay, int | float) or not 0.0 <= float(delay) <= 600.0
        ):
            raise ValueError("behavior.kinds_transient_retry_delay_seconds must be between 0 and 600")
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

    def kinds_overrides(self) -> dict[str, Any]:
        """非空的 ``kinds_*`` 字段 → ``BehaviorKindConfig`` 的同名字段（按前缀派生，不双份维护）。"""

        return {
            field.name[len("kinds_") :]: getattr(self, field.name)
            for field in fields(self)
            if field.name.startswith("kinds_") and getattr(self, field.name) is not None
        }

    @classmethod
    def from_mapping(cls, value: object) -> BehaviorConfig:
        return construct_config(cls, value, "config.behavior")


__all__ = ["BehaviorConfig"]
