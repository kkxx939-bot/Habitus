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
    # 融合覆盖索引 / 回执 / 消费账本的过期窗口（天）：等于上游可能补发多久以前的观测。
    # 上游契约未定前取 7 天；它只影响"重投去重"，不影响树上的数据。
    coverage_window_days: int = 7
    # 一次融合最多接受的片段数。逐帧归属表按片段数线性增长、判断本体按判断数增长；在 8k 输出
    # 预算下 512/160/100 条的段实测都会截断，60 条才稳（BHV-REALDATA-001 第 6 条）。
    max_fragments_per_segment: int = 60
    # 行为类型词表（BHV-KINDS-002）：留空取 ``behavior/kinds/config.py`` 的默认值。
    # - 批形状：一次归一调用判几个未知名字；每个名字给多少候选（向量最近邻 ∪ 命中天数最多的）。
    # - 存活期：``last_hit_day + max(base_days, gap_multiplier × 自己量出的最长命中间隔)``——
    #   一次性名字基础期后删，周期行为按自己的节奏续命；过期按数据时钟（最新行为日）判。
    # - 容量：防失控的护栏；WP4 之后一天二百多个不同名字、大多跨天重复。
    kinds_batch_size: int | None = None
    kinds_vector_candidates: int | None = None
    kinds_frequent_candidates: int | None = None
    kinds_base_days: int | None = None
    kinds_gap_multiplier: int | None = None
    kinds_max_kinds: int | None = None
    kinds_max_aliases_per_kind: int | None = None

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
        for name, upper in (
            ("kinds_batch_size", 100),
            ("kinds_vector_candidates", 500),
            ("kinds_frequent_candidates", 500),
            ("kinds_base_days", 3_650),
            ("kinds_gap_multiplier", 100),
            ("kinds_max_kinds", 1_000_000),
            ("kinds_max_aliases_per_kind", 100_000),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= upper:
                raise ValueError(f"behavior.{name} must be an integer between 0 and {upper}")
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

    def kinds_overrides(self) -> dict[str, int]:
        """非空的 ``kinds_*`` 字段 → ``BehaviorKindConfig`` 的同名字段（去掉前缀）。"""

        pairs = (
            ("batch_size", self.kinds_batch_size),
            ("vector_candidates", self.kinds_vector_candidates),
            ("frequent_candidates", self.kinds_frequent_candidates),
            ("base_days", self.kinds_base_days),
            ("gap_multiplier", self.kinds_gap_multiplier),
            ("max_kinds", self.kinds_max_kinds),
            ("max_aliases_per_kind", self.kinds_max_aliases_per_kind),
        )
        return {name: value for name, value in pairs if value is not None}

    @classmethod
    def from_mapping(cls, value: object) -> BehaviorConfig:
        return construct_config(cls, value, "config.behavior")


__all__ = ["BehaviorConfig"]
