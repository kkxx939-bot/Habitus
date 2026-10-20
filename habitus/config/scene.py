"""语义关联层的运行配置。

本模块只有纯标量、不 import ``scene``（与 ``config/behavior.py`` 同一纪律）。

旧语义树（按候选累积的关联）那一组参数——提示词字符上限、重试、每轮调用数、每批目标数、前因/前提
行数预算——随它整块删掉（2026-09-26）。新语义树按功能分支重建，每支的旋钮在那一支落地时再进配置。

``relations``：关系检验的统计门槛（裁定 26）。每一项都可以不写，不写就是 ``scene.relations.config.RelationThresholds``
的默认值（组合根只把写了的换算过去，默认值不在这里抄第二份）。和时间有关的尺子（槽宽、合并窗口）不在这一组：
它们只认 ``config.temporal``（裁定 59），语义树不另配一份。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from habitus.config.loader import construct_config

# (字段, 下界, 上界)：与 scene/relations/config.py 的校验一致。
_PROBABILITIES = ("fdr", "forward_alpha", "forward_power", "heterogeneity_alpha", "interval_level")
_LIFTS = ("min_absolute_lift", "min_relative_lift")
_COUNTS = (
    "forward_max_antecedents",
    "bootstrap_rounds",
    "exact_strata_limit",
    "first_seen_min_days",
    "first_seen_min_blocks",
    "condition_ask_min_antecedents",
    "max_combined_conditions",
)


@dataclass(frozen=True)
class SceneRelationsConfig:
    """关系检验的统计门槛覆盖值；``None`` = 用默认值。"""

    fdr: float | None = None
    forward_alpha: float | None = None
    forward_power: float | None = None
    forward_max_antecedents: int | None = None
    min_absolute_lift: float | None = None
    min_relative_lift: float | None = None
    heterogeneity_alpha: float | None = None
    interval_level: float | None = None
    bootstrap_rounds: int | None = None
    exact_strata_limit: int | None = None
    first_seen_min_days: int | None = None
    first_seen_min_blocks: int | None = None
    condition_ask_min_antecedents: int | None = None
    max_blind_share: float | None = None
    max_combined_conditions: int | None = None

    def __post_init__(self) -> None:
        # 非法值在配置层就拒，不让它拖到组装期以裸异常爆。
        for name in _PROBABILITIES:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < value < 1.0
            ):
                raise ValueError(f"scene.relations.{name} must lie strictly between 0 and 1")
        for name in _LIFTS:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value < 1.0
            ):
                raise ValueError(f"scene.relations.{name} must lie in [0, 1)")
        share = self.max_blind_share
        if share is not None and (
            isinstance(share, bool) or not isinstance(share, int | float) or not 0.0 < share <= 1.0
        ):
            raise ValueError("scene.relations.max_blind_share must lie in (0, 1]")
        for name in _COUNTS:
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"scene.relations.{name} must be a positive integer")

    def overrides(self) -> dict[str, Any]:
        """写了的字段 → ``RelationThresholds`` 的同名字段。"""

        return {item.name: getattr(self, item.name) for item in fields(self) if getattr(self, item.name) is not None}


@dataclass(frozen=True)
class SceneConfig:
    """语义关联层是否启用、关系检验的统计门槛，以及整晚那一趟的节拍（语义树没开就没有整晚，所以节拍归这里）。"""

    enabled: bool = False
    relations: SceneRelationsConfig = field(default_factory=SceneRelationsConfig)
    # 常态按峰分份时，峰的两边各放宽几槽的容差（待定档）。原来借预测树的 pool_half_width，预测树删除后归语义树自己（裁定 59）。
    baseline_slack_slots: int = 3
    # 整晚那一趟多久跑一次、停机时等它多久（运维数）。原来挂在预测树的重建节拍上，预测树删除后归这里（裁定 59，清单 5.1）。
    nightly_interval_seconds: float = 86_400.0
    nightly_shutdown_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("scene.enabled must be a boolean")
        if (
            isinstance(self.baseline_slack_slots, bool)
            or not isinstance(self.baseline_slack_slots, int)
            or not 0 <= self.baseline_slack_slots <= 96
        ):
            raise ValueError("scene.baseline_slack_slots must be an integer between 0 and 96")
        if not isinstance(self.relations, SceneRelationsConfig):
            raise TypeError("scene.relations must be SceneRelationsConfig")
        for label, low, high in (
            ("nightly_interval_seconds", 60.0, 7 * 86_400.0),
            ("nightly_shutdown_timeout_seconds", 1.0, 3_600.0),
        ):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int | float) or not low <= float(value) <= high:
                raise ValueError(f"scene.{label} must be between {low:g} and {high:g}")

    @classmethod
    def from_mapping(cls, value: Any) -> SceneConfig:
        data = dict(value) if isinstance(value, dict) else value
        if isinstance(data, dict) and "relations" in data:
            data["relations"] = construct_config(SceneRelationsConfig, data["relations"], "config.scene.relations")
        return construct_config(cls, data, "config.scene")


__all__ = ["SceneConfig", "SceneRelationsConfig"]
