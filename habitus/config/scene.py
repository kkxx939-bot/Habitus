"""语义关联层的运行配置。

本模块只有纯标量、不 import ``scene``（与 ``config/behavior.py`` 同一纪律）。

旧语义树（按候选累积的关联）那一组参数——提示词字符上限、重试、每轮调用数、每批目标数、前因/前提
行数预算——随它整块删掉（2026-09-26）。新语义树按功能分支重建，每支的旋钮在那一支落地时再进配置。

``relations``：关系检验的统计门槛（裁定 26）。每一项都可以不写，不写就是 ``scene.relations.config.RelationThresholds``
的默认值（组合根只把写了的换算过去，默认值不在这里抄第二份）。和时间有关的三个数（槽宽、转移窗口、久别重来）不在这一组：
它们只认预测树的配置，语义树不另配一份。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from habitus.config.loader import construct_config

# (字段, 下界, 上界)：与 scene/relations/config.py 的校验一致。
_PROBABILITIES = ("fdr", "forward_alpha", "forward_power", "heterogeneity_alpha", "interval_level")
_LIFTS = ("min_absolute_lift", "min_relative_lift")
_COUNTS = (
    "forward_min_antecedents",
    "forward_max_antecedents",
    "maintenance_blocks",
    "bootstrap_rounds",
    "exact_strata_limit",
    "first_seen_min_days",
    "first_seen_min_blocks",
    "moderation_min_antecedents",
)


@dataclass(frozen=True)
class SceneRelationsConfig:
    """关系检验的统计门槛覆盖值；``None`` = 用默认值。"""

    fdr: float | None = None
    forward_min_antecedents: int | None = None
    forward_alpha: float | None = None
    forward_power: float | None = None
    forward_max_antecedents: int | None = None
    min_absolute_lift: float | None = None
    min_relative_lift: float | None = None
    maintenance_blocks: int | None = None
    heterogeneity_alpha: float | None = None
    interval_level: float | None = None
    bootstrap_rounds: int | None = None
    exact_strata_limit: int | None = None
    first_seen_min_days: int | None = None
    first_seen_min_blocks: int | None = None
    moderation_min_antecedents: int | None = None
    max_blind_share: float | None = None

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
        if share is not None and (isinstance(share, bool) or not isinstance(share, int | float) or not 0.0 < share <= 1.0):
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
    """语义关联层是否启用，以及关系检验的统计门槛。"""

    enabled: bool = False
    relations: SceneRelationsConfig = field(default_factory=SceneRelationsConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("scene.enabled must be a boolean")
        if not isinstance(self.relations, SceneRelationsConfig):
            raise TypeError("scene.relations must be SceneRelationsConfig")

    @classmethod
    def from_mapping(cls, value: Any) -> SceneConfig:
        data = dict(value) if isinstance(value, dict) else value
        if isinstance(data, dict) and "relations" in data:
            data["relations"] = construct_config(SceneRelationsConfig, data["relations"], "config.scene.relations")
        return construct_config(cls, data, "config.scene")


__all__ = ["SceneConfig", "SceneRelationsConfig"]
