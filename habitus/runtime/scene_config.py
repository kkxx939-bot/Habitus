"""语义树关系检验的配置换算：组合根里三处要用（语义树夜批、预测层的证据、启动校验），只在这里拼一次。"""

from __future__ import annotations

from habitus.config import HabitusConfig
from habitus.config.loader import construct_overrides
from habitus.foresight.strength import MIN_HISTORY
from habitus.scene.relations import RelationConfig, RelationThresholds


def relation_config(config: HabitusConfig) -> RelationConfig:
    """关系检验的配置：槽宽与"同一条链"的槽数只认 ``config.temporal``（裁定 59）；统计门槛见 ``relation_thresholds``。
    久别重来不再借固定天数窗，按 temporal 的逐次判法。关系行退休的"久没来"与账的久别同一把尺：H 取预测层的遗忘半衰期、
    最少间隔数取 ``foresight.strength.MIN_HISTORY``，不另开 YAML 项。"""

    return RelationConfig(
        slot_minutes=config.temporal.slot_minutes,
        transition_window_slots=config.temporal.transition_window_slots,
        absence_typical_multiple=config.foresight.half_life_opportunities,
        absence_min_gaps=MIN_HISTORY,
        thresholds=relation_thresholds(config),
    )


def relation_thresholds(config: HabitusConfig) -> RelationThresholds:
    """统计门槛取 ``config.scene.relations``，没写的用 ``RelationThresholds`` 的默认值；值域只由它校验，拒了报配置键。

    ``build_runtime`` 不论语义树开没开都先调一次：写错的门槛在启动时拒，不因为语义树没开就被静默收下。"""

    return construct_overrides(RelationThresholds, config.scene.relations.overrides(), path="config.scene.relations")


__all__ = ["relation_config", "relation_thresholds"]
