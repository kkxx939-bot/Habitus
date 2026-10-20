"""证据的两个值（裁定 49、57）：α 借关系检验前向验证的显著性，遗忘的半衰期是几次机会（``config.foresight.half_life_opportunities``）。
窗口 M = ⌈H · log₂(1/α)⌉ 由它们算出：账的读数、行为单元、temporal 的时段读数都用这一份，只在这里拼一次。"""

from __future__ import annotations

from habitus.config import HabitusConfig
from habitus.foresight import Evidence
from habitus.runtime.scene_night import relation_config


def evidence_of(config: HabitusConfig) -> Evidence:
    return Evidence(
        alpha=relation_config(config).thresholds.forward_alpha,
        half_life=config.foresight.half_life_opportunities,
    )


__all__ = ["evidence_of"]
