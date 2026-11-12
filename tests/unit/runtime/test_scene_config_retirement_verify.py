"""组合根给关系退休的两把尺（实时化阶段 3 独立验证；共享说明 3.2）：H 取 ``config.foresight.half_life_opportunities``、
最少间隔数取 ``foresight.strength.MIN_HISTORY``，不另开 YAML 项。"""

from __future__ import annotations

from pathlib import Path

from habitus.config import HabitusConfig
from habitus.foresight.strength import MIN_HISTORY
from habitus.runtime.scene_config import relation_config
from tests.unit.config.test_config_loading import valid_mapping


def test_relation_config_takes_the_absence_rulers_from_the_foresight_layer(tmp_path: Path) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["behavior"] = {"primary_subject": "用户"}
    mapping["foresight"] = {"half_life_opportunities": 13}
    config = HabitusConfig.from_mapping(mapping)
    built = relation_config(config)
    assert built.absence_typical_multiple == config.foresight.half_life_opportunities == 13
    assert built.absence_min_gaps == MIN_HISTORY == 2


def test_relation_config_follows_the_default_half_life(tmp_path: Path) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["behavior"] = {"primary_subject": "用户"}
    config = HabitusConfig.from_mapping(mapping)
    assert relation_config(config).absence_typical_multiple == config.foresight.half_life_opportunities
