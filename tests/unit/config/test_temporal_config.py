"""时间尺子只此一处（裁定 59）：槽宽整除一天、合并窗口是槽的整数倍；预测树还在时它的槽必须和这里一致。"""

from __future__ import annotations

import pytest

from habitus.config import ConfigError, HabitusConfig, TemporalConfig
from tests.unit.config.test_config_loading import valid_mapping


def test_defaults_and_derived_rulers() -> None:
    config = TemporalConfig()
    assert (config.slot_minutes, config.merge_window_minutes, config.pool_half_width) == (15, 45, 3)
    assert config.merge_window_seconds == 2700
    assert config.transition_window_slots == 3
    assert config.min_peak_mass == 0.10


@pytest.mark.parametrize(
    "overrides",
    [
        {"slot_minutes": 7},  # 不整除一天
        {"merge_window_minutes": 40},  # 不是槽的整数倍
        {"pool_half_width": 0},
        {"shrink_slot_to_pool": -1.0},
        {"min_peak_mass": 1.0},
        {"slot_minutes": True},
    ],
)
def test_bad_rulers_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        TemporalConfig(**overrides)  # type: ignore[arg-type]


def test_the_prediction_section_is_gone_and_temporal_is_the_only_clock(tmp_path) -> None:
    """预测树删了（裁定 59，清单 S6）：配置里写 ``prediction`` 是未知的一段，硬拒；槽宽只有 ``temporal`` 一份，改了照样能装。"""

    payload = valid_mapping(tmp_path)
    payload["temporal"] = {"slot_minutes": 30, "merge_window_minutes": 90}
    assert HabitusConfig.from_mapping(payload).temporal.slot_minutes == 30
    payload["prediction"] = {"enabled": False}
    with pytest.raises(ConfigError, match="prediction"):
        HabitusConfig.from_mapping(payload)


def test_example_yaml_carries_the_temporal_section(tmp_path) -> None:
    config = HabitusConfig.from_mapping(valid_mapping(tmp_path))
    assert config.temporal == TemporalConfig()
    assert config.foresight.half_life_opportunities == 8.0
    assert config.scene.baseline_slack_slots == 3
