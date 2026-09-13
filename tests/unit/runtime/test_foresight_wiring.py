"""预测层的组合根接线：可选启用、钉住一代、此刻按主体时区落钟面、窗口只有一处出处。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.config import HabitusConfig
from habitus.config.loader import ConfigError
from habitus.foresight import ForesightError
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.runtime.foresight import build_foresight_components
from habitus.runtime.prediction import build_prediction_components
from habitus.scene import SceneTree
from tests.integration.test_runtime_assembly import REPOSITORY_ROOT
from tests.unit.behavior.tree_payloads import occurrence_payload
from tests.unit.runtime.test_behavior_pipeline import SUBJECT
from tests.unit.runtime.test_prediction_wiring import STARTUP_PARAMETERS

CST = timezone(timedelta(hours=8))
FIRST = datetime(2026, 8, 3, tzinfo=CST)  # 周一
EVENING = FIRST + timedelta(days=21, hours=19, minutes=5)


def raw_config(tmp_path: Path, *, scene: bool = True, foresight: bool = True, **overrides: Any) -> dict[str, Any]:
    """example.yaml 之上把路由换成 fake，并把行为 / 预测 / 情景 / 预测层四组填满。"""

    raw = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    raw["storage"]["root"] = str(tmp_path / "data")
    for route, adapter in (
        (raw["models"]["chat"]["route"], "fake_chat"),
        (raw["models"]["embedding"]["route"], "fake_embedding"),
        (raw["models"]["rerank"]["route"], "fake_rerank"),
        (raw["memory"]["vector_store"]["route"], "fake_vector"),
        (raw["conversation"]["summary_vector_store"]["route"], "fake_vector"),
    ):
        route.update(provider="fake", adapter=adapter, credential_ref="")
    raw["behavior"] = {"primary_subject": SUBJECT}
    raw["prediction"] = dict(STARTUP_PARAMETERS)
    raw["scene"] = {"enabled": scene}
    raw["foresight"] = {"enabled": foresight, **overrides}
    return raw


def trees(config: HabitusConfig, *, weeks: int = 4) -> tuple[BehaviorTree, SceneTree]:
    """一棵有"每周一晚上打球"的行为树，外加一棵（尚未归组的）情景树。"""

    behavior_tree = BehaviorTree(config.behavior_root / "tree")
    writer = BehaviorDocumentWriter(
        behavior_tree, ProcessLocalLockStore(), clock=lambda: FIRST + timedelta(days=7 * weeks)
    )
    for week in range(weeks):
        started = FIRST + timedelta(days=7 * week, hours=19)
        writer.publish(
            BehaviorKind.OCCURRENCE,
            occurrence_payload(
                occurred_on=started.date(),
                name="打球",
                kind_token="打球",
                started_at=started,
                last_observed_at=started + timedelta(minutes=60),
                onset_available_at=started + timedelta(seconds=2),
                basis=(),
                goal=None,
            ),
        )
    return behavior_tree, SceneTree(config.scene_root / "tree")


def assembled(tmp_path: Path, *, now: datetime = EVENING):
    """走完真实顺序：行为树 → 发布一代预测树 → 组装预测层。"""

    config = HabitusConfig.from_mapping(raw_config(tmp_path))
    behavior_tree, scene_tree = trees(config)
    prediction = build_prediction_components(config, behavior_tree=behavior_tree, clock=lambda: now)
    assert prediction is not None
    asyncio.run(prediction.worker.run_once())
    components = build_foresight_components(
        config, behavior_tree=behavior_tree, scene_tree=scene_tree, store=prediction.store, clock=lambda: now
    )
    assert components is not None
    return config, components


def test_foresight_is_absent_until_it_is_switched_on(tmp_path) -> None:
    config = HabitusConfig.from_mapping(raw_config(tmp_path, foresight=False))
    behavior_tree, scene_tree = trees(config)
    assert config.foresight.enabled is False
    assert build_foresight_components(
        config, behavior_tree=behavior_tree, scene_tree=scene_tree, store=None
    ) is None


def test_enabling_foresight_without_the_scene_tree_is_refused(tmp_path) -> None:
    """配置自相矛盾要在启动时炸：数字取自预测树、与之对应的历史取自情景树，缺一边装配不出证据。

    放过去的下场是启动一切正常、健康面报 foresight 已启用、什么都没发生、无处可查。
    """

    with pytest.raises(ConfigError, match="config.foresight is enabled"):
        HabitusConfig.from_mapping(raw_config(tmp_path, scene=False))


def test_the_window_has_one_source_the_prediction_tree(tmp_path) -> None:
    """窗口宽度与转移窗只从预测树的参数来，预测层这一组只有保护闸。

    配两份的下场是数字按 ±3 槽算、背景按另一个宽度取，两边都对不上却都不报错。
    """

    config, components = assembled(tmp_path)
    assembler = components.assembler
    assert assembler.half_width == config.prediction.pool_half_width == 3
    assert assembler.transition_window_seconds == config.prediction.transition_window_seconds
    assert assembler.max_days_per_layer == config.foresight.max_days_per_layer
    assert assembler.window_days == config.scene.lookback_days


def test_the_moment_lands_on_the_clock_face_in_the_subject_timezone(tmp_path) -> None:
    """传 UTC 进来也要先换回主体本地时刻再落钟面——错一个时区就整体错一天、错一个槽。"""

    _config, components = assembled(tmp_path)
    evidence = components.assembler.assemble(["打球"], now=EVENING.astimezone(UTC))
    assert len(evidence) == 1
    moment = evidence[0].moment
    assert moment.at.utcoffset() == timedelta(hours=8)
    assert (moment.weekday, moment.slot) == (0, 76)  # 周一 19:00–19:15
    assert moment.day_note is None  # 没有当地日历数据时是显式的空，不是漏了字段
    assert [item.layer.name for item in evidence[0].layers] == ["slot", "pool", "cross_weekday", "all_day"]
    # 情景树一天都没归组：四层都"有数、没背景"，如实说出来而不是让背景默默空着。
    assert evidence[0].ungrouped == evidence[0].layers[3].layer.days
    assert all(item.views == () for item in evidence[0].layers)


def test_a_generation_built_with_other_parameters_is_refused(tmp_path) -> None:
    """跨参数混读永不允许：组合出来的数字互不一致，而且看不出来。"""

    _config, components = assembled(tmp_path)
    components.assembler.expected_digest = "另一套参数"
    with pytest.raises(ForesightError, match="different estimation"):
        components.assembler.assemble(["打球"], now=EVENING)


def test_without_a_published_generation_it_says_so_instead_of_answering_zero(tmp_path) -> None:
    """"还没算过"与"什么都不会发生"必须分得清——后者会让上层安心闭嘴。"""

    config = HabitusConfig.from_mapping(raw_config(tmp_path))
    behavior_tree, scene_tree = trees(config)
    prediction = build_prediction_components(config, behavior_tree=behavior_tree)
    assert prediction is not None
    components = build_foresight_components(
        config, behavior_tree=behavior_tree, scene_tree=scene_tree, store=prediction.store
    )
    assert components is not None
    with pytest.raises(ForesightError, match="no prediction generation"):
        components.assembler.assemble(["打球"], now=EVENING)
