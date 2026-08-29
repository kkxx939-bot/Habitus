"""预测夜批的组合根接线：可选启用、参数注入、手动触发一次全量重建。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from behavior import BehaviorDocumentWriter
from behavior.model import BehaviorKind
from behavior.tree import BehaviorTree
from Config import HabitusConfig
from Config.loader import ConfigError
from Config.prediction import PredictionConfig
from infrastructure.store.locks import ProcessLocalLockStore
from prediction.config import PredictionTreeConfig
from prediction.errors import PredictionTreeError
from Runtime.prediction import build_prediction_components
from tests.integration.test_runtime_assembly import REPOSITORY_ROOT
from tests.unit.behavior.tree_payloads import occurrence_payload
from tests.unit.runtime.test_behavior_pipeline import SUBJECT, behavior_enabled_config

CST = timezone(timedelta(hours=8))
STARTUP_PARAMETERS = {
    "enabled": True,
    "slot_minutes": 15,
    "decay_half_life_days": 60,
    "recent_half_life_days": 14,
    "recurrence_half_life_days": 365,
    "pool_half_width": 2,
    "shrink_slot_to_pool": 5,
    "shrink_pool_to_weekday": 5,
    "shrink_weekday_to_all_day": 5,
    "laplace_epsilon": 0.5,
    "transition_window_seconds": 1800,
    "shrink_edge": 5,
    "recurrence_window_days": 90,
    "rebuild_interval_seconds": 86400,
    "published_generations": 7,
}


def prediction_enabled_config(tmp_path: Path) -> HabitusConfig:
    """在行为侧已启用的 fake 路由配置之上，再把预测夜批的十三个启动档填进去。"""

    raw = yaml.safe_load((REPOSITORY_ROOT / "Config" / "example.yaml").read_text(encoding="utf-8"))
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
    return HabitusConfig.from_mapping(raw)


def test_prediction_is_absent_until_it_is_switched_on(tmp_path) -> None:
    config = behavior_enabled_config(tmp_path)
    assert config.prediction.enabled is False
    assert build_prediction_components(config, behavior_tree=_tree(config)) is None


def test_enabled_prediction_requires_every_tree_parameter() -> None:
    with pytest.raises(ConfigError, match="slot_minutes"):
        PredictionConfig.from_mapping({"enabled": True, "decay_half_life_days": 60})


def test_startup_parameters_survive_into_the_tree_config(tmp_path) -> None:
    """YAML 里的启动档必须原样落到 PredictionTreeConfig；范围校验在那里做，不在配置层重写一遍。"""

    config = prediction_enabled_config(tmp_path)
    components = build_prediction_components(config, behavior_tree=_tree(config))
    assert components is not None
    assert components.tree_config.slot_minutes == 15
    assert components.tree_config.transition_window_seconds == 1800
    assert components.store.retained_generations == 7
    assert components.store.root == config.prediction_root


def test_a_self_contradictory_parameter_set_fails_at_assembly() -> None:
    """短窗不比长窗短时，"趋势"比的是两个同样的东西。

    这类自洽校验只写在 ``PredictionTreeConfig`` 一处，配置层不重写一遍——两处各写一份，
    迟早只改一处。
    """

    contradictory = PredictionConfig(**{**STARTUP_PARAMETERS, "recent_half_life_days": 60})
    with pytest.raises(PredictionTreeError, match="shorter than"):
        PredictionTreeConfig(**contradictory.tree_parameters())


def test_manual_rebuild_publishes_a_generation_from_the_behaviour_tree(tmp_path) -> None:
    """手动触发一次全量重建：行为树上有东西就该出一代，读回来能查。"""

    config = prediction_enabled_config(tmp_path)
    behavior_tree = _tree(config)
    _publish_occurrences(behavior_tree, days=10)
    # 固定时钟：_calendar_days 的跨度是"最早记录 → 基准日"，跟着真实时间走会让这个测试
    # 随日期漂移（今天绿、半年后越跑越慢）。
    components = build_prediction_components(
        config, behavior_tree=behavior_tree, clock=lambda: datetime(2026, 8, 16, 23, tzinfo=CST)
    )
    assert components is not None

    published = asyncio.run(components.worker.run_once())
    assert published is not None
    assert components.store.active() == published
    tree = components.store.load()
    assert tree is not None
    assert "洗手" in tree.actions


def test_an_empty_behaviour_tree_publishes_nothing(tmp_path) -> None:
    """发布一棵空树会让读侧把"还没有数据"误当成"什么都不会发生"。"""

    config = prediction_enabled_config(tmp_path)
    behavior_tree = _tree(config)
    behavior_tree.initialize()
    components = build_prediction_components(config, behavior_tree=behavior_tree)
    assert components is not None
    assert asyncio.run(components.worker.run_once()) is None
    assert components.store.active() is None


def _tree(config: HabitusConfig) -> BehaviorTree:
    return BehaviorTree(config.behavior_root / "tree")


def _publish_occurrences(tree: BehaviorTree, *, days: int) -> None:
    clock = datetime(2026, 8, 16, 23, 0, tzinfo=CST)
    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: clock)
    for offset in range(days):
        started_at = datetime(2026, 8, 1, 12, 0, tzinfo=CST) + timedelta(days=offset)
        writer.publish(
            BehaviorKind.OCCURRENCE,
            occurrence_payload(
                occurred_on=started_at.date(),
                started_at=started_at,
                last_observed_at=started_at + timedelta(minutes=1),
                onset_available_at=started_at + timedelta(seconds=2),
                basis=(),
                goal=None,
            ),
        )
