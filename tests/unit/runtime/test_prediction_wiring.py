"""预测夜批的组合根接线：可选启用、参数注入、手动触发一次全量重建。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from habitus.behavior import BehaviorDocumentWriter
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.config import HabitusConfig
from habitus.config.loader import ConfigError
from habitus.config.prediction import PredictionConfig
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.prediction.config import PredictionTreeConfig
from habitus.prediction.errors import PredictionTreeError
from habitus.runtime.prediction import build_prediction_components
from tests.integration.test_runtime_assembly import REPOSITORY_ROOT
from tests.unit.behavior.tree_payloads import occurrence_payload
from tests.unit.runtime.fixtures import STARTUP_PARAMETERS
from tests.unit.runtime.test_behavior_pipeline import SUBJECT, behavior_enabled_config

CST = timezone(timedelta(hours=8))


def prediction_enabled_config(tmp_path: Path) -> HabitusConfig:
    """在行为侧已启用的 fake 路由配置之上，再把预测夜批的十三个启动档填进去。"""

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


def test_the_nightly_stage_runs_after_the_rebuild_and_sees_the_new_generation() -> None:
    """顺序与已删的按天归组相反：关联的待办是树上的出处日，必须等这一代树落地才算得出来。"""

    from habitus.runtime.prediction import PredictionRebuildWorker

    order: list[str] = []

    class _Rebuilder:
        def run_once(self) -> str:
            order.append("rebuild")
            return "generation"

    async def stage() -> None:
        order.append("association")

    worker = PredictionRebuildWorker(
        _Rebuilder(),  # type: ignore[arg-type]
        interval_seconds=60.0,
        shutdown_timeout_seconds=1.0,
        after_rebuild=stage,
    )

    assert asyncio.run(worker.run_once()) == "generation"
    assert order == ["rebuild", "association"]


def test_a_failing_nightly_stage_does_not_block_the_rebuild() -> None:
    """派生层不做级联：关联那一拍失败只留观测，下一轮重建照常。"""

    from habitus.runtime.prediction import PredictionRebuildWorker

    rebuilt: list[str] = []

    class _Rebuilder:
        def run_once(self) -> str:
            rebuilt.append("once")
            return "generation"

    async def stage() -> None:
        raise RuntimeError("association is down")

    worker = PredictionRebuildWorker(
        _Rebuilder(),  # type: ignore[arg-type]
        interval_seconds=60.0,
        shutdown_timeout_seconds=1.0,
        after_rebuild=stage,
    )

    assert asyncio.run(worker.run_once()) == "generation"
    assert rebuilt == ["once"]


def test_the_nightly_stage_result_reaches_the_observation_event() -> None:
    """一夜全败而事件写着 SUCCESS、属性为空的话，运维只能靠猜。"""

    from habitus.foundation.observability import Observer
    from habitus.runtime.prediction import PredictionRebuildWorker
    from habitus.scene import AssociationRefreshReport

    seen: list[object] = []

    class _Observer(Observer):
        def record(self, observation) -> None:  # type: ignore[no-untyped-def]
            seen.append(observation)

    class _Rebuilder:
        def run_once(self) -> str:
            return "generation"

    async def stage() -> AssociationRefreshReport:
        return AssociationRefreshReport(
            associated=("a/2026-09-04",), failed=("b/2026-09-04", "c/2026-09-04"), signals=("x",), model_calls=3
        )

    worker = PredictionRebuildWorker(
        _Rebuilder(),  # type: ignore[arg-type]
        interval_seconds=60.0,
        shutdown_timeout_seconds=1.0,
        observer=_Observer(),
        after_rebuild=stage,
    )
    asyncio.run(worker.run_once())

    (event,) = [item for item in seen if getattr(item, "operation", "") == "nightly_stage"]
    assert event.attributes["associated"] == 1 and event.attributes["failed"] == 2
    assert event.attributes["model_calls"] == 3 and event.attributes["signals"] == 1


def test_a_failing_nightly_stage_is_remembered_by_the_worker() -> None:
    """不设 last_error 的话，健康检查会一直说这个 worker 健康，哪怕关联每夜都抛。"""

    from habitus.runtime.prediction import PredictionRebuildWorker

    class _Rebuilder:
        def run_once(self) -> str:
            return "generation"

    async def stage() -> None:
        raise RuntimeError("association is down")

    worker = PredictionRebuildWorker(
        _Rebuilder(),  # type: ignore[arg-type]
        interval_seconds=60.0,
        shutdown_timeout_seconds=1.0,
        after_rebuild=stage,
    )
    asyncio.run(worker.run_once())

    assert isinstance(worker.last_error, RuntimeError)
