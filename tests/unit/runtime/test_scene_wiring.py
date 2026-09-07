"""语义关联层的组合根接线：可选启用、参数注入、与归约 sweep 共享封口窗口与实例。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from habitus.config import HabitusConfig
from habitus.config.loader import ConfigError
from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.model_client import StructuredChatClient
from habitus.runtime.behavior import build_behavior_components, refresh_scene_days
from habitus.scene import SceneTree
from tests.integration.test_runtime_assembly import REPOSITORY_ROOT
from tests.unit.behavior.test_kinds import ScriptedProvider
from tests.unit.runtime.test_behavior_pipeline import SUBJECT, behavior_enabled_config
from tests.unit.scene.fixtures import structured_client


def scene_enabled_config(tmp_path: Path, **scene: object) -> HabitusConfig:
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
    raw["scene"] = {"enabled": True, **scene}
    return HabitusConfig.from_mapping(raw)


def _build(config: HabitusConfig, bodies: list[dict] | None = None):
    client: StructuredChatClient = structured_client(ScriptedProvider(bodies or [{}]))
    lock_store = ProcessLocalLockStore()
    return build_behavior_components(config, structured_chat=client, lock_store=lock_store, path_lock=PathLock(lock_store))


def test_scene_is_absent_until_it_is_switched_on(tmp_path) -> None:
    components = _build(behavior_enabled_config(tmp_path))
    assert components is not None
    assert components.scene_tree is None and components.scene_refresher is None and components.scene_worker is None


def test_enabled_scene_without_prediction_gets_its_own_nightly_tick(tmp_path) -> None:
    config = scene_enabled_config(tmp_path, lookback_days=5, pending_expiry_days=45, retained_generations=2, max_occurrences_per_call=300, refresh_interval_seconds=3600)
    components = _build(config)
    assert components is not None
    refresher = components.scene_refresher
    assert refresher is not None and components.scene_tree is not None
    assert refresher.scene_tree is components.scene_tree
    assert refresher.behavior_tree is components.tree
    assert components.scene_tree.root == config.scene_root / "tree"
    assert refresher.root == config.scene_root / "refresh"
    assert components.scene_tree.tree_config.retained_generations == 2
    assert refresher.config.lookback_days == 5 and refresher.config.pending_expiry_days == 45
    assert refresher.grouper.config.max_occurrences_per_call == 300  # type: ignore[attr-defined]
    assert refresher.subject == SUBJECT
    worker = components.scene_worker
    assert worker is not None and worker.refresher is refresher and worker.runner is components.reduction_runner
    assert worker.interval_seconds == 3600.0
    assert asyncio.run(worker.run_once()).published == ()  # 还没有定稿日


def test_with_prediction_enabled_the_scene_stage_runs_before_the_rebuild(tmp_path) -> None:
    """夜批顺序：情景阶段（定稿日归组）→ 预测重建；同一拍，由预测 Worker 依次调用。"""

    from habitus.runtime.assembly import build_runtime
    from tests.unit.runtime.test_behavior_pipeline import scripted_dependencies
    from tests.unit.runtime.test_prediction_wiring import STARTUP_PARAMETERS

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
    raw["scene"] = {"enabled": True}
    raw["prediction"] = dict(STARTUP_PARAMETERS)
    providers, vectors = scripted_dependencies([{}])
    runtime = build_runtime(HabitusConfig.from_mapping(raw), providers=providers, vector_stores=vectors)
    behavior = runtime.components.behavior
    prediction = runtime.components.prediction
    assert behavior is not None and prediction is not None
    assert behavior.scene_worker is None  # 不另起节拍
    assert prediction.worker.before_rebuild is not None
    order: list[str] = []
    original_stage = prediction.worker.before_rebuild
    original_rebuild = prediction.rebuilder.run_once

    async def stage():
        order.append("scene")
        return await original_stage()

    prediction.worker.before_rebuild = stage
    prediction.rebuilder.run_once = lambda: (order.append("prediction"), original_rebuild())[1]  # type: ignore[method-assign]
    asyncio.run(prediction.worker.run_once())
    assert order == ["scene", "prediction"]


def test_component_identity_checks_catch_a_foreign_scene_tree(tmp_path) -> None:
    components = _build(scene_enabled_config(tmp_path))
    assert components is not None
    with pytest.raises(ValueError, match="share one scene tree"):
        replace(components, scene_tree=SceneTree(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="enabled together"):
        replace(components, scene_tree=None)


def test_scene_cannot_be_enabled_without_the_behaviour_side(tmp_path) -> None:
    raw = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    raw["storage"]["root"] = str(tmp_path / "data")
    raw["scene"] = {"enabled": True}
    with pytest.raises(ConfigError, match="config.scene is enabled"):
        HabitusConfig.from_mapping(raw)


def test_refresh_scene_days_forces_named_days_and_backfills_closed_days(tmp_path) -> None:
    """正门：指定日子强制重算（不问定稿）；不指定则与夜批同一工作集（已定稿、没有当前版本一代）。"""

    from datetime import date, datetime, timedelta, timezone

    from habitus.behavior import BehaviorDocumentWriter
    from habitus.behavior.model import BehaviorKind
    from tests.unit.behavior.tree_payloads import occurrence_payload

    config = scene_enabled_config(tmp_path)
    components = _build(config, bodies=[{"scenes": [], "assignments": [{"no": 1, "scene_no": None, "role": None}]}])
    assert components is not None
    cst = timezone(timedelta(hours=8))
    started = datetime(2026, 8, 1, 12, 0, tzinfo=cst)
    writer = BehaviorDocumentWriter(components.tree, ProcessLocalLockStore(), clock=lambda: started)
    writer.publish(
        BehaviorKind.OCCURRENCE,
        occurrence_payload(
            occurred_on=started.date(), started_at=started, last_observed_at=started + timedelta(minutes=1),
            onset_available_at=started + timedelta(seconds=2), basis=(), goal=None,
        ),
    )

    assert asyncio.run(refresh_scene_days(components)).published == ()  # 归约还没把任何一天定稿
    report = asyncio.run(refresh_scene_days(components, [date(2026, 8, 1)]))

    assert report.published == (date(2026, 8, 1),)
    assert components.scene_tree is not None and components.scene_tree.day_state(date(2026, 8, 1)) is not None
    assert asyncio.run(refresh_scene_days(components, [date(2026, 8, 1)])).unchanged == (date(2026, 8, 1),)
    disabled = _build(behavior_enabled_config(tmp_path / "off"))
    assert disabled is not None
    with pytest.raises(ValueError, match="not enabled"):
        asyncio.run(refresh_scene_days(disabled))
