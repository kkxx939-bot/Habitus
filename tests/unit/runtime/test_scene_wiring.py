"""语义关联层的组合根接线。

按天归组与整棵日情景树都已经删掉；按候选累积的关联编排接在预测树重建**之后**——待办来自树上的
出处日。
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from habitus.config import HabitusConfig
from habitus.config.loader import ConfigError
from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.model_client import ChatClient, StructuredChatClient
from habitus.runtime.behavior import build_behavior_components
from habitus.scene.regularity import RegularityTree
from tests.integration.test_runtime_assembly import REPOSITORY_ROOT
from tests.unit.behavior.test_kinds import ScriptedProvider
from tests.unit.runtime.test_behavior_pipeline import SUBJECT, behavior_enabled_config
from tests.unit.scene.association_payloads import model_config


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
    client: StructuredChatClient = StructuredChatClient(ChatClient(model_config(), ScriptedProvider(bodies or [{}])), validation_retries=1)
    lock_store = ProcessLocalLockStore()
    return build_behavior_components(config, structured_chat=client, lock_store=lock_store, path_lock=PathLock(lock_store))








def test_scene_cannot_be_enabled_without_the_behaviour_side(tmp_path) -> None:
    raw = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    raw["storage"]["root"] = str(tmp_path / "data")
    raw["scene"] = {"enabled": True}
    with pytest.raises(ConfigError, match="config.scene is enabled"):
        HabitusConfig.from_mapping(raw)


def test_an_enabled_scene_assembles_the_regularity_tree_and_its_refresher(tmp_path) -> None:
    """关联的编排接在组合根上：规律级树、模型触点、刷新器三者共用同一批实例。"""

    components = _build(scene_enabled_config(tmp_path, max_targets_per_call=5, max_cause_rows=4))
    assert components is not None
    refresher = components.association_refresher
    assert refresher is not None and components.regularity_tree is not None
    assert refresher.regularity_tree is components.regularity_tree
    assert refresher.behavior_tree is components.tree
    assert refresher.max_cause_rows == 4
    assert refresher.associator.config.max_targets_per_call == 5  # type: ignore[attr-defined]
    assert components.regularity_tree.root == (tmp_path / "data" / "scene" / "regularity").resolve()
    assert refresher.root == (tmp_path / "data" / "scene" / "association").resolve()


def test_the_refresher_and_its_tree_are_enabled_together(tmp_path) -> None:
    components = _build(scene_enabled_config(tmp_path))
    assert components is not None
    with pytest.raises(ValueError, match="enabled together"):
        replace(components, regularity_tree=None)
    with pytest.raises(ValueError, match="share one regularity tree"):
        replace(components, regularity_tree=RegularityTree(tmp_path / "elsewhere"))


def test_the_association_stage_runs_after_the_rebuild_not_before(tmp_path) -> None:
    """顺序与已删的按天归组相反：关联的待办是树上的出处日，必须等这一代树落地才算得出来。"""

    from habitus.runtime.prediction import PredictionRebuildWorker

    signature = inspect.signature(PredictionRebuildWorker.__init__)
    assert "after_rebuild" in signature.parameters and "before_rebuild" not in signature.parameters


def test_nothing_is_assembled_until_the_semantic_layer_is_switched_on(tmp_path) -> None:
    components = _build(behavior_enabled_config(tmp_path))
    assert components is not None
    assert components.regularity_tree is None and components.association_refresher is None


def test_the_completed_days_source_is_one_object_not_a_new_one_each_time(tmp_path) -> None:
    """组合根拿它做实例同一性校验。每次新建的话，那条校验永远不成立，把唯一正确的也挡掉。"""

    components = _build(scene_enabled_config(tmp_path))
    assert components is not None
    refresher = components.association_refresher
    assert refresher is not None
    assert refresher.associated_days is refresher.associated_days
