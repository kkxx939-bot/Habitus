"""语义关联层的组合根接线。

旧语义树（规律级树、关联刷新器）随 2026-09-26 的重构整块删掉，新树按功能分支重建、逐支接回组合根。
这一刀留下的只有跨域配置校验：语义层从行为树派生，行为侧没开就没有输入。
"""

from __future__ import annotations

import pytest
import yaml

from habitus.config import HabitusConfig
from habitus.config.loader import ConfigError
from tests.integration.test_runtime_assembly import REPOSITORY_ROOT


def test_scene_cannot_be_enabled_without_the_behaviour_side(tmp_path) -> None:
    raw = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    raw["storage"]["root"] = str(tmp_path / "data")
    raw["scene"] = {"enabled": True}
    with pytest.raises(ConfigError, match="config.scene is enabled"):
        HabitusConfig.from_mapping(raw)


def test_the_semantic_stages_hook_in_after_the_rebuild_not_before() -> None:
    """夜批是线性的：树重建 → 结算 → 新语义树的各拍。钩子只有 ``after_rebuild`` 一个位置。"""

    import inspect

    from habitus.runtime.prediction import PredictionRebuildWorker

    signature = inspect.signature(PredictionRebuildWorker.__init__)
    assert "after_rebuild" in signature.parameters and "before_rebuild" not in signature.parameters
