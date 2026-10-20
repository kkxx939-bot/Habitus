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


def test_the_nightly_worker_only_paces_the_night_and_takes_no_stage_hooks() -> None:
    """夜批是线性的：整晚的串法自己排先后（结算 → 语义树 → 机会表 → 找原因，``runtime.nightly``），
    常驻节拍只管节奏，不另开前置 / 后置钩子（原来预测树重建 worker 上的 ``after_rebuild`` 随它删了，裁定 59）。"""

    import inspect

    from habitus.runtime.nightly_worker import NightlyWorker

    parameters = set(inspect.signature(NightlyWorker.__init__).parameters)
    assert not {name for name in parameters if name.startswith(("before", "after"))}
