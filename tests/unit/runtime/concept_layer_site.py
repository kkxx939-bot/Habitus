"""概念层只留"词表节点 + 日历"（2026-11-14）的测试共用现场：一棵真实行为树、一份真实词表、按组合根装好的语义树。

只通过会留下来的口子驱动：组合根 ``build_scene_nightly`` 装语义树、``generate`` 做换代的语义树一步、命中仓读命中记录与日完成章。
不碰映射器、映射这一段与模型队列——改动之后命中记录在封口时由代码直接生成，这些口子都不在了。

模型一律用"被调用就报错"的替身：它把每次调用的名字记下来再抛错。换代里本来就不该调模型。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from habitus.behavior.kinds.ids import ClassId, Lane
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.conditions import LocalCalendar
from habitus.config import HabitusConfig
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.model_client import ChatClient
from habitus.model_client.structured import StructuredChatClient
from habitus.runtime.scene import build_scene_nightly
from habitus.runtime.scene_night import NightReport, SceneNightlyRun
from habitus.scene.occurrences.model import ConceptHits
from habitus.series.reader import read_series
from tests.unit.foresight.scripted_model import ScriptedProvider, model_config
from tests.unit.scene.fixtures import Site

CST = timezone(timedelta(hours=8))
SUBJECT = "家庭成员A"
RUN, EXPENSE, DIARY = "s-k0023", "s-k0016", "s-k0031"
CLASSES = {
    RUN: ("跑步", "户外或跑步机上跑步"),
    EXPENSE: ("记账", "记一笔支出或收入"),
    DIARY: ("写日记", "写当天的日记"),
}
LEGACY = Path(__file__).resolve().parents[2] / "fixtures" / "scene_legacy"
EXAMPLE_CONFIG = Path(__file__).resolve().parents[3] / "habitus" / "config" / "example.yaml"
NOW = datetime(2026, 4, 1, 12, 0, tzinfo=CST)


class ModelMustNotBeCalled(StructuredChatClient):
    """被调用就报错的结构化客户端：先记下这次调用的名字（``name``）再抛。"""

    def __init__(self) -> None:
        super().__init__(ChatClient(model_config(), ScriptedProvider([{}])), validation_retries=0)
        self.names: list[str] = []

    def _refuse(self, kwargs: dict[str, Any]) -> None:
        self.names.append(str(kwargs.get("name")))
        raise RuntimeError(f"the dialogue model must not be called (name={kwargs.get('name')!r})")

    def complete_json(self, *args: Any, **kwargs: Any) -> Any:
        self._refuse(kwargs)

    async def complete_json_async(self, *args: Any, **kwargs: Any) -> Any:
        self._refuse(kwargs)

    def complete_model(self, *args: Any, **kwargs: Any) -> Any:
        self._refuse(kwargs)

    async def complete_model_async(self, *args: Any, **kwargs: Any) -> Any:
        self._refuse(kwargs)


def example_mapping(tmp_path: Path) -> dict[str, Any]:
    """example.yaml 原样读进来，存储根换到 ``tmp_path / "data"``。"""

    payload: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    return payload


def scene_config(tmp_path: Path) -> HabitusConfig:
    """语义树开着、主体是 ``SUBJECT`` 的一份配置（其余全是 example.yaml 的默认）。"""

    payload = example_mapping(tmp_path)
    payload["behavior"] = {"primary_subject": SUBJECT}
    payload["scene"] = {"enabled": True}
    return HabitusConfig.from_mapping(payload)


def vocabulary(tmp_path: Path) -> BehaviorKindStore:
    """一份真实词表（会话 lane 的初始词表）：``CLASSES`` 那三个编号在用（词表方案 2026-11-14；名字、定义取初始词表的）。"""

    store = BehaviorKindStore(
        tmp_path / "kinds",
        lane=Lane.SESSION,
        lock_store=SQLiteLockStore(tmp_path / "locks.sqlite3"),
        lock_ttl_seconds=30,
    )
    active = {ClassId.parse(token): NOW.date() for token in CLASSES}
    store.update(lambda current: current.with_active(active))
    return store


def scene_run(tmp_path: Path, site: Site, client: StructuredChatClient) -> SceneNightlyRun:
    """一支语义树，由组合根装（``runtime.scene.build_scene_nightly``）；日历用 ``LocalCalendar``（只给周几、几号）。"""

    run = build_scene_nightly(
        scene_config(tmp_path),
        behavior_tree=site.behavior_tree,
        kind_store=vocabulary(tmp_path),
        structured_chat=client,
        calendar=LocalCalendar(),
    )
    assert run is not None
    return run


def generate(run: SceneNightlyRun, night: date, now: datetime) -> NightReport:
    """换代的语义树一步，序列截止到 ``night``（``night`` 之前的日子已封口）。"""

    return asyncio.run(
        run.generate(series=read_series(run.behavior_tree, cutoff=night), words=run.syncing.catalog.current(), now=now)
    )


def hits_on(run: SceneNightlyRun, day: date) -> Sequence[ConceptHits]:
    return run.hits.read_day(day)


__all__ = [
    "CLASSES",
    "CST",
    "DIARY",
    "EXPENSE",
    "LEGACY",
    "RUN",
    "SUBJECT",
    "ModelMustNotBeCalled",
    "example_mapping",
    "generate",
    "hits_on",
    "scene_config",
    "scene_run",
    "vocabulary",
]
