"""裁定 94 第 8 条：两条 lane 每份观测逐份都落回执（成了落判断，失败落「错误」），不留"凭据在、回执永远不来"。

核出来的缺口在融合的共用写入（``fusion.writing.publish_judgement``）：派生不过的单位（判断形状不对）以前先落了凭据再抛——
物理 lane 没有"重试账放弃后写「错误」"那一步，上游每次重发都在同一处抛，凭据永远等不来回执、把封口视界钉在它那一刻（会话 lane
靠重试账放弃后的 ``abandon`` 最终落「错误」，不会永远钉住）。现在先派生、后落盘：派生不过的一个文件都不留。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.behavior.fusion.lanes.session import TurnDescription, split_turns
from tests.unit.behavior.session_fixtures import Stores, prompt, reply


def _turn():  # type: ignore[no-untyped-def]
    (turn,) = split_turns("c-1", (prompt(0, "统计一下行数"), reply(1, "好的", minutes=2)))
    return turn


def _description() -> TurnDescription:
    return TurnDescription(name="统计代码行数", summary="他问行数", goal=None, steps=("统计",))


def test_a_unit_that_cannot_be_derived_leaves_no_observation_behind(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    recorder = stores.recorder()
    with pytest.raises(BehaviorFusionError):
        # 判定时刻没带时区：派生不过
        recorder.record(_turn(), _description(), validation_attempts=1, judged_at=datetime(2026, 7, 26, 18, 0))
    assert list(stores.observations.list()) == [] and list(stores.judgements.list()) == []
