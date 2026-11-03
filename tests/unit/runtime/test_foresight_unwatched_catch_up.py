"""没在看的那天保持中性，之后做了也不被改判成「后补」（由 ``37-材料/readme/repro-catchup/repro_catchup.py`` 改成）。

没在看（会话 lane 整天没有记录，或上游报了全天「未观测」）那天结出的是中性事实「过期待决、强度 0」（``foresight.facts``）：
谈不上抓到缺失，之后在下一次机会之前做了，也不该被 ``rule_overdue`` 裁成「后补、强度 1」——那会凭空多一次机会、一笔坏
（裁定 54 第 3 条③、裁定 89 第 2 条）。真漏做的那天（有别的会话、他没做）照常裁后补。

走真实链路：真实行为树、每晚 04:00 先结算（接法同组合根：带机会表与线）再建机会表、白天每 15 分钟一拍；「离上次够久」的账种在
提醒档，出口一直活着；他每天 15:00 另有一条别的会话，只有空白那天按场景不同。空白次日 08:30 他做了。
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from tests.unit.runtime.test_foresight_daily_miss_and_silent_day import (
    HABIT,
    OTHER,
    START,
    fact_of,
    interval_items,
    morning,
    nightly,
    settle,
    world,
)
from tests.unit.scene.fixtures import publish, publish_gap


def _blank_fact(tmp: Path, *, step: int, history: int, blank: str) -> tuple[str, bool, float]:
    """每 ``step`` 天 08:30 做一次、先做 ``history`` 次；第 ``history`` 个该做的日子是空白（``blank``），次日 08:30 补做。
    返回空白那天那条「离上次够久」预期最后的事实（种类、好坏、强度）。"""

    w = world(tmp)
    blank_day = START + timedelta(days=step * history)
    catch_day = blank_day + timedelta(days=1)
    last = blank_day + timedelta(days=step * 2 + 3)
    day = START
    while day <= last:
        index = (day - START).days
        if day > START:
            settle(w, day - timedelta(days=1))
        nightly(w, day)
        if day == blank_day and blank == "gap":
            publish_gap(w.site.behavior_tree, day, (0, 0), (23, 59))
        if not (day == blank_day and blank in ("silent", "gap")):
            publish(w.site.behavior_tree, day, OTHER, 15, 0)
        done = (index % step == 0 and day != blank_day) or day == catch_day
        if day >= blank_day - timedelta(days=2):
            morning(w, day, done=done)
        elif done:
            publish(w.site.behavior_tree, day, HABIT, 8, 30)
        day += timedelta(days=1)
    (item,) = interval_items(w, blank_day)
    fact = fact_of(w, item)
    assert fact is not None
    return fact.kind, fact.good, round(fact.strength, 3)


@pytest.mark.parametrize("blank", ["silent", "gap"])
@pytest.mark.parametrize(("step", "history"), [(7, 8), (1, 16)], ids=["weekly", "daily"])
def test_an_unwatched_day_stays_neutral_after_a_later_doing(
    tmp_path: Path, step: int, history: int, blank: str
) -> None:
    """每周一次、天天做两种节奏，整天没会话、上游报全天未观测两种空白：空白那天结成中性，次日做了也不改判。"""

    assert _blank_fact(tmp_path, step=step, history=history, blank=blank) == ("过期待决", False, 0.0)


def test_a_watched_weekly_miss_is_still_ruled_catch_up(tmp_path: Path) -> None:
    """对照：每周一次的那天有别的会话、他真没做，次日补做——照常裁「后补」（下一次机会之前做了，抓到缺失）。"""

    kind, good, _strength = _blank_fact(tmp_path, step=7, history=8, blank="watched")
    assert (kind, good) == ("后补", False)


def test_a_watched_daily_miss_stays_a_full_miss(tmp_path: Path) -> None:
    """对照：天天做的那天真没做，次日补做时下一次机会（次日那条）已经形成，补做属于下一段，那天仍是一笔整坏（不裁后补）。"""

    assert _blank_fact(tmp_path, step=1, history=16, blank="watched") == ("过期待决", False, 1.0)
