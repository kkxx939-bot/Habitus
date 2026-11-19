"""一条记录只属于它选中的那个词；对它的每一个上级，这条记录是"不知道"（裁定 105，用户 2026-11-19）。

走组合根装的语义树一步与预测层（``site.World``）。判定口径：

- 上级不靠下级凑够：下级的记录很多、直接选中上级的不够——上级不建日子预期；
- 兄弟各自判断：一个够、一个不够——只给够的那个建；
- 直接选中上级的照常：直接记录够——上级照建；
- 做了下级不算上级漏做：上级的预期到期，窗里只有下级的记录——不记命中也不记漏做（中性，不算一次机会）。

第一、二、四条与"窗里上级自己也做了"那条在旧代码（下级的记录也记在上级账上，裁定 101 第 8 条旧文）上失败。
"下级长出来上级不被读成停了"在时间线那一层测（``tests/unit/scene/test_relation_child_unknown.py``）。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from habitus.foresight.clock import is_opportunity
from tests.unit.hierarchy_downstream.site import (
    PLAN,
    WEEKLY,
    WRITE_DOC,
    World,
    at,
    expectation,
)

DAYS = (date(2026, 4, 1), date(2026, 4, 3), date(2026, 4, 6), date(2026, 4, 8))
NIGHT = date(2026, 4, 10)


def _day_kinds(world: World) -> set[str]:
    """换代那一晚的语义树、机会表与当天的日子预期走一遍，返回建了日子预期的类。"""

    report = world.generate(NIGHT)
    world.opportunities(report)
    world.day_expectations()
    return {item.kind_token for item in world.ledger.expectations_on(NIGHT) if item.source == "日子"}


def test_a_parent_does_not_reach_enough_on_its_childrens_records(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    for day in DAYS:
        world.record(WEEKLY, at(day, 9))
    assert WRITE_DOC not in _day_kinds(world)


def test_siblings_are_judged_each_on_their_own_records(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY, PLAN)
    for day in DAYS:
        world.record(WEEKLY, at(day, 9))
    world.record(PLAN, at(DAYS[0], 15))
    assert {kind for kind in _day_kinds(world) if kind in (WRITE_DOC, WEEKLY, PLAN)} == {WEEKLY}


def test_a_parent_picked_directly_often_enough_gets_its_expectation(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    for day in DAYS:
        world.record(WRITE_DOC, at(day, 9))
        world.record(WEEKLY, at(day, 15))
    assert {WRITE_DOC, WEEKLY} <= _day_kinds(world)


def test_a_parent_expectation_with_only_the_child_done_is_neither_a_hit_nor_a_miss(tmp_path: Path) -> None:
    day = date(2026, 4, 14)
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    item = expectation("doc", WRITE_DOC, at(day, 9))
    world.ledger.record_expectation(item)
    world.record(WEEKLY, at(day, 9, 20))
    world.set_clock(at(day + timedelta(days=1), 4))
    world.settle(day)
    (settled,) = world.ledger.settlements_on(day)
    (fact,) = world.ledger.facts_of(item.account)
    assert (settled.outcome, settled.occurrence_uri) == ("过期待决", None)
    assert (fact.good, fact.strength, is_opportunity(fact)) == (False, 0.0, False)


def test_a_parent_expectation_with_its_own_record_is_a_hit_even_with_the_child_done(tmp_path: Path) -> None:
    """窗里先做了下级、后做了上级自己：认的是上级自己那一条，照常结准时。"""

    day = date(2026, 4, 14)
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    world.ledger.record_expectation(expectation("doc", WRITE_DOC, at(day, 9)))
    world.record(WEEKLY, at(day, 9, 15))
    uri = world.record(WRITE_DOC, at(day, 9, 30))
    world.set_clock(at(day + timedelta(days=1), 4))
    world.settle(day)
    (settled,) = world.ledger.settlements_on(day)
    assert (settled.outcome, settled.occurrence_uri) == ("准时", uri)
