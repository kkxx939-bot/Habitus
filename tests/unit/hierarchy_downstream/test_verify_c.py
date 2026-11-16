"""丙一阶段 C 补的测试：阶段 A 没覆盖的几行（6d 后果一侧的时间线条件、C26 的"被删"、C29 只靠启动就报错），以及 S1、S4 按
设计口径的对照版本（阶段 A 那两条的断言不动，差别与原因见阶段 C 报告）。
"""

from __future__ import annotations

import asyncio
import gzip
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from habitus.behavior.kinds.ids import ClassId
from habitus.behavior.kinds.model import Vocabulary
from habitus.scene.relations import LaneTests
from habitus.series.reader import read_series
from tests.unit.hierarchy_downstream.site import (
    ACCOUNT,
    BASELINE_IN_USE,
    BASELINE_NIGHTS,
    HERE,
    MAIL,
    PLAN,
    QUERY,
    WEEKLY,
    WRITE_DOC,
    World,
    assert_contains,
    at,
    dump,
    expectation,
    load,
    plant_baseline,
    run_baseline,
)
from tests.unit.hierarchy_downstream.test_bookkeeping import STORY_DAYS, STORY_NIGHTS, _parent_child_story
from tests.unit.hierarchy_downstream.test_foresight import DAY, _ready, _ticks
from tests.unit.hierarchy_downstream.test_no_conditions import OLD_BOOK
from tests.unit.word_grow.site import GrowSite


def _normal(value: object) -> Any:
    return json.loads(dump(value))


def _drop(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return {key: _drop(item, field) for key, item in value.items() if key != field}
    if isinstance(value, list):
        return [_drop(item, field) for item in value]
    return value


def test_s1_every_product_is_as_before_except_the_code_fingerprint_on_expectations(tmp_path: Path) -> None:
    """S1 按设计的口径：预期上记的 ``implementation`` 是预测层源码的指纹（``runtime.foresight_implementation``，代码一改就变），
    不算产物；其余逐项相同。"""

    world = World(tmp_path)
    plant_baseline(world)
    current = _normal(run_baseline(world))
    expected = json.loads(gzip.decompress((HERE / "baseline_s1.json.gz").read_bytes()))
    assert {item["implementation"] for item in expected["expectations"]} != {
        item["implementation"] for item in current["expectations"]
    }
    assert_contains(_drop(expected, "implementation"), _drop(current, "implementation"))


def _lane_tests(world: World, night: date) -> LaneTests:
    series = read_series(world.tree, cutoff=night)
    concepts = world.scene.concepts.read_all()
    lines = asyncio.run(world.scene.relation_stage.timelines(series, concepts))
    return LaneTests(lines["session"], concepts, world.scene.relation_config, night)


def test_s4_sibling_readings_are_unchanged_including_rows_that_turn_sparse_in_a_larger_family(
    tmp_path: Path,
) -> None:
    """S4 按读数比：上级写文档在用以后检验族变大（共享说明第 4 节接受的代价），兄弟之间个别一行判成稀疏、不进关系表，但它的
    读数（链数、p、最小可达 p、方向）与改动前逐项相同。直接看这一晚检验族里兄弟之间的每一条。"""

    expected = load("baseline_s4.json")
    for variant, in_use in (("parent_idle", BASELINE_IN_USE), ("parent_in_use", (*BASELINE_IN_USE, WRITE_DOC))):
        world = World(tmp_path / variant)
        plant_baseline(world, in_use=in_use)
        for night in BASELINE_NIGHTS:
            world.generate(night)
            tested = {
                f"{item.key.antecedent}->{item.key.consequent}|{item.key.segment.value}|{item.key.condition}": item
                for item in _lane_tests(world, night).night().tests
                if {item.key.antecedent, item.key.consequent} == {WEEKLY, PLAN}
            }
            for key, reading in expected[variant][night.isoformat()].items():
                assert key in tested, f"{variant}/{night}/{key}"
                item = tested[key]
                assert (item.antecedents, float(f"{item.p_value:.10g}"), float(f"{item.minimum_p:.10g}")) == (
                    reading["antecedents"],
                    reading["p_value"],
                    reading["minimum_p"],
                ), f"{variant}/{night}/{key}"


def test_6d_a_parent_and_its_child_are_not_each_others_timeline_condition_on_the_consequent_side(
    tmp_path: Path,
) -> None:
    """后果一侧（落点表 6d）：前因是查数据时，"之前做过写周报"不是"查数据 → 写文档"的条件（写周报是后果写文档的下级，读的是
    同一批记录），"之前做过写文档"也不是"查数据 → 写周报"的；对别的后果（记账）照常列。看的是这一晚枚举的全部关系键。"""

    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY, QUERY, MAIL, ACCOUNT)
    _parent_child_story(world, STORY_DAYS, primed=True)
    night = STORY_NIGHTS[1]
    world.generate(STORY_NIGHTS[0])
    world.generate(night)
    listed = {
        (key.antecedent, key.consequent, key.condition)
        for key in _lane_tests(world, night).keys()
        if key.condition.startswith("timeline:")
    }
    assert {(QUERY, ACCOUNT, f"timeline:{WEEKLY}"), (QUERY, ACCOUNT, f"timeline:{WRITE_DOC}")} <= listed
    assert (QUERY, WRITE_DOC, f"timeline:{WEEKLY}") not in listed
    assert (QUERY, WEEKLY, f"timeline:{WRITE_DOC}") not in listed


def test_6c_a_parent_and_its_child_are_not_each_others_timeline_condition_on_the_antecedent_side(
    tmp_path: Path,
) -> None:
    """前因一侧（6c）按枚举的全部关系键看（C6 看的是关系表，表里只有判过的行）：前因是写文档时没有"之前做过写周报"这个条件，
    前因是写周报时没有"之前做过写文档"；写文档对别的条件（看邮件、查数据）照常列。"""

    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY, QUERY, MAIL, ACCOUNT)
    _parent_child_story(world, STORY_DAYS, primed=True)
    night = STORY_NIGHTS[1]
    world.generate(STORY_NIGHTS[0])
    world.generate(night)
    listed = {
        (key.antecedent, key.condition)
        for key in _lane_tests(world, night).keys()
        if key.condition.startswith("timeline:")
    }
    assert {(WRITE_DOC, f"timeline:{MAIL}"), (WRITE_DOC, f"timeline:{QUERY}")} <= listed
    assert (WRITE_DOC, f"timeline:{WEEKLY}") not in listed
    assert (WEEKLY, f"timeline:{WRITE_DOC}") not in listed


def test_c26_after_the_node_is_deleted_nothing_more_is_said_and_the_waiting_request_is_withdrawn(
    tmp_path: Path,
) -> None:
    """C26 的"被删"那一半：自己长的节点在用、有一条「待取」，之后被删（``Vocabulary.without``）。"""

    world = World(tmp_path)
    grown = world.grow(WRITE_DOC, "写需求书", written=at(DAY - timedelta(days=30), 9))
    _ready(world, grown)
    world.ledger.record_expectation(expectation("early", grown, at(DAY, 9), deadline_minutes=90))
    world.ledger.record_expectation(expectation("later", grown, at(DAY, 10, 45), deadline_minutes=120))
    ticker = world.ticker()
    _ticks(world, at(DAY, 9, 15), at(DAY, 9, 45), ticker)
    assert world.deliveries(DAY)["early"][-1] == "待取"

    def delete(current: Vocabulary) -> Vocabulary:
        return current.without(ClassId.parse(grown), on=DAY)

    world.set_clock(at(DAY, 9, 50))
    world.store.update(delete)
    world.tick(at(DAY, 10), ticker)
    outlet = world.foresight.outlet
    claimed = outlet.claim(outlet.tools[0].host, "after", at(DAY, 10, 5), 3)
    _ticks(world, at(DAY, 10, 15), at(DAY, 11, 30), ticker)
    made = world.deliveries(DAY)
    assert (made["early"][-1], made["later"][-1], [item.request_id for item in claimed]) == ("已停用", "已停用", [])


def test_c29_starting_the_runtime_alone_refuses_an_old_proposal_book(tmp_path: Path) -> None:
    """C29 只靠启动（``Runtime.initialize``，``GrowSite`` 的构造里做）就报出来，不等换代或模型队列。"""

    book = tmp_path / "data" / "scene" / "advice" / "conditions" / "session.json"
    book.parent.mkdir(parents=True)
    book.write_text(OLD_BOOK, encoding="utf-8")
    with pytest.raises(Exception, match=r"conditions[/\\]session\.json"):  # noqa: PT011 - 报错类型由实现定，只要点出文件
        GrowSite(tmp_path).close()
