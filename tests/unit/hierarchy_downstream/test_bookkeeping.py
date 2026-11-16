"""丙一"应当变"的记账一段（预期表 D1 与场景 C1–C9；共享说明 3.1、3.2）：一条记录算给它自己的概念与每一个在用上级的概念；
上级和下级之间不量关系；下级增减改上级概念管的编号。

走换代的语义树一步（组合根装的 ``build_scene_nightly``，``site.World.generate``：同步词表 → 封口 → 时间线 → 关系），看它交出来的
材料（概念集、时间线上的记号、复现）、关系表与命中记录。改动前（``780e298``）一个行为概念只管它自己那一个类、一条记录只给它自己的
概念留记号，这里标"应当变"的在旧代码上应当失败；C3、C7 是"反过来不认""兄弟照常"的守门，旧代码上本来就成立。
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

from habitus.runtime.scene_night import NightMaterials, NightReport
from tests.unit.hierarchy_downstream.site import (
    ACCOUNT,
    BASELINE_IN_USE,
    BASELINE_NIGHTS,
    MAIL,
    PLAN,
    QUERY,
    WEEKLY,
    WRITE_DOC,
    World,
    at,
    plant_baseline,
    workdays,
)

DAY = date(2026, 4, 13)
NIGHT = DAY + timedelta(days=1)


def _materials(report: NightReport) -> NightMaterials:
    materials = report.materials
    assert materials is not None
    return materials


def _marks(report: NightReport, identity: str) -> list[str]:
    line = _materials(report).timelines["session"]
    return [mark.uri for mark in line.marks.get(identity, [])]


def _all_marked(report: NightReport) -> set[str]:
    return {mark.uri for marks in _materials(report).timelines["session"].marks.values() for mark in marks}


# ── D1：在用节点一个概念，管的编号 = 自己 + 全部活着的下级，第一个是自己 ─────────────


def test_d1_a_concept_in_use_manages_itself_and_every_live_descendant_itself_first(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    grown = world.grow(WEEKLY, "写需求书")  # 第 3 层候选：写文档的下级的下级
    world.record(WRITE_DOC, at(DAY, 9))
    report = world.generate(NIGHT)
    concepts = _materials(report).concepts
    doc = concepts[WRITE_DOC]
    assert doc.classes[0] == WRITE_DOC
    assert set(doc.classes) == {WRITE_DOC} | world.descendants(WRITE_DOC)
    assert grown in doc.classes and len(doc.classes) == len(set(doc.classes))
    assert concepts[WEEKLY].classes == (WEEKLY, grown)
    assert concepts.base_for(WRITE_DOC) == WRITE_DOC  # 身份仍是节点自己的编号


# ── C1–C4：一条记录给谁留记号 ─────────────────────────────────────────────


def test_c1_a_record_on_a_child_marks_both_concepts_and_the_hit_keeps_its_own_id(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    uri = world.record(WEEKLY, at(DAY, 10))
    report = world.generate(NIGHT)
    assert uri in _marks(report, WEEKLY)
    assert uri in _marks(report, WRITE_DOC)
    (hit,) = world.scene.hits.read_day(DAY)
    assert (hit.occurrence_uri, hit.kind_token) == (uri, WEEKLY)


def test_c2_a_record_on_an_idle_child_marks_only_the_parent_in_use(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC)
    uri = world.record(WEEKLY, at(DAY, 10))
    report = world.generate(NIGHT)
    assert _marks(report, WRITE_DOC) == [uri]
    assert WEEKLY not in _materials(report).concepts.behaviors()


def test_c3_a_record_on_a_parent_is_not_counted_downwards(tmp_path: Path) -> None:
    """守门：上级的记录不往下算（旧代码上本来就成立）。"""

    world = World(tmp_path)
    world.use(WEEKLY)
    uri = world.record(WRITE_DOC, at(DAY, 10))
    report = world.generate(NIGHT)
    assert uri not in _all_marked(report)
    assert _marks(report, WEEKLY) == []


def test_c4_a_record_on_a_third_level_node_marks_the_first_level_in_use(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC)
    grown = world.grow(WEEKLY, "写需求书")
    uri = world.record(grown, at(DAY, 10))
    report = world.generate(NIGHT)
    assert _marks(report, WRITE_DOC) == [uri]


# ── C5：复现按直接记录加全部下级的记录算 ─────────────────────────────────────


def test_c5_the_recurrence_of_a_parent_counts_its_own_and_its_descendants_records(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC)
    grown = world.grow(WEEKLY, "写需求书")
    world.record(WRITE_DOC, at(date(2026, 4, 1), 10))
    world.record(WEEKLY, at(date(2026, 4, 3), 10))
    world.record(grown, at(date(2026, 4, 6), 10))
    report = world.generate(date(2026, 4, 8))
    recurrence = _materials(report).recurrences[WRITE_DOC]
    assert [item.day for item in recurrence.days] == [date(2026, 4, 1), date(2026, 4, 3), date(2026, 4, 6)]


# ── C6–C8：上下级之间不量关系；兄弟照常；下级的记录算上级来了 ─────────────────────────


STORY_DAYS = 60
STORY_NIGHTS = tuple(workdays(date(2026, 3, 2), count)[-1] + timedelta(days=1) for count in (30, STORY_DAYS))


def _parent_child_story(world: World, days: int, *, primed: bool) -> None:
    """``days`` 个工作日：09:00 看邮件、17:50 记账；上午一回写文档、15–30 分钟后九成接着查数据；下午一回查数据、15–30 分钟后
    九成接着写文档。``primed``：两回各有一半之前 15–25 分钟先写了周报。固定种子。"""

    generator = random.Random(11)
    for day in workdays(date(2026, 3, 2), days):
        world.record(MAIL, at(day, 9, 0))
        world.record(ACCOUNT, at(day, 17, 50))
        for first, second, earliest, latest in (
            (WRITE_DOC, QUERY, 9 * 60 + 40, 12 * 60),
            (QUERY, WRITE_DOC, 14 * 60, 16 * 60),
        ):
            minute = generator.randint(earliest, latest)
            if primed and generator.random() < 0.5:
                world.record(WEEKLY, at(day, *divmod(minute - generator.randint(15, 25), 60)))
            world.record(first, at(day, *divmod(minute, 60)))
            if generator.random() < 0.9:
                world.record(second, at(day, *divmod(minute + generator.randint(15, 30), 60)))


def test_c6_a_parent_and_its_child_form_no_relation_key_and_are_not_each_others_timeline_condition(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY, QUERY, MAIL, ACCOUNT)
    _parent_child_story(world, STORY_DAYS, primed=True)
    night = STORY_NIGHTS[1]
    world.generate(STORY_NIGHTS[0])
    world.generate(night)
    keys = [relation.key for relation in world.scene.relations.read("session", night).relations]
    pair = {WRITE_DOC, WEEKLY}
    paired = [key for key in keys if {key.antecedent, key.consequent} == pair]
    conditioned = [
        key
        for key in keys
        if (key.condition, key.antecedent) in {(f"timeline:{WEEKLY}", WRITE_DOC), (f"timeline:{WRITE_DOC}", WEEKLY)}
    ]
    assert (paired, conditioned) == ([], [])


def test_c7_siblings_form_relation_keys_as_before_even_with_their_parent_in_use(tmp_path: Path) -> None:
    """守门：兄弟照常成键（旧代码上本来就成立）。"""

    world = World(tmp_path)
    plant_baseline(world, in_use=(*BASELINE_IN_USE, WRITE_DOC))
    for night in BASELINE_NIGHTS:
        world.generate(night)
    keys = {
        (relation.key.antecedent, relation.key.consequent)
        for relation in world.scene.relations.read("session", BASELINE_NIGHTS[1]).relations
    }
    assert (WEEKLY, PLAN) in keys and (PLAN, WEEKLY) in keys


def test_c8_after_query_then_weekly_the_parent_relation_counts_as_come(tmp_path: Path) -> None:
    """查数据 → 写文档 成立（写周报不在用）；那一天他查了数据以后写了周报：时间线上写文档在那一刻有记号，事件预期结成做到了。"""

    world = World(tmp_path)
    world.use(WRITE_DOC, QUERY, MAIL, ACCOUNT)
    _parent_child_story(world, STORY_DAYS, primed=False)
    night = STORY_NIGHTS[1]
    world.generate(STORY_NIGHTS[0])
    report = world.generate(night)
    established = {
        (relation.key.antecedent, relation.key.consequent, relation.key.condition)
        for relation in world.scene.relations.read("session", night).relations
        if relation.status.value == "established"
    }
    assert (QUERY, WRITE_DOC, "") in established
    world.opportunities(report)
    world.day_expectations()
    world.record(QUERY, at(night, 10, 0))
    ticker = world.ticker()
    world.tick(at(night, 10, 15), ticker)
    (formed,) = [
        item
        for item in world.ledger.expectations_on(night)
        if item.source == "事件" and item.kind_token == WRITE_DOC and f"|{QUERY}|" in item.trigger
    ]
    weekly = world.record(WEEKLY, at(night, 10, 20))
    world.set_clock(at(night + timedelta(days=1), 4))
    world.settle(night)
    (settled,) = [item for item in world.ledger.settlements_on(night) if item.expectation_id == formed.expectation_id]
    later = world.generate(night + timedelta(days=1))
    assert (settled.occurrence_uri, weekly in _marks(later, WRITE_DOC)) == (weekly, True)


# ── C9：下级增减改上级概念管的编号，身份不变 ─────────────────────────────────────


def test_c9_a_new_candidate_under_the_child_adds_one_id_to_both_concepts_on_the_next_sync(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.use(WRITE_DOC, WEEKLY)
    world.record(WEEKLY, at(DAY, 10))
    before = _materials(world.generate(NIGHT)).concepts
    grown = world.grow(WEEKLY, "写需求书")
    after = _materials(world.generate(NIGHT + timedelta(days=1))).concepts
    for identity in (WRITE_DOC, WEEKLY):
        assert after[identity].name == before[identity].name
        assert after[identity].classes[0] == before[identity].classes[0] == identity
        assert sorted(after[identity].classes) == sorted((*before[identity].classes, grown))
