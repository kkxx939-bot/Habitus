"""第 1 道的检验族分三族、带条件的按层级进族（V2 分族 + 层级，用户 2026-11-08 定；V2.2 加第二条放行路，用户 2026-11-09 定）。

三族各用同一个 q 各控各的：不带条件的短跨度一族；长跨度一族；带条件的短跨度一族，只放进两条路任一放行的：至少有一条直接
父关系（``RelationKey.parents()``）当晚在第 1 族过了第 1 道、或跨夜带着为候选 / 成立；或它所在的（前因, 后果, 段）那一组
（组内键的 Simes 合并 p）过了组间 Tarone + BH。其余这一晚判稀疏（层级假发现率控制，Yekutieli 2008；裁定 95 第 2 条）。

这里只看对外结果：``LaneTests.night`` 的判定与族的大小、``fold`` 产出的候选 / 成立与"靠先验"标记，不看内部函数。

合成数据（``_series``）：40 或 60 个工作日，每天 9:00 调研、17:50 写文档，随机时刻各一次修改代码、调研，三成的日子提交推送；
讨论方案一天 1–2 次，``primed`` 的概率在它之前 15–30 分钟先调研，先调研过的以 ``on``、没调研过的以 ``off`` 的概率在
15–30 分钟后接修改代码。``people`` 个同事各以一半的概率出现在每条记录上——这些"同在的人"是机械列举的条件，
每多一个同事，每个前因就多一批带条件的键，用来把旧的单一短跨度族撑大。同事用另一个随机数发生器，行为时间线与同事多少无关。
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import replace
from datetime import date, timedelta

import pytest

from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import (
    LaneNight,
    LaneState,
    LaneTests,
    Relation,
    RelationKey,
    RelationTest,
    Status,
    Verdict,
    build_timelines,
    fold,
)
from habitus.scene.relations.conditions import CONDITION_JOIN, Condition, Field, Template
from habitus.scene.relations.longspan import MEASURE_VERSIONS
from habitus.series import EventSeries
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.relation_fixtures import (
    CODE,
    CONCEPTS,
    CONFIG,
    DISCUSS,
    RESEARCH,
    TITLES,
    key,
    record,
    watching,
    workdays,
)

#: 讨论方案之前 15–30 分钟先调研过。
PRIMED = Condition(Template.TIMELINE, RESEARCH).text
PARENT = key(DISCUSS, CODE)
CHILD = key(DISCUSS, CODE, condition=PRIMED)


def _series(
    days: int,
    *,
    primed: float,
    on: float,
    off: float,
    people: int = 0,
    seed: int = 7,
    extra: Iterable[tuple[int, int, int, str]] = (),
) -> EventSeries:
    """见模块说明。``extra``：另加的记录（第几个工作日, 时, 分, 类名）。"""

    generator = random.Random(seed)
    company = random.Random(seed + 1000)
    names = [f"同事{index}" for index in range(people)]
    calendar = workdays(days)
    records = []
    for day in calendar:
        number = 0

        def add(hour: int, minute: int, title: str, day: date = day) -> None:
            nonlocal number
            number += 1
            subjects = tuple(name for name in names if company.random() < 0.5)
            records.append(replace(record(day, hour, minute, title, number=number), subjects=subjects))

        def anytime() -> tuple[int, int]:
            return divmod(generator.randint(9 * 60 + 5, 17 * 60 + 15), 60)

        add(9, 0, "调研")
        add(17, 50, "写文档")
        for _ in range(generator.randint(1, 2)):
            hour, minute = anytime()
            primed_here = generator.random() < primed
            if primed_here:
                add(*divmod(hour * 60 + minute - generator.randint(15, 30), 60), "调研")
            add(hour, minute, "讨论方案")
            if generator.random() < (on if primed_here else off):
                add(*divmod(hour * 60 + minute + generator.randint(15, 30), 60), "修改代码")
        add(*anytime(), "修改代码")
        add(*anytime(), "调研")
        if generator.random() < 0.3:
            add(*anytime(), "提交推送")
    for index, (offset, hour, minute, title) in enumerate(extra):
        records.append(record(calendar[offset], hour, minute, title, number=900 + index))
    records.sort(key=lambda item: (item.started_at, item.uri))
    return EventSeries(cutoff=calendar[-1] + timedelta(days=1), records=tuple(records))


def _tests(
    series: EventSeries, *, concepts: ConceptSet = CONCEPTS, established: Iterable[RelationKey] = ()
) -> LaneTests:
    line = build_timelines(series, concepts, {}, CONFIG)["session"]
    return LaneTests(line, concepts, CONFIG, series.cutoff, established=established)


def _by_key(night: LaneNight) -> dict[RelationKey, Verdict]:
    return {test.key: test.verdict for test in night.tests}


def _orphans(night: LaneNight, outside: frozenset[RelationKey] = frozenset()) -> list[str]:
    """进了族（不是稀疏）、本身不是跨夜带着的带条件短跨度关系里，没有一条直接父关系"当晚在不带条件那一族过了第 1 道
    或跨夜带着为候选 / 成立"的。"""

    plain_passed = {
        test.key
        for test in night.tests
        if test.verdict is Verdict.SIGNIFICANT and not test.key.condition and not test.key.segment.long
    }
    found = []
    for test in night.tests:
        item = test.key
        if not item.condition or item.segment.long or item in outside or test.verdict is Verdict.SPARSE:
            continue
        if not any(parent in outside or parent in plain_passed for parent in item.parents()):
            found.append(item.text)
    return found


def _family_matches_verdicts(night: LaneNight) -> bool:
    """没有跨夜带着的行时：族的大小 = 当晚判为"进族"（没过 / 过了第 1 道）的条数。"""

    return night.family == sum(1 for test in night.tests if test.verdict is not Verdict.SPARSE)


def _previous(series: EventSeries, *relations: Relation) -> LaneState:
    return LaneState(
        lane="session",
        night=series.cutoff - timedelta(days=1),
        family=0,
        relations=relations,
        measure_versions=MEASURE_VERSIONS,
    )


def _carried(item: RelationKey, status: Status, days: list[date]) -> Relation:
    """上一晚带着的一行（与 ``test_relation_exact_shortcuts._carried_previous`` 同样的写法）。"""

    if status is Status.CANDIDATE:
        return Relation(key=item, status=status, upward=True, discovered_on=days[20], forward_needed=5)
    if status is Status.ESTABLISHED:
        return Relation(
            key=item,
            status=status,
            upward=True,
            discovered_on=days[5],
            forward_passed_on=days[15],
            established_on=days[15],
            maintained_through=days[30],
            maintenance_needed=4,
        )
    if status is Status.EXPIRED:
        return Relation(
            key=item, status=status, upward=True, discovered_on=days[2], established_on=days[6], expired_on=days[10]
        )
    return Relation(key=item, status=status, upward=True, discovered_on=days[2], rejected_on=days[3])


def _statuses(state: LaneState) -> dict[RelationKey, Status]:
    """存了行的关系 → 状态（稀疏的只计数、不在里面）。"""

    return {item.key: item.status for item in state.relations}


def _tonight(state: LaneState, item: RelationKey) -> RelationTest:
    """存了行的关系当晚的读数。"""

    (row,) = [relation for relation in state.relations if relation.key == item]
    assert row.tonight is not None
    return row.tonight


# ── 1. 不带条件的强关系不再被大批带条件的键挤出族 ─────────────────────────────────


def test_a_plain_relation_is_not_crowded_out_by_a_flood_of_moderated_keys() -> None:
    """讨论方案之后两成的时候接修改代码（不管先调研没有）。没有同事时它过第 1 道、成为候选；同一份行为时间线再加 30 个
    同事，每个前因多出几百条带条件的键——单一的短跨度族被撑到两百多条、门槛压到 q / 250 上下，这条关系就掉出去了。
    分族之后不带条件的自成一族，它照样过第 1 道、成为候选。"""

    quiet = _tests(_series(40, primed=0.0, on=0.2, off=0.2))
    alone = quiet.night()
    assert _by_key(alone)[PARENT] is Verdict.SIGNIFICANT  # 前提：没有大批带条件的键时它本来就过得了
    crowded = _tests(_series(40, primed=0.0, on=0.2, off=0.2, people=30))
    moderated = [item for item in crowded.keys() if item.condition and not item.segment.long]
    assert len(moderated) > 2000  # 前提：带条件的键确实成批
    night = crowded.night()
    assert _by_key(night)[PARENT] is Verdict.SIGNIFICANT
    assert _family_matches_verdicts(night)
    again = _series(40, primed=0.0, on=0.2, off=0.2, people=30)
    state, _ = fold(None, _tests(again), watching=watching(again))
    assert _statuses(state)[PARENT] is Status.CANDIDATE


# ── 2. 父关系没过、也没带着：只有所在的组过了才进族 ──────────────────────────────


def test_a_moderated_relation_whose_parent_did_not_pass_stays_sparse_unless_its_group_passes() -> None:
    """只有先调研过的讨论方案之后才接修改代码（纯交互：不带条件的那条整体上不显著）。带条件的那条唯一的父关系当晚没过
    第 1 道、也没跨夜带着，只剩（讨论方案, 修改代码, 同一条链）这一组的放行路：

    - 先调研过之后六成接修改代码：它自己 p 约 0.045，组的 Simes p 过不了组间 BH——这一晚不进任何族、判稀疏，不成为候选；
    - 先调研过之后必接修改代码：它自己 p 约 5e-5，组过了，它经组进族、过第 1 道、成为候选（只在条件下成立、整体看不出的
      关系不再被漏掉）。"""

    weak = _series(40, primed=0.3, on=0.6, off=0.0)
    night = _tests(weak).night()
    tests = {test.key: test for test in night.tests}
    assert tests[PARENT].verdict is not Verdict.SIGNIFICANT  # 前提：父关系没过第 1 道
    assert tests[CHILD].p_value < 0.05  # 读数照量：它自己不弱
    assert tests[CHILD].verdict is Verdict.SPARSE
    assert _family_matches_verdicts(night)
    state, _ = fold(None, _tests(weak), watching=watching(weak))
    assert CHILD not in _statuses(state)  # 稀疏的只计数

    strong = _series(40, primed=0.3, on=1.0, off=0.0)
    night = _tests(strong).night()
    tests = {test.key: test for test in night.tests}
    assert tests[PARENT].verdict is not Verdict.SIGNIFICANT  # 前提：父关系照旧没过、也没带着
    assert tests[CHILD].p_value < 1e-3
    assert tests[CHILD].verdict is Verdict.SIGNIFICANT
    assert _family_matches_verdicts(night)
    state, _ = fold(None, _tests(strong), watching=watching(strong))
    assert _statuses(state)[CHILD] is Status.CANDIDATE


# ── 3. 父关系当晚过了第 1 道：带条件的进族 ───────────────────────────────────────


def test_a_moderated_relation_enters_when_its_parent_passes_tonight() -> None:
    """60 个工作日、一半的讨论方案先调研过：不带条件的那条当晚过了第 1 道，带条件的那条随之进族，也过了第 1 道、成为候选。"""

    series = _series(60, primed=0.5, on=1.0, off=0.0)
    night = _tests(series).night()
    verdicts = _by_key(night)
    assert verdicts[PARENT] is Verdict.SIGNIFICANT
    assert verdicts[CHILD] is Verdict.SIGNIFICANT
    assert _family_matches_verdicts(night)
    state, _ = fold(None, _tests(series), watching=watching(series))
    assert _statuses(state)[CHILD] is Status.CANDIDATE


# ── 4. 父关系跨夜带着：候选 / 成立的算，失效 / 前向未复现的不算 ─────────────────────


@pytest.mark.parametrize(
    ("status", "enters"),
    [
        (Status.CANDIDATE, True),
        (Status.ESTABLISHED, True),
        (Status.EXPIRED, False),
        (Status.REJECTED, False),
    ],
)
def test_a_parent_carried_as_candidate_or_established_lets_the_moderated_relation_in(
    status: Status, enters: bool
) -> None:
    """与第 2 条"组过不了"那份数据（父关系当晚过不了第 1 道，组也放不进来），上一晚的状态里带着父关系：带着为候选 / 成立时，
    带条件的那条进族、过第 1 道、成为候选；父关系是失效 / 前向未复现时不算"带着"，带条件的那条照旧判稀疏。"""

    series = _series(40, primed=0.3, on=0.6, off=0.0)
    previous = _previous(series, _carried(PARENT, status, workdays(40)))
    tests = _tests(series, established=tuple(item.key for item in previous.of(Status.ESTABLISHED)))
    state, _ = fold(previous, tests, watching=watching(series))
    rows = {item.key: item for item in state.relations}
    assert _tonight(state, PARENT).verdict is not Verdict.SIGNIFICANT  # 前提：当晚它自己没过
    if enters:
        assert rows[CHILD].status is Status.CANDIDATE
    else:
        assert CHILD not in rows


# ── 5. 两个条件的组合：单条件的父关系已成立 ───────────────────────────────────────


def test_a_combination_enters_when_its_single_condition_parent_is_established() -> None:
    """第 2 条"组过不了"那份数据（先调研过之后六成接修改代码）再加一个同事（于是有"同在的人"这个条件）。上一晚
    "讨论方案 → 修改代码｜先调研过"已成立，于是今晚长出"｜先调研过 + 同事0 在"的组合（裁定 62 第 1 条）。组合的直接父关系是
    两条单条件的；不带条件的那条既没过第 1 道也没带着，（讨论方案, 修改代码, 同一条链）这一组也过不了组间检验——组合只能靠
    单条件的"先调研过"跨夜带着为成立进族（不是稀疏）。反例：同一份数据、同一批键，"先调研过"不跨夜带着时，组合判稀疏。"""

    series = _series(40, primed=0.3, on=0.6, off=0.0, people=1)
    previous = _previous(series, _carried(CHILD, Status.ESTABLISHED, workdays(40)))
    tests = _tests(series, established=(CHILD,))
    with_colleague = Condition(Template.FIELD, "同事0", Field.WITH).text
    combination = replace(CHILD, condition=CONDITION_JOIN.join(sorted((PRIMED, with_colleague))))
    assert combination in tests.keys()  # 前提：组合长出来了
    assert CHILD in combination.parents()
    alone = _by_key(_tests(series, established=(CHILD,)).night())
    assert alone[PARENT] is not Verdict.SIGNIFICANT
    assert alone[combination] is Verdict.SPARSE  # 反例：父关系都不放行、组也过不了，组合不进族
    state, _ = fold(previous, tests, watching=watching(series))
    rows = {item.key: item for item in state.relations}
    assert PARENT not in previous.carried()
    assert _tonight(state, PARENT).verdict is not Verdict.SIGNIFICANT
    assert rows[CHILD].status is Status.ESTABLISHED
    assert combination in rows
    assert _tonight(state, combination).verdict is not Verdict.SPARSE


# ── 6. 长跨度：族与结果与分族前一致 ─────────────────────────────────────────────

#: 只在集中期出现的一类（长跨度要有个平时不出现的后果才量得出状态的效应）。
REVIEW = "复盘"
LONG_CONCEPTS = ConceptSet([base(title) for title in (*TITLES, REVIEW)])


def _with_burst(people: int) -> EventSeries:
    """第 1 条那份时间线，第 15–22 个工作日提交推送一天 6 次（一段集中期），集中期里每天 16:45 复盘、平时从不复盘。"""

    extra = [(offset, 10 + index, 40, "提交推送") for offset in range(15, 23) for index in range(5)]
    extra += [(offset, 16, 45, REVIEW) for offset in range(15, 23)]
    return _series(40, primed=0.0, on=0.2, off=0.2, people=people, extra=extra)


def test_long_span_family_and_results_are_unchanged_by_the_split() -> None:
    """短跨度那边加了 30 个同事、族的组成变了，长跨度的那一族不受影响：每条长跨度关系的判定、p、最小可达 p、方向与还要几次
    与分族前（7dffb51 + E1–E5 工作区）逐字相同，过第 1 道的那几条照旧成为候选。"""

    series = _with_burst(30)
    night = _tests(series, concepts=LONG_CONCEPTS).night()
    found = sorted(
        (test.key.text, test.verdict.value, test.p_value, test.minimum_p, test.upward, test.more_needed)
        for test in night.tests
        if test.key.segment.long
    )
    assert len(found) == 20
    assert [row for row in found if row[1] != "sparse"] == GOLDEN_LONG_FAMILY
    assert sum(1 for row in found if row[1] == "sparse") == 17
    state, _ = fold(None, _tests(series, concepts=LONG_CONCEPTS), watching=watching(series))
    assert sorted(
        item.key.text for item in state.relations if item.key.segment.long and item.status is Status.CANDIDATE
    ) == sorted(row[0] for row in GOLDEN_LONG_FAMILY if row[1] == "significant")


# ── 7. 靠先验：不加权的那一遍按同一层级规则判 ─────────────────────────────────────


def test_prior_only_follows_the_same_hierarchy_in_the_unweighted_pass() -> None:
    """40 个工作日、六成先调研过、先调研过之后 65% 接修改代码：不带条件的那条不加权过不了第 1 道（p 约 0.04），先验给它 2、
    其余不带条件的 0.5 时过得了。加权那一遍父关系过了，带条件的那条随之进族、过第 1 道、成为候选；不加权那一遍父关系没过、
    它所在的组也过不了，带条件的那条按同一规则不进族——所以它和父关系一样打"靠先验"（不加权就过不了）。带条件的权重恒为 1
    （``runtime/scene_advice.weights`` 只给不带条件的）。"""

    series = _series(40, primed=0.6, on=0.65, off=0.0)
    unweighted = _tests(series).night()
    assert _by_key(unweighted)[PARENT] is not Verdict.SIGNIFICANT  # 前提：不加权时父关系过不了
    tests = _tests(series)
    weights = {item: (2.0 if item == PARENT else 0.5) for item in tests.keys() if not item.condition}
    night = _tests(series).night(weights)
    assert _by_key(night)[PARENT] is Verdict.SIGNIFICANT
    assert _by_key(night)[CHILD] is Verdict.SIGNIFICANT
    assert CHILD not in night.plain_significant
    # 不加权那一遍：过了第 1 道的带条件关系，都有一条直接父关系在不加权那一遍也过了（这份数据上组那条路一条也没放）
    assert not any(
        item.condition and not any(parent in night.plain_significant for parent in item.parents())
        for item in night.plain_significant
    )
    state, _ = fold(None, tests, weights, watching=watching(series))
    rows = {item.key: item for item in state.relations}
    assert rows[PARENT].status is Status.CANDIDATE and rows[PARENT].prior_only
    assert rows[CHILD].status is Status.CANDIDATE and rows[CHILD].prior_only


# ── 8. 层级规则本身：进了族的带条件关系，要么有父关系撑腰，要么所在的组过了 ─────────────────


@pytest.mark.parametrize(
    "case",
    ["crowded", "pure_interaction", "group_fails", "parent_passes", "parent_carried", "combination", "long_span"],
)
def test_no_moderated_relation_enters_the_family_without_a_parent_or_a_group_that_passed(case: str) -> None:
    """上面几份数据逐一看当晚的判定：进了族（没过 / 过了第 1 道）、本身不是跨夜带着的带条件短跨度关系里，没有一条直接父关系
    当晚在不带条件那一族过了第 1 道、也没跨夜带着为候选 / 成立的（"孤儿"），只能是经组进来的：纯交互那份数据上恰好是组过了的
    那条"讨论方案 → 修改代码｜先调研过"，组过不了的那份（``group_fails``）与其余各份一条没有；族的大小 = 进族的条数
    （跨夜带着的不计）。"""

    outside: frozenset[RelationKey] = frozenset()
    concepts = CONCEPTS
    established: tuple[RelationKey, ...] = ()
    if case == "crowded":
        series = _series(40, primed=0.0, on=0.2, off=0.2, people=30)
    elif case == "pure_interaction":
        series = _series(40, primed=0.3, on=1.0, off=0.0)
    elif case == "group_fails":
        series = _series(40, primed=0.3, on=0.6, off=0.0)
    elif case == "parent_passes":
        series = _series(60, primed=0.5, on=1.0, off=0.0)
    elif case == "parent_carried":
        series = _series(40, primed=0.3, on=1.0, off=0.0)
        outside = frozenset((PARENT,))
    elif case == "combination":
        series = _series(40, primed=0.3, on=1.0, off=0.0, people=1)
        outside = frozenset((CHILD,))
        established = (CHILD,)
    else:
        series = _with_burst(30)
        concepts = LONG_CONCEPTS
    night = _tests(series, concepts=concepts, established=established).night(outside=outside, keep=outside)
    assert _orphans(night, outside) == ([CHILD.text] if case == "pure_interaction" else [])
    carried = sum(1 for test in night.tests if test.key in outside and test.verdict is not Verdict.SPARSE)
    assert night.family == sum(1 for test in night.tests if test.verdict is not Verdict.SPARSE) - carried


# ── 在当前代码（7dffb51 + E1–E5 工作区，分族前）上跑出的长跨度期望值 ─────────────────

#: （键, 判定, p, 最小可达 p, 朝上, 还要几次）——进了长跨度族的那几条。
GOLDEN_LONG_FAMILY: list[tuple[str, str, float, float, bool, int | None]] = [
    ("session|s-k621352|s-k659225|high_state", "significant", 2.719345834166133e-06, 2.719345834166133e-06, True, None),
    ("session|s-k659225|s-k621352|first_seen", "tested", 0.12693498452012383, 0.0004422821760283061, True, None),
    (
        "session|s-k659225|s-k621352|high_state",
        "significant",
        0.00044869206263741196,
        0.00044869206263741196,
        True,
        None,
    ),
]
