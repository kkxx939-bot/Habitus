"""V2.2：带条件的同一条链也配两道对照（(a)），带条件那一族多一条按（前因, 后果, 段）分组的放行路（(b)）。

(a) 不带条件的同一条链早就配两道对照（``Pairing.SAME_DAY`` + ``Pairing.SAME_SLOT``，两道都显著才算）；带条件的同一条链
两道各按条件分组一次、合起来读：p 与最小可达 p 取两道大的，某一道某一边不到 ``MIN_PER_SIDE`` 就开不了测。别的段的带条件关系、
不带条件的关系、长跨度关系读数不变。

(b) 父关系那条路（裁定 95 第 2 条）没放行的带条件短跨度键，按（前因, 后果, 段）分组，组内只算最小可达 p < 1 的键，组 p 取
Simes 合并 ``min_i(k · p_(i) / i)``；所有组之间做一次 Tarone + BH；过了的组整组进带条件那一族。父关系已放行的照旧进族，不进分组。

这里只看对外结果（``LaneTests.night`` 的读数与判定）与第 1 道分族函数 ``_families`` 的返回，不 mock 被测逻辑。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from datetime import date, timedelta

import pytest

from habitus.scene.relations import LaneNight, LaneTests, RelationKey, Segment, Verdict, build_timelines, stats
from habitus.scene.relations.conditions import (
    CONDITION_JOIN,
    Condition,
    Field,
    ModeratedMeasure,
    Template,
    ready_to_test,
    split,
)
from habitus.scene.relations.engine import RelationTest
from habitus.scene.relations.evaluation import _families
from habitus.scene.relations.measures import ChainMeasure, Measure, PairedMeasure
from habitus.scene.relations.spans import Pairing
from habitus.series import EventSeries
from tests.unit.scene.relation_fixtures import (
    CODE,
    CONCEPTS,
    CONFIG,
    DISCUSS,
    DOCS,
    PUSH,
    RESEARCH,
    key,
    record,
    workdays,
)
from tests.unit.scene.test_relation_family_hierarchy import (
    LONG_CONCEPTS,
    _orphans,
    _series,
    _tests,
    _with_burst,
)

ALPHA = CONFIG.thresholds.forward_alpha
EXACT = CONFIG.thresholds.exact_strata_limit
Q = CONFIG.thresholds.fdr

#: 讨论方案之前 W 槽内先调研过（定点作息里就是"那天整串晨间作息都在"）。
PRIMED = Condition(Template.TIMELINE, RESEARCH).text
PARENT = key(DISCUSS, CODE)
CHILD = key(DISCUSS, CODE, condition=PRIMED)


# ── 造数：定点晨间作息 ──────────────────────────────────────────────────────────


def _routine(
    days: int = 40,
    *,
    seed: int = 3,
    on_routine: float = 1.0,
    off_routine: float = 0.7,
    fixed_off: bool = False,
) -> EventSeries:
    """每个工作日 7:00 写文档开工、17:50 写文档收工，上午随机一次调研、一半的日子随机一次提交推送。
    一半的日子"整串晨间作息都在"：7:25 调研、7:50 讨论方案，再以 ``on_routine`` 的概率 8:20 修改代码。其余日子 8:20 以
    ``off_routine`` 的概率照样修改代码（修改代码是定点作息，与讨论方案无关）；讨论方案在下午随机时刻
    （``fixed_off`` 时改成 7:50 讨论方案、另在下午随机时刻再讨论一次）。

    条件"讨论方案之前先调研过"只在晨间作息齐全的日子成立，于是与"那天 8:20 有修改代码"相关：第 1 道对照（同一天别的事做完的
    时刻）在两组之间抵不掉这个钟点偏差，第 2 道对照（别的周同周几同钟点、没做讨论方案）抵得掉。"""

    generator = random.Random(seed)
    calendar = workdays(days)
    records = []
    for day in calendar:
        number = 0

        def add(hour: int, minute: int, title: str, day: date = day) -> None:
            nonlocal number
            number += 1
            records.append(record(day, hour, minute, title, number=number))

        def afternoon() -> tuple[int, int]:
            return divmod(generator.randint(13 * 60, 16 * 60), 60)

        add(7, 0, "写文档")
        if generator.random() < 0.5:
            add(7, 25, "调研")
            add(7, 50, "讨论方案")
            if generator.random() < on_routine:
                add(8, 20, "修改代码")
        else:
            if generator.random() < off_routine:
                add(8, 20, "修改代码")
            if fixed_off:
                add(7, 50, "讨论方案")
            add(*afternoon(), "讨论方案")
        add(*divmod(generator.randint(10 * 60, 11 * 60 + 30), 60), "调研")
        if generator.random() < 0.5:
            add(*divmod(generator.randint(10 * 60, 16 * 60), 60), "提交推送")
        add(17, 50, "写文档")
    records.sort(key=lambda item: (item.started_at, item.uri))
    return EventSeries(cutoff=calendar[-1] + timedelta(days=1), records=tuple(records))


def _lane(series: EventSeries) -> LaneTests:
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    return LaneTests(line, CONCEPTS, CONFIG, series.cutoff)


def _split_by(tests: LaneTests, item: RelationKey, pairing: Pairing) -> ModeratedMeasure:
    """只按一道对照把层按条件分组（V2.2 之前带条件的只做第 1 道这一份）。"""

    judges = [tests.condition_judge(item.antecedent, condition) for condition in item.conditions]
    return split(ChainMeasure(tests.strata(item, pairing=pairing)), judges)


def _p(measure: Measure) -> float:
    return stats.two_sided(*measure.tails(exact_limit=EXACT))


def _by_key(night: LaneNight) -> dict[RelationKey, RelationTest]:
    return {test.key: test for test in night.tests}


# ── 1. 不带条件、长跨度、非同一条链的带条件：读数与 V2 逐字相同 ─────────────────────


def _rounded(value: object) -> object:
    """浮点保留 10 位有效数字（各平台 libm 末位可能差一点），其余原样。"""

    if isinstance(value, float):
        return float(f"{value:.10g}")
    if isinstance(value, dict):
        return {name: _rounded(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    if isinstance(value, Verdict | Segment):
        return value.value
    return value


def _unchanged_rows(night: LaneNight) -> list[object]:
    """V2.2 不该动的读数：不带条件的、长跨度的整条检验（含判定、第 2/3 道读数）；非同一条链的带条件键只看读数本身
    （p、最小可达 p、方向、效应、次数与块数）——它们的判定会随 (b) 的放行路变，不在这里看。"""

    rows: list[object] = []
    for test in sorted(night.tests, key=lambda item: item.key.text):
        item = test.key
        if item.condition and item.segment is Segment.CHAIN:
            continue
        if item.condition:
            rows.append(
                [
                    item.text,
                    test.p_value,
                    test.minimum_p,
                    test.upward,
                    dataclasses.asdict(test.effect),
                    test.antecedents,
                    test.blocks,
                ]
            )
        else:
            whole = dataclasses.asdict(test)
            whole["key"] = item.text
            rows.append(whole)
    return rows


def _digest(rows: list[object]) -> str:
    text = json.dumps(_rounded(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


#: 在改动前的工作区（7dffb51 + E1–E5 + V2，未提交）上跑出的（行数, 摘要）。
GOLDEN_UNCHANGED = {
    "pure_interaction_with_colleague": (173, "e9b8d34403cfebd4"),
    "long_span_burst": (367, "7756b41bc26bd3e9"),
    "routine": (110, "c671d00b2f26d580"),
}


def _unchanged_case(case: str) -> LaneNight:
    if case == "pure_interaction_with_colleague":
        return _tests(_series(40, primed=0.3, on=1.0, off=0.0, people=1)).night()
    if case == "long_span_burst":
        return _tests(_with_burst(2), concepts=LONG_CONCEPTS).night()
    return _lane(_routine()).night()


@pytest.mark.parametrize("case", sorted(GOLDEN_UNCHANGED))
def test_readings_outside_conditioned_chain_keys_are_unchanged(case: str) -> None:
    """场景 1：不带条件的（任何段）、长跨度的、非同一条链的带条件键，读数与 V2 逐字相同（浮点 10 位有效数字）。"""

    rows = _unchanged_rows(_unchanged_case(case))
    assert (len(rows), _digest(rows)) == GOLDEN_UNCHANGED[case]


def test_measure_kinds_outside_conditioned_chain_keys_are_unchanged() -> None:
    """场景 1（量法种类）：不带条件的同一条链是两道成对（``PairedMeasure``），不带条件的别的段一道（``ChainMeasure``）；
    非同一条链的带条件键仍只有"别的周同周几"一种对照，读数与只按这一道分组逐字相同。"""

    tests = _tests(_series(40, primed=0.3, on=1.0, off=0.0, people=1))
    checked = 0
    for item in tests.keys():
        if item.segment.long or item.segment is Segment.SITUATION:
            continue
        measure = tests.measure(item)
        if not item.condition:
            assert isinstance(measure, PairedMeasure if item.segment is Segment.CHAIN else ChainMeasure), item.text
        elif item.segment is not Segment.CHAIN and tests.sides_reachable(item):
            single = _split_by(tests, item, Pairing.SAME_DAY)
            assert measure.tails(exact_limit=EXACT) == single.tails(exact_limit=EXACT), item.text
            assert measure.minimum_p() == single.minimum_p(), item.text
            checked += 1
    assert checked > 0  # 前提：真有非同一条链、开得了门的带条件键


# ── 2. 定点作息：第 1 道分组显著、第 2 道分组不显著 → 整条不显著 ───────────────────


def test_a_fixed_routine_is_not_read_as_stronger_under_the_condition() -> None:
    """场景 2：讨论方案、修改代码都是定点晨间作息，"先调研过"只在作息齐全的日子成立。只按第 1 道对照分组时
    "讨论方案 → 修改代码｜先调研过"极显著（钟点偏差在两组之间抵不掉）；按第 2 道对照（别的周同周几同钟点）分组时两组一样、
    不显著。两道合起来读：p 取两道大的，整条不显著。"""

    tests = _lane(_routine())
    first = _split_by(tests, CHILD, Pairing.SAME_DAY)
    second = _split_by(tests, CHILD, Pairing.SAME_SLOT)
    # 前提：第 1 道分组显著、第 2 道分组开得了测但不显著
    assert ready_to_test(first, alpha=ALPHA) and _p(first) < 1e-4
    assert ready_to_test(second, alpha=ALPHA) and _p(second) > 0.5
    assert _p(tests.measure(CHILD)) >= _p(second)
    reading = _by_key(tests.night())[CHILD]
    assert reading.p_value >= _p(second)
    assert reading.minimum_p >= max(first.minimum_p(), second.minimum_p())
    assert reading.verdict is not Verdict.SIGNIFICANT


# ── 3. 第 2 道某一边不到 MIN_PER_SIDE → 开不了测，读数 (1, 1) ─────────────────────


def test_a_conditioned_chain_short_on_one_side_of_the_second_pairing_is_not_ready() -> None:
    """场景 3：讨论方案每天 7:50 都有（作息齐全的日子先调研过，其余日子没有），其余日子下午另讨论一次。条件下那几次在别的周
    同一钟点都做了讨论方案，第 2 道对照一个都找不到——第 2 道条件下那一边 0 次；第 1 道两边都够、单看极显著。两道合读开不了测，
    读数 (1, 1)、不进检验族。"""

    tests = _lane(_routine(off_routine=0.0, fixed_off=True))
    first = _split_by(tests, CHILD, Pairing.SAME_DAY)
    second = _split_by(tests, CHILD, Pairing.SAME_SLOT)
    # 前提：第 1 道够、显著；第 2 道条件下不到 MIN_PER_SIDE，条件外够
    assert ready_to_test(first, alpha=ALPHA) and _p(first) < 1e-4
    assert len(second.inside) == 0 and all(len(group) >= 3 for group in second.outsides)
    reading = _by_key(tests.night())[CHILD]
    assert (reading.p_value, reading.minimum_p) == (1.0, 1.0)
    assert reading.verdict is Verdict.SPARSE


# ── 第 1 道分族函数的造数 ────────────────────────────────────────────────────────

#: 另一个条件："讨论方案之前写过文档"。
DOCUMENTED = Condition(Template.TIMELINE, DOCS).text
SIBLING = key(DISCUSS, CODE, condition=DOCUMENTED)
COMBINED = key(DISCUSS, CODE, condition=CONDITION_JOIN.join(sorted((PRIMED, DOCUMENTED))))
OTHER_PARENT = key(PUSH, RESEARCH)
OTHER_CHILD = key(PUSH, RESEARCH, condition=Condition(Template.TIMELINE, DOCS).text)


def _reading(p: float, minimum: float) -> tuple[Measure, float, float, bool]:
    return (ChainMeasure(()), p, minimum, True)


def _family(
    readings: dict[RelationKey, tuple[Measure, float, float, bool]], *, outside: frozenset[RelationKey] = frozenset()
) -> tuple[set[str], set[str], dict[str, int]]:
    inside = [item for item in readings if item not in outside]
    return _families(readings, inside, {}, q=Q, outside=outside)[:3]


# ── 4. 父关系没放行，同组有一条极强的 → 经 Simes 组放行进族 ───────────────────────


def test_a_strong_moderated_key_enters_through_its_group_when_its_parent_did_not_pass() -> None:
    """场景 4（真实时间线）：只有先调研过的讨论方案之后才接修改代码（纯交互）。不带条件的那条没过第 1 道、也不在 ``outside``；
    带条件的那条两道对照都极显著，它那一组（讨论方案, 修改代码, 同一条链）Simes 过了组间 BH：进带条件那一族、过第 1 道。
    （V2 下它判稀疏。）"""

    tests = _tests(_series(40, primed=0.3, on=1.0, off=0.0))
    for pairing in (Pairing.SAME_DAY, Pairing.SAME_SLOT):  # 前提：两道各自都极显著
        assert _p(_split_by(tests, CHILD, pairing)) < 1e-4
    night = tests.night()
    found = _by_key(night)
    assert found[PARENT].verdict is not Verdict.SIGNIFICANT
    assert found[CHILD].verdict is Verdict.SIGNIFICANT
    assert night.family == sum(1 for test in night.tests if test.verdict is not Verdict.SPARSE)


def test_the_group_route_admits_the_whole_group_of_an_unadmitted_parent() -> None:
    """场景 4（分族函数）：父关系 p = 0.5 没过；它的两条带条件子键一条 p = 1e-6、一条 p = 0.6，同组 Simes = 2e-6 过了组间 BH：
    两条都进带条件那一族，强的那条过第 1 道。别的（前因, 后果, 段）组只有一条 p = 0.4 的，不过、不进族。"""

    readings = {
        PARENT: _reading(0.5, 1e-4),
        CHILD: _reading(1e-6, 1e-8),
        SIBLING: _reading(0.6, 1e-3),
        OTHER_PARENT: _reading(0.7, 1e-4),
        OTHER_CHILD: _reading(0.4, 1e-3),
    }
    family, passed, limits = _family(readings)
    assert PARENT.text not in passed
    assert {CHILD.text, SIBLING.text} <= family
    assert OTHER_CHILD.text not in family
    assert CHILD.text in passed and SIBLING.text not in passed
    assert limits[CHILD.text] == 2


def test_keys_that_cannot_reach_the_opening_gate_do_not_dilute_the_group() -> None:
    """场景 4（只算最小可达 p < 1 的键）：同组一条 p = 0.002 的，另 60 条开测门到不了（最小可达 p = 1、p = 1）。只算够得着的
    那一条时组 p = 0.002 < q；把 60 条到不了的也算进 k，组 p = 61 × 0.002 = 0.122 > q = 0.10，白白抬高门槛。"""

    readings = {PARENT: _reading(0.5, 1e-4), CHILD: _reading(0.002, 1e-5)}
    for index in range(60):
        condition = Condition(Template.FIELD, f"人{index}", Field.WITH).text
        readings[key(DISCUSS, CODE, condition=condition)] = _reading(1.0, 1.0)
    _family_set, passed, _limits = _family(readings)
    assert CHILD.text in passed


# ── 5. 组内全是无信号的 → 组不过，与 V2 相同 ────────────────────────────────────


def test_a_group_of_noise_keys_admits_nothing() -> None:
    """场景 5（真实时间线）：讨论方案之后从不特意接修改代码（无信号）。父关系没过、没带着，带条件的组都过不了组间 BH：
    进了族的带条件关系每一条都仍有一条父关系撑腰（与 V2 相同），族的大小 = 进族的条数。"""

    night = _tests(_series(40, primed=0.3, on=0.0, off=0.0)).night()
    assert _by_key(night)[PARENT].verdict is not Verdict.SIGNIFICANT
    assert _orphans(night) == []
    assert night.family == sum(1 for test in night.tests if test.verdict is not Verdict.SPARSE)


def test_a_group_without_signal_stays_out_in_the_family_function() -> None:
    """场景 5（分族函数）：父关系没过；同组三条带条件键 p 都在 0.3–0.9：组 Simes ≥ 0.3，过不了，三条都不进族。"""

    readings = {
        PARENT: _reading(0.5, 1e-4),
        CHILD: _reading(0.3, 1e-3),
        SIBLING: _reading(0.6, 1e-3),
        COMBINED: _reading(0.9, 1e-3),
    }
    family, passed, _limits = _family(readings)
    assert family == {PARENT.text}
    assert passed == set()


# ── 6. 父关系已放行的：照旧进族，不因 (b) 重复计数 ──────────────────────────────


def test_parent_admitted_keys_enter_once_and_do_not_pull_their_group_in() -> None:
    """场景 6（分族函数）：父关系 p = 1e-6 过了第 1 道，两条单条件子键随之进族（一条 p = 1e-6、一条 p = 0.6）。两个条件的
    组合 p = 0.5，它的父是那两条单条件键——不在不带条件那一族、也不在 ``outside``，父关系这条路放不了它；它那一组只剩它
    自己（父关系已放行的两条不进分组），Simes = 0.5 过不了：带条件那一族仍是那两条（族大小 2，不是 3）。"""

    readings = {
        PARENT: _reading(1e-6, 1e-8),
        CHILD: _reading(1e-6, 1e-8),
        SIBLING: _reading(0.6, 1e-3),
        COMBINED: _reading(0.5, 1e-3),
    }
    family, passed, limits = _family(readings)
    assert {CHILD.text, SIBLING.text} <= family
    assert COMBINED.text not in family
    assert limits[CHILD.text] == limits[SIBLING.text] == 2
    assert CHILD.text in passed and PARENT.text in passed


def test_a_parent_carried_outside_still_admits_its_children() -> None:
    """场景 6（``outside``）：父关系跨夜带着（不进任何族）：两条子键照旧进族，族大小 2。"""

    readings = {PARENT: _reading(0.5, 1e-4), CHILD: _reading(1e-6, 1e-8), SIBLING: _reading(0.6, 1e-3)}
    family, passed, limits = _family(readings, outside=frozenset((PARENT,)))
    assert family == {CHILD.text, SIBLING.text}
    assert passed == {CHILD.text}
    assert limits[CHILD.text] == 2


def test_a_moderated_relation_whose_parent_passes_still_enters_on_a_real_timeline() -> None:
    """场景 6（真实时间线）：60 个工作日、一半先调研过——不带条件的那条过第 1 道，带条件的那条照旧进族、过第 1 道；
    族的大小 = 进族的条数（没有一条被数两遍）。"""

    night = _tests(_series(60, primed=0.5, on=1.0, off=0.0)).night()
    found = _by_key(night)
    assert found[PARENT].verdict is Verdict.SIGNIFICANT
    assert found[CHILD].verdict is Verdict.SIGNIFICANT
    assert night.family == sum(1 for test in night.tests if test.verdict is not Verdict.SPARSE)
