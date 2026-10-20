"""几个条件合起来的关系（裁定 61 第 6 条、62 第 1、3 条、63 第 1 条）：条件集合的键、合起来判、每个条件各对"其余成立、它不成立"分组；
第二个条件起只长在已成立的带条件关系上、最多几个做配置；父退回组合跟着清；成立的关系每晚也判效应大小。"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from habitus.scene.relations import (
    LaneState,
    LaneTests,
    Relation,
    RelationKey,
    RelationTest,
    Segment,
    Status,
    build_timelines,
    fold,
)
from habitus.scene.relations.conditions import (
    CONDITION_JOIN,
    Condition,
    Template,
    evaluator,
    evaluator_all,
    join_conditions,
    parse_conditions,
    split,
)
from habitus.scene.relations.config import RelationThresholds
from habitus.scene.relations.measures import ChainMeasure
from tests.unit.scene.relation_fixtures import CODE, CONCEPTS, CONFIG, DISCUSS, PUSH, RESEARCH, key, planted

PRIMED = Condition(Template.TIMELINE, RESEARCH)
WROTE = Condition(Template.TIMELINE, PUSH)


def lane_tests(series, *, established=(), config=CONFIG):  # type: ignore[no-untyped-def]
    line = build_timelines(series, CONCEPTS, {}, config)["session"]
    return LaneTests(line, CONCEPTS, config, series.cutoff, established=established)


def test_condition_sets_have_one_key_and_combine_when_judged() -> None:
    both = join_conditions((WROTE, PRIMED))
    assert both == join_conditions((PRIMED, WROTE)) == CONDITION_JOIN.join(sorted((PRIMED.text, WROTE.text)))
    assert set(parse_conditions(both)) == {PRIMED, WROTE} and parse_conditions("") == ()
    composite = RelationKey("session", DISCUSS, CODE, Segment.CHAIN, both)
    assert set(composite.conditions) == {PRIMED, WROTE}
    assert set(composite.parents()) == {
        replace(composite, condition=PRIMED.text),
        replace(composite, condition=WROTE.text),
    }
    assert replace(composite, condition=PRIMED.text).parents() == (replace(composite, condition=""),)
    assert replace(composite, condition="").parents() == ()

    series = planted(30, moderated=True)
    tests = lane_tests(series)
    # 第二个条件取"之前改过代码"：合成数据里它在讨论方案之前有时成立有时不成立（只为了判法，这里不管它是不是后果自己）
    coded = Condition(Template.TIMELINE, CODE)
    judge = evaluator_all(tests.line, (PRIMED, coded), window=CONFIG.transition_window_slots)
    single = {
        condition: evaluator(tests.line, condition, window=CONFIG.transition_window_slots)
        for condition in (PRIMED, coded)
    }
    flags = {chain: (single[PRIMED](chain), single[coded](chain)) for chain in tests.chains(DISCUSS)}
    for chain, (first, second) in flags.items():
        expected = False if False in (first, second) else (None if None in (first, second) else True)
        assert judge(chain) is expected
    # 分组：每个条件一组对照——"全成立"对"其余成立、它不成立"（Webb 2007 附录 A）；两边都不成立的链不进任何一组
    measure = split(ChainMeasure(tests.strata(key(DISCUSS, CODE))), [single[PRIMED], single[coded]])
    inside = {item.chain for item in measure.inside}
    assert inside and all(flags[chain] == (True, True) for chain in inside)
    assert len(measure.outsides) == 2
    assert all(flags[item.chain] == (False, True) for item in measure.outsides[0])
    assert all(flags[item.chain] == (True, False) for item in measure.outsides[1])
    grouped = inside | {item.chain for group in measure.outsides for item in group}
    assert all(chain not in grouped for chain, pair in flags.items() if pair == (False, False))


def test_each_condition_is_read_against_its_own_outside_so_a_free_rider_condition_cannot_pass() -> None:
    """裁定 63 第 1 条（Webb 2007 附录 A）：{C1, C2} 要 C1 在 C2 成立时加得上东西、C2 在 C1 成立时也加得上东西。C1 自己抬升
    1.0、C2 什么都不解释：把两组对照混成一池（旧法）时组合看着比池子强；按条件各比时 C2 那组交互为 0，整条关系按最难过的
    那组读——不显著，区间近端 0、够不上"影响够大"。"""

    from habitus.scene.relations import stats
    from habitus.scene.relations.conditions import ModeratedMeasure
    from habitus.scene.relations.spans import Stratum

    def side(count: int, treated: int, start: int) -> tuple[Stratum, ...]:
        return tuple(
            Stratum(chain=None, treated=treated, positives=15, controls=50, block=start + index)  # type: ignore[arg-type]
            for index in range(count)
        )

    inside = side(8, 1, 0)  # C1 且 C2：抬升 0.7
    without_c1 = side(8, 0, 100)  # C2 成立、C1 不成立：抬升 −0.3 → C1 加了 1.0
    without_c2 = side(8, 1, 200)  # C1 成立、C2 不成立：抬升 0.7 → C2 加了 0
    exact = CONFIG.thresholds.exact_strata_limit
    alpha = CONFIG.thresholds.forward_alpha
    options = dict(level=CONFIG.thresholds.interval_level, rounds=CONFIG.thresholds.bootstrap_rounds, seed=7)
    only_c1 = ModeratedMeasure(inside, (without_c1,), enough=True)
    assert stats.two_sided(*only_c1.tails(exact_limit=exact)) < alpha
    assert only_c1.interval(**options)[0] == pytest.approx((1.0, 1.0))
    pooled = ModeratedMeasure(inside, ((*without_c1, *without_c2),), enough=True)
    assert stats.two_sided(*pooled.tails(exact_limit=exact)) < alpha  # 混成一池：C2 搭着 C1 的便车也"显著"
    both = ModeratedMeasure(inside, (without_c1, without_c2), enough=True)
    assert both.units == 24 and both.blocks == 24
    assert stats.two_sided(*both.tails(exact_limit=exact)) >= 0.99  # C2 那组交互 0，按最难过的那组读
    assert both.interval(**options)[0] == pytest.approx((0.0, 1.0))  # 近端取最不利的那组：够不上 min_absolute_lift
    assert not both.leave_one_out(upward=True)  # C2 那组去掉任何一块交互仍是 0
    assert only_c1.leave_one_out(upward=True)
    assert both.minimum_p() <= only_c1.minimum_p()  # 两组对照都走到极端时 C2 那组也能显著：开测
    assert both.subset().units == both.units and both.subset(since=None).outsides[1] == without_c2


def test_a_second_condition_grows_only_on_an_established_conditional_relation_and_is_capped() -> None:
    series = planted(60, moderated=True)
    parent = key(DISCUSS, CODE, condition=PRIMED.text)
    plain = lane_tests(series).keys()
    assert parent in plain
    assert not any(CONDITION_JOIN in item.condition for item in plain)
    grown = lane_tests(series, established=(parent,)).keys()
    combos = [item for item in grown if CONDITION_JOIN in item.condition]
    assert combos and all(
        item.antecedent == DISCUSS and item.consequent == CODE and item.segment is Segment.CHAIN for item in combos
    )
    assert all(PRIMED in item.conditions and len(item.conditions) == 2 for item in combos)
    assert all(parent in item.parents() for item in combos)
    # 不带条件的成立关系不往上长单条件（单条件照旧对每个前因都列，不靠它）；上限 1 时一个组合都不长
    assert lane_tests(series, established=(key(DISCUSS, CODE),)).keys() == plain
    capped = replace(CONFIG, thresholds=RelationThresholds(max_combined_conditions=1))
    assert not any(
        CONDITION_JOIN in item.condition for item in lane_tests(series, established=(parent,), config=capped).keys()
    )


def test_a_combination_whose_parent_stopped_being_established_is_cleared_with_its_reason() -> None:
    series = planted(60, moderated=True)
    night = series.cutoff - timedelta(days=1)
    parent = key(DISCUSS, CODE, condition=PRIMED.text)
    combo = key(DISCUSS, CODE, condition=join_conditions((PRIMED, WROTE)))
    held = Relation(
        key=parent,
        status=Status.ESTABLISHED,
        upward=True,
        discovered_on=night - timedelta(days=20),
        established_on=night,
        forward_passed_on=night,
        maintained_through=night,
        maintenance_needed=40,
    )
    growing = Relation(key=combo, status=Status.CANDIDATE, upward=True, discovered_on=night, forward_needed=40)
    with_parent = LaneState(lane="session", night=night, family=1, relations=(held, growing))
    state, moved = fold(with_parent, lane_tests(series, established=(parent,)))
    kept = {item.key: item for item in state.relations}
    assert kept[parent].status is Status.ESTABLISHED and kept[combo].status is Status.CANDIDATE
    # 父关系不在了（退回候选）：组合整行清掉，迁移写清楚为什么
    orphaned = LaneState(
        lane="session", night=night, family=1, relations=(replace(held, status=Status.CANDIDATE), growing)
    )
    state, moved = fold(orphaned, lane_tests(series, established=()))
    assert combo not in {item.key for item in state.relations}
    dropped = [item for item in moved if item.key == combo]
    assert dropped and dropped[0].before is Status.CANDIDATE and "一起退回" in dropped[0].reason


def test_an_established_relation_is_assessed_every_night_so_its_effect_size_is_known() -> None:
    series = planted(60)
    night = series.cutoff - timedelta(days=1)
    held = Relation(
        key=key(DISCUSS, CODE),
        status=Status.ESTABLISHED,
        upward=True,
        discovered_on=night - timedelta(days=30),
        established_on=night - timedelta(days=10),
        forward_passed_on=night - timedelta(days=10),
        maintained_through=night,
        maintenance_needed=40,
    )
    state, _ = fold(LaneState(lane="session", night=night, family=1, relations=(held,)), lane_tests(series))
    (relation,) = (item for item in state.relations if item.key == held.key)
    assert relation.status is Status.ESTABLISHED
    assert relation.tonight is not None and relation.tonight.large_enough is True
    assert relation.tonight.interval is not None and relation.tonight.stable is not None


def test_the_nightly_effect_size_of_an_established_relation_has_hysteresis() -> None:
    """裁定 63 第 3 条：线附近的成立关系别一晚是原因、一晚是巧合。进用区间近端过线（和成立那一关一样）；昨晚够大的，今晚要
    远端也掉到线下（``gone``）才翻成不够；近端没过、远端过了的维持昨晚。"""

    series = planted(60)
    night = series.cutoff - timedelta(days=1)
    held = Relation(
        key=key(DISCUSS, CODE),
        status=Status.ESTABLISHED,
        upward=True,
        discovered_on=night - timedelta(days=30),
        established_on=night - timedelta(days=10),
        forward_passed_on=night - timedelta(days=10),
        maintained_through=night,
        maintenance_needed=40,
    )

    def tonight(previous: Relation, line: float) -> RelationTest:
        config = replace(CONFIG, thresholds=replace(CONFIG.thresholds, min_absolute_lift=line, min_relative_lift=0.0))
        state, _ = fold(
            LaneState(lane="session", night=night, family=1, relations=(previous,)), lane_tests(series, config=config)
        )
        (relation,) = (item for item in state.relations if item.key == held.key)
        assert relation.status is Status.ESTABLISHED and relation.tonight is not None
        return relation.tonight

    first = tonight(held, CONFIG.thresholds.min_absolute_lift)
    assert first.interval is not None and first.large_enough is True
    low, high = first.interval
    between = (low + high) / 2.0
    # 近端没过线：没有昨晚（第一次判）就是不够大
    assert tonight(held, between).large_enough is False
    assert tonight(replace(held, tonight=replace(first, large_enough=False)), between).large_enough is False
    # 昨晚够大、今晚近端没过但远端过了：维持
    assert tonight(replace(held, tonight=replace(first, large_enough=True)), between).large_enough is True
    # 远端也掉到线下：昨晚够大也翻成不够
    assert tonight(replace(held, tonight=replace(first, large_enough=True)), high + 0.01).large_enough is False
