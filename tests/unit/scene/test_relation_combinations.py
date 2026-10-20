"""几个条件合起来的关系（裁定 61 第 6 条、62 第 1、3 条）：条件集合的键、合起来判、"全成立"对"少任意一个"分组；
第二个条件起只长在已成立的带条件关系上、最多几个做配置；父退回组合跟着清；成立的关系每晚也判效应大小。"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from habitus.scene.relations import LaneState, LaneTests, Relation, RelationKey, Segment, Status, build_timelines, fold
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
    # 分组："全成立"对"少任意一个"——两边都不成立的链不进任何一组
    measure = split(ChainMeasure(tests.strata(key(DISCUSS, CODE))), [single[PRIMED], single[coded]])
    inside = {item.chain for item in measure.inside}
    outside = {item.chain for item in measure.outside}
    assert inside and all(flags[chain] == (True, True) for chain in inside)
    assert all(sum(1 for flag in flags[chain] if flag is False) == 1 for chain in outside)
    assert all(chain not in inside | outside for chain, pair in flags.items() if pair == (False, False))


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
