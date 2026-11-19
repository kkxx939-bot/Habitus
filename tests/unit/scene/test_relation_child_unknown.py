"""下级的记录对上级是"不知道"（裁定 105 第 2 条）：同一件事改记到下级以后，上级不被读成"他不做了"。

摆法：第 0–29 天每天 9:20 记在上级「开发」上，第 30 天起同一件事每天记在它的下级「写函数」上；每天 8:00 调研、17:00 写文档
垫底让人整天在场。走真实入口（``build_timelines`` → ``LaneTests``、``retirement``），只断言读数。

只数直接记录、不记"不知道"时，这几条都失败：上级当后果第 30 天起读成"没来"、当前因被切出一段"停了一阵"、第 30 天起的
日子当"没做上级"的对照、条件窗里只有下级时读成"条件不成立"、关系退休判它久没来。
"""

from __future__ import annotations

from dataclasses import replace

from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import LaneTests, Segment, build_timelines, evaluator, parse_condition
from habitus.scene.relations.retirement import absence, visits
from habitus.series import EventSeries
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.relation_fixtures import CONFIG, DISCUSS, key, record, watching
from tests.unit.scene.test_relation_first_seen_unknown import DAYS, day_of

PARENT = base("开发")
CHILD = base("写函数")
NESTED = ConceptSet(
    [
        base("讨论方案"),
        base("调研"),
        base("写文档"),
        replace(PARENT, classes=(PARENT.name, CHILD.name)),
        CHILD,
    ]
)
SWITCH = 30


def _series(discuss: tuple[int, ...] = ()) -> EventSeries:
    found = []
    for offset in range(DAYS):
        found += [record(day_of(offset), 8, 0, "调研"), record(day_of(offset), 17, 0, "写文档")]
        found.append(record(day_of(offset), 9, 20, "开发" if offset < SWITCH else "写函数"))
    found += [record(day_of(offset), 9, 0, "讨论方案") for offset in discuss]
    found.sort(key=lambda item: (item.started_at, item.uri))
    return EventSeries(cutoff=day_of(DAYS), records=tuple(found))


def _tests(series: EventSeries) -> LaneTests:
    return LaneTests(build_timelines(series, NESTED, {}, CONFIG)["session"], NESTED, CONFIG, series.cutoff)


def test_a_day_with_only_the_child_is_unknown_for_the_parent_as_consequent() -> None:
    """第 40 天讨论方案，次日只有下级：对上级既不算来了、也不算没来，这一层不存在；第 20 天那次照常（次日上级来了）。"""

    tests = _tests(_series(discuss=(20, 40)))
    strata = tests.strata(key(DISCUSS, PARENT.name, Segment.NEXT_DAY))
    assert [(item.chain.day, item.treated) for item in strata] == [(day_of(20), 1)]


def test_the_parent_is_not_read_as_having_dropped_once_the_child_takes_over() -> None:
    tests = _tests(_series())
    assert tests.episodes(PARENT.name, Segment.LOW_STATE) == ()


def test_the_parent_is_not_retired_as_away_while_its_child_keeps_coming() -> None:
    series = _series()
    line = build_timelines(series, NESTED, {}, CONFIG)["session"]
    assert absence(visits(line, PARENT.name), watching(series), CONFIG) is None


def test_a_day_with_only_the_child_is_no_control_for_the_parent_as_antecedent() -> None:
    """上级当前因：对照取"别的周同一个周几、没做上级"的日子；第 30 天起那些日子只有下级，对上级是"不知道"，不当对照——
    第 0–29 天又天天有上级，这一层没有对照。"""

    tests = _tests(_series(discuss=tuple(range(0, DAYS, 3))))
    assert tests.strata(key(PARENT.name, DISCUSS, Segment.NEXT_DAY)) == ()


def test_a_timeline_condition_on_the_parent_with_only_the_child_in_the_window_cannot_be_judged() -> None:
    """条件"之前出现过上级"：第 40 天 9:30 讨论方案，前面窗里只有下级 9:20 那一条——判不了（None），第 20 天照常成立。"""

    series = _series()
    found = [*series.records, record(day_of(20), 9, 30, "讨论方案"), record(day_of(40), 9, 30, "讨论方案")]
    series = EventSeries(
        cutoff=series.cutoff, records=tuple(sorted(found, key=lambda item: (item.started_at, item.uri)))
    )
    tests = _tests(series)
    judge = evaluator(tests.line, parse_condition(f"timeline:{PARENT.name}"), window=CONFIG.transition_window_slots)
    assert [(chain.day, judge(chain)) for chain in tests.chains(DISCUSS)] == [(day_of(20), True), (day_of(40), None)]


def test_days_with_only_the_child_are_neither_state_days_nor_ordinary_days_when_measuring() -> None:
    """上级第 30–39 天一天 4 次（一段集中期），第 40 天起只有下级：量"集中期 → 讨论方案"时，第 40 天起的日子对上级是"不知道"，
    既不当状态日、也不当紧挨着的平常日子。"""

    found = []
    for offset in range(DAYS):
        found += [record(day_of(offset), 8, 0, "调研"), record(day_of(offset), 17, 0, "写文档")]
        found.append(record(day_of(offset), 9, 0, "讨论方案"))
        if offset < 40:
            found += [
                record(day_of(offset), 10 + index, 0, "开发", number=index) for index in range(4 if offset >= 30 else 1)
            ]
        else:
            found.append(record(day_of(offset), 10, 0, "写函数"))
    found.sort(key=lambda item: (item.started_at, item.uri))
    tests = _tests(EventSeries(cutoff=day_of(DAYS), records=tuple(found)))
    assert tests.episodes(PARENT.name, Segment.HIGH_STATE) != ()
    measure = tests.measure(key(PARENT.name, DISCUSS, Segment.HIGH_STATE))
    assert measure.days and max(measure.days) < day_of(40)
