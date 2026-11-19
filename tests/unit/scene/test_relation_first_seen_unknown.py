"""一件事第一次自然发生之前的日子，对它一律记"不知道"（Q4，场景号见甲批测试输入 B 节）。

那天既不算来了、也不算没来，不进任何一边的统计。"第一次"含下级的记录，不含被提醒之后才做的；第一次那天起照常量。

全部走真实入口（``build_timelines`` → ``LaneTests``），只断言读数，不看时间线内部怎么表示"不知道"。

造数的约定：从 ``START``（周一）起第 i 个日历日叫"第 i 天"；每天 8:00 调研、17:00 写文档两件垫底的事让人整天在场。
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import date, timedelta

import pytest

from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import LaneTests, Segment, Verdict, build_timelines
from habitus.scene.relations.longspan import Episode, EpisodeMeasure, unit_length
from habitus.scene.relations.retirement import absence, visits
from habitus.scene.relations.spans import Pairing, Stratum
from habitus.series import EventSeries, SeriesRecord
from habitus.temporal.absence import Visit
from tests.unit.scene.concept_fixtures import base
from tests.unit.scene.relation_fixtures import (
    CODE,
    CONCEPTS,
    CONFIG,
    DISCUSS,
    RESEARCH,
    START,
    key,
    record,
    watching,
)

#: 后果第一次自然发生的那天（第 30 天，周三）。
FIRST = 30
#: 序列一共多少天（截止日是第 60 天）。
DAYS = 60
#: 前因只在单数周的周四 9:00 做，双数周的周四留作对照（B3、B4）。
ODD_WEEK_THURSDAYS = (10, 24, 38, 52)


def day_of(offset: int) -> date:
    return START + timedelta(days=offset)


def _series(
    antecedent: Iterable[int] = (),
    consequent: Iterable[int] = (),
    *,
    reminded: Iterable[int] = (),
    absent: Collection[int] = (),
    begin: int = 0,
    consequent_at: tuple[int, int] = (9, 20),
    consequent_title: str = "修改代码",
    per_day: Mapping[int, int] | None = None,
    extra: Sequence[SeriesRecord] = (),
) -> EventSeries:
    """``antecedent`` 那几天 9:00 讨论方案；``consequent`` 那几天在 ``consequent_at`` 修改代码；``reminded`` 那几天同一时刻
    修改代码、但是被提醒之后才做的。``per_day`` 给了，讨论方案改成第 i 天 10:00 起每小时一次、一共 ``per_day[i]`` 次。
    ``absent`` 那几天和 ``begin`` 之前的日子没有垫底的事（人不在场）。"""

    hour, minute = consequent_at
    found: list[SeriesRecord] = list(extra)
    for offset in range(begin, DAYS):
        if offset not in absent:
            found += [record(day_of(offset), 8, 0, "调研"), record(day_of(offset), 17, 0, "写文档")]
    found += [record(day_of(offset), 9, 0, "讨论方案") for offset in antecedent]
    for offset, count in (per_day or {}).items():
        found += [record(day_of(offset), 10 + index, 0, "讨论方案", number=index) for index in range(count)]
    found += [record(day_of(offset), hour, minute, consequent_title) for offset in consequent]
    found += [replace(record(day_of(offset), hour, minute, consequent_title), reminded=True) for offset in reminded]
    found.sort(key=lambda item: (item.started_at, item.uri))
    return EventSeries(cutoff=day_of(DAYS), records=tuple(found))


def _tests(series: EventSeries, concepts: ConceptSet = CONCEPTS) -> LaneTests:
    return LaneTests(build_timelines(series, concepts, {}, CONFIG)["session"], concepts, CONFIG, series.cutoff)


def _on(strata: Iterable[Stratum], offset: int) -> list[Stratum]:
    """前因落在第 ``offset`` 天的那些层。"""

    return [item for item in strata if item.chain.day == day_of(offset)]


def _burst(*, burst: range) -> dict[int, int]:
    """讨论方案平时一天 1 次，``burst`` 那几天一天 6 次：一段集中期。"""

    return {offset: (6 if offset in burst else 1) for offset in range(DAYS)}


# --- 长跨度 --------------------------------------------------------------------------------------


def test_b1_controls_before_the_first_time_leave_the_long_span_table_without_a_control() -> None:
    """T1 / B1：前因第 20 天第一次出现（状态从这天起），后果第 24 天第一次出现、之后天天来。状态前的对照全是"不知道"，
    这张表没有对照：没有读数，当晚样本不够。"""

    tests = _tests(_series(range(20, DAYS), range(24, DAYS)))
    relation = key(DISCUSS, CODE, Segment.FIRST_SEEN)
    assert tests.episodes(DISCUSS, Segment.FIRST_SEEN) == (Episode(day_of(20), day_of(DAYS)),)
    assert tests.measure(relation).units == 0
    (reading,) = [test for test in tests.night().tests if test.key == relation]
    assert reading.verdict is Verdict.SPARSE and reading.antecedents == 0


def test_b1_only_the_controls_after_the_first_time_are_compared() -> None:
    """T1 / B1 的另一半：集中期第 30–49 天，前后各取 20 个平常日子当对照（第 10–29、50–59 天）。后果第 20 天第一次出现、之后隔天
    来，状态里天天来。第 10–19 天是"不知道"，对照只剩第 20–29、50–59 天，一半的日子来了；算进去就会被稀释成三分之一。"""

    outside = [offset for offset in [*range(20, 30), *range(50, DAYS)] if offset % 2 == 0]
    series = _series(consequent=[*outside, *range(30, 50)], consequent_at=(16, 0), per_day=_burst(burst=range(30, 50)))
    tests = _tests(series)
    assert tests.episodes(DISCUSS, Segment.HIGH_STATE) == (Episode(day_of(30), day_of(50)),)
    effect = tests.measure(key(DISCUSS, CODE, Segment.HIGH_STATE)).effect()
    assert (effect.antecedents, effect.treated_rate) == (20, 1.0)
    assert effect.control_rate == pytest.approx(0.5)


def test_b2_a_consequent_seen_from_the_first_day_reads_exactly_as_before() -> None:
    """T2 / B2（防回归）：后果从第 0 天起就有，第 20–27 天天天来。没有"第一次之前"的日子，读数和拿 0/1 手写的日序列算的逐字相同。"""

    came = {*range(0, DAYS, 5), *range(20, 28)}
    series = _series(consequent=sorted(came), consequent_at=(16, 0), per_day=_burst(burst=range(20, 28)))
    tests = _tests(series)
    found = (Episode(day_of(20), day_of(28)),)
    assert tests.episodes(DISCUSS, Segment.HIGH_STATE) == found
    days = tuple(day_of(offset) for offset in range(DAYS))
    outcomes = tuple(int(offset in came) for offset in range(DAYS))
    by_hand = EpisodeMeasure(days, outcomes, found, unit_days=unit_length(days, outcomes, found))
    measure = tests.measure(key(DISCUSS, CODE, Segment.HIGH_STATE))
    assert measure.units == by_hand.units > 0
    assert measure.effect() == by_hand.effect()
    assert measure.tails(exact_limit=400) == by_hand.tails(exact_limit=400)
    assert measure.minimum_p() == by_hand.minimum_p()


# --- 短跨度：窗口全在第一次之前 --------------------------------------------------------------------


def test_b3_rest_of_day_before_the_first_time_is_not_a_stratum() -> None:
    """T3 / B3：前因只在单数周做，后果第 30 天第一次出现。第 10、24 天那两次之后的当天剩下的都是"不知道"，这两层不存在；
    第一次之后的那两次照常成层。"""

    tests = _tests(_series(ODD_WEEK_THURSDAYS, range(FIRST, DAYS), consequent_at=(15, 0)))
    strata = tests.strata(key(DISCUSS, CODE, Segment.REST_OF_DAY))
    assert _on(strata, 10) == [] and _on(strata, 24) == []
    assert [item.treated for item in [*_on(strata, 38), *_on(strata, 52)]] == [1, 1]


@pytest.mark.parametrize(
    ("segment", "before"),
    [(Segment.NEXT_DAY, FIRST - 2), (Segment.DAYS_2_7, FIRST - 8)],
    ids=["B6-next-day", "B7-days-2-7"],
)
def test_b6_b7_a_daily_window_entirely_before_the_first_time_is_not_a_stratum(segment: Segment, before: int) -> None:
    """T3 / B6、B7：次日窗口（前因在第一次的前两天）、第 2–7 天窗口（前因在第一次的前八天）全在第一次之前，这一层不存在。"""

    tests = _tests(_series((before,), range(FIRST, DAYS)))
    assert tests.strata(key(DISCUSS, CODE, segment)) == ()


def test_b4_a_peer_day_before_the_first_time_is_not_a_control() -> None:
    """T4 / B4：第 38 天那次前因的第 2 道对照，别的周同周几、同钟点、没做前因的日子是第 17、31、45、59 天（第 3 天人不在场）。
    第 17 天在第一次之前，不进对照：只剩后三天，三天都来了。"""

    tests = _tests(_series(ODD_WEEK_THURSDAYS, range(FIRST, DAYS), absent=(3,)))
    (stratum,) = _on(tests.strata(key(DISCUSS, CODE), pairing=Pairing.SAME_SLOT), 38)
    assert (stratum.treated, stratum.controls, stratum.positives) == (1, 3, 3)


@pytest.mark.parametrize(
    ("segment", "before"),
    [(Segment.NEXT_DAY, FIRST - 1), (Segment.DAYS_2_7, FIRST - 7), (Segment.DAYS_2_7, FIRST - 2)],
    ids=["next-day-eve", "days-2-7-last-day-of-window", "days-2-7-first-day-of-window"],
)
def test_b5_a_window_that_reaches_the_first_time_saw_it_come(segment: Segment, before: int) -> None:
    """T5 / B5（防回归）：窗口里含第一次那天，窗口里有真实的后果，这一次就是来了。"""

    tests = _tests(_series((before,), range(FIRST, DAYS)))
    (stratum,) = tests.strata(key(DISCUSS, CODE, segment))
    assert stratum.treated == 1 and stratum.controls >= 1


# --- 从没自然发生过 --------------------------------------------------------------------------------


@pytest.mark.parametrize("reminded", [(), (25, 40)], ids=["only-an-empty-place", "only-reminded-records"])
@pytest.mark.parametrize(
    ("segment", "pairing"),
    [
        (Segment.CHAIN, Pairing.SAME_DAY),
        (Segment.CHAIN, Pairing.SAME_SLOT),
        (Segment.REST_OF_DAY, Pairing.SAME_DAY),
        (Segment.NEXT_DAY, Pairing.SAME_DAY),
        (Segment.DAYS_2_7, Pairing.SAME_DAY),
    ],
    ids=["chain-same-day", "chain-same-slot", "rest-of-day", "next-day", "days-2-7"],
)
def test_b8_a_consequent_that_never_came_naturally_has_no_strata(
    segment: Segment, pairing: Pairing, reminded: tuple[int, ...]
) -> None:
    """T6 / B8：后果从没自然发生过（只有空位，或只有被提醒才做的记录），短跨度四段一层都没有。"""

    tests = _tests(_series(range(20, DAYS, 3), reminded=reminded))
    assert tests.strata(key(DISCUSS, CODE, segment), pairing=pairing) == ()


@pytest.mark.parametrize("reminded", [(), (25, 40)], ids=["only-an-empty-place", "only-reminded-records"])
def test_b8_a_consequent_that_never_came_naturally_has_no_long_span_table(reminded: tuple[int, ...]) -> None:
    """T6 / B8：长跨度同样没有表，读数是样本不够。"""

    tests = _tests(_series(range(20, DAYS, 3), reminded=reminded))
    relation = key(DISCUSS, CODE, Segment.FIRST_SEEN)
    assert tests.episodes(DISCUSS, Segment.FIRST_SEEN) != ()
    assert tests.measure(relation).units == 0
    (reading,) = [test for test in tests.night().tests if test.key == relation]
    assert reading.verdict is Verdict.SPARSE


# --- 第一次当天 ------------------------------------------------------------------------------------


def test_b14_earlier_hours_of_the_first_day_are_a_known_no() -> None:
    """T7 / B14（防回归）：后果第一次发生在第 30 天 15:00。那天 9:00 的前因之后窗口里没有后果，是实实在在的"没来"，
    同一天别的事做完的时刻（8:00 调研）也照常当对照。第一次当天整天都算已知。"""

    series = _series((FIRST,), range(FIRST, DAYS), begin=FIRST - 2, consequent_at=(15, 0))
    (stratum,) = _on(_tests(series).strata(key(DISCUSS, CODE)), FIRST)
    assert (stratum.treated, stratum.positives) == (0, 0) and stratum.controls >= 1


def test_b14_the_morning_of_the_first_day_is_a_comparable_control_for_a_later_week() -> None:
    """T7 / B14（防回归）：第 37 天 9:00 的前因，第 2 道对照是第 30、44、51、58 天的 9:00（第 30 天之前不在场）。第 30 天 9:00 比
    当天的第一次（15:00）早，仍是可比的对照，结果是没来。"""

    series = _series((FIRST + 7,), range(FIRST, DAYS), begin=FIRST - 2, consequent_at=(15, 0))
    (stratum,) = _tests(series).strata(key(DISCUSS, CODE), pairing=Pairing.SAME_SLOT)
    assert (stratum.treated, stratum.controls, stratum.positives) == (0, 4, 0)


# --- 第一次之后被提醒才做的 --------------------------------------------------------------------------


def test_b15_a_reminded_time_after_the_first_time_is_unknown_for_the_treated_side() -> None:
    """T8 / B15（防回归）：后果从第 0 天起天天自然发生，只有第 17 天是被提醒才做的。第 16 天那次前因的次日只有这条记录：
    结果不知道，这一层不存在。"""

    natural = [offset for offset in range(DAYS) if offset != 17]
    tests = _tests(_series((16,), natural, reminded=(17,)))
    assert tests.strata(key(DISCUSS, CODE, Segment.NEXT_DAY)) == ()


def test_b15_a_reminded_time_after_the_first_time_is_unknown_for_the_control_side() -> None:
    """T8 / B15（防回归）：第 30 天那次前因的次日对照是别的周三（第 2、9、16、23、37、44、51、58 天）。第 16 天的次日只有被提醒才做的
    记录，不进对照：8 天里算 7 天，7 天都来了。"""

    natural = [offset for offset in range(DAYS) if offset != 17]
    tests = _tests(_series((30,), natural, reminded=(17,)))
    (stratum,) = tests.strata(key(DISCUSS, CODE, Segment.NEXT_DAY))
    assert (stratum.treated, stratum.controls, stratum.positives) == (1, 7, 7)


# --- "第一次"怎么算 -------------------------------------------------------------------------------


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


def test_a_record_of_a_child_is_not_the_first_time_of_its_parent() -> None:
    """裁定 105：下级的记录对上级是"不知道"，不算上级的第一次。第 10 天第一次有下级的记录，上级自己的记录第 30 天才有；
    第 20 天那次前因的次日仍在上级第一次之前，对上级是"不知道"，这一层不存在。"""

    child = record(day_of(10), 14, 0, "写函数")
    series = _series((20,), range(FIRST, DAYS), consequent_title="开发", extra=(child,))
    assert _tests(series, NESTED).strata(key(DISCUSS, PARENT.name, Segment.NEXT_DAY)) == ()


def test_days_before_the_first_record_of_a_child_are_unknown_for_its_parent() -> None:
    """下级更早出现也不改变：第 5 天那次前因的次日在上级第一次之前，这一层不存在。"""

    child = record(day_of(10), 14, 0, "写函数")
    series = _series((5,), range(FIRST, DAYS), consequent_title="开发", extra=(child,))
    assert _tests(series, NESTED).strata(key(DISCUSS, PARENT.name, Segment.NEXT_DAY)) == ()


def test_a_reminded_record_of_a_child_is_not_the_first_time_of_its_parent() -> None:
    """B 节"第一次"的定义：被提醒才做的下级记录不算上级的第一次：第 10 天那条是被提醒的，上级的第一次还是第 30 天，第 20 天那次前因的次日是"不知道"。"""

    child = replace(record(day_of(10), 14, 0, "写函数"), reminded=True)
    series = _series((20,), range(FIRST, DAYS), consequent_title="开发", extra=(child,))
    assert _tests(series, NESTED).strata(key(DISCUSS, PARENT.name, Segment.NEXT_DAY)) == ()


def test_a_reminded_record_is_not_the_first_time() -> None:
    """B 节"第一次"的定义：被提醒才做的不算第一次：第 10 天被提醒做过一次，第一次自然发生是第 30 天。第 20 天那次前因的次日是"不知道"，这一层不存在。"""

    tests = _tests(_series((20,), range(FIRST, DAYS), reminded=(10,)))
    assert tests.strata(key(DISCUSS, CODE, Segment.NEXT_DAY)) == ()


def test_the_long_span_starts_counting_from_the_first_natural_time_not_the_reminded_one() -> None:
    """B1、B 节"第一次"的定义：长跨度同一个口径：后果第 5 天被提醒做过一次、第 24 天才第一次自然发生，前因第 20 天起的状态前面仍然没有已知的对照。"""

    tests = _tests(_series(range(20, DAYS), range(24, DAYS), reminded=(5,)))
    assert tests.measure(key(DISCUSS, CODE, Segment.FIRST_SEEN)).units == 0


# --- 规则只管后果 ----------------------------------------------------------------------------------


def test_b18_the_antecedent_side_does_not_depend_on_the_rule() -> None:
    """T12 / B18（防回归）：前因第 20 天第一次出现，第 30–37 天集中。链数、块长、新类第一次出现的状态段、集中期都照前因自己的记录算，
    第一次之前的日子不影响它们。"""

    counts = {offset: (6 if offset in range(30, 38) else 1) for offset in range(20, DAYS)}
    tests = _tests(_series(per_day=counts))
    assert len(tests.chains(DISCUSS)) == sum(counts.values())
    blocks = tests.blocks(DISCUSS)
    assert blocks is not None and blocks.days == 1.0 and blocks.origin == day_of(20)
    assert tests.episodes(DISCUSS, Segment.FIRST_SEEN) == (Episode(day_of(20), day_of(DAYS)),)
    assert Episode(day_of(30), day_of(38)) in tests.episodes(DISCUSS, Segment.HIGH_STATE)


def test_b18_a_reminded_antecedent_still_counts_as_an_antecedent() -> None:
    """T12 / B18（防回归）：被提醒才做的不算后果的第一次，当前因照常：第 15 天那次被提醒的前因自成一条链、标出新类第一次出现，
    之后的后果（从第 0 天起天天有）照常量。之后每 3 天自然做一次，块长 3 天，第 15 天已过"开始记录"那段保护期。"""

    prompted = replace(record(day_of(15), 9, 0, "讨论方案"), reminded=True)
    natural = range(18, DAYS, 3)
    tests = _tests(_series(natural, range(DAYS), extra=(prompted,)))
    assert [chain.day for chain in tests.chains(DISCUSS)] == [day_of(offset) for offset in (15, *natural)]
    assert tests.episodes(DISCUSS, Segment.FIRST_SEEN) == (Episode(day_of(15), day_of(DAYS)),)
    (stratum,) = _on(tests.strata(key(DISCUSS, CODE, Segment.NEXT_DAY)), 15)
    assert stratum.treated == 1


def test_b18_the_antecedent_side_of_a_late_consequent_keeps_every_chain() -> None:
    """T12 / B18（防回归）：后果第 30 天才第一次出现，第一次之前的前因少了层，但前因的链一条不少。"""

    tests = _tests(_series(ODD_WEEK_THURSDAYS, range(FIRST, DAYS)))
    assert [chain.day for chain in tests.chains(DISCUSS)] == [day_of(offset) for offset in ODD_WEEK_THURSDAYS]


def test_b19_the_long_absence_reading_ignores_the_days_before_the_first_time() -> None:
    """T12 / B19（防回归）：关系退休的"久没来"只看来过的日子：第 20–40 天隔天来、之后再没来。读数是久没来，来过的日子一天不差，
    而且把第一次之前的日子整段去掉，读数逐字相同。"""

    came = range(20, 41, 2)
    whole = _series(consequent=came)
    late = _series(consequent=came, begin=20)
    line = build_timelines(whole, CONCEPTS, {}, CONFIG)["session"]
    cut = build_timelines(late, CONCEPTS, {}, CONFIG)["session"]
    assert visits(line, CODE) == tuple(Visit(day_of(offset), prompted=False) for offset in came)
    reading = absence(visits(line, CODE), watching(whole), CONFIG)
    assert reading is not None and (reading.longest, reading.typical) == (2.0, 2.0)
    assert reading == absence(visits(cut, CODE), watching(late), CONFIG)


def test_b2_another_consequent_is_untouched_by_a_late_one() -> None:
    """B2（防回归）：规则按后果各算各的：修改代码第 30 天才出现，不影响从第 0 天起就有的调研，第 10 天那次前因的次日照常量（来了）。"""

    tests = _tests(_series(ODD_WEEK_THURSDAYS, range(FIRST, DAYS)))
    (stratum,) = _on(tests.strata(key(DISCUSS, RESEARCH, Segment.NEXT_DAY)), 10)
    assert stratum.treated == 1
