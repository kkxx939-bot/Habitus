"""长跨度标志与调节条件（语义树新方案 ``13`` ①②，第 6 步）。"""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import LaneTests, RelationKey, Segment, Verdict, build_timelines, examine_night
from habitus.scene.relations.conditions import (
    Condition,
    ConditionError,
    Field,
    ModeratedMeasure,
    Template,
    compile_proposal,
    parse,
)
from habitus.scene.relations.longspan import Episode, EpisodeMeasure, episodes, measure_of, unit_length
from habitus.series import EventSeries, SeriesRecord
from tests.unit.scene.concept_fixtures import situation
from tests.unit.scene.relation_fixtures import (
    CODE,
    CONCEPTS,
    CONFIG,
    DISCUSS,
    DOCS,
    PUSH,
    RESEARCH,
    START,
    planted,
    record,
)

WEEKEND = situation("出差中", "人在常住地之外过夜")


def daily(counts: dict[int, int], *, filler: str = "调研", title: str = "提交推送") -> EventSeries:
    """第 i 天（从 START 起的日历日）：一条 ``filler``（让那天人在场）+ ``counts[i]`` 条 ``title``。"""

    records: list[SeriesRecord] = []
    last = max(counts)
    for offset in range(last + 1):
        day = START + timedelta(days=offset)
        records.append(record(day, 9, 0, filler))
        for index in range(counts.get(offset, 0)):
            records.append(record(day, 10 + index, 0, title, number=index))
    records.sort(key=lambda item: (item.started_at, item.uri))
    return EventSeries(cutoff=START + timedelta(days=last + 1), records=tuple(records))


def line_of(series: EventSeries, concepts: ConceptSet = CONCEPTS):  # type: ignore[no-untyped-def]
    return build_timelines(series, concepts, {}, CONFIG)["session"]


def found(
    series: EventSeries, segment: Segment, concept: str = PUSH, *, block_days: float = 1.0
) -> tuple[Episode, ...]:
    return episodes(
        line_of(series),
        concept,
        segment,
        block_days=block_days,
        first_seen=(CONFIG.thresholds.first_seen_min_days, CONFIG.thresholds.first_seen_min_blocks),
        recurrence_days=CONFIG.recurrence_window_days,
    )


# --- 标志 ----------------------------------------------------------------------------------------


def test_a_burst_is_found_by_change_points_and_lasts_at_least_a_block() -> None:
    """一天 1 次的提交推送，第 20–27 天一天 6 次：一段集中期；常态是最长的那段，没有"停了一阵"。"""

    counts = {day: (6 if 20 <= day < 28 else 1) for day in range(48)}
    series = daily(counts)
    (burst,) = found(series, Segment.HIGH_STATE)
    assert burst == Episode(START + timedelta(days=20), START + timedelta(days=28))
    assert found(series, Segment.LOW_STATE) == ()
    # 状态最短不少于 1 个块长：块长 10 天时，那段集中期被放宽到至少 10 天（把它包在里面）
    (wide,) = found(series, Segment.HIGH_STATE, block_days=10.0)
    assert (wide.end - wide.start).days >= 10 and wide.start <= burst.start and burst.end <= wide.end


def test_first_seen_only_counts_after_recording_has_settled() -> None:
    late = daily({**{day: 0 for day in range(30)}, **{day: 1 for day in range(20, 30)}})
    (first,) = found(late, Segment.FIRST_SEEN)
    assert first.start == START + timedelta(days=20)
    # 第 5 天就出现的是"开始记录"，不是"第一次发生"
    early = daily({**{day: 0 for day in range(30)}, **{day: 1 for day in range(5, 30)}})
    assert found(early, Segment.FIRST_SEEN) == ()


def test_a_return_after_more_than_the_recurrence_window_starts_a_state() -> None:
    counts = {day: 0 for day in range(130)}
    counts.update({day: 1 for day in range(0, 5)})
    counts.update({day: 1 for day in range(110, 130)})
    (back,) = found(daily(counts), Segment.RETURN)
    assert back.start == START + timedelta(days=110) and back.end == START + timedelta(days=130)


def test_situation_days_make_runs() -> None:
    series = daily({day: 1 for day in range(10)})
    line = line_of(series)
    line.situations["出差中"] = {START + timedelta(days=day): day in (2, 3, 4, 7) for day in range(10)}
    runs = episodes(line, "出差中", Segment.SITUATION, block_days=1.0, first_seen=(14, 3), recurrence_days=90)
    assert runs == (
        Episode(START + timedelta(days=2), START + timedelta(days=5)),
        Episode(START + timedelta(days=7), START + timedelta(days=8)),
    )


# --- 长跨度的量法 --------------------------------------------------------------------------------


def burst_with_docs() -> EventSeries:
    """一天 1 次的提交推送，第 20–27 天一天 6 次（一段集中期）；集中期里每天都写文档，平时从不写。"""

    series = daily({day: (6 if 20 <= day < 28 else 1) for day in range(48)})
    records = list(series.records) + [record(START + timedelta(days=day), 16, 0, "写文档") for day in range(20, 28)]
    records.sort(key=lambda item: (item.started_at, item.uri))
    return EventSeries(cutoff=series.cutoff, records=tuple(records))


def test_a_clean_state_effect_is_compared_with_the_days_around_it() -> None:
    """8 天集中期对前后各 8 个平常日子：状态里 8 天全写、前后 16 天一天没写——零假设下这种排法的概率是 1 / C(24, 8)。
    原先的循环平移在 48 天里最小只能到 2 / 48。"""

    series = burst_with_docs()
    measure = measure_of(line_of(series), DOCS, found(series, Segment.HIGH_STATE))
    assert measure.unit_days == 1  # 写文档只在状态里出现，扣掉状态内外之差后没有扎堆
    effect = measure.effect()
    assert (effect.antecedents, effect.treated_rate, effect.control_rate) == (8, 1.0, 0.0)
    upper, lower = measure.tails(exact_limit=0)
    assert upper == pytest.approx(1 / math.comb(24, 8)) and lower == 1.0
    assert measure.minimum_p() == pytest.approx(2 / math.comb(24, 8))
    spread, relative = measure.interval(level=0.95, rounds=200, seed=0)
    assert spread == (1.0, 1.0) and relative is None  # 对照率是 0：说不清相对差
    # 只有一段状态：切成前后两半，两半各自都比对照多
    assert measure.leave_one_out(upward=True) and not measure.leave_one_out(upward=False)


def test_long_span_keys_are_their_own_family() -> None:
    """长跨度自成一族（S4）：48 天里那段集中期带来的写文档过了第 1 道，族的大小只数长跨度。"""

    (night,) = examine_night(burst_with_docs(), CONCEPTS, {}, CONFIG).values()
    key = RelationKey("session", PUSH, DOCS, Segment.HIGH_STATE)
    (burst,) = [test for test in night.tests if test.key == key]
    assert burst.verdict is Verdict.SIGNIFICANT and burst.upward
    long_family = [test for test in night.tests if test.key.segment.long and test.verdict is not Verdict.SPARSE]
    assert long_family and night.family >= len(long_family)


def test_bursty_consequents_are_counted_in_longer_units() -> None:
    """B 一阵一阵来（连着 6 天天天有、再连着 6 天没有），与状态无关：相邻几天不独立，几天并成一个单位再数。"""

    days = tuple(START + timedelta(days=day) for day in range(84))
    bursty = tuple(1 if (day // 6) % 2 == 0 else 0 for day in range(84))
    episode = (Episode(days[30], days[40]),)
    assert unit_length(days, bursty, episode) > 1
    alternating = tuple(day % 2 for day in range(84))
    assert unit_length(days, alternating, episode) == 1


def test_an_episode_measure_cut_to_new_data_keeps_only_new_state_days() -> None:
    """前向验证只数 ``since`` 之后的状态日；对照还是那段状态前后的平常日子（新的一段状态往往从旧数据延续过来）。"""

    days = tuple(START + timedelta(days=day) for day in range(30))
    measure = EpisodeMeasure(
        days=days,
        outcomes=tuple(1 if 3 <= day < 6 or 20 <= day < 24 else 0 for day in range(30)),
        episodes=(Episode(days[3], days[6]), Episode(days[20], days[24])),
    )
    assert measure.units == 7
    later = measure.subset(since=days[22])
    assert later.units == 2 and later.effect().control_rate == 0.0
    assert measure.subset(last_blocks=3).units == 3


# --- 调节条件 ------------------------------------------------------------------------------------


def test_a_moderated_relation_is_found_under_its_condition() -> None:
    """只有先调研过的讨论方案之后才接修改代码：带"之前出现过调研"这个条件的那条显著，而且条件下比条件外强得多。"""

    (night,) = examine_night(planted(80, moderated=True), CONCEPTS, {}, CONFIG).values()
    condition = Condition(Template.TIMELINE, RESEARCH).text
    moderated = next(
        test for test in night.tests if test.key == RelationKey("session", DISCUSS, CODE, Segment.CHAIN, condition)
    )
    assert moderated.verdict is Verdict.SIGNIFICANT and moderated.upward and moderated.large_enough
    assert moderated.interval is not None and moderated.interval[0] > 0.2  # 条件下比对照多 20 个百分点以上
    # "后果刚出现过"不当条件（那是后果自己扎堆，不是调节）
    assert not any(
        parse(test.key.condition).value == test.key.consequent
        for test in night.tests
        if test.key.condition and parse(test.key.condition).template is Template.TIMELINE
    )


def test_moderation_is_only_opened_for_antecedents_with_enough_samples() -> None:
    series = planted(20)
    tests = LaneTests(line_of(series), CONCEPTS, CONFIG, series.cutoff)
    assert len(tests.chains(DISCUSS)) < CONFIG.thresholds.moderation_min_antecedents
    assert not any(key.condition for key in tests.keys() if key.antecedent == DISCUSS)
    assert ModeratedMeasure((), (), enough=False).minimum_p() == 1.0


def test_proposals_compile_into_one_of_three_templates_or_are_refused() -> None:
    concepts = ConceptSet([*CONCEPTS.values(), WEEKEND])
    assert compile_proposal({"template": "field", "field": "weekend"}, concepts, lane="session") == Condition(
        Template.FIELD, "", Field.WEEKEND
    )
    assert compile_proposal(
        {"template": "field", "field": "place", "value": "公司"}, concepts, lane="session"
    ).text == ("field:place=公司")
    assert compile_proposal({"template": "timeline", "concept": "调研"}, concepts, lane="session") == Condition(
        Template.TIMELINE, RESEARCH
    )
    assert (
        compile_proposal({"template": "concept", "concept": "出差中"}, concepts, lane="session").text
        == "concept:出差中"
    )
    for bad, reason in (
        ({"template": "vibes"}, "unknown template"),
        ({"template": "field", "field": "mood"}, "unknown field"),
        ({"template": "field", "field": "place"}, "plain value"),
        ({"template": "timeline", "concept": "不存在"}, "no concept"),
        ({"template": "timeline", "concept": "出差中"}, "behaviour concept"),
        ({"template": "concept", "concept": "调研"}, "situation concept"),
    ):
        with pytest.raises(ConditionError, match=reason):
            compile_proposal(bad, concepts, lane="session")
    with pytest.raises(ConditionError, match="another lane"):
        compile_proposal({"template": "timeline", "concept": "调研"}, concepts, lane="physical")


def test_condition_text_round_trips() -> None:
    for condition in (
        Condition(Template.FIELD, "", Field.WEEKEND),
        Condition(Template.FIELD, "调休", Field.CALENDAR),
        Condition(Template.TIMELINE, RESEARCH),
        Condition(Template.CONCEPT, "出差中"),
    ):
        assert parse(condition.text) == condition


def test_a_proposal_that_repeats_a_mechanical_condition_keeps_its_full_history() -> None:
    """模型提的条件与机械列举撞上（「之前出现过调研」机械列举里本来就有）：照旧在全部历史上量，不被截成"提出之后"（裁定 27 第 10 条，E5）。"""

    series = planted(80, moderated=True)
    proposed_on = series.cutoff - timedelta(days=10)
    condition = Condition(Template.TIMELINE, RESEARCH)
    line = line_of(series)
    mechanical_only = LaneTests(line, CONCEPTS, CONFIG, series.cutoff)
    proposed = LaneTests(line, CONCEPTS, CONFIG, series.cutoff, proposals={DISCUSS: ((condition, proposed_on),)})
    key = RelationKey("session", DISCUSS, CODE, Segment.CHAIN, condition.text)
    full = next(test for test in mechanical_only.night().tests if test.key == key)
    repeated = next(test for test in proposed.night().tests if test.key == key)
    assert repeated.antecedents == full.antecedents


def test_a_new_proposed_condition_is_measured_only_after_it_was_proposed() -> None:
    """机械列举里没有的条件（地点：记录上本来没有这个字段）才是模型新提的：只量提出之后的数据。"""

    series = planted(80, moderated=True)
    proposed_on = series.cutoff - timedelta(days=10)
    condition = Condition(Template.FIELD, "家里", Field.PLACE)
    line = line_of(series)
    proposed = LaneTests(line, CONCEPTS, CONFIG, series.cutoff, proposals={DISCUSS: ((condition, proposed_on),)})
    key = RelationKey("session", DISCUSS, CODE, Segment.CHAIN, condition.text)
    assert key in proposed.keys()
    assert key not in LaneTests(line, CONCEPTS, CONFIG, series.cutoff).keys()
    late = proposed.measure(key, since=proposed_on)
    assert all(item.chain.day >= proposed_on for item in late.inside + late.outside)  # type: ignore[attr-defined]
