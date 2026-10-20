"""``Controls.arrival``：预测层机会表的只读口——与 ``stratum`` 的实验组同一把尺，只是把"来了"的那条也找出来。"""

from __future__ import annotations

from datetime import UTC, timedelta

from habitus.scene.relations import Controls, Segment, build_timelines, chains_of, merge_for
from habitus.series import EventSeries, SeriesRecord
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


def line_of(*records: SeriesRecord, days: int = 1):  # type: ignore[no-untyped-def]
    series = EventSeries(
        cutoff=START + timedelta(days=days), records=tuple(sorted(records, key=lambda r: (r.started_at, r.uri)))
    )
    return build_timelines(series, CONCEPTS, {}, CONFIG)["session"]


def test_the_arrival_in_the_chain_window_names_the_first_consequent_after_the_antecedent_finished() -> None:
    day = START
    line = line_of(
        record(day, 9, 0, "调研"),
        record(day, 10, 0, "讨论方案"),
        record(day, 10, 20, "修改代码", number=1),
        record(day, 10, 40, "修改代码", number=2),
        record(day, 14, 0, "提交推送"),
    )
    controls = Controls(line, CONFIG.transition_window_slots)
    (chain,) = chains_of(line, DISCUSS, CONFIG.transition_window_slots)
    arrival = controls.arrival(Segment.CHAIN, chain, CODE)
    assert arrival.came is True and arrival.uri is not None and "修改代码1" in arrival.uri
    assert arrival.started_at is not None and arrival.started_at.hour == 10 and arrival.started_at.minute == 20
    # 与 stratum 的实验组同一个读数
    stratum = controls.stratum(Segment.CHAIN, chain, CODE, antecedent=DISCUSS)
    assert stratum is not None and stratum.treated == 1
    # 窗口里没有后果：没来（不是未知）
    assert controls.arrival(Segment.CHAIN, chain, RESEARCH).came is False
    assert controls.arrival(Segment.CHAIN, chain, RESEARCH).uri is None


def test_the_last_thing_before_leaving_has_an_unknown_chain_result() -> None:
    """提交推送 14:00 是当天最后一件：之后人不在场，同一条链的窗口没整段在场 → 结果未知。"""

    day = START
    line = line_of(record(day, 9, 0, "调研"), record(day, 14, 0, "提交推送"))
    controls = Controls(line, CONFIG.transition_window_slots)
    (chain,) = chains_of(line, PUSH, CONFIG.transition_window_slots)
    assert controls.arrival(Segment.CHAIN, chain, CODE).came is None


def test_rest_of_day_and_daily_segments_find_the_consequent_in_their_own_windows() -> None:
    day = START
    line = line_of(
        record(day, 9, 0, "调研"),
        record(day, 10, 0, "讨论方案"),
        record(day, 15, 30, "修改代码"),
        record(day, 17, 50, "写文档"),
        record(day + timedelta(days=1), 9, 0, "调研"),
        record(day + timedelta(days=1), 11, 0, "修改代码"),
        record(day + timedelta(days=3), 9, 0, "调研"),
        record(day + timedelta(days=3), 16, 0, "修改代码"),
        days=9,
    )
    controls = Controls(line, CONFIG.transition_window_slots)
    (chain,) = chains_of(line, DISCUSS, CONFIG.transition_window_slots)
    rest = controls.arrival(Segment.REST_OF_DAY, chain, CODE)
    assert rest.came is True and rest.started_at is not None and rest.started_at.hour == 15
    (docs,) = merge_for(Segment.NEXT_DAY, chains_of(line, DOCS, CONFIG.transition_window_slots))
    tomorrow = controls.arrival(Segment.NEXT_DAY, docs, CODE)
    assert (
        tomorrow.came is True
        and tomorrow.started_at is not None
        and tomorrow.started_at.date() == day + timedelta(days=1)
    )
    week = controls.arrival(Segment.DAYS_2_7, docs, CODE)
    assert week.came is True and week.started_at is not None and week.started_at.date() == day + timedelta(days=3)
    # 次日段里没有调研之外的东西 → 写文档 → 讨论方案 次日没来
    assert controls.arrival(Segment.NEXT_DAY, docs, DISCUSS).came is False
    # 与按天的实验组读数一致
    stratum = controls.stratum(Segment.NEXT_DAY, docs, CODE, antecedent=DOCS)
    assert stratum is None or stratum.treated == 1


def test_a_window_past_the_cutoff_is_unknown() -> None:
    line = build_timelines(planted(days=3), CONCEPTS, {}, CONFIG)["session"]
    controls = Controls(line, CONFIG.transition_window_slots)
    last = merge_for(Segment.DAYS_2_7, chains_of(line, DOCS, CONFIG.transition_window_slots))[-1]
    assert controls.arrival(Segment.DAYS_2_7, last, CODE).came is None


def test_a_reminded_consequent_is_not_a_natural_arrival_but_still_starts_a_chain() -> None:
    """被系统提醒之后才做的（行为树 ``reminded``，裁定 54）：当后果按"看不到"，这一次两边都不计；当前因照常成链。"""

    from dataclasses import replace

    day = START
    code = replace(record(day, 10, 20, "修改代码", number=1), reminded=True)
    line = line_of(record(day, 9, 0, "调研"), record(day, 10, 0, "讨论方案"), code, record(day, 14, 0, "提交推送"))
    controls = Controls(line, CONFIG.transition_window_slots)
    (chain,) = chains_of(line, DISCUSS, CONFIG.transition_window_slots)
    assert controls.arrival(Segment.CHAIN, chain, CODE).came is None  # 不是来了，也不是没来
    assert controls.stratum(Segment.CHAIN, chain, CODE, antecedent=DISCUSS) is None  # 这一层两边都不计
    assert [mark.uri for mark in line.marks[CODE]] == [code.uri] and line.outcomes(CODE) == ()
    assert chains_of(line, CODE, CONFIG.transition_window_slots)  # 当前因照常
    # 同一窗口里另有一条自然发生的：照常算来了
    natural = record(day, 10, 30, "修改代码", number=2)
    both = line_of(
        record(day, 9, 0, "调研"), record(day, 10, 0, "讨论方案"), code, natural, record(day, 14, 0, "提交推送")
    )
    (again,) = chains_of(both, DISCUSS, CONFIG.transition_window_slots)
    arrival = Controls(both, CONFIG.transition_window_slots).arrival(Segment.CHAIN, again, CODE)
    assert arrival.came is True and arrival.uri == natural.uri


def test_a_reminded_consequent_still_ended_a_b_so_it_is_no_same_day_control_moment() -> None:
    """同日对照不取"一件 B 做完"的时刻（它那一类之后暂时不会再来）：被提醒才做的 B 一样是 B 做完了。按 ``outcomes`` 排除
    会把它放进对照，添一个几乎必然为 0 的对照、压低对照率，把 A → B 的"更多"读大。"""

    from dataclasses import replace

    day = START

    def stratum_with(reminded: bool):  # type: ignore[no-untyped-def]
        late = replace(record(day, 13, 0, "修改代码", number=9), reminded=reminded)
        late = replace(late, last_observed_at=late.started_at + timedelta(minutes=45))
        line = line_of(
            record(day, 9, 0, "调研"),
            record(day, 10, 0, "讨论方案"),
            record(day, 10, 15, "修改代码", number=1),
            record(day, 11, 0, "写文档"),
            record(day, 11, 15, "修改代码", number=2),
            late,
            record(day, 14, 0, "调研", number=3),
            record(day, 14, 30, "写文档", number=4),
            record(day, 16, 0, "提交推送"),
        )
        (chain,) = chains_of(line, DISCUSS, CONFIG.transition_window_slots)
        return Controls(line, CONFIG.transition_window_slots).stratum(Segment.CHAIN, chain, CODE, antecedent=DISCUSS)

    natural, reminded = stratum_with(False), stratum_with(True)
    assert natural is not None and reminded is not None
    assert reminded.controls == natural.controls


def test_the_outcomes_view_follows_the_marks_and_never_shares_their_list() -> None:
    from datetime import datetime

    from habitus.scene.relations import LaneTimeline, Mark

    at = datetime(2026, 6, 1, 9, tzinfo=UTC)
    line = LaneTimeline(lane="session", slots_per_day=96, horizon=10**9)
    assert line.outcomes(CODE) == ()
    line.marks.setdefault(CODE, []).append(Mark(slot=1, started_at=at, uri="u1"))
    assert [mark.uri for mark in line.outcomes(CODE)] == ["u1"]  # 先问过一次空的，之后加的照样看得到
    line.marks[CODE].append(Mark(slot=2, started_at=at, uri="u2", reminded=True))
    assert [mark.uri for mark in line.outcomes(CODE)] == ["u1"]  # 被提醒的不漏进来
