"""结算规则：起算之后、最晚时限之前第一次发生落在哪里；过期待决之后补做了算后补；发生前没有在时预期的是漏报。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone

from habitus.foresight.ledger import (
    AntecedentRow,
    Expectation,
    antecedent_holds_when_delivering,
    antecedent_holds_when_settling,
    first_occurrence,
    omissions,
    rule_overdue,
    settle,
)
from habitus.foresight.opportunities import RelationOpportunities
from habitus.scene.views import FlowRow

CST = timezone(timedelta(hours=8))
FORMED = datetime(2026, 12, 10, 14, 7, tzinfo=CST)
HOTEL = "s-k0002"


def expectation(**overrides: object) -> Expectation:
    values: dict[str, object] = dict(
        expectation_id="hotel-1210",
        kind_token=HOTEL,
        day=FORMED.date(),
        formed_at=FORMED,
        source="事件",
        trigger="session|s-k0001|s-k0002|chain",
        trigger_instance="behavior://flight.md",
        window=(FORMED + timedelta(minutes=4), FORMED + timedelta(minutes=51)),
        deadline=FORMED + timedelta(hours=3),
        basis=(),
        opportunities=13,
        hits=10,
        observed_from=FORMED,
    )
    values.update(overrides)
    return Expectation(**values)  # type: ignore[arg-type]


def row(uri: str, at: datetime, kind: str = HOTEL) -> FlowRow:
    return FlowRow(
        uri=uri, name=uri, kind_token=kind, at=at, last_observed_at=at + timedelta(minutes=5), summary="", day_count=1
    )


def unchanged(token: str) -> tuple[str, ...]:
    return (token,)


NOW = FORMED + timedelta(days=1)


def test_on_time_early_and_late_are_measured_against_the_window() -> None:
    item = expectation()
    on_time = settle(
        item, [row("a", FORMED + timedelta(minutes=20))], spoken_at=None, settled_at=NOW, current=unchanged
    )
    assert (on_time.outcome, on_time.occurrence_uri, on_time.offset_minutes) == ("准时", "a", None)
    early = settle(item, [row("b", FORMED + timedelta(minutes=2))], spoken_at=None, settled_at=NOW, current=unchanged)
    assert (early.outcome, early.offset_minutes) == ("早到", -2)
    late = settle(item, [row("c", FORMED + timedelta(minutes=70))], spoken_at=None, settled_at=NOW, current=unchanged)
    assert (late.outcome, late.offset_minutes) == ("晚了但做了", 19)
    assert on_time.hit and early.hit and late.hit


def test_only_the_first_occurrence_after_observed_from_counts_and_other_kinds_do_not() -> None:
    item = expectation()
    rows = [
        row("before", FORMED - timedelta(minutes=1)),
        row("other", FORMED + timedelta(minutes=10), kind="s-k0009"),
        row("first", FORMED + timedelta(minutes=30)),
        row("second", FORMED + timedelta(minutes=40)),
    ]
    found = first_occurrence(item, rows, current=unchanged)
    assert found is not None and found.uri == "first"
    # 词表拆改之后按"现在对应哪些编号"认
    split = settle(
        item,
        [row("s", FORMED + timedelta(minutes=30), kind="s-k0099")],
        spoken_at=None,
        settled_at=NOW,
        current=lambda _t: ("s-k0099",),
    )
    assert split.outcome == "准时"


def test_an_occurrence_after_the_reminder_is_not_a_natural_hit() -> None:
    item = expectation()
    spoken = FORMED + timedelta(minutes=10)
    result = settle(
        item, [row("a", FORMED + timedelta(minutes=25))], spoken_at=spoken, settled_at=NOW, current=unchanged
    )
    assert (result.outcome, result.offset_minutes, result.hit) == ("提醒后发生", 15, False)
    # 说之前就做了的，照常按时窗记
    before = settle(
        item, [row("b", FORMED + timedelta(minutes=5))], spoken_at=spoken, settled_at=NOW, current=unchanged
    )
    assert before.outcome == "准时"


def test_nothing_by_the_deadline_is_overdue_pending_and_one_done_before_forming_still_counts() -> None:
    item = expectation()
    none = settle(
        item, [row("late", item.deadline + timedelta(minutes=1))], spoken_at=None, settled_at=NOW, current=unchanged
    )
    assert none.outcome == "过期待决" and none.occurrence_uri is None and not none.hit
    # 到了钟点才建、形成时他已经做了：从当天零点起算，照常按时窗结（裁定 49：每个机会都进账，不再有「多余」）
    clock = expectation(
        source="日子",
        trigger="weekday=3",
        window=(FORMED - timedelta(minutes=20), FORMED + timedelta(minutes=10)),
        observed_from=FORMED.replace(hour=0, minute=0),
    )
    done = settle(
        clock, [row("ran", FORMED - timedelta(minutes=12))], spoken_at=None, settled_at=NOW, current=unchanged
    )
    assert (done.outcome, done.occurrence_uri) == ("准时", "ran")
    earlier = settle(
        clock, [row("ran", FORMED - timedelta(minutes=50))], spoken_at=None, settled_at=NOW, current=unchanged
    )
    assert (earlier.outcome, earlier.offset_minutes) == ("早到", -30)


def test_overdue_is_ruled_as_caught_up_only_before_the_next_opportunity() -> None:
    item = expectation()
    later = row("caught", item.deadline + timedelta(hours=5))
    ruling = rule_overdue(item, [later], before=None, ruled_at=NOW, current=unchanged)
    assert ruling is not None and (ruling.ruling, ruling.occurrence_uri, ruling.delay_minutes) == (
        "后补",
        "caught",
        300,
    )
    # 下一次同类机会已经形成：那之后做的属于下一次，不算补做
    assert (
        rule_overdue(item, [later], before=item.deadline + timedelta(hours=1), ruled_at=NOW, current=unchanged) is None
    )
    # 时限之前的行不算（那是结算的事），没有补做不裁
    assert (
        rule_overdue(item, [row("x", item.deadline - timedelta(hours=1))], before=None, ruled_at=NOW, current=unchanged)
        is None
    )


def test_omissions_are_class_rows_without_a_covering_expectation() -> None:
    item = expectation()
    rows = [
        row("covered", FORMED + timedelta(minutes=30)),
        row("uncovered", item.deadline + timedelta(hours=2)),
        row("other-kind", FORMED + timedelta(minutes=30), kind="s-k0007"),
        row("pending", FORMED + timedelta(minutes=31), kind="s-待定"),
    ]
    found = omissions(rows, [item], noticed_at=NOW, current=unchanged)
    assert [(o.occurrence_uri, o.kind_token, o.had_expectation) for o in found] == [
        ("uncovered", HOTEL, True),
        ("other-kind", "s-k0007", False),
    ]
    assert all(o.day == date(2026, 12, 10) for o in found)


def test_an_abandoned_occurrence_is_not_a_hit_nor_a_catch_up_nor_an_omission() -> None:
    """只认真做成的（裁定 54：假命中）：放弃了的那一行不算这件事发生了；被打断的算（他做了，只是没做完）。"""

    from dataclasses import replace as swap

    item = expectation()
    gave_up = swap(row("gave-up", FORMED + timedelta(minutes=20)), status="abandoned")
    assert settle(item, [gave_up], spoken_at=None, settled_at=NOW, current=unchanged).outcome == "过期待决"
    stopped = swap(row("stopped", FORMED + timedelta(minutes=20)), status="interrupted")
    assert settle(item, [stopped], spoken_at=None, settled_at=NOW, current=unchanged).outcome == "准时"
    later = swap(row("later", item.deadline + timedelta(hours=1)), status="abandoned")
    assert rule_overdue(item, [later], before=None, ruled_at=NOW, current=unchanged) is None
    assert omissions([later], [], noticed_at=NOW, current=unchanged) == ()


def test_an_event_expectation_whose_antecedent_turned_into_something_else_does_not_hold() -> None:
    from habitus.foresight.ledger import antecedent_holds

    item = expectation()
    flight = ("s-k0001",)
    assert antecedent_holds(item, "s-k0001", flight, current=unchanged)
    assert not antecedent_holds(item, "s-k0009", flight, current=unchanged)  # 封口后归成了别的类
    assert not antecedent_holds(item, None, flight, current=unchanged)  # 成了非事件 / 被并掉：找不到了
    assert antecedent_holds(item, "s-k0011", flight, current=lambda t: ("s-k0011",) if t == "s-k0001" else (t,))  # 拆改
    assert antecedent_holds(item, None, None, current=unchanged)  # 关系已不在表里：说不清，不判作废
    clock = expectation(source="日子", trigger="weekday=3")
    assert antecedent_holds(clock, None, flight, current=unchanged)  # 只管事件触发


def _relation(**overrides: object) -> RelationOpportunities:
    values: dict[str, object] = dict(
        key="session|s-k0001|s-k0002|chain",
        antecedent="flight",
        consequent="hotel",
        antecedent_label="订机票",
        consequent_label="订酒店",
        antecedent_classes=("s-k0001",),
        consequent_classes=(HOTEL,),
        segment="chain",
        status="established",
        upward=True,
        items=(),
    )
    values.update(overrides)
    return RelationOpportunities(**values)  # type: ignore[arg-type]


FLIGHT_AT = FORMED - timedelta(minutes=30)


def _event(**overrides: object) -> Expectation:
    return expectation(trigger_instance=FLIGHT_AT.astimezone(UTC).isoformat(), **overrides)


def _rows_on(rows: list[FlowRow]) -> Callable[[date], list[AntecedentRow]]:
    return lambda day: [
        AntecedentRow(item.at, item.last_observed_at, item.kind_token) for item in rows if item.at.date() == day
    ]


def _no_calendar(_day: date) -> tuple[tuple[str, str], ...]:
    raise AssertionError("an event trigger does not read the calendar")


def test_settling_finds_the_antecedent_by_its_start_then_by_the_row_spanning_that_moment() -> None:
    relation = _relation()
    exact = [row("flight", FLIGHT_AT, kind="s-k0001")]
    holds = antecedent_holds_when_settling(
        _event(), relation, official=False, calendar_on=_no_calendar, rows_on=_rows_on(exact), current=unchanged
    )
    assert holds
    # 封口时开始被更正挪了几分钟：跨过那一刻的那一行
    moved = [row("flight", FLIGHT_AT - timedelta(minutes=3), kind="s-k0001")]
    assert antecedent_holds_when_settling(
        _event(), relation, official=False, calendar_on=_no_calendar, rows_on=_rows_on(moved), current=unchanged
    )
    # 封口后归成了别的类：前因不成立；词表把它拆改成了前因类之一：成立
    other = [row("flight", FLIGHT_AT, kind="s-k0007")]
    assert not antecedent_holds_when_settling(
        _event(), relation, official=False, calendar_on=_no_calendar, rows_on=_rows_on(other), current=unchanged
    )
    assert antecedent_holds_when_settling(
        _event(),
        relation,
        official=False,
        calendar_on=_no_calendar,
        rows_on=_rows_on(other),
        current=lambda token: ("s-k0007",) if token == "s-k0001" else (token,),
    )
    # 找不到：前因不成立；关系不在表里、实例读不出时刻：说不清，不判作废
    assert not antecedent_holds_when_settling(
        _event(), relation, official=False, calendar_on=_no_calendar, rows_on=_rows_on([]), current=unchanged
    )
    assert antecedent_holds_when_settling(
        _event(), None, official=False, calendar_on=_no_calendar, rows_on=_rows_on([]), current=unchanged
    )
    assert antecedent_holds_when_settling(
        expectation(), relation, official=False, calendar_on=_no_calendar, rows_on=_rows_on([]), current=unchanged
    )


def test_settling_rechecks_a_situation_against_the_latest_calendar() -> None:
    relation = _relation(
        key="session|工作日|s-k0002|situation",
        antecedent="工作日",
        antecedent_classes=("工作日",),
        segment="situation",
        calendar_key="日历.日型",
        calendar_values=("工作日",),
    )
    item = expectation(source="情境", trigger=relation.key, trigger_instance=FORMED.date().isoformat())

    def no_rows(_day: date) -> list[FlowRow]:
        raise AssertionError("a situation trigger does not read the rows")

    def on(value: str) -> Callable[[date], tuple[tuple[str, str], ...]]:
        return lambda _day: (("日历.日型", value),)

    assert antecedent_holds_when_settling(
        item, relation, official=False, calendar_on=on("工作日"), rows_on=no_rows, current=unchanged
    )
    assert not antecedent_holds_when_settling(
        item, relation, official=False, calendar_on=on("节假日"), rows_on=no_rows, current=unchanged
    )
    assert antecedent_holds_when_settling(
        item, None, official=False, calendar_on=on("节假日"), rows_on=no_rows, current=unchanged
    )


def test_delivering_finds_the_antecedent_the_same_way_as_settling() -> None:
    """段 6 Q4：递送开口前与结算对"前因还成立吗"读同一份——按开始时刻找、找不到就找跨过那一刻的那一行（封口时开始被更正挪了
    几分钟），类按 ``current`` 认词表拆改。原来递送只按开始时刻原样找、只认原编号：开始被更正或前因拆改后递送说不成立（不说），
    结算却说成立（记过期待决，账上记一笔坏）。情境要不要按最新的日历重核是两边唯一的不同（递送不重核，待定）。"""

    relation = _relation()
    split = lambda token: ("s-k0007",) if token == "s-k0001" else (token,)  # noqa: E731
    cases = [
        ([row("flight", FLIGHT_AT, kind="s-k0001")], unchanged),
        ([row("flight", FLIGHT_AT - timedelta(minutes=3), kind="s-k0001")], unchanged),  # 开始被更正
        ([row("flight", FLIGHT_AT, kind="s-k0007")], unchanged),  # 归成了别的类
        ([row("flight", FLIGHT_AT, kind="s-k0007")], split),  # 拆改成了前因类之一
        ([], unchanged),  # 找不到
    ]
    for rows, current in cases:
        settling = antecedent_holds_when_settling(
            _event(), relation, official=False, calendar_on=_no_calendar, rows_on=_rows_on(rows), current=current
        )
        delivering = antecedent_holds_when_delivering(_event(), relation, rows_on=_rows_on(rows), current=current)
        assert delivering is settling
    assert [
        antecedent_holds_when_delivering(_event(), relation, rows_on=_rows_on(rows), current=current)
        for rows, current in cases
    ] == [True, True, False, True, False]

    def never(_day: date) -> list[AntecedentRow]:
        raise AssertionError("not an event trigger, or no relation: nothing to look up")

    assert antecedent_holds_when_delivering(_event(), None, rows_on=never, current=unchanged)
    weekday = expectation(source="日子", trigger="weekday=3")
    assert antecedent_holds_when_delivering(weekday, relation, rows_on=never, current=unchanged)


# --- 段 6 Q18：存活变异补的边界断言 ------------------------------------------------------------------------------


def test_a_reminder_spoken_at_the_very_moment_it_happened_is_not_after_the_reminder() -> None:
    """说出口与发生同一刻：不算提醒之后才做（M01：``<`` 改 ``<=``）。"""

    item = expectation()
    at = FORMED + timedelta(minutes=20)
    same = settle(item, [row("a", at)], spoken_at=at, settled_at=NOW, current=unchanged)
    assert same.outcome == "准时"
    after = settle(item, [row("a", at)], spoken_at=at - timedelta(minutes=1), settled_at=NOW, current=unchanged)
    assert (after.outcome, after.offset_minutes) == ("提醒后发生", 1)


def test_an_occurrence_exactly_at_the_window_end_is_on_time() -> None:
    """时窗终点那一刻算窗内（M03：``<=`` 改 ``<``）。"""

    item = expectation()
    end = settle(item, [row("a", item.window[1])], spoken_at=None, settled_at=NOW, current=unchanged)
    assert (end.outcome, end.offset_minutes) == ("准时", None)


def test_a_row_still_carrying_the_old_number_mid_migration_counts() -> None:
    """词表迁移做到一半：那天的行可能还是原编号，原编号总认（M05：去掉原编号）。"""

    item = expectation()
    moved = lambda token: ("s-k0020",) if token == HOTEL else (token,)  # noqa: E731
    found = first_occurrence(item, [row("old", FORMED + timedelta(minutes=20))], current=moved)
    assert found is not None and found.uri == "old"


def test_the_antecedent_is_also_looked_for_on_the_neighbouring_day() -> None:
    """前因那一行按实例在那天与前后各一天里找（M08：只找那一天）：实例按形成时的时区落在 12-10，树上那一行的日子是 12-09。"""

    instance = datetime(2026, 12, 10, 7, 0, tzinfo=CST)  # = 12-09 23:00 UTC
    item = expectation(trigger_instance=instance.astimezone(UTC).isoformat())
    yesterday = [row("flight", instance.astimezone(UTC), kind="s-k0001")]
    assert yesterday[0].at.date() == date(2026, 12, 9)
    assert antecedent_holds_when_settling(
        item, _relation(), official=False, calendar_on=_no_calendar, rows_on=_rows_on(yesterday), current=unchanged
    )


def test_a_catch_up_needs_to_come_strictly_after_the_deadline_and_before_the_next_chance() -> None:
    """后补：恰在最晚时限那一刻的发生不算补做（结算已经认过它，M11：``<=`` 改 ``<``）；恰在下一次机会形成那一刻的归下一次
    （M12：``>=`` 改 ``>``）。"""

    item = expectation()
    deadline = item.deadline
    assert rule_overdue(item, [row("at", deadline)], before=None, ruled_at=NOW, current=unchanged) is None
    later = rule_overdue(
        item, [row("late", deadline + timedelta(minutes=1))], before=None, ruled_at=NOW, current=unchanged
    )
    assert later is not None and later.ruling == "后补"
    next_chance = deadline + timedelta(hours=2)
    assert rule_overdue(item, [row("n", next_chance)], before=next_chance, ruled_at=NOW, current=unchanged) is None
