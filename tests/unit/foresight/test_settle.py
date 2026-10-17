"""结算规则：起算之后、最晚时限之前第一次发生落在哪里；过期待决之后补做了算后补；发生前没有在时预期的是漏报。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from habitus.foresight.ledger import Expectation, first_occurrence, omissions, rule_overdue, settle
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
        source="钟面",
        trigger="weekday=3",
        window=(FORMED - timedelta(minutes=20), FORMED + timedelta(minutes=10)),
        observed_from=FORMED.replace(hour=0, minute=0),
    )
    done = settle(clock, [row("ran", FORMED - timedelta(minutes=12))], spoken_at=None, settled_at=NOW, current=unchanged)
    assert (done.outcome, done.occurrence_uri) == ("准时", "ran")
    earlier = settle(clock, [row("ran", FORMED - timedelta(minutes=50))], spoken_at=None, settled_at=NOW, current=unchanged)
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
    clock = expectation(source="钟面", trigger="weekday=3")
    assert antecedent_holds(clock, None, flight, current=unchanged)  # 只管事件触发
