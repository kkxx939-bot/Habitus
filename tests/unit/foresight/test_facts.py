"""结算写哪一笔事实的四条短路（裁定 54、61、91）：前因不成立记作废；过期待决、窗里有观测空白记中性；过期待决、他回了「已做」
记中性；别的账的提醒引出的发生记强度 0 的「提醒后发生」；都不是交给按结果与强度的那一笔。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from habitus.foresight.facts import short_circuit_fact
from habitus.foresight.ledger import VOID, Expectation, Settlement

CST = timezone(timedelta(hours=8))
FORMED = datetime(2026, 9, 1, 9, 0, tzinfo=CST)
ITEM = Expectation(
    expectation_id="run-0901",
    kind_token="s-k0003",
    day=FORMED.date(),
    formed_at=FORMED,
    source="日子",
    trigger="weekday=1",
    trigger_instance=FORMED.date().isoformat(),
    window=(FORMED + timedelta(minutes=10), FORMED + timedelta(minutes=40)),
    deadline=FORMED + timedelta(hours=2),
    basis=(),
    opportunities=10,
    hits=7,
    observed_from=FORMED,
)


def result(outcome: str, uri: str | None = None) -> Settlement:
    offset = 5 if outcome in ("早到", "晚了但做了", "提醒后发生") else None
    return Settlement(ITEM.expectation_id, ITEM.kind_token, ITEM.day, FORMED + timedelta(days=1), outcome, uri, offset)


def short(  # type: ignore[no-untyped-def]
    outcome: str,
    uri: str | None = None,
    *,
    holds: bool = True,
    unobserved: bool = False,
    said_done: bool = False,
    prompted: bool = False,
):
    return short_circuit_fact(
        ITEM,
        result(outcome, uri),
        antecedent_holds=holds,
        unobserved=unobserved,
        said_done=said_done,
        prompted=prompted,
    )


def test_an_antecedent_that_no_longer_holds_voids_the_opportunity_whatever_happened() -> None:
    voided = short("准时", "behavior://run.md", holds=False, prompted=True)
    assert voided is not None and (voided.kind, voided.good, voided.strength, voided.at) == (VOID, False, 0.0, FORMED)


def test_an_unobserved_stretch_makes_only_an_overdue_neutral() -> None:
    neutral = short("过期待决", unobserved=True)
    assert neutral is not None and (neutral.kind, neutral.good, neutral.strength) == ("过期待决", False, 0.0)
    assert short("准时", "behavior://run.md", unobserved=True) is None


def test_a_said_done_reply_makes_only_an_overdue_neutral() -> None:
    """裁定 91 第 3 条②（落实 54 第 3 条③）：他回了「已做」、到时限没看到做 → 中性（强度 0、不算机会），不是一笔坏；
    看到做了的照常按结果写；前因不成立照旧作废。"""

    neutral = short("过期待决", said_done=True)
    assert neutral is not None and (neutral.kind, neutral.good, neutral.strength) == ("过期待决", False, 0.0)
    assert short("过期待决") is None  # 没回已做：交给按结果写（坏、强度 1）
    assert short("准时", "behavior://run.md", said_done=True) is None
    assert short("提醒后发生", "behavior://run.md", said_done=True) is None
    voided = short("过期待决", holds=False, said_done=True)
    assert voided is not None and voided.kind == VOID


def test_a_happening_prompted_by_another_reminder_is_done_but_no_evidence() -> None:
    prompted = short("准时", "behavior://run.md", prompted=True)
    assert prompted is not None and (prompted.kind, prompted.good, prompted.strength) == ("提醒后发生", True, 0.0)
    assert prompted.offset_minutes is None
    # 这条自己的提醒引出的、或者根本没发生的：不走这一条
    assert short("提醒后发生", "behavior://run.md", prompted=True) is None
    assert short("过期待决", prompted=True) is None


def test_nothing_special_leaves_the_fact_to_the_outcome_and_its_strength() -> None:
    assert short("准时", "behavior://run.md") is None
    assert short("过期待决") is None


# --- 段 6 Q18：存活变异补的边界断言 ------------------------------------------------------------------------------


def _every_three_days(*days: int):  # type: ignore[no-untyped-def]
    """这一类在 8 月这几天各做过一次（全是自然发生），相邻两次的间隔登记在机会表里。"""

    from datetime import date

    from habitus.foresight.opportunities import KindDays, OpportunityTable
    from habitus.temporal import DayInterval, OccurrenceDay

    found = tuple(
        OccurrenceDay(
            date(2026, 8, day),
            f"u{day}",
            datetime(2026, 8, day, 9, tzinfo=CST),
            1,
            1,
            f"u{day}",
            datetime(2026, 8, day, 9, tzinfo=CST),
        )
        for day in days
    )
    intervals = tuple(
        DayInterval(a, b, (b.day - a.day).days, 1.0, False) for a, b in zip(found, found[1:], strict=False)
    )
    return OpportunityTable(
        "session", FORMED.date(), FORMED.date(), 15, 3, (), (KindDays(ITEM.kind_token, "跑步", found, intervals),)
    )


def test_the_offset_history_only_takes_chances_strictly_before_this_one() -> None:
    """强度的历史只看这次机会之前的：同一刻的另一次机会不算（M43：``<`` 改 ``<=``）。历史只剩一次：只有方向、强度 1。"""

    from habitus.foresight.facts import settlement_fact
    from habitus.foresight.ledger import AccountFact

    earlier = AccountFact(ITEM.account, FORMED - timedelta(days=7), True, 1.0, "准时", "old", offset_minutes=0)
    same = AccountFact(ITEM.account, FORMED, True, 1.0, "晚了但做了", "twin", offset_minutes=50)
    fact = settlement_fact(ITEM, result("早到", "behavior://run.md"), known=[earlier, same])
    assert fact.strength == 1.0


def test_an_overdue_is_settled_once_the_next_chance_formed_on_a_closed_day_or_a_full_normal_gap_passed() -> None:
    """裁决不再等：下一次机会形成那天已经定稿（含当天，M44：``>=`` 改 ``>``）；从上一次发生到定稿的尽头已隔了不短于最长的
    正常间隔（含相等，M45：``>=`` 改 ``>``）。"""

    from habitus.foresight.facts import ruling_settled

    table = _every_three_days(25, 28, 31)  # 间隔 3、3；上一次 8 月 31 日
    closed = FORMED.date() + timedelta(days=1)  # 9 月 2 日：离上一次 2 天，还短于正常间隔
    assert not ruling_settled(ITEM, None, closed, table)
    next_chance = datetime.combine(closed, datetime.min.time(), tzinfo=CST).replace(hour=8)
    assert ruling_settled(ITEM, next_chance, closed, table)
    assert ruling_settled(ITEM, None, closed + timedelta(days=1), table)  # 9 月 3 日：正好隔了 3 天


def test_the_catch_up_strength_counts_the_days_from_the_last_occurrence_to_the_catch_up() -> None:
    """补做的坏按"补做那天离上一次几天"在历史间隔里的位置（M47：少算一天）：隔 3 天（等于正常间隔）就和没做一样坏，2 天不算坏。"""

    from habitus.foresight.facts import catch_up_fact
    from habitus.foresight.ledger import Ruling

    table = _every_three_days(25, 28, 31)
    sep_3 = Ruling(ITEM.expectation_id, FORMED + timedelta(days=3), "后补", "u", 46 * 60)  # 9 月 3 日 09:00 → 隔 3 天
    sep_2 = Ruling(ITEM.expectation_id, FORMED + timedelta(days=3), "后补", "u", 22 * 60)  # 9 月 2 日 09:00 → 隔 2 天
    assert catch_up_fact(ITEM, sep_3, table).strength == 1.0
    assert catch_up_fact(ITEM, sep_2, table).strength == 0.0
