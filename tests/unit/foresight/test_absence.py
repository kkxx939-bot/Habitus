"""账的久别（26 号清单 3.3）与"一次机会"的口径（4.1）：纯函数，不读盘。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from habitus.foresight.absence import opportunity_days, returned_after_absence
from habitus.foresight.clock import is_opportunity
from habitus.foresight.ledger import AccountFact, Expectation, account_key
from habitus.foresight.opportunities import DayWatch, OpportunityTable

CST = timezone(timedelta(hours=8))
START = date(2026, 9, 1)  # 周二
ACCOUNT = account_key("s-k0002", "日子", "rel")


def at(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time(), tzinfo=CST) + timedelta(hours=9)


def fact(day: date, identity: str, *, kind: str = "准时", good: bool = True, strength: float = 1.0) -> AccountFact:
    return AccountFact(ACCOUNT, at(day), good, strength, kind, identity)


def facts_on(days: list[date]) -> list[AccountFact]:
    return [fact(day, f"e-{day.isoformat()}") for day in days]


def expectation(day: date, identity: str = "now") -> Expectation:
    formed = at(day)
    return Expectation(
        expectation_id=identity,
        kind_token="s-k0002",
        day=day,
        formed_at=formed,
        source="日子",
        trigger="rel",
        trigger_instance=f"behavior://{identity}.md",
        window=(formed + timedelta(minutes=10), formed + timedelta(minutes=40)),
        deadline=formed + timedelta(hours=2),
        basis=(),
        opportunities=3,
        hits=3,
        observed_from=formed,
    )


def table(watch: dict[date, tuple[float, str | None]]) -> OpportunityTable:
    rows = tuple(DayWatch(day, share, day_type) for day, (share, day_type) in sorted(watch.items()))
    return OpportunityTable("session", max(watch), None, 15, 3, (), (), watch=rows)


def every_day(until: date, share: float = 1.0) -> dict[date, tuple[float, str | None]]:
    return {START + timedelta(days=offset): (share, None) for offset in range((until - START).days + 1)}


# --- 一次机会 -------------------------------------------------------------------------------------


def test_void_and_unobserved_facts_are_no_opportunities_but_prompted_ones_are() -> None:
    assert not is_opportunity(fact(START, "a", kind="作废", good=False, strength=0.0))
    assert is_opportunity(fact(START, "b", kind="提醒后发生", strength=0.0))
    assert not is_opportunity(fact(START, "c", kind="过期待决", good=False, strength=0.0))  # 窗里有观测空白：不算
    assert is_opportunity(fact(START, "e", kind="过期待决", good=False, strength=1.0))  # 真没做：算


def test_opportunity_days_are_deduplicated_and_stop_before_this_expectation() -> None:
    known = [
        fact(START, "a"),
        fact(START + timedelta(days=1), "b", kind="作废", good=False, strength=0.0),
        fact(START + timedelta(days=2), "c"),
        fact(START + timedelta(days=2), "c2"),
        fact(START + timedelta(days=5), "later"),
    ]
    assert opportunity_days(known, expectation(START + timedelta(days=4))) == (START, START + timedelta(days=2))


# --- 久别 ----------------------------------------------------------------------------------------


def test_a_gap_longer_than_every_earlier_one_is_a_return_after_absence() -> None:
    days = [START + timedelta(days=offset) for offset in (0, 2, 4, 6)]  # 隔一天一次
    today = START + timedelta(days=20)
    view = table(every_day(today - timedelta(days=1)))
    assert returned_after_absence(facts_on(days), expectation(today), view)
    assert not returned_after_absence(facts_on(days), expectation(START + timedelta(days=8)), view)


def test_days_nobody_was_watching_do_not_stretch_the_gap() -> None:
    """中间那几天设备没上传：日历上隔了 14 天，看着的只有 2 天，不比往常的隔一天两天长。"""

    days = [START + timedelta(days=offset) for offset in (0, 2, 4, 6)]
    today = START + timedelta(days=20)
    watch = every_day(today - timedelta(days=1))
    watch.update({START + timedelta(days=offset): (0.0, None) for offset in range(7, 19)})
    assert not returned_after_absence(facts_on(days), expectation(today), table(watch))


def test_only_day_types_the_account_had_opportunities_on_are_counted() -> None:
    """只在工作日做的事：一个周末加五天节假日夹在中间，不算久别；同样长的工作日空白算。"""

    workdays = [START + timedelta(days=offset) for offset in (0, 1, 2, 3)]  # 周二到周五
    watch: dict[date, tuple[float, str | None]] = {
        day: (1.0, "工作日") for day in (START + timedelta(days=offset) for offset in range(30))
    }
    for offset in range(4, 11):  # 周六起七天不上班
        watch[START + timedelta(days=offset)] = (1.0, "周末" if offset in (4, 5) else "节假日")
    view = table(watch)
    assert not returned_after_absence(facts_on(workdays), expectation(START + timedelta(days=11)), view)
    assert returned_after_absence(facts_on(workdays), expectation(START + timedelta(days=14)), view)


def test_too_few_gaps_or_no_table_decide_nothing() -> None:
    today = START + timedelta(days=20)
    view = table(every_day(today))
    assert not returned_after_absence(facts_on([START, START + timedelta(days=1)]), expectation(today), view)
    days = [START + timedelta(days=offset) for offset in (0, 2, 4)]
    assert returned_after_absence(facts_on(days), expectation(today), view)  # 两个间隔就够
    assert not returned_after_absence(facts_on(days), expectation(today), None)
    blind = OpportunityTable("session", today, None, 15, 3, (), ())  # 表上没有覆盖：只剩机会日算长度
    assert not returned_after_absence(facts_on(days), expectation(today), blind)
    assert not returned_after_absence([], expectation(today), view)
    earlier = AccountFact(ACCOUNT, at(today) - timedelta(hours=1), True, 1.0, "准时", "earlier")
    assert not returned_after_absence(
        [*facts_on(days), earlier], expectation(today), view
    )  # 今天早些时候已有过一次机会


def test_a_compacted_account_judges_absence_like_the_whole_one() -> None:
    """压实了的那一段只剩间隔个数与最长的、最后一次机会那天：久别照样判得一样（26 号清单 4.8）。"""

    from habitus.foresight import ranked_account
    from habitus.foresight.compaction import compact, compactable

    days = [START + timedelta(days=2 * index) for index in range(20)]
    facts = facts_on(days)
    view = table(every_day(START + timedelta(days=80)))
    now = at(START + timedelta(days=81))
    account = ranked_account(facts, visible_at=now)
    plan = compactable(account, open_day=None, diagnosis_since=now.date(), window=5)
    base = compact(account, plan, table=view)
    assert base.gap_count == 14 and base.longest_gap == 2.0 and base.last_day == days[14]
    remaining = [fact for fact in facts if not base.covers(fact.at, fact.expectation_id)]
    for today in (days[-1] + timedelta(days=2), days[-1] + timedelta(days=9)):
        whole = returned_after_absence(facts, expectation(today), view)
        folded = returned_after_absence(remaining, expectation(today), view, base)
        assert folded == whole
    assert returned_after_absence(remaining, expectation(days[-1] + timedelta(days=9)), view, base)
