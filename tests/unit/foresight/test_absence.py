"""账的久别（26 号清单 3.3；判法按裁定 88 第 2 条：比历史最长还长，且至少 H 个典型间隔）与"一次机会"的口径（4.1）：纯函数，不读盘。"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone

from habitus.foresight.absence import opportunity_days, returned_after_absence
from habitus.foresight.clock import is_opportunity
from habitus.foresight.ledger import AccountFact, Expectation, account_key
from habitus.foresight.ledger.compaction import AccountCompaction
from habitus.foresight.opportunities import DayWatch, OpportunityTable

CST = timezone(timedelta(hours=8))
START = date(2026, 9, 1)  # 周二
ACCOUNT = account_key("s-k0002", "日子", "rel")
H = 8.0  # 遗忘半衰期（``config.foresight.half_life_opportunities`` 的默认值）


def absent(
    facts: Iterable[AccountFact],
    item: Expectation,
    view: OpportunityTable | None,
    base: AccountCompaction | None = None,
    unsettled: Iterable[Expectation] = (),
) -> bool:
    return returned_after_absence(facts, item, view, base, unsettled, half_life=H)


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


def test_an_every_other_day_habit_returns_after_absence_only_after_sixteen_days() -> None:
    """隔天做：典型间隔 2 天，空白 16 天（8 个典型间隔）才算久别；比历来都长的 8 天、15 天不算。"""

    days = [START + timedelta(days=offset) for offset in (0, 2, 4, 6)]  # 隔一天一次
    view = table(every_day(START + timedelta(days=30)))
    assert absent(facts_on(days), expectation(START + timedelta(days=22)), view)
    assert not absent(facts_on(days), expectation(START + timedelta(days=21)), view)
    assert not absent(facts_on(days), expectation(START + timedelta(days=8)), view)


def test_days_nobody_was_watching_do_not_stretch_the_gap() -> None:
    """中间那几天设备没上传：日历上隔了 24 天，看着的只有 2 天，不比往常的隔一天两天长。"""

    days = [START + timedelta(days=offset) for offset in (0, 2, 4, 6)]
    today = START + timedelta(days=30)
    watch = every_day(today - timedelta(days=1))
    assert absent(facts_on(days), expectation(today), table(watch))  # 都在看：24 天是久别
    watch.update({START + timedelta(days=offset): (0.0, None) for offset in range(7, 29)})
    assert not absent(facts_on(days), expectation(today), table(watch))


def test_only_day_types_the_account_had_opportunities_on_are_counted() -> None:
    """只在工作日做的事：一个周末加五天节假日夹在中间不算长度；之后的工作日空白满 8 天才算。"""

    workdays = [START + timedelta(days=offset) for offset in (0, 1, 2, 3)]  # 周二到周五
    watch: dict[date, tuple[float, str | None]] = {
        day: (1.0, "工作日") for day in (START + timedelta(days=offset) for offset in range(30))
    }
    for offset in range(4, 11):  # 周六起七天不上班
        watch[START + timedelta(days=offset)] = (1.0, "周末" if offset in (4, 5) else "节假日")
    view = table(watch)
    assert not absent(facts_on(workdays), expectation(START + timedelta(days=11)), view)  # 日历上 8 天、算数的 1 天
    assert not absent(facts_on(workdays), expectation(START + timedelta(days=17)), view)
    assert absent(facts_on(workdays), expectation(START + timedelta(days=18)), view)


def test_too_few_gaps_or_no_table_decide_nothing() -> None:
    today = START + timedelta(days=20)
    view = table(every_day(today))
    assert not absent(facts_on([START, START + timedelta(days=1)]), expectation(today), view)
    days = [START + timedelta(days=offset) for offset in (0, 2, 4)]
    assert absent(facts_on(days), expectation(today), view)  # 两个间隔就够
    assert not absent(facts_on(days), expectation(today), None)
    blind = OpportunityTable("session", today, None, 15, 3, (), ())  # 表上没有覆盖：只剩机会日算长度
    assert not absent(facts_on(days), expectation(today), blind)
    assert not absent([], expectation(today), view)
    earlier = AccountFact(ACCOUNT, at(today) - timedelta(hours=1), True, 1.0, "准时", "earlier")
    assert not absent([*facts_on(days), earlier], expectation(today), view)  # 今天早些时候已有过一次机会


def test_a_daily_habit_missing_one_day_is_not_a_return_after_absence() -> None:
    """裁定 88 第 1 条（E1）：天天做的事漏一天，下一次机会比历来都长（2 天），但不是久别、照常。停 7 天也不是，停 8 天是。"""

    days = _daily(0, 19)
    view = table(every_day(START + timedelta(days=40)))
    last = days[-1]
    assert not absent(facts_on(days), expectation(last + timedelta(days=2)), view)
    assert not absent(facts_on(days), expectation(last + timedelta(days=7)), view)
    assert absent(facts_on(days), expectation(last + timedelta(days=8)), view)
    assert absent(facts_on(days), expectation(last + timedelta(days=12)), view)


def test_a_weekly_habit_stopped_three_weeks_is_not_a_return_after_absence_but_eight_weeks_is() -> None:
    weekly = [START + timedelta(weeks=week) for week in range(8)]
    view = table(every_day(START + timedelta(weeks=16)))
    last = weekly[-1]
    assert not absent(facts_on(weekly), expectation(last + timedelta(weeks=3)), view)
    assert not absent(facts_on(weekly), expectation(last + timedelta(days=55)), view)
    assert absent(facts_on(weekly), expectation(last + timedelta(weeks=8)), view)


def test_a_compacted_account_judges_absence_like_the_whole_one() -> None:
    """压实了的那一段只剩间隔的长度分布与最后一次机会那天：久别照样判得一样（26 号清单 4.8）。"""

    from habitus.foresight import ranked_account
    from habitus.foresight.compaction import compact, compactable

    days = [START + timedelta(days=2 * index) for index in range(20)]
    facts = facts_on(days)
    view = table(every_day(START + timedelta(days=80)))
    now = at(START + timedelta(days=81))
    account = ranked_account(facts, visible_at=now)
    plan = compactable(account, open_day=None, diagnosis_since=now.date(), window=5)
    base = compact(account, plan, table=view)
    assert base.gap_lengths == ((2.0, 14),) and base.last_day == days[14]
    remaining = [fact for fact in facts if not base.covers(fact.at, fact.expectation_id)]
    for blank in (2, 9, 15, 16):
        today = days[-1] + timedelta(days=blank)
        assert absent(remaining, expectation(today), view, base) == absent(facts, expectation(today), view)
    assert not absent(remaining, expectation(days[-1] + timedelta(days=15)), view, base)
    assert absent(remaining, expectation(days[-1] + timedelta(days=16)), view, base)


def test_a_compacted_daily_past_still_sets_the_typical_gap() -> None:
    """前 40 天天天做（中间漏过两天）、已压实；窗口里只剩最近几次每三天一次。典型间隔按全历史（1 天）：空白 10 天是久别。
    压实那一段若只剩个数与最长的（3 天），典型间隔就成了 3 天，10 天不算了——所以压实存的是长度分布。"""

    from habitus.foresight import ranked_account
    from habitus.foresight.compaction import compact, compactable

    past = [day for day in _daily(0, 39) if day not in (START + timedelta(days=20), START + timedelta(days=21))]
    days = [*past, *(START + timedelta(days=offset) for offset in (42, 45, 48, 51, 54))]
    facts = facts_on(days)
    view = table(every_day(START + timedelta(days=80)))
    now = at(START + timedelta(days=79))
    account = ranked_account(facts, visible_at=now)
    plan = compactable(account, open_day=None, diagnosis_since=now.date(), window=5)
    base = compact(account, plan, table=view)
    assert base.gap_lengths == ((1.0, 36), (3.0, 1)) and base.last_day == past[-1]
    remaining = [fact for fact in facts if not base.covers(fact.at, fact.expectation_id)]
    today = START + timedelta(days=64)
    assert absent(remaining, expectation(today), view, base)
    assert absent(remaining, expectation(today), view, base) == absent(facts, expectation(today), view)
    assert not absent(remaining, expectation(START + timedelta(days=61)), view, base)  # 7 天：不到 8 个典型间隔


# --- 已形成、还没结算的预期也算机会（G3）------------------------------------------------------------
# 事实要到结算才写：D 日那条最晚时限 D+1 00:00，D+1 定稿后的夜批（D+2 04:00）才结；第 T 天白天账上最新的事实只到 T−2。


def _daily(first: int, last: int) -> list[date]:
    return [START + timedelta(days=offset) for offset in range(first, last + 1)]


def _formed(days: list[date]) -> list[Expectation]:
    return [expectation(day, f"e-{day.isoformat()}") for day in days]


def test_a_daily_habit_whose_last_days_are_formed_but_unsettled_is_not_a_return_after_absence() -> None:
    """天天做 30 天，事实只到 T−2、T−1 已形成未结算：空白是 T−1 → T（1 天），不是久别。"""

    today = START + timedelta(days=30)
    view = table(every_day(today - timedelta(days=1)))
    settled = facts_on(_daily(0, 28))
    item = expectation(today)
    unsettled = [*_formed(_daily(29, 29)), item]  # 这一条自己也还没结算
    assert not absent(settled, item, view, None, unsettled)
    assert not absent(settled, item, view)  # 只数已结算的空白也只有 2 天：不到 8 个典型间隔（裁定 88）


def test_a_real_absence_is_still_a_return_after_absence_with_unsettled_expectations_counted() -> None:
    """天天做 20 天后停下、8 天、10 天再回来，照样是久别：未结算的只补上已形成的机会，不掩盖真空白。"""

    # 停的第 1 天（20）那条照建、还没结算；之后没建（"离上次够久"在没见过的间隔上不建）；28 回来（空白 8 天）
    back = START + timedelta(days=28)
    view = table(every_day(back - timedelta(days=1)))
    item = expectation(back)
    later = expectation(back + timedelta(days=1), "later")  # 这一条之后才形成的不算
    pending = [*_formed(_daily(20, 20)), item, later]
    assert absent(facts_on(_daily(0, 19)), item, view, None, pending)
    # 停 10 天：20 那条已结成过期待决（真没做，算机会），之后不再建；30 回来
    back = START + timedelta(days=30)
    view = table(every_day(back - timedelta(days=1)))
    item = expectation(back)
    settled = [*facts_on(_daily(0, 19)), fact(START + timedelta(days=20), "e-20", kind="过期待决", good=False)]
    assert absent(settled, item, view, None, [item])


def test_once_an_unsettled_expectation_settles_as_void_the_settled_fact_decides() -> None:
    """未结算时先按机会算；结成「作废」后以事实为准，回到原规则。最后八天的预期已形成未结算：先当机会，不是久别；
    八条都结成「作废」后，空白是第 21 天 → 第 30 天（9 天），是久别。"""

    today = START + timedelta(days=30)
    view = table(every_day(today - timedelta(days=1)))
    item = expectation(today)
    last = _formed(_daily(22, 29))
    before = facts_on(_daily(0, 21))
    assert not absent(before, item, view, None, [*last, item])
    void = [fact(other.day, other.expectation_id, kind="作废", good=False, strength=0.0) for other in last]
    after = [*before, *void]
    # 读到结算前取的那份未结算清单也一样：账上已有事实的以事实为准
    assert absent(after, item, view, None, [*last, item]) == absent(after, item, view)
    assert absent(after, item, view, None, [*last, item])
