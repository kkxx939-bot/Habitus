"""结算往账上写哪一笔事实（裁定 48–51、54、61）：纯函数，不读盘、不读时钟。组合根（``runtime/foresight_settlement.py``）
读账本与行为树、把结果交给这里，再把这里给出的事实落账（2026-10-17 四方评审：原来这些规则与读写混在结算那一拍里，只能落盘来测）。

- **结算**：一笔"做"的事实——准时 / 早到 / 晚了但做了是好，过期待决是坏；提醒后才做的是中性（强度 0：提醒的功劳不算习惯本身）。
  提醒反馈不在账上，挂在行为单元上（``foresight.feedback``）。
- **后补**：覆盖那条过期待决的坏，按"补做那天离上一次发生隔了几天"在这一类**这次机会之前**的历史间隔里的位置打折。
- **裁不裁**：以后还会不会被裁出不同的结果（下一次机会已形成且定稿 / 已隔得比最长的正常间隔还久 / 没有间隔可比）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta

from habitus.foresight.clock import current_facts
from habitus.foresight.ledger.model import NATURAL_HITS, AccountFact, Expectation, Ruling, Settlement
from habitus.foresight.opportunities.model import OpportunityTable
from habitus.foresight.strength import MIN_HISTORY, catch_up_strength, offset_strength


def settlement_fact(
    item: Expectation,
    result: Settlement,
    *,
    known: Iterable[AccountFact],
    earlier_offsets: Sequence[int] = (),
) -> AccountFact:
    """一次结算的事实。``known`` 是这本账上结算时已经写下的事实（结算取最晚时限之前写下的：哪一晚结、断没断过重来，算出的都是
    同一份；重结取此刻）；强度的历史分布只看这本账在这次机会**之前**的预期（之后的机会拿来比这一次是用未来解释过去）。
    事实的时刻是这次机会形成的时刻（不是结算那一刻）：先后按机会算。``earlier_offsets`` 是这本账压实了的那一段里自然命中的
    偏差（压实只压早于最早一个没结完的日子的，那一段全在这次机会之前）。"""

    account, at = item.account, item.formed_at
    prior = tuple(fact for fact in current_facts(known) if fact.expectation_id != item.expectation_id and fact.at < at)
    outcome = result.outcome
    if outcome in NATURAL_HITS:
        offset = 0 if outcome == "准时" else int(result.offset_minutes or 0)
        history = [
            *earlier_offsets,
            *(fact.offset_minutes or 0 for fact in prior if fact.good and fact.kind in NATURAL_HITS),
        ]
        strength = offset_strength(offset, history)
        return AccountFact(account, at, True, strength, outcome, item.expectation_id, offset_minutes=offset)
    if outcome == "提醒后发生":
        delay = int(result.offset_minutes or 0)
        return AccountFact(account, at, True, 0.0, outcome, item.expectation_id, offset_minutes=delay)
    return AccountFact(account, at, False, 1.0, "过期待决", item.expectation_id)


def catch_up_fact(item: Expectation, ruling: Ruling, table: OpportunityTable | None) -> AccountFact:
    """后补：覆盖那条过期待决的坏（隔得比他最短的正常间隔还短就不算坏；已经隔了一个正常间隔那么久，就和没做一样坏）。"""

    delay = int(ruling.delay_minutes or 0)
    return AccountFact(
        item.account,
        item.formed_at,
        False,
        _catch_up_strength(item, delay, table),
        "后补",
        item.expectation_id,
        offset_minutes=delay,
        written_at=ruling.ruled_at,
    )


def last_before(item: Expectation, table: OpportunityTable | None) -> tuple[date | None, tuple[int, ...]]:
    """这条预期之前这一类最后一次发生在哪天，与这一类**在那之前**相邻两次隔几天（机会表里这个类的复现，由 ``temporal`` 算，
    裁定 59：同一套合并、按日历日相减；以被提醒结尾的截尾那几段不算）。没有表就是空。
    间隔只取这次机会之前的：今晚的复现里还有这次补做本身和之后的发生，拿来比是用未来解释过去，哪一晚裁出的强度也就不同、
    回放不可重复（2026-10-17 四方评审）。"""

    kind = table.kind(item.kind_token) if table is not None else None
    if kind is None:
        return None, ()
    earlier = [entry.day for entry in kind.days if entry.first_at < item.observed_from]
    gaps = tuple(
        interval.days
        for interval in kind.intervals
        if interval.following.first_at < item.observed_from and not interval.censored
    )
    return (max(earlier) if earlier else None), gaps


def ruling_settled(
    item: Expectation, before: datetime | None, last_closed: date, table: OpportunityTable | None
) -> bool:
    """这条过期待决以后还会不会被裁出不同的结果：下一次机会已经形成且那天已定稿（之后的发生归下一次），或者从上一次发生到
    已定稿的尽头已经隔了不短于最长的正常间隔（再补做强度也是 1），或者没有间隔可比（强度恒为 1）。"""

    if before is not None and last_closed >= before.date():
        return True
    if ruling_moot(item, table):
        return True
    last, gaps = last_before(item, table)
    assert last is not None
    return (last_closed - last).days >= max(gaps)


def ruling_moot(item: Expectation, table: OpportunityTable | None) -> bool:
    """没有上一次或没有间隔可比：补做的强度恒为 1，裁不裁账上都一样坏（裁了只多一个「后补」的名字）。"""

    last, gaps = last_before(item, table)
    return last is None or len(gaps) < MIN_HISTORY


def slice_of(item: Expectation) -> str:
    """校准表的切片：触发的种类（事件按跨度段，情境按日历键，离上次够久一种）。"""

    parts = item.trigger.split("|")
    if item.source == "情境":
        return f"情境:{parts[1] if len(parts) >= 4 else item.trigger}"
    if item.source == "事件":
        # 关系键 ``lane|前因|后果|段[|条件]``：段在第 4 个字段（条件文本里不许有 ``|``），带条件的账照旧按段切片
        return f"事件:{parts[3] if len(parts) >= 4 else item.trigger}"
    return f"日子:{item.trigger}"


def same_fact(old: AccountFact, new: AccountFact) -> bool:
    """重结时两笔是不是同一个结论（种类、方向、强度、偏差）。"""

    return (old.kind, old.good, round(old.strength, 9), old.offset_minutes) == (
        new.kind,
        new.good,
        round(new.strength, 9),
        new.offset_minutes,
    )


def _catch_up_strength(item: Expectation, delay_minutes: int, table: OpportunityTable | None) -> float:
    last, gaps = last_before(item, table)
    if last is None:
        return 1.0
    caught = (item.deadline + timedelta(minutes=delay_minutes)).date()
    return catch_up_strength(float(max(0, (caught - last).days)), gaps)


__all__ = [
    "catch_up_fact",
    "last_before",
    "ruling_moot",
    "ruling_settled",
    "same_fact",
    "settlement_fact",
    "slice_of",
]
