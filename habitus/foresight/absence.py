"""账的久别（裁定 57、60 第 1 条；26 号清单 3.3）：这本账的前因隔了比它历史上最长的一次还久才又出现，这一次只记影子、不开口。

问的是"他还在不在做这件事"：三周没碰的事今天又来了，往常那个钟点未必还对，先看一次再说。它只管"醒"的这一侧——不停建预期、
不改账的级、不写 ``UnitLevel``，下一次照常按账的档走。

- **序列**是这本账相邻两次**机会**（``clock.is_opportunity``，与遗忘的时钟同一口径）形成的那几天；这一次的空白 = 上一次机会
  形成那天 → 这条预期形成那天。
- **有效长度**按机会表上每天看着的比例算（人不在、设备没上传的日子不算长度），只数这本账有过机会的那几种日型（只在工作日
  有的事，国庆七天不算超过；没有放假数据时日型都答不出，日子都算）。形成过机会的日子（含这一次那天）按看着、按算数：前因
  就是那天看到的。表是昨夜的，昨夜之后到今天之间的日子按没在看算；表上没有覆盖时只剩机会日算长度，判不出久别（宁可少判）。
- **样本**：没截尾的间隔不到 ``MIN_HISTORY`` 个不判（代价只是一次影子）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import date

from habitus.foresight.clock import current_facts, is_opportunity
from habitus.foresight.ledger.compaction import AccountCompaction
from habitus.foresight.ledger.model import AccountFact, Expectation
from habitus.foresight.opportunities.model import OpportunityTable
from habitus.foresight.strength import MIN_HISTORY
from habitus.temporal.absence import (
    AbsenceVerdict,
    EffectiveGap,
    GapSummary,
    Visit,
    effective_gaps,
    exceeds_longest,
    open_gap,
)


def opportunity_days(
    facts: Iterable[AccountFact], item: Expectation, base: AccountCompaction | None = None
) -> tuple[date, ...]:
    """这条预期之前这本账（压实以外的）形成过机会的日子（升序、不重；按形成时刻的当地日期）。"""

    return tuple(
        sorted(
            {
                fact.at.date()
                for fact in current_facts(facts)
                if is_opportunity(fact)
                and fact.expectation_id != item.expectation_id
                and fact.at < item.formed_at
                and (base is None or not base.covers(fact.at, fact.expectation_id))
            }
        )
    )


def returned_after_absence(
    facts: Iterable[AccountFact],
    item: Expectation,
    table: OpportunityTable | None,
    base: AccountCompaction | None = None,
) -> bool:
    """这条预期是不是久别之后的第一次。没有机会表（不知道哪天在看）、之前没有机会时不判。``base`` 是压实了的那一段：
    它的间隔只剩个数与最长的，最后一次机会那天接着当序列的起点。"""

    if table is None:
        return False
    days = opportunity_days(facts, item, base)
    anchors = [*((base.last_day,) if base is not None and base.last_day is not None else ()), *days]
    today = item.formed_at.date()
    if not anchors or anchors[-1] >= today:
        return False
    types = {table.day_type(day) for day in days} | set(base.day_types if base is not None else ())
    prior, current = _gaps(anchors, today, table=table, types=types)
    earlier = GapSummary(base.gap_count, base.longest_gap) if base is not None else None
    return exceeds_longest(prior, current, min_samples=MIN_HISTORY, earlier=earlier) is AbsenceVerdict.EXCEEDS


def compacted_gaps(
    previous: AccountCompaction | None, days: Sequence[date], table: OpportunityTable | None
) -> tuple[GapSummary, date | None, tuple[str | None, ...]]:
    """压实时这一段的机会间隔：接着上一次压实的最后一天，算到这一段最后一次机会；返回（间隔摘要、最后一天、出现过的日型）。
    没有机会表时间隔照日子数算（全看着）。"""

    summary = GapSummary(previous.gap_count, previous.longest_gap) if previous is not None else GapSummary()
    anchors = [*((previous.last_day,) if previous is not None and previous.last_day is not None else ()), *days]
    known = set(previous.day_types if previous is not None else ())
    types = known | {table.day_type(day) if table is not None else None for day in days}
    if len(anchors) >= 2:
        seen = set(anchors)
        watched, counts = _rulers(table, seen, types)
        summary = summary.including(effective_gaps([Visit(day) for day in anchors], watched=watched, counts=counts))
    ordered = tuple(sorted(types, key=lambda value: (value is not None, value or "")))
    return summary, (anchors[-1] if anchors else None), ordered


def _gaps(
    anchors: Sequence[date], today: date, *, table: OpportunityTable, types: set[str | None]
) -> tuple[tuple[EffectiveGap, ...], EffectiveGap]:
    """形成过机会的日子（含今天）一律按看着、按算数：前因就是那天看到的。"""

    watched, counts = _rulers(table, {*anchors, today}, types)
    prior = effective_gaps([Visit(day) for day in anchors], watched=watched, counts=counts)
    current = open_gap(anchors[-1], today, watched=watched, counts=counts)
    assert current is not None
    return prior, current


def _rulers(
    table: OpportunityTable | None, seen: set[date], types: set[str | None]
) -> tuple[Callable[[date], float], Callable[[date], bool]]:
    def watched(day: date) -> float:
        if day in seen:
            return 1.0
        return 1.0 if table is None else table.watched(day)

    def counts(day: date) -> bool:
        return day in seen or table is None or table.day_type(day) in types

    return watched, counts


__all__ = ["compacted_gaps", "opportunity_days", "returned_after_absence"]
