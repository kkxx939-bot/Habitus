"""按账精确压实的策略（裁定 57、61 第 7 条；26 号清单 4.8）：哪些事实文件可以删、留什么摘要。纯函数，读写在组合根。

**只压同时满足三条的**（每条都是"以后再也不会读它原样"的理由）：

1. 预期已经结算、再也不会被后补或改判：早于账本里最早一个还没结完的日子（``mark_final`` 的定义就是"全部结算过、过期待决的
   裁决再也不会变"）；
2. 早于诊断回看的那一段（30 + 30 天）；
3. 窗口之外：这本账最近 M 次机会（``Evidence.window``）一律留原样（单元历史的原料从结完的日子记，压实之前已记下）。

实际留下的是三条里要求最多的那一截。压实不再折权重（权重归单元，从单元历史各刻数），只留全历史摘要：自然命中的偏差、
机会间隔。重结只作用于没压实的窗口：压实的那一段本来就不会再变。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date

from habitus.foresight.absence import compacted_gaps
from habitus.foresight.clock import FactKey, RankedAccount, fact_key, is_opportunity
from habitus.foresight.ledger.compaction import AccountCompaction
from habitus.foresight.ledger.model import NATURAL_HITS, AccountFact
from habitus.foresight.opportunities.model import OpportunityTable


@dataclass(frozen=True)
class CompactionPlan:
    """压到哪一次机会为止（没有可压的为 None）与这一次要并进去的那几笔（按机会排）。"""

    boundary: FactKey | None
    facts: tuple[AccountFact, ...]


def compactable(account: RankedAccount, *, open_day: date | None, diagnosis_since: date, window: int) -> CompactionPlan:
    """``open_day`` 是账本里最早一个还没结完的日子（全结完了为 None），``diagnosis_since`` 是诊断回看的第一天，``window`` 是 M。"""

    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    keys = account.prediction.keys
    allowed = [
        key
        for key in keys[: max(0, len(keys) - window)]
        if key[0].date() < diagnosis_since and (open_day is None or key[0].date() < open_day)
    ]
    if not allowed:
        return CompactionPlan(None, ())
    boundary = allowed[-1]
    return CompactionPlan(boundary, tuple(fact for fact in account.facts if fact_key(fact) <= boundary))


def compact(account: RankedAccount, plan: CompactionPlan, *, table: OpportunityTable | None) -> AccountCompaction:
    """把 ``plan`` 的那几笔并进 ``account.base``（上一次压实的那一段），得到新的压实段。``account`` 是整本账（含窗内）。"""

    if plan.boundary is None:
        raise ValueError("nothing to compact")
    previous = account.base
    offsets = Counter(dict(previous.offsets) if previous is not None else {})
    for fact in plan.facts:
        if fact.good and fact.kind in NATURAL_HITS:
            offsets[abs(fact.offset_minutes or 0)] += 1
    days = sorted({fact.at.date() for fact in plan.facts if is_opportunity(fact)})
    gaps, last_day, day_types = compacted_gaps(previous, days, table)
    return AccountCompaction(
        account=_account_of(account, plan),
        through=plan.boundary,
        offsets=tuple(sorted(offsets.items())),
        gap_count=gaps.count,
        longest_gap=gaps.longest,
        last_day=last_day,
        day_types=day_types,
    )


def _account_of(account: RankedAccount, plan: CompactionPlan) -> str:
    if account.base is not None:
        return account.base.account
    return plan.facts[0].account


__all__ = ["CompactionPlan", "compact", "compactable"]
