"""一本账的流水怎么读（裁定 57、61）：账只是事实的流水——预期、结算、改判——不再从它折权重。

- **当时可见**：一刻能读到的是那一刻已经写下的事实（``precedence`` 不晚于它）；同一条预期只留最后写的那一笔
  （``current_facts``）。先按可见筛、再去重：之后才写的改判不能回头盖住那一刻的原判。
- **一次机会**（``is_opportunity``）：去重后的一条预期；作废、观测空白不算，「提醒后发生」算。账的久别、压实的边界、
  单元历史的原料都用这一个口径。
- **时钟**只给压实定边界（压到第几次机会为止）：权重归单元，从单元历史的各刻按边数（``foresight.edges``）。

本模块纯函数、对存储零知识。
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.compaction import AccountCompaction
from habitus.foresight.ledger.model import VOID, AccountFact

#: 改判一条已有事实的种类：同一时刻落账时排在原判之后（同一晚先结成过期待决、又裁成后补，后补要覆盖它）。
REVISIONS = frozenset({"后补"})
#: 观测空白结出的事实种类（强度 0）；真没做的同一种类、强度 1。
UNOBSERVED_KIND = "过期待决"

type FactKey = tuple[datetime, str]


def current_facts(facts: Iterable[AccountFact]) -> tuple[AccountFact, ...]:
    """同一条预期只留最后写的那一条（按时刻；同刻时改判排在原判之后，再按给出的先后）。"""

    latest: dict[str, AccountFact] = {}
    ordered = sorted(
        enumerate(facts),
        key=lambda pair: (pair[1].precedence, pair[1].kind in REVISIONS, pair[0]),
    )
    for item in ordered:
        latest[item[1].expectation_id] = item[1]
    return tuple(sorted(latest.values(), key=lambda fact: fact.at))


def is_opportunity(fact: AccountFact) -> bool:
    """这笔（去重后的）事实算不算这本账的一次机会。算不算按种类，不按强度给机会打折：

    - 「作废」不算：前因不成立，这次机会不存在；
    - 观测空白不算：窗里有空白又没看到发生，结算记的是强度 0 的「过期待决」（真没做的是强度 1）。情境触发不要求当天有记录，
      空白期照样每周建预期，算进去的话，看不见的那几周会把久别的空白切碎；
    - 「提醒后发生」算（强度 0 只是不给习惯记功）。"""

    if fact.kind == VOID:
        return False
    return not (fact.kind == UNOBSERVED_KIND and fact.strength == 0.0)


def fact_key(fact: AccountFact) -> FactKey:
    return (fact.at, fact.expectation_id)


@dataclass(frozen=True)
class Clock:
    """一条时钟：按（形成时刻, 预期身份）排好的机会。"""

    keys: tuple[FactKey, ...]

    def __post_init__(self) -> None:
        if list(self.keys) != sorted(set(self.keys)):
            raise ForesightError("a clock holds distinct opportunities in order")

    def rank(self, fact: AccountFact) -> int:
        """这笔事实在这条时钟上的名次：它那一刻（含）之前有几次机会减一（最早的机会之前的也记 0）。"""

        return max(0, bisect_right(self.keys, fact_key(fact)) - 1)

    @property
    def last(self) -> int:
        return max(0, len(self.keys) - 1)


@dataclass(frozen=True)
class RankedAccount:
    """一本账在 ``visible_at`` 那一刻可见的事实（去重、按机会排好）与机会的时钟。``base`` 是压实了的那一段：``facts`` 与时钟
    只含它之后的。"""

    visible_at: datetime
    facts: tuple[AccountFact, ...]
    prediction: Clock
    base: AccountCompaction | None = None


def ranked_account(
    facts: Iterable[AccountFact], *, visible_at: datetime, base: AccountCompaction | None = None
) -> RankedAccount:
    """``visible_at`` 那一刻可见的整本账，编好时钟。``base`` 以内残留的事实文件一律忽略（先写累计值、再删文件，
    中途崩溃不会数两遍）。"""

    if not isinstance(visible_at, datetime) or visible_at.utcoffset() is None:
        raise ForesightError("visible_at must be a timezone-aware datetime")
    visible = (
        fact
        for fact in facts
        if fact.precedence <= visible_at and (base is None or not base.covers(fact.at, fact.expectation_id))
    )
    live = tuple(sorted(current_facts(visible), key=fact_key))
    return RankedAccount(
        visible_at=visible_at,
        facts=live,
        prediction=Clock(tuple(fact_key(fact) for fact in live if is_opportunity(fact))),
        base=base,
    )


__all__ = [
    "REVISIONS",
    "UNOBSERVED_KIND",
    "Clock",
    "RankedAccount",
    "current_facts",
    "fact_key",
    "is_opportunity",
    "ranked_account",
]
