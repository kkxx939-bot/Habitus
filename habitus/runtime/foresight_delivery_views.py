"""递送那一拍的只读可见面（裁定 50）：全部账、全部行为单元、最近的漏报此刻的样子。

只读账本与单元历史，不决定、不落账；可见面本身在层外。边此刻的位置用递送引擎同一个读法（``DeliveryStage._standing``），
由引擎交进来，两边不各算一份。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from habitus.foresight import Standing
from habitus.foresight.ledger import parse_account, relation_backed
from habitus.foresight.opportunities import OpportunityTable, TreeReading
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_units import UnitPlanner, UnitView

#: 一条边此刻的位置：(账, 行为, 此刻, 机会表) → 位置。
StandingReader = Callable[[str, str, datetime, OpportunityTable | None], Standing]


@dataclass(frozen=True)
class AccountView:
    """可见面读的一本账（裁定 50）：是什么、此刻在哪一档、两条轴的读数、设置、最近几条事实。"""

    account: str
    standing: Standing
    facts: int
    recent: tuple[str, ...]


@dataclass(frozen=True)
class OmissionView:
    """可见面里"还没看懂的事"（裁定 50）：一类行为最近一段里发生了、而发生前没有任何一条在时预期的次数。
    ``with_expectation`` 是其中那天这一类有过别的预期的次数（时窗没对上）；其余是根本没有预期（还没认出规律）。"""

    kind_token: str
    count: int
    with_expectation: int
    latest_uri: str


class DeliveryViews:
    def __init__(self, ledger: ForesightLedgerStore, *, standing: StandingReader) -> None:
        self.ledger = ledger
        self._standing = standing

    def account_views(
        self, now: datetime, table: OpportunityTable | None = None, *, recent: int = 5
    ) -> tuple[AccountView, ...]:
        """全部账此刻的样子（可见面的只读口；可见面本身在层外）：边的位置从单元历史读，流水里最近几笔给人看。"""

        views = []
        for account in self.ledger.accounts():
            facts = self.ledger.facts_of(account)
            kind, _source, _trigger = parse_account(account)
            standing = self._standing(account, kind, now, table)
            tail = tuple(f"{item.at:%m-%d} {'好' if item.good else '坏'} {item.kind}" for item in facts[-recent:])
            views.append(AccountView(account, standing, len(facts), tail))
        return tuple(views)

    def unit_views(self, now: datetime, table: OpportunityTable | None) -> tuple[UnitView, ...]:
        """全部行为单元此刻的样子（可见面的只读口）：每个行为的每个输入——边此刻的位置、语义树读数、两个数对不对得上。"""

        by_kind: dict[str, list[tuple[str, Standing, TreeReading | None]]] = {}
        for account in self.ledger.accounts():
            kind, source, trigger = parse_account(account)
            standing = self._standing(account, kind, now, table)
            relation = table.relation(trigger) if table is not None and relation_backed(source) else None
            by_kind.setdefault(kind, []).append((account, standing, relation.reading if relation is not None else None))
        return UnitPlanner.views(by_kind)

    def omission_views(self, now: datetime, *, days: int = 30) -> tuple[OmissionView, ...]:
        """最近 ``days`` 天的漏报按类汇总，次数多的在前（可见面的只读口）。"""

        found: dict[str, list[tuple[datetime, str, bool]]] = {}
        for offset in range(days, -1, -1):
            for item in self.ledger.omissions_on(now.date() - timedelta(days=offset)):
                found.setdefault(item.kind_token, []).append(
                    (item.noticed_at, item.occurrence_uri, item.had_expectation)
                )
        views = [
            OmissionView(
                kind_token=kind,
                count=len(items),
                with_expectation=sum(1 for _at, _uri, had in items if had),
                latest_uri=max(items)[1],
            )
            for kind, items in found.items()
        ]
        return tuple(sorted(views, key=lambda view: (-view.count, view.kind_token)))


__all__ = ["AccountView", "DeliveryViews", "OmissionView"]
