"""递送那一拍的计数（``runtime.foresight_delivery``）：决定了几条、说了几条、今天说过几次、今天说过哪几条预期。"同类已说"看
它们的类，"上下级已说"看它们的类与说中的范围（裁定 101 第 8 条）；今天说过的从账本读（``spoken_today``），跨拍、重启都在。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from habitus.foresight.delivery import related_spoken
from habitus.foresight.hierarchy import KindHierarchy
from habitus.foresight.ledger import AWAITING_HOST, BASIS_WITHDRAWN, NODE_RETIRED, Delivery, Expectation
from habitus.runtime.foresight_ledger import ForesightLedgerStore


def spoken_today(ledger: ForesightLedgerStore, now: datetime) -> dict[str, Expectation | None]:
    """今天（按决定的日子）说过的那几条预期（条数就是说过几次；"同类已说"看它们的类，"上下级已说"看它们的类与说中的范围，
    读的是账本，跨拍、重启都在）——「待取」也算（话已经写好、等着问出口），同一条预期「待取」之后又来
    「说了」只算一次。说出口的递送挂在预期所在的日子下：今天决定的，那条预期还没结（结算在它最晚时限那天定稿之后），所以只看
    还没结完的日子。"""

    today = now.date()
    counted: dict[str, Expectation | None] = {}
    for day in ledger.open_days():
        made = ledger.deliveries_on(day)
        # 「待取」之后因节点不在用撤下的（「已停用」）没对人说出口：不再算今天说过（不然它会替还在用的上级压掉"上下级已说"）
        retired = {item.expectation_id for item in made if item.reason == NODE_RETIRED}
        deliveries = [
            item
            for item in made
            if item.spoken or (item.reason == AWAITING_HOST and item.expectation_id not in retired)
        ]
        if not deliveries:
            continue
        by_id = {item.expectation_id: item for item in ledger.expectations_on(day)}
        for delivery in deliveries:
            if delivery.decided_at.astimezone(now.tzinfo).date() == today and delivery.expectation_id not in counted:
                counted[delivery.expectation_id] = by_id.get(delivery.expectation_id)
    return counted


class Tally:
    """这一拍的计数：决定了几条、说了几条、今天说过几次、今天说过哪些类。"""

    def __init__(
        self,
        now: datetime,
        spoken: Mapping[str, Expectation | None],
        *,
        alive: bool = False,
        same_moment: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        self.now: datetime = now
        #: 今天说过（含「待取」）的预期（账上读不到那条预期的是 None：照样占名额）。
        self.counted: dict[str, Expectation | None] = dict(spoken)
        self.spoken_today: int = len(self.counted)
        self.spoken_kinds: set[str] = {item.kind_token for item in self.counted.values() if item is not None}
        #: 接手的成员 → 它那一刻的全部输入：算「同类已说」与名额时，同一刻里没说成的那几条不算。
        self.same_moment: dict[str, frozenset[str]] = dict(same_moment or {})
        #: 有没有宿主活着（这一拍开头看一次）。
        self.alive: bool = alive
        self.decided: int = 0
        self.spoken: int = 0
        self.by_reason: dict[str, int] = {}
        self.by_level: dict[int, int] = {}
        #: 这一拍刚记「待取」的（拍末交给出口）。
        self.awaiting: list[tuple[Expectation, Delivery]] = []
        self.withdrawn: set[str] = set()  # 这一拍撤下的「待取」（依据失效）：不再交给出口

    def count(self, delivery: Delivery, item: Expectation) -> None:
        if delivery.reason in (BASIS_WITHDRAWN, NODE_RETIRED):
            self.withdrawn.add(delivery.expectation_id)
        self.decided += 1
        self.by_reason[delivery.reason] = self.by_reason.get(delivery.reason, 0) + 1
        self.by_level[delivery.level] = self.by_level.get(delivery.level, 0) + 1
        if delivery.spoken:
            self.spoken += 1
        if delivery.spoken or delivery.reason == AWAITING_HOST:
            self.spoken_today += 1  # 「待取」也占名额、也算"同类已说""上下级已说"
            self.spoken_kinds.add(item.kind_token)
            self.counted[delivery.expectation_id] = item

    def spoken_for(self, expectation_id: str) -> tuple[int, frozenset[str], tuple[Expectation, ...]]:
        """这一条决定时看到的「今天说过几次、说过哪些类、说过哪几条」：接手的成员不算它那一刻里没说成的几条（它们一句也没说出口）。"""

        skip = self.same_moment.get(expectation_id)
        rest = {key: item for key, item in self.counted.items() if not skip or key not in skip}
        items = tuple(item for item in rest.values() if item is not None)
        if not skip:
            return self.spoken_today, frozenset(self.spoken_kinds), items
        return self.spoken_today - (len(self.counted) - len(rest)), frozenset(item.kind_token for item in items), items

    def room_for(self, item: Expectation, daily_cap: int, kinds: KindHierarchy) -> bool:
        """名额没满、同一类今天还没说过、今天也没对它的上级或下级在重叠的范围里说过（模型只问这一拍真要开口的）。"""

        spoken_today, spoken_kinds, spoken = self.spoken_for(item.expectation_id)
        return (
            spoken_today < daily_cap and item.kind_token not in spoken_kinds and not related_spoken(item, spoken, kinds)
        )


__all__ = ["Tally", "spoken_today"]
