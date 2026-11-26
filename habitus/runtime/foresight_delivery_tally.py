"""递送那一拍的计数（``runtime.foresight_delivery``）：决定了几条、说了几条、今天说过几次、今天说过哪几条预期。"同类已说"看
它们的类；今天说过的从账本读（``spoken_today``），跨拍、重启都在。
宿主没取走的不算说过（裁定 104）：记了「宿主没取」的，和记在最晚时限那一刻或之后、出口不给取的「待取」。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from habitus.foresight.ledger import (
    AWAITING_HOST,
    BASIS_WITHDRAWN,
    HOST_NEVER_CLAIMED,
    NODE_RETIRED,
    Delivery,
    Expectation,
)
from habitus.runtime.foresight_ledger import ForesightLedgerStore

#: 「待取」之后记了这两种的，话没对人说出口：不再算今天说过。
_NOT_SAID_AFTER_ALL = (NODE_RETIRED, HOST_NEVER_CLAIMED)


def _claimable(delivery: Delivery, item: Expectation | None) -> bool:
    """这条「待取」宿主取得走：记在最晚时限之前（出口从最晚时限那一刻起不给取）。账上读不到那条预期的照旧占名额。"""

    return item is None or delivery.decided_at < item.deadline


def spoken_today(ledger: ForesightLedgerStore, now: datetime) -> dict[str, Expectation | None]:
    """今天（按决定的日子）说过的那几条预期（条数就是说过几次；"同类已说"看它们的类，
    读的是账本，跨拍、重启都在）——「待取」也算（话已经写好、等着问出口），同一条预期「待取」之后又来
    「说了」只算一次；宿主取不走、没取走的不算（裁定 104）。说出口的递送挂在预期所在的日子下：今天决定的，那条预期还没结
    （结算在它最晚时限那天定稿之后），所以只看还没结完的日子。"""

    today = now.date()
    counted: dict[str, Expectation | None] = {}
    for day in ledger.open_days():
        made = ledger.deliveries_on(day)
        # 「待取」之后撤下的（「已停用」）、宿主一直没取的（「宿主没取」）没对人说出口：不再算今天说过
        unsaid = {item.expectation_id for item in made if item.reason in _NOT_SAID_AFTER_ALL}
        candidates = [
            item for item in made if item.spoken or (item.reason == AWAITING_HOST and item.expectation_id not in unsaid)
        ]
        if not candidates:
            continue
        by_id = {item.expectation_id: item for item in ledger.expectations_on(day)}
        deliveries = [item for item in candidates if item.spoken or _claimable(item, by_id.get(item.expectation_id))]
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
        if delivery.reason in _NOT_SAID_AFTER_ALL and delivery.expectation_id in self.counted:
            # 这一拍才知道没说成：同一拍后面决定的那几条就不再算它
            del self.counted[delivery.expectation_id]
            self.spoken_today = len(self.counted)
            self.spoken_kinds = {found.kind_token for found in self.counted.values() if found is not None}
        if delivery.spoken or (delivery.reason == AWAITING_HOST and _claimable(delivery, item)):
            self.spoken_today += 1  # 取得走的「待取」也占名额、也算"同类已说"
            self.spoken_kinds.add(item.kind_token)
            self.counted[delivery.expectation_id] = item

    def spoken_for(self, expectation_id: str) -> tuple[int, frozenset[str]]:
        """这一条决定时看到的「今天说过几次、说过哪些类」：接手的成员不算它那一刻里没说成的几条（它们一句也没说出口）。"""

        skip = self.same_moment.get(expectation_id)
        if not skip:
            return self.spoken_today, frozenset(self.spoken_kinds)
        rest = {key: item for key, item in self.counted.items() if key not in skip}
        kinds = frozenset(item.kind_token for item in rest.values() if item is not None)
        return self.spoken_today - (len(self.counted) - len(rest)), kinds

    def room_for(self, item: Expectation, daily_cap: int) -> bool:
        """名额没满、同一类今天还没说过（模型只问这一拍真要开口的）。"""

        spoken_today, spoken_kinds = self.spoken_for(item.expectation_id)
        return spoken_today < daily_cap and item.kind_token not in spoken_kinds


__all__ = ["Tally", "spoken_today"]
