"""上级让给还在等时窗末的下级（裁定 104 第 3 条）的取材料：规则在 ``foresight.delivery.waits_for_finer``，这里读账、算下级此刻能不能开口。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from habitus.foresight import Standing
from habitus.foresight.delivery import waits_for_finer, yield_along
from habitus.foresight.hierarchy import FLAT, KindHierarchy
from habitus.foresight.ledger import Delivery, Expectation
from habitus.runtime.foresight_ledger import ForesightLedgerStore

#: 一条预期那本账此刻的位置。
StandingOf = Callable[[Expectation], Standing]


class _Materials(Protocol):
    """递送那一拍的材料（``foresight_delivery.JudgeMaterials``）里这里要的两样：这件事这回做了没有、这一拍的上下级。"""

    @property
    def happened(self) -> Callable[[Expectation], bool]: ...

    @property
    def kinds(self) -> KindHierarchy: ...


def finer_waiting(
    ledger: ForesightLedgerStore,
    closing: Sequence[tuple[Expectation, Delivery]],
    materials: _Materials | None,
    since: datetime,
    now: datetime,
    tick: timedelta,
    deferred: Collection[str],
    unit_level: Callable[[Expectation], int],
) -> frozenset[str]:
    """这一拍到了时窗末、先让给下级的那几条上级的身份。这一拍到时窗末的都没有下级时不读账。``tick`` 是配置的拍长，``deferred`` 是这一拍被重读推迟的预期。``unit_level`` 现算单元此刻的档：
    账上存的档是上一次决定时记的，提醒反馈已经把单元压回影子时它还是旧的。"""

    kinds = materials.kinds if materials is not None else FLAT
    parents = [item for item, _pending in closing if kinds.below.get(item.kind_token)]
    if materials is None or not parents:
        return frozenset()
    others: list[tuple[Expectation, list[Delivery]]] = []
    for day in ledger.open_days():
        made: dict[str, list[Delivery]] = {}
        for delivery in ledger.deliveries_on(day):
            made.setdefault(delivery.expectation_id, []).append(delivery)
        others.extend((item, made.get(item.expectation_id, [])) for item in ledger.unsettled_expectations(day))

    def speaks(item: Expectation) -> bool:
        return kinds.active(item.kind_token) and not materials.happened(item) and unit_level(item) >= 1

    instant = now.astimezone(UTC)
    found = frozenset(
        item.expectation_id
        for item in parents
        if waits_for_finer(
            item, others, since=since, instant=instant, tick=tick, deferred=deferred, kinds=kinds, speaks=speaks
        )
    )
    return yield_along((item for item, _pending in closing), found, kinds)  # 三层以上：让位传递到底


async def deciding_closing[M: _Materials](
    ledger: ForesightLedgerStore,
    closing: Sequence[tuple[Expectation, Delivery]],
    materials: M | None,
    since: datetime,
    now: datetime,
    tick: timedelta,
    reread: Callable[[Sequence[tuple[Expectation, Delivery]], M | None], Awaitable[frozenset[str]]],
    unit_level: Callable[[Expectation], int],
) -> tuple[tuple[Expectation, Delivery], ...]:
    """这一拍到了时窗末、真要决定的那几条：当场重读（``reread``）只给真要决定的，让着下级的上级不重读、不花模型钱；有下级被重读
    推迟到下一拍时再看一遍，让着它的上级也留到下一拍。"""

    yielding = finer_waiting(ledger, closing, materials, since, now, tick, frozenset(), unit_level)
    deferred = await reread([pair for pair in closing if pair[0].expectation_id not in yielding], materials)
    if deferred:
        yielding |= finer_waiting(ledger, closing, materials, since, now, tick, deferred, unit_level)
    return tuple(pair for pair in closing if pair[0].expectation_id not in yielding | deferred)


__all__ = ["StandingOf", "deciding_closing", "finer_waiting"]
