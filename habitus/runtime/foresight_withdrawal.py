"""「待取」之后每拍记什么（递送那一拍，``runtime.foresight_delivery``）：做了「已发生」、过了最晚时限还没人问出口
「宿主没取」（``foresight.delivery.on_awaiting``）；都不是时看依据——话已写好、宿主还没取走，而依据此刻已不能开口（换代之后关系变回
候选、从机会表上消失），撤下请求、记「依据失效」；节点已不在用的同样撤下、记「已停用」（``foresight.delivery.on_withdrawn``）；同一刻最多挂一条请求，成员照现有规则接手。
已被取走的不撤（宿主可能已在问），盘上一张表都没有（读不出来）时认不出依据还成不成立，也不撤。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from habitus.foresight.delivery import on_awaiting, on_withdrawn
from habitus.foresight.hierarchy import KindHierarchy
from habitus.foresight.ledger import Delivery, Expectation
from habitus.foresight.opportunities import OpportunityTable
from habitus.foresight.units import voiced


class _Materials(Protocol):
    """递送那一拍的材料（``foresight_delivery.JudgeMaterials``）里这里要的三样：此刻的机会表、这件事这回做了没有、这一拍的上下级。"""

    @property
    def table(self) -> OpportunityTable | None: ...

    @property
    def happened(self) -> Callable[[Expectation], bool]: ...

    @property
    def kinds(self) -> KindHierarchy: ...


def awaiting_outcome(
    claimed: Callable[[str], bool],
    item: Expectation,
    pending: Delivery,
    materials: _Materials | None,
    now: datetime,
) -> Delivery | None:
    """这条「待取」这一拍追加哪一条递送（不追加是 None）。``claimed`` 是出口上这条请求被宿主取走过没有；节点此刻已不在用、
    宿主还没取走的撤下，记「已停用」（裁定 101）。"""

    happened = materials is not None and materials.happened(item)
    ended = on_awaiting(item, pending, now=now, happened=happened)
    if ended is not None:
        return ended
    taken = claimed(item.expectation_id)
    if materials is not None and not materials.kinds.active(item.kind_token):
        return on_withdrawn(item, pending, now=now, voiced=True, taken=taken, active=False)
    table = None if materials is None else materials.table
    if table is None:
        return None
    return on_withdrawn(item, pending, now=now, voiced=voiced(table, item), taken=taken)


__all__ = ["awaiting_outcome"]
