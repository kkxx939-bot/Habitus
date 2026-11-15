"""按天缓存的行为树行：结算（写）与诊断（只读）共用的两种取法。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

from habitus.behavior.uri import BehaviorURI
from habitus.scene.views import DayIndexCache, FlowRow
from habitus.series.reader import admitted

if TYPE_CHECKING:
    from habitus.runtime.foresight_settlement import BehaviorReadSide


def basis_row(cache: DayIndexCache, uri: str) -> FlowRow | None:
    try:
        day = BehaviorURI.parse(uri).to_address().occurred_on
    except (TypeError, ValueError):
        return None
    return cache.day(day).by_uri.get(uri)


def rows_between(cache: DayIndexCache, first: date, last: date) -> tuple[FlowRow, ...]:
    rows: list[FlowRow] = []
    cursor = first
    while cursor <= last:
        rows.extend(cache.day(cursor).rows)
        cursor += timedelta(days=1)
    return tuple(rows)


def day_cache(read_side: BehaviorReadSide) -> DayIndexCache:
    """一次查询用的按天索引：撞车重复、「非事件」不算（与事件序列同一条规则），树上的编号过同一个换算口。"""

    return DayIndexCache(
        read_side.behavior_tree,
        subject=read_side.subject,
        admits=admitted,
        calendar=read_side.calendar,
        kinds=read_side.kinds,
    )


__all__ = ["basis_row", "day_cache", "rows_between"]
