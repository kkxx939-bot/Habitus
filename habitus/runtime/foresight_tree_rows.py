"""按天缓存的行为树行：结算（写）与诊断（只读）共用的两种取法。"""

from __future__ import annotations

from datetime import date, timedelta

from habitus.behavior.uri import BehaviorURI
from habitus.scene.views import DayIndexCache, FlowRow


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


__all__ = ["basis_row", "rows_between"]
