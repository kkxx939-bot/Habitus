"""按天缓存的行为树行：结算（写）与诊断（只读）共用的两种取法。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

from habitus.behavior.model import BehaviorKind
from habitus.behavior.uri import BehaviorURI, BehaviorURIError
from habitus.foresight.hierarchy import FLAT, KindHierarchy
from habitus.scene.views import DayIndexCache, FlowRow
from habitus.series.reader import admitted

if TYPE_CHECKING:
    from habitus.runtime.foresight_settlement import BehaviorReadSide


def tree_day(source: str) -> date | None:
    """出处若是树上一条事件的地址，给它那一天；不是树上的出处（``unsealed:<开始时刻>``、事件触发的开始时刻、情境触发的
    日期——都不带 ``://``）给 None。带 ``://`` 却不是一支里的事件地址（例如退出了的旧协议写法）直接报错，不当成"找不到"，
    否则换了地址写法之后漏改的地方只会悄悄取不到行（裁定 112）。"""

    if "://" not in source:
        return None
    address = BehaviorURI.parse(source).to_address()
    if address.kind is not BehaviorKind.OCCURRENCE:
        # 出处只会是一条事件；空白的地址拿去按天找事件行，只会悄悄找不到
        raise BehaviorURIError(f"a source on the tree must be an occurrence, not a {address.kind.value}: {source}")
    return address.occurred_on


def basis_row(cache: DayIndexCache, uri: str) -> FlowRow | None:
    day = tree_day(uri)
    return None if day is None else cache.day(day).by_uri.get(uri)


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


def hierarchy_of(read_side: BehaviorReadSide) -> KindHierarchy:
    """此刻的上下级（组合根装的口读一次；没接词表是没有上下级、全都在用）。一次结算 / 一拍开头取一份，整次用同一份。"""

    return FLAT if read_side.hierarchy is None else read_side.hierarchy()


__all__ = ["basis_row", "day_cache", "hierarchy_of", "rows_between"]
