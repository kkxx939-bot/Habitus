"""连续前缀水位：``through`` 及以前的全部已处理，``through`` 之后零散处理过的记在 ``above``。

处理顺序可以乱（定稿日不按顺序、同一地址的会话源可能后到先终态），但"可以放心删 / 不用再扫"
只能按连续前缀推进——只记"最晚处理到哪"会把中间没处理的跳过去。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, Self


class Ordered(Protocol):
    """能排先后的键（日期、整数序号）。"""

    def __lt__(self, other: Self, /) -> bool: ...

    def __hash__(self) -> int: ...

    def __eq__(self, other: Any, /) -> bool: ...


@dataclass(frozen=True)
class Watermark[K: Ordered]:
    """``through`` 由调用方给初值（第一个键的前一个），之后只经 ``mark`` 推进。"""

    through: K
    above: frozenset[K] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if any(not self.through < key for key in self.above):
            raise ValueError("keys above the watermark must be later than through")


def covers[K: Ordered](watermark: Watermark[K], key: K) -> bool:
    """这个键处理过没有。"""

    return not watermark.through < key or key in watermark.above


def mark[K: Ordered](watermark: Watermark[K], key: K, *, successor: Callable[[K], K]) -> Watermark[K]:
    """记下 ``key`` 处理完了；能连上的就把 ``through`` 往后推。"""

    if covers(watermark, key):
        return watermark
    above = set(watermark.above)
    above.add(key)
    through = watermark.through
    while (following := successor(through)) in above:
        above.remove(following)
        through = following
    return Watermark(through=through, above=frozenset(above))


__all__ = ["Ordered", "Watermark", "covers", "mark"]
