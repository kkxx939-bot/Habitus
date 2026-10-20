"""按名次遗忘：一笔证据的分量只看它之后这条轴又来了几次机会，不看隔了几天（裁定 57）。

按天遗忘会把一个月一次的事压到只剩两三次有效样本；按名次遗忘对节奏不同的事一视同仁。名次在调用方
认定的"当时可见"的全部证据上编一次，再往下传——只在子集上编，等于每个子集各自计时。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from habitus.foundation.lifecycle._validation import require_half_life

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison


@dataclass(frozen=True)
class Ranked[T]:
    """一笔证据与它的名次：从 0 起，越晚越大。"""

    item: T
    rank: int

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")


def rank_by[T](items: Iterable[T], *, key: Callable[[T], SupportsRichComparison]) -> tuple[Ranked[T], ...]:
    """按 ``key`` 稳定排序后编名次；``key`` 相同的保持输入顺序，所以调用方要把并列的先后写进 ``key``。"""

    return tuple(Ranked(item, index) for index, item in enumerate(sorted(items, key=key)))


def decay(rank_gap: int, *, half_life: float) -> float:
    """名次差为 ``rank_gap`` 的那一笔还剩多少分量：0.5^(rank_gap / half_life)。"""

    if isinstance(rank_gap, bool) or not isinstance(rank_gap, int) or rank_gap < 0:
        raise ValueError("rank_gap must be a non-negative integer")
    require_half_life(half_life)
    return 0.5 ** (rank_gap / half_life)


def effective_cap(*, half_life: float) -> float:
    """按名次遗忘时有效样本的上限：每一笔都满分量、无穷多笔时 Σ 0.5^(i / H) = 1 / (1 − 2^(−1/H))（H = 8 时约 12.05）。"""

    require_half_life(half_life)
    return 1.0 / (1.0 - 0.5 ** (1.0 / half_life))


__all__ = ["Ranked", "decay", "effective_cap", "rank_by"]
