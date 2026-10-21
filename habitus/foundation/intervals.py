"""时间区间上的小判断（纯函数）。"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from datetime import datetime


def starts_within(begin: datetime, end: datetime, starts: Sequence[datetime]) -> bool:
    """``starts``（升序）里有没有一个落在 ``[begin, end)``。"""

    stamps = [moment.timestamp() for moment in starts]
    index = bisect_left(stamps, begin.timestamp())
    return index < len(stamps) and stamps[index] < end.timestamp()


__all__ = ["starts_within"]
