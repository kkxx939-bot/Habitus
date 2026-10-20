"""扎堆合并（裁定 59 第 6 条第 1 步）：同一类相邻两条开始时刻相差不超过合并窗口（按秒），就算同一次；一条接一条往下连。

全系统只用这一把尺：语义树的链（``chains_of``）、预测层的事件触发、``temporal`` 自己的"一次一次"都调它。
只吃一串开始时刻，不认识类、概念、树——谁的记号由调用方传。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime


def merge_starts(starts: Sequence[datetime], *, window_seconds: int) -> tuple[tuple[int, ...], ...]:
    """``starts`` 须按时刻升序；返回每一次包含的下标（升序）。相邻两条相差 ``<= window_seconds`` 秒并进同一次。

    例：17:00、17:40、18:20（窗口 2700 秒）是一次；17:00、18:30 是两次。一条记录持续多久与合并无关。
    """

    if isinstance(window_seconds, bool) or not isinstance(window_seconds, int) or window_seconds < 0:
        raise ValueError("window_seconds must be a non-negative integer")
    found: list[list[int]] = []
    previous: datetime | None = None
    for index, moment in enumerate(starts):
        if not isinstance(moment, datetime) or moment.utcoffset() is None:
            raise ValueError("starts must be timezone-aware datetimes")
        if previous is not None and moment < previous:
            raise ValueError("starts must be sorted ascending")
        if previous is not None and (moment - previous).total_seconds() <= window_seconds:
            found[-1].append(index)
        else:
            found.append([index])
        previous = moment
    return tuple(tuple(group) for group in found)


__all__ = ["merge_starts"]
