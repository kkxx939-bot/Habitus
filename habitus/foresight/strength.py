"""反馈的强度（裁定 51）：一件事做得多反常，就记多重的一笔；反常拿这本账自己的历史比，不拿别人、不拿手写数。

每个强度都是"这一次在这本账自己历史分布里的位置"。历史不到 ``MIN_HISTORY`` 次时只有方向、强度为 1（样本少就没有读数，
与触发层"只出现过一次不建"同一条规矩）。纯函数，对存储与 scene 零知识。

- 窗外发生（早到 / 晚了但做了）：好，强度 = 历史上偏差不小于这一次的比例（窗内发生记偏差 0）。
- 没发生、之后补做：坏，强度 = 历史间隔里不长于补做延迟的比例（隔得比他最短的正常间隔还短，就不算坏）。

提醒组的事实是单位权重（裁定 58 第 6 条）：拒绝、落空、提醒后做了量纲不同，不再给强度。
"""

from __future__ import annotations

from collections.abc import Sequence

from habitus.foresight.errors import ForesightError

#: 历史少于这么多次时只有方向、强度为 1。
MIN_HISTORY = 2


def offset_strength(offset_minutes: int, history: Sequence[int]) -> float:
    """窗外发生的好：历史上（窗内记 0）偏差绝对值不小于这一次的比例。"""

    if offset_minutes == 0 or len(history) < MIN_HISTORY:
        return 1.0
    size = abs(offset_minutes)
    return sum(1 for item in history if abs(item) >= size) / len(history)


def catch_up_strength(delay_days: float, gaps_days: Sequence[int]) -> float:
    """补做的坏：历史上相邻两次的间隔里，不长于这次补做延迟的比例。"""

    if delay_days < 0:
        raise ForesightError("a catch-up delay cannot be negative")
    if len(gaps_days) < MIN_HISTORY:
        return 1.0
    return sum(1 for gap in gaps_days if gap <= delay_days) / len(gaps_days)


__all__ = ["MIN_HISTORY", "catch_up_strength", "offset_strength"]
