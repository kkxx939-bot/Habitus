"""测试里一次性读行为序列的便捷入口；生产换代用常驻的 ``SeriesReader``（带缓存与脏日水位）。"""

from __future__ import annotations

from datetime import date

from habitus.behavior.tree import BehaviorTree
from habitus.series.model import EventSeries, KindResolution
from habitus.series.reader import SeriesReader


def read_series(tree: BehaviorTree, *, cutoff: date, kinds: KindResolution | None = None) -> EventSeries:
    """截至 ``cutoff`` 的序列，读一次不缓存。"""

    return SeriesReader(tree, kinds=kinds).read(cutoff=cutoff)
