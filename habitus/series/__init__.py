"""事件序列：行为树每晚被读成的那一份输入，语义树、``temporal`` 与机会表共用。

包根只导出纯值（``series.model``）；读行为树的 ``series.reader`` 不从这里转手——包根一 import 它，
两棵树 ``from habitus.series import EventSeries`` 就会把行为树与词表连带进来。组合根按模块路径直接 import 它。
"""

from habitus.series.model import (
    SILENT_DAY_UNWATCHED_LANES,
    EventSeries,
    GapKind,
    SeriesError,
    SeriesGap,
    SeriesRecord,
    SkippedRecord,
    SkipReason,
    silent_day_unwatched,
    standing_parts,
)

__all__ = [
    "EventSeries",
    "GapKind",
    "SeriesError",
    "SeriesGap",
    "SeriesRecord",
    "SkipReason",
    "SILENT_DAY_UNWATCHED_LANES",
    "SkippedRecord",
    "silent_day_unwatched",
    "standing_parts",
]
