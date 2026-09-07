"""行为上下文视图：投影（历史）、此刻、对比表。读时计算，不落盘，零 LLM。"""

from habitus.scene.views.compare import ComparisonTable, SlotComparison, Verdict, compare, render_table
from habitus.scene.views.index import DayIndex, DayIndexCache
from habitus.scene.views.model import (
    COMPARED_SLOTS,
    SLOT_LABELS,
    SLOT_NAMES,
    ActionRef,
    ContextView,
    LastTime,
    Precondition,
    SceneRef,
)
from habitus.scene.views.now import now_context
from habitus.scene.views.projection import context_view, history_contexts, last_time

__all__ = [
    "COMPARED_SLOTS",
    "SLOT_LABELS",
    "SLOT_NAMES",
    "ActionRef",
    "ComparisonTable",
    "ContextView",
    "DayIndex",
    "DayIndexCache",
    "LastTime",
    "Precondition",
    "SceneRef",
    "SlotComparison",
    "Verdict",
    "compare",
    "context_view",
    "history_contexts",
    "last_time",
    "now_context",
    "render_table",
]
