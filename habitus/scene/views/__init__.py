"""⑤ ``scene/views/``：行为树的读侧投影。读时计算、零 LLM、不是权威，预测层的证据包用。

- ``history_contexts`` / ``context_view``：某个候选落在钟面邻域里的每次发生，投影成视图。
- ``slot_neighbourhood`` / ``slot_neighbourhood_until``：一次发生（或此刻）前后 ±k 槽内的原子行为序列。

语义树的关系读数按新方案（``13-语义树新方案``）重建，不住在这里。
"""

from habitus.scene.views.index import GAP_LOOKBACK_DAYS, DayIndex, DayIndexCache
from habitus.scene.views.model import (
    ActionRef,
    ContextView,
    FlowRow,
    LastTime,
    Neighbour,
    ObservationGap,
)
from habitus.scene.views.neighbourhood import slot_neighbourhood, slot_neighbourhood_until, slot_window
from habitus.scene.views.projection import context_view, history_contexts, last_time

__all__ = [
    "GAP_LOOKBACK_DAYS",
    "ActionRef",
    "ContextView",
    "DayIndex",
    "DayIndexCache",
    "FlowRow",
    "LastTime",
    "Neighbour",
    "ObservationGap",
    "context_view",
    "history_contexts",
    "last_time",
    "slot_neighbourhood",
    "slot_neighbourhood_until",
    "slot_window",
]
