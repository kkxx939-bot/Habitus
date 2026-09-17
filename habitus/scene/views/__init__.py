"""行为上下文的读侧：读时计算，不落盘，零 LLM。

三组入口，全部按预测树算候选的维度取（树用哪些维度算候选，语义侧就沿同样的维度取回）：

- ``history_contexts`` / ``context_view``：某个候选落在钟面邻域里的每次发生，投影成视图
  （起因、上一次、同时在做、和谁、当天首次、紧邻上下条、日型），只读行为树。
- ``slot_neighbourhood`` / ``slot_neighbourhood_until``：一次发生（或此刻）前后 ±k 槽内的原子行为序列，
  历史侧与此刻侧共用一处实现，只读行为树。
- ``association_glosses`` / ``situations_of``：规律树上该候选的关联记录与情形列表，按 ``occurrence_uri``
  贴到上面的发生上；这是 ``views`` 里唯一读规律树的模块。

投影的零件（``first_of_day``、``neighbours``、``concurrent_refs`` 等）留在各自模块里，不从这里导出。
"""

from habitus.scene.views.gloss import AssociationGloss, Situation, association_glosses, situations_of
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
    "AssociationGloss",
    "ContextView",
    "DayIndex",
    "DayIndexCache",
    "FlowRow",
    "LastTime",
    "Neighbour",
    "ObservationGap",
    "Situation",
    "association_glosses",
    "context_view",
    "history_contexts",
    "last_time",
    "situations_of",
    "slot_neighbourhood",
    "slot_neighbourhood_until",
    "slot_window",
]
