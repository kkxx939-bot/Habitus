"""行为上下文视图：投影（历史）、此刻、对比表、全历史聚合画像、相似情景反查。读时计算，不落盘，零 LLM。

五个入口沿预测树算候选的维度各取一类上下文（树用哪些维度算候选，语义侧就沿同样的维度取回）：
``history_contexts`` 带槽过滤对应时刻邻域、不带槽过滤喂 ``history_profile`` 对应跨周几 / 全天；
视图上的 ``preceding`` / ``following`` 与 ``similar_scenes`` 对应转移边；"同时在做"对应并行边；
"上一次"对应复发间隔；``now_context`` 的 ``today_kinds`` 与 ``gaps`` 对应累积率与曝光。

投影的零件（``first_of_day``、``neighbours``、``concurrent_refs``、``gaps_until`` 等）留在各自模块里，
不从这里导出。
"""

from habitus.scene.views.index import DayIndex, DayIndexCache
from habitus.scene.views.model import (
    ActionRef,
    ContextView,
    LastTime,
    Neighbour,
    ObservationGap,
)
from habitus.scene.views.projection import context_view, history_contexts, last_time

__all__ = [
    "ActionRef",
    "ContextView",
    "DayIndex",
    "DayIndexCache",
    "LastTime",
    "Neighbour",
    "ObservationGap",
    "context_view",
    "history_contexts",
    "last_time",
]
