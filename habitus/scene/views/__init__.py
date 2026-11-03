"""⑤ ``scene/views/``：行为树的读侧投影。读时计算、零 LLM、不是权威，预测层取行与观测空白用。

- ``DayIndexCache``：一次查询里按日读一次行为树；``DayIndex.rows`` / ``by_uri`` 是一天的原子行为（``FlowRow``），
  ``gaps`` / ``DayIndexCache.has_gap`` / ``gaps_within`` 是观测空白（``ObservationGap``），``day_note`` 是当地日历的说法。

语义树的关系读数按新方案（``13-语义树新方案``）重建，不住在这里。
"""

from habitus.scene.views.index import GAP_LOOKBACK_DAYS, DayIndex, DayIndexCache
from habitus.scene.views.model import FlowRow, ObservationGap

__all__ = [
    "GAP_LOOKBACK_DAYS",
    "DayIndex",
    "DayIndexCache",
    "FlowRow",
    "ObservationGap",
]
