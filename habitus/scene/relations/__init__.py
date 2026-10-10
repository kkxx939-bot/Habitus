"""④ ``scene/relations/``：关系检验（语义树新方案 ``13`` ①③）。纯算法、零模型调用、不落盘。

读事件序列（``habitus.series``）、概念集与概念命中，每晚每条 lane 把全部两两都量一遍：对照取同一天 / 同一周里人在场的
其他时刻，四段互不重叠的跨度，剔除不可能显著的，第 1 道加权 BH、第 2 道按块看稳、第 3 道区间够大。
跨夜的状态（候选 → 成立 → 失效）由 ``state.fold`` 逐晚折叠；落盘在 ``relations.store``（包根不转手它：
存储只给组合根用，检验与折叠保持纯算法）。
"""

from habitus.scene.relations.config import RelationConfig, RelationConfigError, RelationThresholds
from habitus.scene.relations.engine import (
    LaneNight,
    LaneTests,
    RelationKey,
    RelationTest,
    Subset,
    Verdict,
    comparable,
    examine_night,
)
from habitus.scene.relations.spans import SEGMENTS, Segment
from habitus.scene.relations.state import LaneState, Relation, StateError, Status, Transition, fold
from habitus.scene.relations.timeline import LaneTimeline, build_timelines

__all__ = [
    "SEGMENTS",
    "LaneNight",
    "LaneState",
    "LaneTests",
    "LaneTimeline",
    "RelationConfig",
    "RelationConfigError",
    "RelationThresholds",
    "Relation",
    "RelationKey",
    "RelationTest",
    "Segment",
    "StateError",
    "Status",
    "Subset",
    "Transition",
    "Verdict",
    "build_timelines",
    "comparable",
    "examine_night",
    "fold",
]
