"""④ ``scene/relations/``：关系检验（语义树新方案 ``13`` ①③）。纯算法、零模型调用、不落盘。

读事件序列（``habitus.series``）、概念集与概念命中，每晚每条 lane 把全部两两都量一遍：对照取同一天 / 同一周里人在场的
其他时刻，四段互不重叠的跨度，剔除不可能显著的，第 1 道加权 BH、第 2 道按块看稳、第 3 道区间够大。
跨夜的状态（候选 → 成立 → 失效）由 ``state.fold`` 逐晚折叠；落盘在 ``relations.store``（包根不转手它：
存储只给组合根用，检验与折叠保持纯算法）。

包根也转手链合并与在场口径（``Chain`` / ``chains_of`` / ``merge_for`` / ``Controls`` / ``Arrival``）：预测层的机会表要在同一条
时间线上按同一把尺列出历史每次机会，只许经这里拿，不另写一份。调节条件的判法（``Condition`` / ``evaluator`` /
``parse_condition``）同理：预测层给落空找原因时要判"这一次条件成立没有"，与检验同一份（裁定 55）。
"""

from habitus.scene.relations.conditions import Condition, ConditionError, Field, Template, evaluator
from habitus.scene.relations.conditions import parse as parse_condition
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
from habitus.scene.relations.spans import SEGMENTS, Arrival, Chain, Controls, Segment, chains_of, merge_for
from habitus.scene.relations.state import LaneState, Relation, StateError, Status, Transition, fold
from habitus.scene.relations.timeline import LaneTimeline, Mark, build_timelines

__all__ = [
    "SEGMENTS",
    "Arrival",
    "Chain",
    "Condition",
    "ConditionError",
    "Controls",
    "Field",
    "LaneNight",
    "LaneState",
    "LaneTests",
    "LaneTimeline",
    "Mark",
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
    "Template",
    "Transition",
    "Verdict",
    "build_timelines",
    "chains_of",
    "comparable",
    "evaluator",
    "examine_night",
    "fold",
    "merge_for",
    "parse_condition",
]
