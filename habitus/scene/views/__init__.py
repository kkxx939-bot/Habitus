"""⑤ ``scene/views/``：读侧。两半都是读时计算、零 LLM、不是权威。

**行为树的读侧**（旧有，预测层的证据包用）：

- ``history_contexts`` / ``context_view``：某个候选落在钟面邻域里的每次发生，投影成视图。
- ``slot_neighbourhood`` / ``slot_neighbourhood_until``：一次发生（或此刻）前后 ±k 槽内的原子行为序列。

**语义树的投影**（第 4 刀，读 concepts / hypotheses / occurrences / ledger 四支，四支反过来不引用这里）：

- ``relations``：每条假设的当前读数——累积度门槛、强度（三方面三套，概率账兼给兑现间隔）、类型、上一级指针、稳定性扫、
  剂量、共享证据的调节对照、干预分层。
- ``residue``：残差候选——没映射到概念的 kind 离升级判据差多少。
- ``people``：profile（稳定成立的关系）与 entities（按对象类情境分账的切片）。
- ``materialize``：把读数落成 ``scene/views/`` 下的 Markdown，整个重写、不是权威。
"""

from habitus.scene.views.behaviours import BehaviourSide, BehaviourView, behaviour_views
from habitus.scene.views.index import GAP_LOOKBACK_DAYS, DayIndex, DayIndexCache
from habitus.scene.views.kinds import KindSpread, concept_kinds, kinds_by_concept
from habitus.scene.views.materialize import VIEWS_SEGMENT, ViewsStore, ViewsStoreError, materialize_views
from habitus.scene.views.model import (
    ActionRef,
    ContextView,
    FlowRow,
    LastTime,
    Neighbour,
    ObservationGap,
)
from habitus.scene.views.neighbourhood import slot_neighbourhood, slot_neighbourhood_until, slot_window
from habitus.scene.views.people import EntitySlice, ProfileView, entity_slices, profile_view
from habitus.scene.views.projection import context_view, history_contexts, last_time
from habitus.scene.views.relations import (
    Account,
    Accumulation,
    DoseReading,
    EvidenceIndex,
    FallbackReading,
    FulfilmentReading,
    Influence,
    Moderation,
    RelationReading,
    SharedEvidence,
    StabilityReport,
    Strength,
    TypeReading,
    TypeReadout,
    ViewsConfig,
    load_account,
    read_relation,
    read_relations,
)
from habitus.scene.views.residue import ResidueCandidate, residue_candidates
from habitus.scene.views.stats import Interval

__all__ = [
    "GAP_LOOKBACK_DAYS",
    "VIEWS_SEGMENT",
    "Account",
    "BehaviourSide",
    "BehaviourView",
    "Accumulation",
    "ActionRef",
    "ContextView",
    "DayIndex",
    "DayIndexCache",
    "DoseReading",
    "EntitySlice",
    "EvidenceIndex",
    "FallbackReading",
    "FlowRow",
    "Interval",
    "KindSpread",
    "LastTime",
    "Moderation",
    "Neighbour",
    "ObservationGap",
    "ProfileView",
    "FulfilmentReading",
    "Influence",
    "RelationReading",
    "ResidueCandidate",
    "SharedEvidence",
    "StabilityReport",
    "Strength",
    "TypeReading",
    "TypeReadout",
    "ViewsConfig",
    "ViewsStore",
    "ViewsStoreError",
    "context_view",
    "entity_slices",
    "history_contexts",
    "last_time",
    "load_account",
    "materialize_views",
    "profile_view",
    "read_relation",
    "behaviour_views",
    "concept_kinds",
    "kinds_by_concept",
    "read_relations",
    "residue_candidates",
    "slot_neighbourhood",
    "slot_neighbourhood_until",
    "slot_window",
]
