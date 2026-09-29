"""Habitus 语义关联层的稳定公开入口。

旧的语义树（按候选累积的规律级：关联记录、L1/L0、待关联清单、前提投影）已于 2026-09-26 整块删掉，
按《语义树重构》方案重建：底层四支 ``concepts / hypotheses / occurrences / ledger`` 按功能分，
``views`` 只做投影。各支从自己的子包引用（``habitus.scene.concepts``、``habitus.scene.occurrences``……），
包根只放新旧设计都要用的三样：

- ``views``  行为树的读侧投影（紧邻的上下条、起因、上一次、观测空白、日型……）与槽位邻域序列，
  读时计算、零 LLM、只读行为树。
- ``calendar``  日型接缝（读时算、不冻结）。
- ``facts``  外部条件的事实门（一刻一组受控键值），给预测层的承诺用。
"""

from habitus.scene.calendar import DayTypeCalendar, NominalCalendar
from habitus.scene.facts import (
    FACT_KINDS,
    CompositeFacts,
    Conditions,
    FactKey,
    FactProvider,
    NoFacts,
    conditions_of,
    require_local,
)
from habitus.scene.views import (
    ActionRef,
    ContextView,
    DayIndex,
    DayIndexCache,
    FlowRow,
    LastTime,
    Neighbour,
    ObservationGap,
    context_view,
    history_contexts,
    slot_neighbourhood,
    slot_neighbourhood_until,
)

__all__ = [
    "FACT_KINDS",
    "ActionRef",
    "CompositeFacts",
    "Conditions",
    "ContextView",
    "DayIndex",
    "DayIndexCache",
    "DayTypeCalendar",
    "FactKey",
    "FactProvider",
    "FlowRow",
    "LastTime",
    "Neighbour",
    "NoFacts",
    "NominalCalendar",
    "ObservationGap",
    "conditions_of",
    "context_view",
    "history_contexts",
    "require_local",
    "slot_neighbourhood",
    "slot_neighbourhood_until",
]
