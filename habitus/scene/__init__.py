"""Habitus 语义关联层的稳定公开入口。

语义树按新方案（``13-语义树新方案``，裁定 22）：``concepts``（概念与词表同步）、``occurrences``（映射与命中）、
``relations``（关系的检验、状态与存储）、``advice``（先验与候选调节条件两个模型触点）。各支从自己的子包引用，
包根只放预测层也要用的 ``views``：行为树的读侧投影（紧邻的上下条、起因、上一次、观测空白……）与槽位邻域序列，
读时计算、零 LLM、只读行为树。日历与外部条件在 ``habitus.conditions``（裁定 59）。
"""

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
    "slot_neighbourhood",
    "slot_neighbourhood_until",
]
