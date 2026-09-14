"""Habitus 语义关联层的稳定公开入口。

**职责只有一条**：把时间预测树算出来的候选行为的上下文关联上。任何带判断性质的东西——该不该做、
变没变、像不像、到期没到期、重不重复——全部属预测层：它手里有四层数字和历次的上下文，自己比得出来。

形状是**按候选累积的规律级**（``regularity``）：一次发生一条记录（指回那条行为、一句"当时是什么
情况"、归属的情境、两类边、两件前提事实），按候选上卷成 L1（有哪几种情形、各覆盖哪些日期）与
L0（一句话）。范围由预测树的出处日决定，不设日历窗口——周频行为的相邻两次正好隔 7 天，按日历
回看的窗口永远卡在边界上。

各模块：

- ``backlog``  待关联清单（树上的出处日 − 已关联完成 − 被挡住的），以及前因排序要的事实。
  **它是本包唯一读预测树的模块**，由架构测试钉死。
- ``association``  关联本身。面向模型的五个文件（形状、schema、提示词、装配校验、服务）只认编号，
  不碰 URI、不落盘；编排那半（输入装配、物化、进度、刷新器、前提投影）读树、写树。
- ``regularity``  产物与存储：记录、L1/L0、受控前向边、完成标记。
- ``views``  读时投影的行为侧上下文（此前几步、紧邻的上下条、观测空白、日型……）。
  **规律级那一半还没有接上**——按候选取记录是读侧改写要做的事。
- ``calendar`` / ``model`` / ``uri`` / ``text``  日型接缝、地址与受控枚举、URI、文本清洗。

按天归组与整棵日情景树已经在 2026-09-13/14 删掉：归组产出的"事"是个冗余的中间容器，而项目
没有上线、没有消费者，所以整块删掉而不是并行保留。
"""

from habitus.scene.association import (
    ASSOCIATION_VERSION,
    AssociationAssembly,
    AssociationConfig,
    AssociationDraft,
    AssociationInput,
    AssociationLimitError,
    AssociationRefreshConfig,
    AssociationRefresher,
    AssociationRefreshReport,
    Associator,
    LLMAssociator,
)
from habitus.scene.backlog import (
    AssociatedDays,
    AssociationTask,
    BlockedDays,
    CauseFact,
    CauseFacts,
    backlog,
)
from habitus.scene.calendar import DayTypeCalendar, NominalCalendar
from habitus.scene.model import AssociationAddress, KindDirectory, RegularityLevel, SceneLinkType
from habitus.scene.regularity import AssociationDocument, RegularityTree, RegularityTreeError
from habitus.scene.regularity.link import SceneStoredLink, parse_link_target
from habitus.scene.uri import SceneURI, SceneURIError, SceneURINodeType
from habitus.scene.views import (
    ActionRef,
    ContextView,
    DayIndex,
    DayIndexCache,
    LastTime,
    Neighbour,
    ObservationGap,
    context_view,
    history_contexts,
)

__all__ = [
    "ASSOCIATION_VERSION",
    "ActionRef",
    "AssociatedDays",
    "AssociationAddress",
    "AssociationAssembly",
    "AssociationConfig",
    "AssociationDocument",
    "AssociationDraft",
    "AssociationInput",
    "AssociationLimitError",
    "AssociationRefreshConfig",
    "AssociationRefreshReport",
    "AssociationRefresher",
    "AssociationTask",
    "Associator",
    "BlockedDays",
    "CauseFact",
    "CauseFacts",
    "ContextView",
    "DayIndex",
    "DayIndexCache",
    "DayTypeCalendar",
    "KindDirectory",
    "LLMAssociator",
    "LastTime",
    "Neighbour",
    "NominalCalendar",
    "ObservationGap",
    "RegularityLevel",
    "RegularityTree",
    "RegularityTreeError",
    "SceneLinkType",
    "SceneStoredLink",
    "SceneURI",
    "SceneURIError",
    "SceneURINodeType",
    "backlog",
    "context_view",
    "history_contexts",
    "parse_link_target",
]
