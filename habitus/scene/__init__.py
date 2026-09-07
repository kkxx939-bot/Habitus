"""Habitus 语义关联层（情景树）的稳定公开入口。

定位：时间预测树之后、只为候选服务的解释层——每封口日一次 LLM 归组产出"事"（情景）文档，
每条行为的上下文由此机械投影。设计定稿见桌面《语义关联层实现方案 v1》；分步落地：

- M1：地址与 URI、文档与 codec、Schema、按天多代存储；``prediction/scene_source`` 读接口。
- M2（本包现状）：归组提示词/schema/装配校验（``grouping``）、定稿日刷新器（``refresh``，由组合根
  排在预测夜批之前）、检查点、读时派生的待用前提清单（``ledger``）。
- M3（本包现状）：上下文视图投影（``views``：context_view / history_contexts）、此刻视图 now_context、
  逐槽三值表 compare。读时计算、零 LLM。
"""

from habitus.scene.document import (
    SceneDocument,
    SceneDocumentCodec,
    SceneDocumentConfig,
    SceneDocumentIntegrityError,
    SceneDocumentLimitError,
    SceneDocumentMetadata,
    SceneStoredLink,
    parse_link_target,
)
from habitus.scene.grouping import LLMSceneGrouper, SceneGrouper, SceneGroupingConfig, SceneGroupingLimitError
from habitus.scene.ledger import PendingItem, pending_before
from habitus.scene.model import SceneAddress, SceneDirectory, SceneLinkType, SceneRole
from habitus.scene.refresh import (
    SceneRefreshBusyError,
    SceneRefreshConfig,
    SceneRefresher,
    SceneRefreshError,
    SceneRefreshReport,
)
from habitus.scene.schema import SceneFieldRole, SceneSchemaError, SceneSchemaRegistry
from habitus.scene.tree import (
    SceneDayGeneration,
    SceneTree,
    SceneTreeConfig,
    SceneTreeConflictError,
    SceneTreeIntegrityError,
)
from habitus.scene.uri import SceneURI, SceneURIError, SceneURINodeType
from habitus.scene.views import (
    ActionRef,
    ComparisonTable,
    ContextView,
    DayIndexCache,
    LastTime,
    Precondition,
    SceneRef,
    SlotComparison,
    Verdict,
    compare,
    context_view,
    history_contexts,
    now_context,
)

__all__ = [
    "ActionRef",
    "ComparisonTable",
    "ContextView",
    "DayIndexCache",
    "LastTime",
    "Precondition",
    "SceneRef",
    "SlotComparison",
    "Verdict",
    "LLMSceneGrouper",
    "PendingItem",
    "SceneAddress",
    "SceneDayGeneration",
    "SceneDirectory",
    "SceneDocument",
    "SceneDocumentCodec",
    "SceneDocumentConfig",
    "SceneDocumentIntegrityError",
    "SceneDocumentLimitError",
    "SceneDocumentMetadata",
    "SceneFieldRole",
    "SceneGrouper",
    "SceneGroupingConfig",
    "SceneGroupingLimitError",
    "SceneLinkType",
    "SceneRefreshBusyError",
    "SceneRefreshConfig",
    "SceneRefreshError",
    "SceneRefreshReport",
    "SceneRefresher",
    "SceneRole",
    "SceneSchemaError",
    "SceneSchemaRegistry",
    "SceneStoredLink",
    "SceneTree",
    "SceneTreeConfig",
    "SceneTreeConflictError",
    "SceneTreeIntegrityError",
    "SceneURI",
    "SceneURIError",
    "SceneURINodeType",
    "parse_link_target",
    "compare",
    "context_view",
    "history_contexts",
    "now_context",
    "pending_before",
]
