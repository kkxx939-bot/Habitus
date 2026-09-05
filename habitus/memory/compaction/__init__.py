"""L2 与 Conversation Summary 共用的字段级语义压缩原语。"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from habitus.memory.compaction.field_ops import (
    SemanticFieldMergePolicy,
    SemanticFieldOperation,
    SemanticFieldOperationBatch,
    SemanticFieldOperationError,
    SemanticFieldOperationKind,
    merge_semantic_fields,
)

if TYPE_CHECKING:
    from habitus.memory.compaction.commit import MemoryLifecycleCommitter
    from habitus.memory.compaction.l2_fields import (
        MemoryFieldCompactionConfig,
        MemoryFieldCompactionError,
        MemoryFieldCompactionResult,
        MemoryFieldCompactor,
    )
    from habitus.memory.compaction.lifecycle import (
        MemoryContextUseResult,
        MemoryLifecycleMaintenanceConfig,
        MemoryLifecycleMaintenanceFailure,
        MemoryLifecycleMaintenanceResult,
        MemoryLifecycleManager,
    )
    from habitus.memory.compaction.operation import (
        MemoryLifecycleOperation,
        MemoryLifecycleOperationError,
        MemoryLifecycleOperationKind,
        MemoryLifecycleOperationPhase,
        MemoryLifecycleOperationStore,
    )
    from habitus.memory.compaction.recovery import (
        MemoryRecoveryError,
        MemoryRecoveryRecord,
        MemoryRecoveryStore,
    )


_LAZY_EXPORTS = {
    "MemoryRecoveryError": ("habitus.memory.compaction.recovery", "MemoryRecoveryError"),
    "MemoryRecoveryRecord": ("habitus.memory.compaction.recovery", "MemoryRecoveryRecord"),
    "MemoryRecoveryStore": ("habitus.memory.compaction.recovery", "MemoryRecoveryStore"),
    "MemoryFieldCompactionConfig": ("habitus.memory.compaction.l2_fields", "MemoryFieldCompactionConfig"),
    "MemoryFieldCompactionError": ("habitus.memory.compaction.l2_fields", "MemoryFieldCompactionError"),
    "MemoryFieldCompactionResult": ("habitus.memory.compaction.l2_fields", "MemoryFieldCompactionResult"),
    "MemoryFieldCompactor": ("habitus.memory.compaction.l2_fields", "MemoryFieldCompactor"),
    "MemoryLifecycleCommitter": ("habitus.memory.compaction.commit", "MemoryLifecycleCommitter"),
    "MemoryLifecycleOperation": ("habitus.memory.compaction.operation", "MemoryLifecycleOperation"),
    "MemoryLifecycleOperationError": (
        "habitus.memory.compaction.operation",
        "MemoryLifecycleOperationError",
    ),
    "MemoryLifecycleOperationKind": (
        "habitus.memory.compaction.operation",
        "MemoryLifecycleOperationKind",
    ),
    "MemoryLifecycleOperationPhase": (
        "habitus.memory.compaction.operation",
        "MemoryLifecycleOperationPhase",
    ),
    "MemoryLifecycleOperationStore": (
        "habitus.memory.compaction.operation",
        "MemoryLifecycleOperationStore",
    ),
    "MemoryLifecycleMaintenanceFailure": (
        "habitus.memory.compaction.lifecycle",
        "MemoryLifecycleMaintenanceFailure",
    ),
    "MemoryContextUseResult": ("habitus.memory.compaction.lifecycle", "MemoryContextUseResult"),
    "MemoryLifecycleMaintenanceConfig": (
        "habitus.memory.compaction.lifecycle",
        "MemoryLifecycleMaintenanceConfig",
    ),
    "MemoryLifecycleMaintenanceResult": (
        "habitus.memory.compaction.lifecycle",
        "MemoryLifecycleMaintenanceResult",
    ),
    "MemoryLifecycleManager": ("habitus.memory.compaction.lifecycle", "MemoryLifecycleManager"),
}


def __getattr__(name: str) -> Any:
    """延迟装载 L2 编排，避免共享字段协议反向依赖 Conversation。"""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "SemanticFieldMergePolicy",
    "SemanticFieldOperation",
    "SemanticFieldOperationBatch",
    "SemanticFieldOperationError",
    "SemanticFieldOperationKind",
    "merge_semantic_fields",
    "MemoryRecoveryError",
    "MemoryRecoveryRecord",
    "MemoryRecoveryStore",
    "MemoryFieldCompactionConfig",
    "MemoryFieldCompactionError",
    "MemoryFieldCompactionResult",
    "MemoryFieldCompactor",
    "MemoryLifecycleCommitter",
    "MemoryLifecycleOperation",
    "MemoryLifecycleOperationError",
    "MemoryLifecycleOperationKind",
    "MemoryLifecycleOperationPhase",
    "MemoryLifecycleOperationStore",
    "MemoryLifecycleMaintenanceFailure",
    "MemoryContextUseResult",
    "MemoryLifecycleMaintenanceConfig",
    "MemoryLifecycleMaintenanceResult",
    "MemoryLifecycleManager",
]
