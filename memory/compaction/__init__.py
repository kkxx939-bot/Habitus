"""L2 与 Conversation Summary 共用的字段级语义压缩原语。"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from memory.compaction.field_ops import (
    SemanticFieldMergePolicy,
    SemanticFieldOperation,
    SemanticFieldOperationBatch,
    SemanticFieldOperationError,
    SemanticFieldOperationKind,
    merge_semantic_fields,
)

if TYPE_CHECKING:
    from memory.compaction.commit import MemoryLifecycleCommitter
    from memory.compaction.l2_fields import (
        MemoryFieldCompactionConfig,
        MemoryFieldCompactionError,
        MemoryFieldCompactionResult,
        MemoryFieldCompactor,
    )
    from memory.compaction.lifecycle import (
        MemoryContextUseResult,
        MemoryLifecycleMaintenanceConfig,
        MemoryLifecycleMaintenanceFailure,
        MemoryLifecycleMaintenanceResult,
        MemoryLifecycleManager,
    )
    from memory.compaction.operation import (
        MemoryLifecycleOperation,
        MemoryLifecycleOperationError,
        MemoryLifecycleOperationKind,
        MemoryLifecycleOperationPhase,
        MemoryLifecycleOperationStore,
    )
    from memory.compaction.recovery import (
        MemoryRecoveryError,
        MemoryRecoveryRecord,
        MemoryRecoveryStore,
    )


_LAZY_EXPORTS = {
    "MemoryRecoveryError": ("memory.compaction.recovery", "MemoryRecoveryError"),
    "MemoryRecoveryRecord": ("memory.compaction.recovery", "MemoryRecoveryRecord"),
    "MemoryRecoveryStore": ("memory.compaction.recovery", "MemoryRecoveryStore"),
    "MemoryFieldCompactionConfig": ("memory.compaction.l2_fields", "MemoryFieldCompactionConfig"),
    "MemoryFieldCompactionError": ("memory.compaction.l2_fields", "MemoryFieldCompactionError"),
    "MemoryFieldCompactionResult": ("memory.compaction.l2_fields", "MemoryFieldCompactionResult"),
    "MemoryFieldCompactor": ("memory.compaction.l2_fields", "MemoryFieldCompactor"),
    "MemoryLifecycleCommitter": ("memory.compaction.commit", "MemoryLifecycleCommitter"),
    "MemoryLifecycleOperation": ("memory.compaction.operation", "MemoryLifecycleOperation"),
    "MemoryLifecycleOperationError": (
        "memory.compaction.operation",
        "MemoryLifecycleOperationError",
    ),
    "MemoryLifecycleOperationKind": (
        "memory.compaction.operation",
        "MemoryLifecycleOperationKind",
    ),
    "MemoryLifecycleOperationPhase": (
        "memory.compaction.operation",
        "MemoryLifecycleOperationPhase",
    ),
    "MemoryLifecycleOperationStore": (
        "memory.compaction.operation",
        "MemoryLifecycleOperationStore",
    ),
    "MemoryLifecycleMaintenanceFailure": (
        "memory.compaction.lifecycle",
        "MemoryLifecycleMaintenanceFailure",
    ),
    "MemoryContextUseResult": ("memory.compaction.lifecycle", "MemoryContextUseResult"),
    "MemoryLifecycleMaintenanceConfig": (
        "memory.compaction.lifecycle",
        "MemoryLifecycleMaintenanceConfig",
    ),
    "MemoryLifecycleMaintenanceResult": (
        "memory.compaction.lifecycle",
        "MemoryLifecycleMaintenanceResult",
    ),
    "MemoryLifecycleManager": ("memory.compaction.lifecycle", "MemoryLifecycleManager"),
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
