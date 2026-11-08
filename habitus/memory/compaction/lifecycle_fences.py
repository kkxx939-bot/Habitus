"""L2 生命周期各协作者共用的围栏判定与热度目标构造。

维护编排、耐久操作续跑和最终 Context 使用确认都要回答同一组问题："当前快照还是 operation 冻结的来源吗？
是计划写出的目标吗？热度状态还对得上这一版文档吗？"这些判定只读入参，不加锁、不落盘。
"""

from __future__ import annotations

from datetime import datetime

from habitus.foundation.timestamps import require_aware_utc
from habitus.memory.compaction.operation import MemoryLifecycleOperation
from habitus.memory.document import MemoryDocument
from habitus.memory.recall import (
    MemoryRecallCandidate,
    MemoryRecallLifecycle,
    MemoryRecallState,
    MemoryRecallTarget,
)
from habitus.memory.snapshot import MemorySnapshot
from habitus.memory.uri import MemoryURI


def lifecycle_timestamp(value: datetime) -> datetime:
    return require_aware_utc(value, "memory lifecycle timestamp must include a timezone")


def lifecycle_state(recall_lifecycle: MemoryRecallLifecycle, uri: MemoryURI) -> MemoryRecallState | None:
    if not recall_lifecycle.config.enabled:
        return None
    states = recall_lifecycle.store.read_many((uri,))
    return states[0] if states else None


def recall_target(document: MemoryDocument) -> MemoryRecallTarget:
    return MemoryRecallTarget(
        MemoryURI.from_address(document.address),
        document.metadata.revision,
        document.metadata.created_at,
    )


def snapshot_recall_target(snapshot: MemorySnapshot) -> MemoryRecallTarget:
    if not snapshot.exists or snapshot.value is None:
        raise ValueError("memory lifecycle target snapshot must exist")
    return recall_target(snapshot.value)


def recall_candidate(document: MemoryDocument) -> MemoryRecallCandidate:
    return MemoryRecallCandidate(
        recall_target(document),
        document.kind,
        document.metadata.updated_at,
        1.0,
    )


def same_snapshot(left: MemorySnapshot, right: MemorySnapshot) -> bool:
    return left.state is right.state and left.revision == right.revision and left.source_digest == right.source_digest


def matches_target(snapshot: MemorySnapshot, target: MemoryRecallTarget) -> bool:
    return (
        snapshot.exists
        and isinstance(snapshot.value, MemoryDocument)
        and snapshot.identity == str(target.uri)
        and snapshot.value.metadata.revision == target.document_revision
        and snapshot.value.metadata.created_at == target.document_created_at
    )


def matches_operation_source(
    snapshot: MemorySnapshot,
    operation: MemoryLifecycleOperation,
) -> bool:
    return (
        snapshot.exists
        and isinstance(snapshot.value, MemoryDocument)
        and snapshot.identity == str(operation.uri)
        and snapshot.revision == operation.source_revision
        and snapshot.source_digest == operation.source_digest
        and snapshot.value.metadata.created_at == operation.source_created_at
    )


def matches_operation_target(
    snapshot: MemorySnapshot,
    operation: MemoryLifecycleOperation,
) -> bool:
    return (
        operation.target_revision is not None
        and snapshot.exists
        and isinstance(snapshot.value, MemoryDocument)
        and snapshot.identity == str(operation.uri)
        and snapshot.revision == operation.target_revision
        and snapshot.source_digest == operation.target_digest
        and snapshot.value.metadata.created_at == operation.source_created_at
    )


def matches_planned_target(
    snapshot: MemorySnapshot,
    operation: MemoryLifecycleOperation,
) -> bool:
    return (
        operation.planned_fields is not None
        and snapshot.exists
        and isinstance(snapshot.value, MemoryDocument)
        and snapshot.identity == str(operation.uri)
        and snapshot.revision == operation.source_revision + 1
        and snapshot.value.metadata.created_at == operation.source_created_at
        and snapshot.value.fields == operation.planned_fields
    )


def compaction_state_matches(
    state: MemoryRecallState | None,
    snapshot: MemorySnapshot,
    expected_version: int,
) -> bool:
    return (
        state is not None
        and state.version == expected_version
        and state.document_revision == snapshot.revision
        and isinstance(snapshot.value, MemoryDocument)
        and state.document_created_at == snapshot.value.metadata.created_at
        and state.compacted_at is not None
        and state.retired_at is None
    )


__all__ = [
    "compaction_state_matches",
    "lifecycle_state",
    "lifecycle_timestamp",
    "matches_operation_source",
    "matches_operation_target",
    "matches_planned_target",
    "matches_target",
    "recall_candidate",
    "recall_target",
    "same_snapshot",
    "snapshot_recall_target",
]
