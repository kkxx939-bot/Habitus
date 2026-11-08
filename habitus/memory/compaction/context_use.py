"""最终模型可见 Context 的 L2 使用确认与 COLD_2 同步恢复。

只有真正进入最终 Context 的 L2 才计一次成功使用；已压缩（COLD_2）的节点必须先经耐久 RESTORE operation 恢复
完整字段，再把恢复后的文档交给调用方。另提供只为 Context 预算临时展开活动基线的只读入口。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from habitus.memory.compaction.lifecycle_fences import (
    lifecycle_state,
    lifecycle_timestamp,
    matches_operation_source,
    matches_target,
)
from habitus.memory.compaction.operation import (
    MemoryLifecycleOperation,
    MemoryLifecycleOperationKind,
    MemoryLifecycleOperationPhase,
    MemoryLifecycleOperationStore,
)
from habitus.memory.compaction.operation_runner import MemoryLifecycleOperationRunner
from habitus.memory.compaction.recovery import MemoryRecoveryStore
from habitus.memory.compaction.scan import MemoryLifecycleScanStore
from habitus.memory.document import MemoryDocument
from habitus.memory.recall import MemoryRecallLifecycle, MemoryRecallState, MemoryRecallTarget
from habitus.memory.snapshot import MemorySnapshot, MemorySnapshotReader
from habitus.memory.uri import MemoryURI


@dataclass(frozen=True)
class MemoryContextUseResult:
    documents: tuple[MemoryDocument, ...]
    rejected_uris: tuple[MemoryURI, ...]


class MemoryContextUseAcknowledger:
    """确认最终 Context 使用；COLD_2 目标交给耐久 RESTORE operation 同步恢复。"""

    def __init__(
        self,
        snapshot_reader: MemorySnapshotReader,
        recall_lifecycle: MemoryRecallLifecycle,
        recovery_store: MemoryRecoveryStore,
        operation_store: MemoryLifecycleOperationStore,
        scan_store: MemoryLifecycleScanStore,
        operation_runner: MemoryLifecycleOperationRunner,
        *,
        clock: Callable[[], datetime],
    ) -> None:
        self.snapshot_reader = snapshot_reader
        self.recall_lifecycle = recall_lifecycle
        self.recovery_store = recovery_store
        self.operation_store = operation_store
        self.scan_store = scan_store
        self.operation_runner = operation_runner
        self.clock = clock

    async def record_context_use(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        used_at: datetime | None = None,
    ) -> MemoryContextUseResult:
        """最终模型可见 Context 的 L2 才计成功；COLD_2 必须先同步恢复。"""

        if len({item.uri for item in targets}) != len(targets):
            raise ValueError("context use targets must be unique")
        timestamp = lifecycle_timestamp(self.clock() if used_at is None else used_at)
        documents: list[MemoryDocument] = []
        rejected: list[MemoryURI] = []
        for target in targets:
            try:
                documents.append(await self._acknowledge(target, timestamp))
            except asyncio.CancelledError:
                raise
            except Exception:
                rejected.append(target.uri)
        return MemoryContextUseResult(tuple(documents), tuple(rejected))

    async def _acknowledge(self, target: MemoryRecallTarget, timestamp: datetime) -> MemoryDocument:
        snapshot, state, detailed = await asyncio.to_thread(
            self._prepare_context_use,
            target,
            timestamp,
        )
        if detailed is None:
            assert isinstance(snapshot.value, MemoryDocument)
            return snapshot.value
        assert state is not None
        operation = await asyncio.to_thread(
            self.operation_store.prepare,
            MemoryLifecycleOperationKind.RESTORE,
            snapshot,
            planned_fields=detailed.fields,
            expected_state_version=state.version,
            prepared_at=timestamp,
        )
        await self.operation_runner.resume(operation, finish_derived=False)
        current = await asyncio.to_thread(self.snapshot_reader.read, target.uri)
        if not current.exists or current.value is None:
            raise RuntimeError("restored final Context memory disappeared")
        return current.value

    def _prepare_context_use(
        self,
        target: MemoryRecallTarget,
        timestamp: datetime,
    ) -> tuple[MemorySnapshot, MemoryRecallState | None, MemoryDocument | None]:
        """在 operation URI 锁内提交普通 use，或冻结待恢复的精确来源。"""

        with self.operation_store.acquire(target.uri, wait_timeout_seconds=5.0):
            pending = self.operation_store.try_read(target.uri)
            snapshot = self.snapshot_reader.read(target.uri)
            if not matches_target(snapshot, target):
                raise RuntimeError("final Context target changed before use acknowledgement")
            if (
                pending is not None
                and pending.kind is MemoryLifecycleOperationKind.RETIRE
                and not matches_operation_source(snapshot, pending)
            ):
                self._discard_stale_retirement(target, pending)
                pending = None
            allowed_pending_restore = (
                pending is not None
                and pending.kind is MemoryLifecycleOperationKind.RESTORE
                and pending.phase is MemoryLifecycleOperationPhase.DERIVED_PENDING
            )
            if pending is not None and not allowed_pending_restore:
                raise RuntimeError("memory has an unfinished lifecycle operation")
            state = self._state_for(target.uri)
            same_state_document = (
                state is not None
                and state.document_revision == target.document_revision
                and state.document_created_at == target.document_created_at
            )
            if state is not None and same_state_document and state.retired_at is not None:
                raise RuntimeError("final Context target entered retirement")
            if state is not None and same_state_document and state.compacted_at is not None:
                if self.recovery_store.for_compacted(snapshot) is None:
                    raise RuntimeError("COLD_2 recovery baseline is missing or stale")
                detailed = self.recovery_store.restore(
                    target.uri,
                    created_at=target.document_created_at,
                    compacted_snapshot=snapshot,
                )
                return snapshot, state, detailed
            self._record_use(target, timestamp)
            if not allowed_pending_restore:
                self.recovery_store.delete(target.uri)
            return snapshot, state, None

    def _discard_stale_retirement(
        self,
        target: MemoryRecallTarget,
        pending: MemoryLifecycleOperation,
    ) -> None:
        """同址已写入新文档时，旧 RETIRE operation 只清理它自己的旧状态与恢复基线（调用方已持锁）。"""

        stale_state = self._state_for(target.uri)
        if (
            stale_state is not None
            and stale_state.document_revision == pending.source_revision
            and stale_state.document_created_at == pending.source_created_at
        ):
            self.recall_lifecycle.forget((target.uri,))
        self.recovery_store.delete(target.uri)
        self.operation_store.complete_owned(pending)
        self.scan_store.clear_failure(target.uri)

    def _record_use(self, target: MemoryRecallTarget, timestamp: datetime) -> None:
        states = self.recall_lifecycle.record_use((target,), used_at=timestamp)
        if self.recall_lifecycle.config.enabled and (
            not states
            or states[0].document_revision != target.document_revision
            or states[0].document_created_at != target.document_created_at
        ):
            raise RuntimeError("final Context use acknowledgement lost its revision fence")

    def expand_for_probe(self, document: MemoryDocument) -> MemoryDocument:
        """只为最终 Context 预算临时展开活动基线，不修改 L2 或热度。"""

        uri = MemoryURI.from_address(document.address)
        snapshot = self.snapshot_reader.read(uri)
        if not snapshot.exists or snapshot.value != document:
            return document
        state = self._state_for(uri)
        if (
            state is None
            or state.document_created_at != document.metadata.created_at
            or state.document_revision != document.metadata.revision
            or state.compacted_at is None
            or state.retired_at is not None
            or self.recovery_store.for_compacted(snapshot) is None
        ):
            return document
        return self.recovery_store.restore(
            uri,
            created_at=document.metadata.created_at,
            compacted_snapshot=snapshot,
        )

    def _state_for(self, uri: MemoryURI) -> MemoryRecallState | None:
        return lifecycle_state(self.recall_lifecycle, uri)


__all__ = ["MemoryContextUseAcknowledger", "MemoryContextUseResult"]
