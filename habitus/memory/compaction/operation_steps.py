"""L2 生命周期三类耐久 operation 的逐相位续跑步骤。

COMPACT、RESTORE、RETIRE 都按 PREPARED → L2_COMMITTED → STATE_COMMITTED → DERIVED_PENDING 推进；每个相位先核对
当前快照与热度状态仍属于这条 operation，再落盘并推进相位，所以进程在任意两步之间中断后，下一次续跑都从耐久
相位接上。收尾（派生刷新后完成 operation）由 ``MemoryLifecycleOperationRunner`` 负责。
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from habitus.memory.compaction.commit import MemoryLifecycleCommitter
from habitus.memory.compaction.lifecycle_fences import (
    compaction_state_matches,
    lifecycle_state,
    matches_operation_source,
    matches_operation_target,
    matches_planned_target,
    snapshot_recall_target,
)
from habitus.memory.compaction.operation import (
    MemoryLifecycleOperation,
    MemoryLifecycleOperationPhase,
    MemoryLifecycleOperationStore,
)
from habitus.memory.compaction.recovery import MemoryRecoveryStore
from habitus.memory.recall import MemoryRecallLifecycle, MemoryRecallState
from habitus.memory.snapshot import MemorySnapshot, MemorySnapshotReader
from habitus.memory.uri import MemoryURI


class _OperationSteps:
    def __init__(
        self,
        snapshot_reader: MemorySnapshotReader,
        recall_lifecycle: MemoryRecallLifecycle,
        recovery_store: MemoryRecoveryStore,
        committer: MemoryLifecycleCommitter,
        operation_store: MemoryLifecycleOperationStore,
    ) -> None:
        self.snapshot_reader = snapshot_reader
        self.recall_lifecycle = recall_lifecycle
        self.recovery_store = recovery_store
        self.committer = committer
        self.operation_store = operation_store

    def _state_for(self, uri: MemoryURI) -> MemoryRecallState | None:
        return lifecycle_state(self.recall_lifecycle, uri)


class MemoryCompactSteps(_OperationSteps):
    """COMPACT：CAS 写入压缩字段、激活恢复基线、发布压缩状态。"""

    async def resume(
        self,
        operation: MemoryLifecycleOperation,
        now: datetime,
    ) -> MemoryLifecycleOperation:
        current = operation
        if current.phase is MemoryLifecycleOperationPhase.PREPARED:
            current = await self._compact_prepared(current, now)
        if current.phase is MemoryLifecycleOperationPhase.L2_COMMITTED:
            current = await self._compact_l2_committed(current, now)
        if current.phase is MemoryLifecycleOperationPhase.STATE_COMMITTED:
            current = await asyncio.to_thread(
                self.operation_store.advance,
                current,
                MemoryLifecycleOperationPhase.DERIVED_PENDING,
                updated_at=now,
            )
        return current

    async def _compact_prepared(
        self,
        current: MemoryLifecycleOperation,
        now: datetime,
    ) -> MemoryLifecycleOperation:
        """PREPARED：CAS 写入计划字段并激活恢复基线；来源已被并发改写时完成并原样返回（仍是 PREPARED）。"""

        snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
        state = await asyncio.to_thread(self._state_for, current.uri)
        state_version = 0 if state is None else state.version
        if matches_operation_source(snapshot, current):
            if state_version != current.expected_state_version:
                await asyncio.to_thread(self.recovery_store.delete, current.uri)
                await asyncio.to_thread(self.operation_store.complete, current)
                return current
            assert current.planned_fields is not None
            await asyncio.to_thread(self.committer.replace_fields, snapshot, current.planned_fields)
            snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
        elif not matches_operation_target(snapshot, current):
            if not matches_planned_target(snapshot, current):
                await asyncio.to_thread(self.recovery_store.delete, current.uri)
                await asyncio.to_thread(self.operation_store.complete, current)
                return current
        if not matches_planned_target(snapshot, current):
            raise RuntimeError("compaction target does not match its durable plan")
        record = await asyncio.to_thread(
            self.recovery_store.latest,
            current.uri,
            created_at=current.source_created_at,
        )
        if record is None:
            raise RuntimeError("compaction recovery baseline is missing")
        await asyncio.to_thread(self.recovery_store.activate, record, snapshot)
        return await asyncio.to_thread(
            self.operation_store.advance,
            current,
            MemoryLifecycleOperationPhase.L2_COMMITTED,
            updated_at=now,
            target_revision=snapshot.revision,
            target_digest=snapshot.source_digest,
        )

    async def _compact_l2_committed(
        self,
        current: MemoryLifecycleOperation,
        now: datetime,
    ) -> MemoryLifecycleOperation:
        """L2_COMMITTED：补齐恢复基线，再把压缩状态发布到热度表。"""

        snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
        if not matches_operation_target(snapshot, current):
            raise RuntimeError("compacted L2 changed before lifecycle state publication")
        if await asyncio.to_thread(self.recovery_store.for_compacted, snapshot) is None:
            record = await asyncio.to_thread(
                self.recovery_store.latest,
                current.uri,
                created_at=current.source_created_at,
            )
            if record is None:
                raise RuntimeError("compaction recovery baseline is missing")
            await asyncio.to_thread(self.recovery_store.activate, record, snapshot)
        state = await asyncio.to_thread(self._state_for, current.uri)
        if not compaction_state_matches(state, snapshot, current.expected_state_version + 1):
            await self._publish_compaction_state(current, snapshot, state)
        return await asyncio.to_thread(
            self.operation_store.advance,
            current,
            MemoryLifecycleOperationPhase.STATE_COMMITTED,
            updated_at=now,
        )

    async def _publish_compaction_state(
        self,
        current: MemoryLifecycleOperation,
        snapshot: MemorySnapshot,
        state: MemoryRecallState | None,
    ) -> None:
        if (0 if state is None else state.version) != current.expected_state_version:
            raise RuntimeError("lifecycle state changed during compaction publication")
        detailed = await asyncio.to_thread(
            self.recovery_store.restore,
            current.uri,
            created_at=current.source_created_at,
            compacted_snapshot=snapshot,
        )
        activity = max(
            value
            for value in (
                detailed.metadata.updated_at,
                None if state is None else state.last_useful_recall_at,
                None if state is None else state.lifecycle_activity_at,
            )
            if value is not None
        )
        await asyncio.to_thread(
            self.recall_lifecycle.mark_compacted,
            snapshot_recall_target(snapshot),
            lifecycle_activity_at=activity,
            compacted_at=current.prepared_at,
            expected_version=current.expected_state_version,
        )


class MemoryRestoreSteps(_OperationSteps):
    """RESTORE：CAS 写回完整字段，再补记那次最终 Context 使用。"""

    async def resume(
        self,
        operation: MemoryLifecycleOperation,
        now: datetime,
    ) -> MemoryLifecycleOperation:
        current = operation
        if current.phase is MemoryLifecycleOperationPhase.PREPARED:
            current = await self._restore_prepared(current, now)
        if current.phase is MemoryLifecycleOperationPhase.L2_COMMITTED:
            current = await self._restore_l2_committed(current, now)
        if current.phase is MemoryLifecycleOperationPhase.STATE_COMMITTED:
            current = await asyncio.to_thread(
                self.operation_store.advance,
                current,
                MemoryLifecycleOperationPhase.DERIVED_PENDING,
                updated_at=now,
            )
        return current

    async def _restore_prepared(
        self,
        current: MemoryLifecycleOperation,
        now: datetime,
    ) -> MemoryLifecycleOperation:
        snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
        state = await asyncio.to_thread(self._state_for, current.uri)
        if not matches_operation_source(snapshot, current):
            if not matches_planned_target(snapshot, current):
                raise RuntimeError("restore source changed before L2 CAS")
        else:
            if state is None or state.version != current.expected_state_version:
                raise RuntimeError("lifecycle state changed before COLD_2 restore")
            assert current.planned_fields is not None
            await asyncio.to_thread(self.committer.replace_fields, snapshot, current.planned_fields)
            snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
        return await asyncio.to_thread(
            self.operation_store.advance,
            current,
            MemoryLifecycleOperationPhase.L2_COMMITTED,
            updated_at=now,
            target_revision=snapshot.revision,
            target_digest=snapshot.source_digest,
        )

    async def _restore_l2_committed(
        self,
        current: MemoryLifecycleOperation,
        now: datetime,
    ) -> MemoryLifecycleOperation:
        snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
        if not matches_operation_target(snapshot, current):
            raise RuntimeError("restored L2 changed before use acknowledgement")
        target = snapshot_recall_target(snapshot)
        state = await asyncio.to_thread(self._state_for, current.uri)
        already_recorded = (
            state is not None
            and state.document_revision == target.document_revision
            and state.document_created_at == target.document_created_at
            and state.last_useful_recall_at is not None
            and state.last_useful_recall_at >= current.prepared_at
            and state.compacted_at is None
        )
        if not already_recorded:
            states = await asyncio.to_thread(
                self.recall_lifecycle.record_use,
                (target,),
                used_at=current.prepared_at,
            )
            if self.recall_lifecycle.config.enabled and (
                not states or states[0].document_revision != target.document_revision
            ):
                raise RuntimeError("restored L2 use acknowledgement lost its revision fence")
        return await asyncio.to_thread(
            self.operation_store.advance,
            current,
            MemoryLifecycleOperationPhase.STATE_COMMITTED,
            updated_at=now,
        )


class MemoryRetireSteps(_OperationSteps):
    """RETIRE：在 URI operation 锁内删除 L2 并逐相位推进。"""

    async def resume(
        self,
        operation: MemoryLifecycleOperation,
        now: datetime,
    ) -> MemoryLifecycleOperation | None:
        current = operation
        if current.phase is MemoryLifecycleOperationPhase.PREPARED:
            resumed = await asyncio.to_thread(self._resume_retire_prepared, current, now)
            if resumed is None:
                return None
            current = resumed
        if current.phase is MemoryLifecycleOperationPhase.L2_COMMITTED:
            resumed = await asyncio.to_thread(
                self._advance_retire_if_current,
                current,
                MemoryLifecycleOperationPhase.STATE_COMMITTED,
                now,
            )
            if resumed is None:
                return None
            current = resumed
        if current.phase is MemoryLifecycleOperationPhase.STATE_COMMITTED:
            resumed = await asyncio.to_thread(
                self._advance_retire_if_current,
                current,
                MemoryLifecycleOperationPhase.DERIVED_PENDING,
                now,
            )
            if resumed is None:
                return None
            current = resumed
        return current

    def _resume_retire_prepared(
        self,
        operation: MemoryLifecycleOperation,
        now: datetime,
    ) -> MemoryLifecycleOperation | None:
        """在同一 URI operation lock 内完成 RETIRE 的读取、校验、CAS 与推进。"""

        with self.operation_store.acquire(operation.uri, wait_timeout_seconds=5.0):
            current = self.operation_store.try_read(operation.uri)
            if current is None or current.operation_id != operation.operation_id:
                return None
            if current.phase is not MemoryLifecycleOperationPhase.PREPARED:
                return current
            snapshot = self.snapshot_reader.read(current.uri)
            if snapshot.exists:
                if not matches_operation_source(snapshot, current):
                    state = self._state_for(current.uri)
                    if (
                        state is not None
                        and state.document_revision == current.source_revision
                        and state.document_created_at == current.source_created_at
                    ):
                        self.recall_lifecycle.forget((current.uri,))
                    self.recovery_store.delete(current.uri)
                    self.operation_store.complete_owned(current)
                    return None
                state = self._state_for(current.uri)
                if state is None or state.retired_at is None:
                    self.recovery_store.delete(current.uri)
                    self.operation_store.complete_owned(current)
                    return None
                self.committer.delete(snapshot)
            return self.operation_store.advance_owned(
                current,
                MemoryLifecycleOperationPhase.L2_COMMITTED,
                updated_at=now,
            )

    def _advance_retire_if_current(
        self,
        operation: MemoryLifecycleOperation,
        phase: MemoryLifecycleOperationPhase,
        now: datetime,
    ) -> MemoryLifecycleOperation | None:
        """串行化 RETIRE 阶段推进；旧 operation 被新 generation 清除时安静结束。"""

        with self.operation_store.acquire(operation.uri, wait_timeout_seconds=5.0):
            current = self.operation_store.try_read(operation.uri)
            if current is None or current.operation_id != operation.operation_id:
                return None
            return self.operation_store.advance_owned(current, phase, updated_at=now)


__all__ = ["MemoryCompactSteps", "MemoryRestoreSteps", "MemoryRetireSteps"]
