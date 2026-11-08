"""L2 生命周期耐久操作的续跑入口与收尾。

按 operation 种类把逐相位续跑交给 ``operation_steps`` 里的三组步骤；到 DERIVED_PENDING 后按需刷新派生层，再由
同一 operation 在锁内收尾。这里只负责续跑与收尾，不决定何时创建 operation。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime

from habitus.memory.compaction.commit import MemoryLifecycleCommitter
from habitus.memory.compaction.lifecycle_fences import lifecycle_timestamp
from habitus.memory.compaction.operation import (
    MemoryLifecycleOperation,
    MemoryLifecycleOperationKind,
    MemoryLifecycleOperationPhase,
    MemoryLifecycleOperationStore,
)
from habitus.memory.compaction.operation_steps import (
    MemoryCompactSteps,
    MemoryRestoreSteps,
    MemoryRetireSteps,
)
from habitus.memory.compaction.recovery import MemoryRecoveryStore
from habitus.memory.recall import MemoryRecallLifecycle
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.uri import MemoryURI


class MemoryLifecycleOperationRunner:
    """从耐久相位续跑一条 L2 生命周期 operation，并在派生刷新后由同一 operation 收尾。"""

    def __init__(
        self,
        snapshot_reader: MemorySnapshotReader,
        recall_lifecycle: MemoryRecallLifecycle,
        recovery_store: MemoryRecoveryStore,
        committer: MemoryLifecycleCommitter,
        operation_store: MemoryLifecycleOperationStore,
        *,
        clock: Callable[[], datetime],
        derived_refresh: Callable[[tuple[MemoryURI, ...]], Awaitable[None]] | None = None,
    ) -> None:
        self.snapshot_reader = snapshot_reader
        self.recall_lifecycle = recall_lifecycle
        self.recovery_store = recovery_store
        self.committer = committer
        self.operation_store = operation_store
        self.clock = clock
        self.derived_refresh = derived_refresh
        steps = (snapshot_reader, recall_lifecycle, recovery_store, committer, operation_store)
        self.compact_steps = MemoryCompactSteps(*steps)
        self.restore_steps = MemoryRestoreSteps(*steps)
        self.retire_steps = MemoryRetireSteps(*steps)

    async def resume(
        self,
        operation: MemoryLifecycleOperation,
        *,
        finish_derived: bool,
    ) -> MemoryLifecycleOperationKind | None:
        current_time = lifecycle_timestamp(self.clock())
        current = await asyncio.to_thread(self.operation_store.try_read, operation.uri)
        if current is None or current.operation_id != operation.operation_id:
            return None
        if current.kind is MemoryLifecycleOperationKind.COMPACT:
            current = await self.compact_steps.resume(current, current_time)
        elif current.kind is MemoryLifecycleOperationKind.RESTORE:
            current = await self.restore_steps.resume(current, current_time)
        else:
            current = await self.retire_steps.resume(current, current_time)
            if current is None:
                return None
        if current.phase is not MemoryLifecycleOperationPhase.DERIVED_PENDING:
            return None
        if not finish_derived:
            return current.kind
        if self.derived_refresh is not None:
            await self.derived_refresh((current.uri,))
        completed = await asyncio.to_thread(self._finish_operation, current)
        return current.kind if completed else None

    def _finish_operation(self, operation: MemoryLifecycleOperation) -> bool:
        """只清理由同一耐久 operation 仍拥有的派生与状态，避免清掉并发新修订。"""

        with self.operation_store.acquire(operation.uri, wait_timeout_seconds=5.0):
            current = self.operation_store.try_read(operation.uri)
            if current is None or current.operation_id != operation.operation_id:
                return False
            if current.phase is not MemoryLifecycleOperationPhase.DERIVED_PENDING:
                return False
            if current.kind in {
                MemoryLifecycleOperationKind.RESTORE,
                MemoryLifecycleOperationKind.RETIRE,
            }:
                self.recovery_store.delete(current.uri)
            if current.kind is MemoryLifecycleOperationKind.RETIRE:
                self.recall_lifecycle.forget((current.uri,))
            self.operation_store.complete_owned(current)
            return True


__all__ = ["MemoryLifecycleOperationRunner"]
