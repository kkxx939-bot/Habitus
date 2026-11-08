"""协调 COLD_2 压缩、最终上下文恢复和终态退休。

退休与压缩互不依赖：COLD_2 且最近一次活动超过该类退休期即标候选，宽限期满删除，不要求先压缩过；
压缩尽力而为，模型答 KEEP 只在扫描账里记"这一版内容判过"，同一版内容不再送模型。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from habitus.memory.compaction.commit import MemoryLifecycleCommitter
from habitus.memory.compaction.context_use import MemoryContextUseAcknowledger, MemoryContextUseResult
from habitus.memory.compaction.l2_fields import MemoryFieldCompactor
from habitus.memory.compaction.lifecycle_fences import (
    lifecycle_timestamp,
    recall_candidate,
    recall_target,
    same_snapshot,
    snapshot_recall_target,
)
from habitus.memory.compaction.maintenance_plan import (
    MemoryMaintenanceAction,
    MemoryMaintenanceQuota,
    plan_maintenance_action,
)
from habitus.memory.compaction.operation import (
    MemoryLifecycleOperationKind,
    MemoryLifecycleOperationStore,
)
from habitus.memory.compaction.operation_runner import MemoryLifecycleOperationRunner
from habitus.memory.compaction.recovery import MemoryRecoveryStore
from habitus.memory.compaction.scan import (
    MemoryCompactionKeep,
    MemoryLifecycleScanFailure,
    MemoryLifecycleScanStore,
)
from habitus.memory.document import MemoryDocument
from habitus.memory.model import MemoryAddress
from habitus.memory.recall import (
    MemoryRecallLifecycle,
    MemoryRecallRanking,
    MemoryRecallTarget,
    MemoryTemperature,
)
from habitus.memory.snapshot import MemorySnapshot, MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI


@dataclass(frozen=True)
class MemoryLifecycleMaintenanceConfig:
    max_scan_items: int = 256
    max_compactions_per_cycle: int = 4
    max_retirements_per_cycle: int = 4

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_scan_items", self.max_scan_items, 100_000),
            ("max_compactions_per_cycle", self.max_compactions_per_cycle, 1_000),
            ("max_retirements_per_cycle", self.max_retirements_per_cycle, 1_000),
        ):
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} is outside its supported range")


@dataclass(frozen=True)
class MemoryLifecycleMaintenanceFailure:
    uri: MemoryURI
    error_type: str
    message: str
    attempts: int
    retry_at: datetime

    @classmethod
    def from_scan(cls, value: MemoryLifecycleScanFailure) -> MemoryLifecycleMaintenanceFailure:
        return cls(value.uri, value.error_type, value.message, value.attempts, value.retry_at)


@dataclass(frozen=True)
class MemoryLifecycleMaintenanceResult:
    scanned: int
    compacted: tuple[MemoryURI, ...]
    retired: tuple[MemoryURI, ...]
    failures: tuple[MemoryLifecycleMaintenanceFailure, ...] = ()


class MemoryLifecycleManager:
    """L2 生命周期编排器；所有业务修改继续经过 Memory Editor CAS。"""

    def __init__(
        self,
        tree: MemoryTree,
        snapshot_reader: MemorySnapshotReader,
        recall_lifecycle: MemoryRecallLifecycle,
        field_compactor: MemoryFieldCompactor,
        recovery_store: MemoryRecoveryStore,
        committer: MemoryLifecycleCommitter,
        operation_store: MemoryLifecycleOperationStore,
        scan_store: MemoryLifecycleScanStore,
        *,
        config: MemoryLifecycleMaintenanceConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        derived_refresh: Callable[[tuple[MemoryURI, ...]], Awaitable[None]] | None = None,
    ) -> None:
        if snapshot_reader.tree is not tree:
            raise ValueError("lifecycle manager must share one tree and snapshot reader")
        if recovery_store.tree is not tree:
            raise ValueError("lifecycle manager and recovery store must share one tree")
        if committer.snapshot_reader is not snapshot_reader:
            raise ValueError("lifecycle manager and committer must share one snapshot reader")
        if operation_store.tree is not tree:
            raise ValueError("lifecycle manager and operation store must share one tree")
        self.tree = tree
        self.snapshot_reader = snapshot_reader
        self.recall_lifecycle = recall_lifecycle
        self.field_compactor = field_compactor
        self.recovery_store = recovery_store
        self.committer = committer
        self.operation_store = operation_store
        self.scan_store = scan_store
        self.config = config or MemoryLifecycleMaintenanceConfig()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.operation_runner = MemoryLifecycleOperationRunner(
            snapshot_reader,
            recall_lifecycle,
            recovery_store,
            committer,
            operation_store,
            clock=self.clock,
            derived_refresh=derived_refresh,
        )
        self.context_use = MemoryContextUseAcknowledger(
            snapshot_reader,
            recall_lifecycle,
            recovery_store,
            operation_store,
            scan_store,
            self.operation_runner,
            clock=self.clock,
        )

    def initialize(self) -> None:
        self.recall_lifecycle.initialize()
        self.recovery_store.initialize()
        self.operation_store.initialize()
        self.scan_store.initialize()

    async def maintain(self) -> MemoryLifecycleMaintenanceResult:
        """先续跑耐久操作，再有界轮转并隔离每个 L2 节点的失败。

        每轮压缩名额按送模型的次数算：压成、KEEP、报错都占一个。
        """

        now = self._timestamp()
        self.initialize()
        tally = _MaintenanceTally()
        await self._resume_pending(tally, now=now)
        addresses = await self._scan_batch()
        for address in addresses:
            await self._maintain_address(address, tally, now=now)
        return MemoryLifecycleMaintenanceResult(
            len(addresses),
            tuple(dict.fromkeys(tally.compacted)),
            tuple(dict.fromkeys(tally.retired)),
            tuple(tally.failures),
        )

    async def _resume_pending(self, tally: _MaintenanceTally, *, now: datetime) -> None:
        pending = await asyncio.to_thread(self.operation_store.pending)
        for operation in pending:
            try:
                if not await asyncio.to_thread(
                    self.scan_store.eligible,
                    operation.uri,
                    now=now,
                ):
                    continue
                outcome = await self.operation_runner.resume(operation, finish_derived=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_failure(tally, operation.uri, exc, now=now)
            else:
                await asyncio.to_thread(self.scan_store.clear_failure, operation.uri)
                if outcome is MemoryLifecycleOperationKind.COMPACT:
                    tally.compacted.append(operation.uri)
                elif outcome is MemoryLifecycleOperationKind.RETIRE:
                    tally.retired.append(operation.uri)

    async def _scan_batch(self) -> tuple[MemoryAddress, ...]:
        cursor = await asyncio.to_thread(self.scan_store.cursor)
        addresses = await asyncio.to_thread(
            self.tree.list_addresses,
            limit=self.config.max_scan_items,
            after=cursor,
        )
        if not addresses and cursor is not None:
            await asyncio.to_thread(self.scan_store.reset_cursor)
            addresses = await asyncio.to_thread(
                self.tree.list_addresses,
                limit=self.config.max_scan_items,
            )
        return addresses

    async def _maintain_address(
        self,
        address: MemoryAddress,
        tally: _MaintenanceTally,
        *,
        now: datetime,
    ) -> None:
        uri = MemoryURI.from_address(address)
        try:
            await self._maintain_uri(uri, tally, now=now)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._record_failure(tally, uri, exc, now=now)
        else:
            await asyncio.to_thread(self.scan_store.clear_failure, uri)
        finally:
            await asyncio.to_thread(self.scan_store.advance, address)

    async def _maintain_uri(self, uri: MemoryURI, tally: _MaintenanceTally, *, now: datetime) -> None:
        eligible = await asyncio.to_thread(self.scan_store.eligible, uri, now=now)
        if not eligible or await asyncio.to_thread(self.operation_store.try_read, uri) is not None:
            return
        snapshot = await asyncio.to_thread(self.snapshot_reader.read, uri)
        ranking = await asyncio.to_thread(self._ranking, snapshot, now)
        action = plan_maintenance_action(
            ranking,
            retirement_eligible=self.recall_lifecycle.retirement_eligible(ranking, now=now),
            retired_count=len(tally.retired),
            compaction_attempts=tally.compaction_attempts,
            quota=MemoryMaintenanceQuota(
                max_retirements=self.config.max_retirements_per_cycle,
                max_compactions=self.config.max_compactions_per_cycle,
            ),
        )
        if action is MemoryMaintenanceAction.RETIRE:
            await self._retire(snapshot, uri, ranking.state.version, tally, now=now)
        elif action is MemoryMaintenanceAction.CONSIDER_RETIREMENT:
            await self._consider_retirement(snapshot, ranking, now=now)
            refreshed = await asyncio.to_thread(self._ranking, snapshot, now)
            if refreshed.state.retired_at is not None:
                await self._retire(snapshot, uri, refreshed.state.version, tally, now=now)
        elif action is MemoryMaintenanceAction.COMPACT:
            version = _compaction_keep(snapshot, ranking, now)
            if await asyncio.to_thread(self.scan_store.compaction_kept, version):
                return
            tally.compaction_attempts += 1
            if await self._compact(snapshot, version, now=now):
                tally.compacted.append(uri)

    async def _retire(
        self,
        snapshot: MemorySnapshot,
        uri: MemoryURI,
        expected_state_version: int,
        tally: _MaintenanceTally,
        *,
        now: datetime,
    ) -> None:
        operation = await asyncio.to_thread(
            self.operation_store.prepare,
            MemoryLifecycleOperationKind.RETIRE,
            snapshot,
            planned_fields=None,
            expected_state_version=expected_state_version,
            prepared_at=now,
        )
        outcome = await self.operation_runner.resume(operation, finish_derived=True)
        if outcome is MemoryLifecycleOperationKind.RETIRE:
            tally.retired.append(uri)

    async def _record_failure(
        self,
        tally: _MaintenanceTally,
        uri: MemoryURI,
        exc: Exception,
        *,
        now: datetime,
    ) -> None:
        failure = await asyncio.to_thread(
            self.scan_store.record_failure,
            uri,
            exc,
            failed_at=now,
        )
        tally.failures.append(MemoryLifecycleMaintenanceFailure.from_scan(failure))

    async def record_context_use(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        used_at: datetime | None = None,
    ) -> MemoryContextUseResult:
        """最终模型可见 Context 的 L2 才计成功；COLD_2 必须先同步恢复。"""

        return await self.context_use.record_context_use(targets, used_at=used_at)

    def expand_for_probe(self, document: MemoryDocument) -> MemoryDocument:
        """只为最终 Context 预算临时展开活动基线，不修改 L2 或热度。"""

        return self.context_use.expand_for_probe(document)

    async def _compact(
        self,
        source: MemorySnapshot,
        version: MemoryCompactionKeep,
        *,
        now: datetime,
    ) -> bool:
        if not source.exists or source.value is None:
            return False
        result = await self.field_compactor.compact(source.value)
        if not result.changed:
            await asyncio.to_thread(self.scan_store.record_compaction_keep, version)
            return False
        current = await asyncio.to_thread(self.snapshot_reader.read, source.identity)
        if not same_snapshot(current, source):
            return False
        ranking = await asyncio.to_thread(self._ranking, current, now)
        if ranking.temperature is not MemoryTemperature.COLD_2 or ranking.state.compacted_at is not None:
            return False
        recovery = await asyncio.to_thread(self.recovery_store.save, source.value, saved_at=now)
        operation = await asyncio.to_thread(
            self.operation_store.prepare,
            MemoryLifecycleOperationKind.COMPACT,
            current,
            planned_fields=result.fields,
            expected_state_version=ranking.state.version,
            prepared_at=now,
        )
        outcome = await self.operation_runner.resume(operation, finish_derived=True)
        return outcome is MemoryLifecycleOperationKind.COMPACT and recovery.uri == ranking.candidate.uri

    async def _consider_retirement(
        self,
        snapshot: MemorySnapshot,
        ranking: MemoryRecallRanking,
        *,
        now: datetime,
    ) -> None:
        if not self.recall_lifecycle.retirement_eligible(ranking, now=now):
            return
        state = ranking.state
        if state.retire_candidate_at is None:
            await asyncio.to_thread(
                self.recall_lifecycle.mark_retire_candidate,
                snapshot_recall_target(snapshot),
                marked_at=now,
                expected_version=state.version,
            )
            return
        if not self.recall_lifecycle.retirement_grace_elapsed(state, now=now):
            return
        await asyncio.to_thread(
            self.recall_lifecycle.mark_retired,
            snapshot_recall_target(snapshot),
            retired_at=now,
            expected_version=state.version,
        )

    def _ranking(self, snapshot: MemorySnapshot, now: datetime) -> MemoryRecallRanking:
        if not snapshot.exists or snapshot.value is None:
            raise ValueError("memory lifecycle target does not exist")
        return self.recall_lifecycle.rank((recall_candidate(snapshot.value),), now=now)[0]

    _target = staticmethod(recall_target)

    def _timestamp(self) -> datetime:
        return lifecycle_timestamp(self.clock())


@dataclass
class _MaintenanceTally:
    """一轮维护的累计：结果列表与已用压缩名额（退休名额按 ``retired`` 条数算）。"""

    compacted: list[MemoryURI] = field(default_factory=list)
    retired: list[MemoryURI] = field(default_factory=list)
    failures: list[MemoryLifecycleMaintenanceFailure] = field(default_factory=list)
    compaction_attempts: int = 0


def _compaction_keep(
    snapshot: MemorySnapshot,
    ranking: MemoryRecallRanking,
    now: datetime,
) -> MemoryCompactionKeep:
    """把送去压缩的这一版内容写成 KEEP 记录的键；判过之后内容或被用到的时刻变了才会再送。"""

    if snapshot.revision is None or snapshot.source_digest is None:
        raise ValueError("memory compaction source snapshot must carry a revision and digest")
    return MemoryCompactionKeep(
        uri=ranking.candidate.uri,
        revision=snapshot.revision,
        source_digest=snapshot.source_digest,
        last_used_at=ranking.state.last_useful_recall_at,
        judged_at=now,
    )


__all__ = [
    "MemoryLifecycleMaintenanceConfig",
    "MemoryLifecycleMaintenanceFailure",
    "MemoryLifecycleMaintenanceResult",
    "MemoryLifecycleManager",
]
