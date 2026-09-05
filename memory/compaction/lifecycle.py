"""协调 COLD_2 压缩、最终上下文恢复和终态退休。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from memory.compaction.commit import MemoryLifecycleCommitter
from memory.compaction.l2_fields import MemoryFieldCompactor
from memory.compaction.operation import (
    MemoryLifecycleOperation,
    MemoryLifecycleOperationKind,
    MemoryLifecycleOperationPhase,
    MemoryLifecycleOperationStore,
)
from memory.compaction.recovery import MemoryRecoveryStore
from memory.compaction.scan import MemoryLifecycleScanFailure, MemoryLifecycleScanStore
from memory.document import MemoryDocument
from memory.retrieval.lifecycle import (
    MemoryRecallCandidate,
    MemoryRecallLifecycle,
    MemoryRecallRanking,
    MemoryRecallState,
    MemoryRecallTarget,
    MemoryTemperature,
)
from memory.snapshot import MemorySnapshot, MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI


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
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
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


@dataclass(frozen=True)
class MemoryContextUseResult:
    documents: tuple[MemoryDocument, ...]
    rejected_uris: tuple[MemoryURI, ...]


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
        *,
        operation_store: MemoryLifecycleOperationStore | None = None,
        scan_store: MemoryLifecycleScanStore | None = None,
        config: MemoryLifecycleMaintenanceConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        derived_refresh: Callable[[tuple[MemoryURI, ...]], Awaitable[None]] | None = None,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be MemoryTree")
        if not isinstance(snapshot_reader, MemorySnapshotReader) or snapshot_reader.tree is not tree:
            raise ValueError("lifecycle manager must share one tree and snapshot reader")
        if not isinstance(recall_lifecycle, MemoryRecallLifecycle):
            raise TypeError("recall_lifecycle must be MemoryRecallLifecycle")
        if not isinstance(field_compactor, MemoryFieldCompactor):
            raise TypeError("field_compactor must be MemoryFieldCompactor")
        if not isinstance(recovery_store, MemoryRecoveryStore) or recovery_store.tree is not tree:
            raise ValueError("lifecycle manager and recovery store must share one tree")
        if not isinstance(committer, MemoryLifecycleCommitter) or committer.snapshot_reader is not snapshot_reader:
            raise ValueError("lifecycle manager and committer must share one snapshot reader")
        if config is not None and not isinstance(config, MemoryLifecycleMaintenanceConfig):
            raise TypeError("config must be MemoryLifecycleMaintenanceConfig")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if derived_refresh is not None and not callable(derived_refresh):
            raise TypeError("derived_refresh must be callable")
        default_operations = MemoryLifecycleOperationStore(tree, committer.transaction.path_lock)
        default_scan = MemoryLifecycleScanStore(tree, committer.transaction.path_lock)
        if operation_store is not None and not isinstance(operation_store, MemoryLifecycleOperationStore):
            raise TypeError("operation_store must be MemoryLifecycleOperationStore")
        if scan_store is not None and not isinstance(scan_store, MemoryLifecycleScanStore):
            raise TypeError("scan_store must be MemoryLifecycleScanStore")
        self.tree = tree
        self.snapshot_reader = snapshot_reader
        self.recall_lifecycle = recall_lifecycle
        self.field_compactor = field_compactor
        self.recovery_store = recovery_store
        self.committer = committer
        self.operation_store = operation_store or default_operations
        self.scan_store = scan_store or default_scan
        self.config = config or MemoryLifecycleMaintenanceConfig()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.derived_refresh = derived_refresh

    def initialize(self) -> None:
        self.recall_lifecycle.initialize()
        self.recovery_store.initialize()
        self.operation_store.initialize()
        self.scan_store.initialize()

    async def maintain(self) -> MemoryLifecycleMaintenanceResult:
        """先续跑耐久操作，再有界轮转并隔离每个 L2 节点的失败。"""

        now = self._timestamp()
        self.initialize()
        compacted: list[MemoryURI] = []
        retired: list[MemoryURI] = []
        failures: list[MemoryLifecycleMaintenanceFailure] = []
        pending = await asyncio.to_thread(self.operation_store.pending)
        for operation in pending:
            try:
                if not await asyncio.to_thread(
                    self.scan_store.eligible,
                    operation.uri,
                    now=now,
                ):
                    continue
                outcome = await self._resume_operation(operation, finish_derived=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure = await asyncio.to_thread(
                    self.scan_store.record_failure,
                    operation.uri,
                    exc,
                    failed_at=now,
                )
                failures.append(MemoryLifecycleMaintenanceFailure.from_scan(failure))
            else:
                await asyncio.to_thread(self.scan_store.clear_failure, operation.uri)
                if outcome is MemoryLifecycleOperationKind.COMPACT:
                    compacted.append(operation.uri)
                elif outcome is MemoryLifecycleOperationKind.RETIRE:
                    retired.append(operation.uri)

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
        for address in addresses:
            uri = MemoryURI.from_address(address)
            try:
                eligible = await asyncio.to_thread(self.scan_store.eligible, uri, now=now)
                if not eligible or await asyncio.to_thread(self.operation_store.try_read, uri) is not None:
                    continue
                snapshot = await asyncio.to_thread(self.snapshot_reader.read, uri)
                ranking = await asyncio.to_thread(self._ranking, snapshot, now)
                if ranking.state.retired_at is not None:
                    if len(retired) >= self.config.max_retirements_per_cycle:
                        continue
                    operation = await asyncio.to_thread(
                        self.operation_store.prepare,
                        MemoryLifecycleOperationKind.RETIRE,
                        snapshot,
                        planned_fields=None,
                        expected_state_version=ranking.state.version,
                        prepared_at=now,
                    )
                    outcome = await self._resume_operation(operation, finish_derived=True)
                    if outcome is MemoryLifecycleOperationKind.RETIRE:
                        retired.append(uri)
                    continue
                if ranking.temperature is not MemoryTemperature.COLD_2:
                    continue
                if ranking.state.compacted_at is None:
                    if len(compacted) >= self.config.max_compactions_per_cycle:
                        continue
                    if await self._compact(snapshot, now=now):
                        compacted.append(uri)
                    continue
                if len(retired) >= self.config.max_retirements_per_cycle:
                    continue
                await self._consider_retirement(snapshot, ranking, now=now)
                refreshed = await asyncio.to_thread(self._ranking, snapshot, now)
                if refreshed.state.retired_at is not None:
                    operation = await asyncio.to_thread(
                        self.operation_store.prepare,
                        MemoryLifecycleOperationKind.RETIRE,
                        snapshot,
                        planned_fields=None,
                        expected_state_version=refreshed.state.version,
                        prepared_at=now,
                    )
                    outcome = await self._resume_operation(operation, finish_derived=True)
                    if outcome is MemoryLifecycleOperationKind.RETIRE:
                        retired.append(uri)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure = await asyncio.to_thread(
                    self.scan_store.record_failure,
                    uri,
                    exc,
                    failed_at=now,
                )
                failures.append(MemoryLifecycleMaintenanceFailure.from_scan(failure))
            else:
                await asyncio.to_thread(self.scan_store.clear_failure, uri)
            finally:
                await asyncio.to_thread(self.scan_store.advance, address)
        return MemoryLifecycleMaintenanceResult(
            len(addresses),
            tuple(dict.fromkeys(compacted)),
            tuple(dict.fromkeys(retired)),
            tuple(failures),
        )

    async def record_context_use(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        used_at: datetime | None = None,
    ) -> MemoryContextUseResult:
        """最终模型可见 Context 的 L2 才计成功；COLD_2 必须先同步恢复。"""

        if not isinstance(targets, tuple) or any(not isinstance(item, MemoryRecallTarget) for item in targets):
            raise TypeError("context use targets must be a tuple of MemoryRecallTarget")
        if len({item.uri for item in targets}) != len(targets):
            raise ValueError("context use targets must be unique")
        timestamp = self._timestamp() if used_at is None else _timestamp(used_at)
        documents: list[MemoryDocument] = []
        rejected: list[MemoryURI] = []
        for target in targets:
            try:
                snapshot, state, detailed = await asyncio.to_thread(
                    self._prepare_context_use,
                    target,
                    timestamp,
                )
                if detailed is not None:
                    assert state is not None
                    operation = await asyncio.to_thread(
                        self.operation_store.prepare,
                        MemoryLifecycleOperationKind.RESTORE,
                        snapshot,
                        planned_fields=detailed.fields,
                        expected_state_version=state.version,
                        prepared_at=timestamp,
                    )
                    await self._resume_operation(operation, finish_derived=False)
                    current = await asyncio.to_thread(self.snapshot_reader.read, target.uri)
                    if not current.exists or not isinstance(current.value, MemoryDocument):
                        raise RuntimeError("restored final Context memory disappeared")
                    documents.append(current.value)
                else:
                    assert isinstance(snapshot.value, MemoryDocument)
                    documents.append(snapshot.value)
            except asyncio.CancelledError:
                raise
            except Exception:
                rejected.append(target.uri)
        return MemoryContextUseResult(tuple(documents), tuple(rejected))

    def _prepare_context_use(
        self,
        target: MemoryRecallTarget,
        timestamp: datetime,
    ) -> tuple[MemorySnapshot, MemoryRecallState | None, MemoryDocument | None]:
        """在 operation URI 锁内提交普通 use，或冻结待恢复的精确来源。"""

        with self.operation_store.acquire(target.uri, wait_timeout_seconds=5.0):
            pending = self.operation_store.try_read(target.uri)
            snapshot = self.snapshot_reader.read(target.uri)
            if not _matches_target(snapshot, target):
                raise RuntimeError("final Context target changed before use acknowledgement")
            if (
                pending is not None
                and pending.kind is MemoryLifecycleOperationKind.RETIRE
                and not _matches_operation_source(snapshot, pending)
            ):
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
            states = self.recall_lifecycle.record_use((target,), used_at=timestamp)
            if self.recall_lifecycle.config.enabled and (
                not states
                or states[0].document_revision != target.document_revision
                or states[0].document_created_at != target.document_created_at
            ):
                raise RuntimeError("final Context use acknowledgement lost its revision fence")
            if not allowed_pending_restore:
                self.recovery_store.delete(target.uri)
            return snapshot, state, None

    async def record_use(
        self,
        uris: tuple[MemoryURI, ...],
        *,
        used_at: datetime | None = None,
    ) -> tuple[MemoryRecallTarget, ...]:
        """兼容进程内显式入口；产品正确性不依赖外部回执。"""

        if not isinstance(uris, tuple):
            raise TypeError("used memory URIs must be a tuple")
        snapshots = tuple(
            [await asyncio.to_thread(self.snapshot_reader.read, uri) for uri in uris]
        )
        targets = tuple(self._target_snapshot(snapshot) for snapshot in snapshots)
        result = await self.record_context_use(targets, used_at=used_at)
        return tuple(self._target(document) for document in result.documents)

    def expand_for_probe(self, document: MemoryDocument) -> MemoryDocument:
        """只为最终 Context 预算临时展开活动基线，不修改 L2 或热度。"""

        if not isinstance(document, MemoryDocument):
            raise TypeError("document must be MemoryDocument")
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

    async def _compact(self, source: MemorySnapshot, *, now: datetime) -> bool:
        if not source.exists or not isinstance(source.value, MemoryDocument):
            return False
        result = await self.field_compactor.compact(source.value)
        if not result.changed:
            return False
        current = await asyncio.to_thread(self.snapshot_reader.read, source.identity)
        if not _same_snapshot(current, source):
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
        outcome = await self._resume_operation(operation, finish_derived=True)
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
                self._target_snapshot(snapshot),
                marked_at=now,
                expected_version=state.version,
            )
            return
        if not self.recall_lifecycle.retirement_grace_elapsed(state, now=now):
            return
        await asyncio.to_thread(
            self.recall_lifecycle.mark_retired,
            self._target_snapshot(snapshot),
            retired_at=now,
            expected_version=state.version,
        )

    async def _resume_operation(
        self,
        operation: MemoryLifecycleOperation,
        *,
        finish_derived: bool,
    ) -> MemoryLifecycleOperationKind | None:
        current_time = self._timestamp()
        current = await asyncio.to_thread(self.operation_store.try_read, operation.uri)
        if current is None or current.operation_id != operation.operation_id:
            return None
        if current.kind is MemoryLifecycleOperationKind.COMPACT:
            current = await self._resume_compact(current, current_time)
        elif current.kind is MemoryLifecycleOperationKind.RESTORE:
            current = await self._resume_restore(current, current_time)
        else:
            current = await self._resume_retire(current, current_time)
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

    async def _resume_compact(
        self,
        operation: MemoryLifecycleOperation,
        now: datetime,
    ) -> MemoryLifecycleOperation:
        current = operation
        if current.phase is MemoryLifecycleOperationPhase.PREPARED:
            snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
            state = await asyncio.to_thread(self._state_for, current.uri)
            state_version = 0 if state is None else state.version
            if _matches_operation_source(snapshot, current):
                if state_version != current.expected_state_version:
                    await asyncio.to_thread(self.recovery_store.delete, current.uri)
                    await asyncio.to_thread(self.operation_store.complete, current)
                    return current
                assert current.planned_fields is not None
                await asyncio.to_thread(self.committer.replace_fields, snapshot, current.planned_fields)
                snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
            elif not _matches_operation_target(snapshot, current):
                if not _matches_planned_target(snapshot, current):
                    await asyncio.to_thread(self.recovery_store.delete, current.uri)
                    await asyncio.to_thread(self.operation_store.complete, current)
                    return current
            if not _matches_planned_target(snapshot, current):
                raise RuntimeError("compaction target does not match its durable plan")
            record = await asyncio.to_thread(
                self.recovery_store.latest,
                current.uri,
                created_at=current.source_created_at,
            )
            if record is None:
                raise RuntimeError("compaction recovery baseline is missing")
            await asyncio.to_thread(self.recovery_store.activate, record, snapshot)
            current = await asyncio.to_thread(
                self.operation_store.advance,
                current,
                MemoryLifecycleOperationPhase.L2_COMMITTED,
                updated_at=now,
                target_revision=snapshot.revision,
                target_digest=snapshot.source_digest,
            )
        if current.phase is MemoryLifecycleOperationPhase.L2_COMMITTED:
            snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
            if not _matches_operation_target(snapshot, current):
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
            if not _compaction_state_matches(state, snapshot, current.expected_state_version + 1):
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
                    self._target_snapshot(snapshot),
                    lifecycle_activity_at=activity,
                    compacted_at=current.prepared_at,
                    expected_version=current.expected_state_version,
                )
            current = await asyncio.to_thread(
                self.operation_store.advance,
                current,
                MemoryLifecycleOperationPhase.STATE_COMMITTED,
                updated_at=now,
            )
        if current.phase is MemoryLifecycleOperationPhase.STATE_COMMITTED:
            current = await asyncio.to_thread(
                self.operation_store.advance,
                current,
                MemoryLifecycleOperationPhase.DERIVED_PENDING,
                updated_at=now,
            )
        return current

    async def _resume_restore(
        self,
        operation: MemoryLifecycleOperation,
        now: datetime,
    ) -> MemoryLifecycleOperation:
        current = operation
        if current.phase is MemoryLifecycleOperationPhase.PREPARED:
            snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
            state = await asyncio.to_thread(self._state_for, current.uri)
            if not _matches_operation_source(snapshot, current):
                if not _matches_planned_target(snapshot, current):
                    raise RuntimeError("restore source changed before L2 CAS")
            else:
                if state is None or state.version != current.expected_state_version:
                    raise RuntimeError("lifecycle state changed before COLD_2 restore")
                assert current.planned_fields is not None
                await asyncio.to_thread(self.committer.replace_fields, snapshot, current.planned_fields)
                snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
            current = await asyncio.to_thread(
                self.operation_store.advance,
                current,
                MemoryLifecycleOperationPhase.L2_COMMITTED,
                updated_at=now,
                target_revision=snapshot.revision,
                target_digest=snapshot.source_digest,
            )
        if current.phase is MemoryLifecycleOperationPhase.L2_COMMITTED:
            snapshot = await asyncio.to_thread(self.snapshot_reader.read, current.uri)
            if not _matches_operation_target(snapshot, current):
                raise RuntimeError("restored L2 changed before use acknowledgement")
            target = self._target_snapshot(snapshot)
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
            current = await asyncio.to_thread(
                self.operation_store.advance,
                current,
                MemoryLifecycleOperationPhase.STATE_COMMITTED,
                updated_at=now,
            )
        if current.phase is MemoryLifecycleOperationPhase.STATE_COMMITTED:
            current = await asyncio.to_thread(
                self.operation_store.advance,
                current,
                MemoryLifecycleOperationPhase.DERIVED_PENDING,
                updated_at=now,
            )
        return current

    async def _resume_retire(
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
                if not _matches_operation_source(snapshot, current):
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

    def _ranking(self, snapshot: MemorySnapshot, now: datetime) -> MemoryRecallRanking:
        if not snapshot.exists or not isinstance(snapshot.value, MemoryDocument):
            raise ValueError("memory lifecycle target does not exist")
        return self.recall_lifecycle.rank((self._candidate(snapshot.value),), now=now)[0]

    def _state_for(self, uri: MemoryURI) -> MemoryRecallState | None:
        if not self.recall_lifecycle.config.enabled:
            return None
        states = self.recall_lifecycle.store.read_many((uri,))
        return states[0] if states else None

    @staticmethod
    def _target(document: MemoryDocument) -> MemoryRecallTarget:
        return MemoryRecallTarget(
            MemoryURI.from_address(document.address),
            document.metadata.revision,
            document.metadata.created_at,
        )

    @classmethod
    def _target_snapshot(cls, snapshot: MemorySnapshot) -> MemoryRecallTarget:
        if not snapshot.exists or not isinstance(snapshot.value, MemoryDocument):
            raise ValueError("memory lifecycle target snapshot must exist")
        return cls._target(snapshot.value)

    @classmethod
    def _candidate(cls, document: MemoryDocument) -> MemoryRecallCandidate:
        return MemoryRecallCandidate(
            cls._target(document),
            document.kind,
            document.metadata.updated_at,
            1.0,
        )

    def _timestamp(self) -> datetime:
        return _timestamp(self.clock())


def _timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("memory lifecycle clock must return datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("memory lifecycle timestamp must include a timezone")
    return value.astimezone(UTC)


def _same_snapshot(left: MemorySnapshot, right: MemorySnapshot) -> bool:
    return (
        left.state is right.state
        and left.revision == right.revision
        and left.source_digest == right.source_digest
    )


def _matches_target(snapshot: MemorySnapshot, target: MemoryRecallTarget) -> bool:
    return (
        snapshot.exists
        and isinstance(snapshot.value, MemoryDocument)
        and snapshot.identity == str(target.uri)
        and snapshot.value.metadata.revision == target.document_revision
        and snapshot.value.metadata.created_at == target.document_created_at
    )


def _matches_operation_source(
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


def _matches_operation_target(
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


def _matches_planned_target(
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


def _compaction_state_matches(
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
    "MemoryContextUseResult",
    "MemoryLifecycleMaintenanceConfig",
    "MemoryLifecycleMaintenanceFailure",
    "MemoryLifecycleMaintenanceResult",
    "MemoryLifecycleManager",
]
