"""跨 Conversation、Summary 与记忆工作流的统一生命周期维护。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from memory.conversation import (
    ConversationAddress,
    ConversationLayout,
    ConversationMessageJournal,
    ConversationRangeSummaryStore,
    ConversationSummaryCompactionConfig,
    ConversationSummaryCompactionResult,
    ConversationSummaryCompactor,
    ConversationSummaryRetirementManifest,
    ConversationSummaryRetirementPhase,
    ConversationSummaryRetirementStore,
    ConversationSummaryStore,
)
from memory.conversation.indexing import PersistentConversationSummaryVectorIndex, summary_reference
from memory.editor import MemoryTransactionJournal, MemoryTransactionJournalState
from memory.workflow.jobs import MemoryJobStatus, MemoryJobStore
from memory.workflow.receipt import (
    MemoryChangeReceiptState,
    MemoryChangeReceiptStore,
    MemoryChangeSource,
)
from pre.conversation import (
    ConversationRangeSummary,
    ConversationRangeSummaryStage,
    ConversationSegment,
    ConversationSegmentSummary,
    ConversationSummarySourceKind,
)


class ConversationLifecycleError(RuntimeError):
    """Conversation 生命周期状态不完整，不能继续压缩或清理。"""


@dataclass(frozen=True)
class MemoryWorkflowLifecycleConfig:
    """终态 Job、回执与一次维护操作的明确生命周期边界。"""

    committed_job_retention_days: int = 30
    committed_receipt_retention_days: int = 365
    cleanup_batch_size: int = 100

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("committed_job_retention_days", self.committed_job_retention_days, 0, 3_650),
            (
                "committed_receipt_retention_days",
                self.committed_receipt_retention_days,
                1,
                36_500,
            ),
            ("cleanup_batch_size", self.cleanup_batch_size, 1, 10_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"workflow lifecycle {name} must be between {minimum} and {maximum}")
        if self.committed_receipt_retention_days < self.committed_job_retention_days:
            raise ValueError("committed receipt retention cannot be shorter than committed job retention")


@dataclass(frozen=True)
class ConversationLifecycleMaintenanceResult:
    """一次显式维护的压缩结果与各层实际物理清理结果。"""

    compaction: ConversationSummaryCompactionResult
    summary_indexed: bool
    purged_history_segment_ids: tuple[str, ...]
    released_history_segment_ids: tuple[str, ...]
    deleted_segment_summary_ids: tuple[str, ...]
    deleted_range_summary_ids: tuple[str, ...]
    deleted_archive_summary_ids: tuple[str, ...]
    deleted_memory_job_sequences: tuple[int, ...]
    deleted_memory_receipt_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.compaction, ConversationSummaryCompactionResult):
            raise TypeError("compaction must be ConversationSummaryCompactionResult")
        if not isinstance(self.summary_indexed, bool):
            raise TypeError("summary_indexed must be boolean")
        for name in (
            "purged_history_segment_ids",
            "released_history_segment_ids",
            "deleted_segment_summary_ids",
            "deleted_range_summary_ids",
            "deleted_archive_summary_ids",
            "deleted_memory_receipt_ids",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(value, str) or not value for value in values):
                raise TypeError(f"{name} must contain non-empty identifiers")
        if not isinstance(self.deleted_memory_job_sequences, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.deleted_memory_job_sequences
        ):
            raise TypeError("deleted_memory_job_sequences must contain positive integers")


class ConversationLifecycleManager:
    """生成不可变父摘要；来源链只在 Archive 终态退休时统一清理。"""

    def __init__(
        self,
        compactor: ConversationSummaryCompactor,
        journal: ConversationMessageJournal,
        segment_store: ConversationSummaryStore,
        range_store: ConversationRangeSummaryStore,
        summary_vector_index: PersistentConversationSummaryVectorIndex,
        jobs: MemoryJobStore,
        receipts: MemoryChangeReceiptStore,
        transaction_journal: MemoryTransactionJournal,
        retirement_store: ConversationSummaryRetirementStore | None = None,
        *,
        summary_config: ConversationSummaryCompactionConfig | None = None,
        workflow_config: MemoryWorkflowLifecycleConfig | None = None,
    ) -> None:
        if not isinstance(compactor, ConversationSummaryCompactor):
            raise TypeError("compactor must be ConversationSummaryCompactor")
        if not isinstance(journal, ConversationMessageJournal):
            raise TypeError("journal must be ConversationMessageJournal")
        if not isinstance(segment_store, ConversationSummaryStore):
            raise TypeError("segment_store must be ConversationSummaryStore")
        if not isinstance(range_store, ConversationRangeSummaryStore):
            raise TypeError("range_store must be ConversationRangeSummaryStore")
        if not isinstance(summary_vector_index, PersistentConversationSummaryVectorIndex):
            raise TypeError("summary_vector_index must be PersistentConversationSummaryVectorIndex")
        if not isinstance(jobs, MemoryJobStore):
            raise TypeError("jobs must be MemoryJobStore")
        if not isinstance(receipts, MemoryChangeReceiptStore):
            raise TypeError("receipts must be MemoryChangeReceiptStore")
        if not isinstance(transaction_journal, MemoryTransactionJournal):
            raise TypeError("transaction_journal must be MemoryTransactionJournal")
        if retirement_store is not None and not isinstance(
            retirement_store,
            ConversationSummaryRetirementStore,
        ):
            raise TypeError("retirement_store must be ConversationSummaryRetirementStore or None")
        if summary_config is not None and not isinstance(
            summary_config,
            ConversationSummaryCompactionConfig,
        ):
            raise TypeError("summary_config must be ConversationSummaryCompactionConfig")
        if workflow_config is not None and not isinstance(
            workflow_config,
            MemoryWorkflowLifecycleConfig,
        ):
            raise TypeError("workflow_config must be MemoryWorkflowLifecycleConfig")
        if not (
            compactor.journal is journal
            and compactor.segment_store is segment_store
            and compactor.range_store is range_store
        ):
            raise ValueError("lifecycle manager must reuse the compactor Conversation stores")
        if summary_vector_index.compactor is not compactor:
            raise ValueError("lifecycle manager must reuse the Summary vector index compactor")
        if receipts.codec is not transaction_journal.codec:
            raise ValueError("lifecycle receipts and transaction journal must share one codec")
        resolved_summary_config = compactor.config if summary_config is None else summary_config
        if resolved_summary_config != compactor.config:
            raise ValueError("lifecycle manager and compactor must share one lifecycle config")
        self.compactor = compactor
        self.journal = journal
        self.segment_store = segment_store
        self.range_store = range_store
        self.summary_vector_index = summary_vector_index
        self.jobs = jobs
        self.receipts = receipts
        self.transaction_journal = transaction_journal
        self.retirement_store = retirement_store or ConversationSummaryRetirementStore(jobs.root)
        source_retirements = summary_vector_index.sources.retirement_store
        if source_retirements is None:
            summary_vector_index.sources.retirement_store = self.retirement_store
        elif source_retirements is not self.retirement_store:
            raise ValueError("lifecycle manager and Summary index must share one retirement store")
        self.summary_config = resolved_summary_config
        self.workflow_config = workflow_config or MemoryWorkflowLifecycleConfig()

    async def maintain_once(
        self,
        address: ConversationAddress,
        *,
        now: datetime | None = None,
    ) -> ConversationLifecycleMaintenanceResult:
        """每次至多生成一个父摘要；普通阶段不做不可逆来源清理。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        current_time = _utc_datetime(now or datetime.now(timezone.utc))
        pending_retirement = self.retirement_store.for_address(address)
        if len(pending_retirement) > 1:
            raise ConversationLifecycleError("one Conversation has multiple retiring Archives")
        if pending_retirement:
            compaction = ConversationSummaryCompactionResult(
                None,
                False,
                "resumed pending Archive retirement before new compaction",
            )
        else:
            compaction = await self.compactor.compact_once(address, now=current_time)
        purged, released, deleted_segments, deleted_ranges, deleted_archives = await (
            self._retire_archive_chain(address, current_time)
        )
        await self.summary_vector_index.synchronize(address)
        deleted_jobs = self._delete_expired_committed_jobs(address, current_time)
        deleted_receipts = self._delete_expired_committed_receipts(address, current_time)
        return ConversationLifecycleMaintenanceResult(
            compaction=compaction,
            summary_indexed=True,
            purged_history_segment_ids=purged,
            released_history_segment_ids=released,
            deleted_segment_summary_ids=deleted_segments,
            deleted_range_summary_ids=deleted_ranges,
            deleted_archive_summary_ids=deleted_archives,
            deleted_memory_job_sequences=deleted_jobs,
            deleted_memory_receipt_ids=deleted_receipts,
        )

    async def _retire_archive_chain(
        self,
        address: ConversationAddress,
        now: datetime,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Archive 只在长期未用且宽限期结束后，按原文前缀分批执行终态删除。"""

        pending = self.retirement_store.for_address(address)
        if len(pending) > 1:
            raise ConversationLifecycleError("one Conversation has multiple retiring Archives")
        if pending:
            return await self._resume_archive_retirement(pending[0], now)
        use_store = self.compactor.use_store
        required = (
            "read_many",
            "mark_retire_candidate",
            "claim_retirement",
            "delete_coverage",
        )
        if use_store is None or any(not callable(getattr(use_store, name, None)) for name in required):
            return (), (), (), (), ()
        archives = self.range_store.list(address, ConversationRangeSummaryStage.ARCHIVE)
        state = self.journal.read_state(address)
        for archive in archives:
            if archive.end_sequence <= state.released_through:
                pass
            elif archive.start_sequence != state.released_through + 1:
                continue
            reference = summary_reference(address, archive)
            states = use_store.read_many((reference,))
            use_state = states[0] if states else None
            last_use = None if use_state is None else use_state.last_useful_recall_at
            activity = max(archive.generated_at, last_use) if last_use is not None else archive.generated_at
            if now - activity < timedelta(days=self.summary_config.archive_retire_days):
                continue
            candidate_at = None if use_state is None else use_state.retire_candidate_at
            if candidate_at is None:
                use_store.mark_retire_candidate(reference, marked_at=now)
                return (), (), (), (), ()
            if now - candidate_at < timedelta(days=self.summary_config.archive_retire_grace_days):
                continue

            ranges, segments = self._archive_sources(address, archive)
            expected_version = 0 if use_state is None else use_state.version
            manifest = self.retirement_store.prepare(
                address,
                archive,
                ranges,
                segments,
                expected_use_version=expected_version,
                prepared_at=now,
            )
            return await self._resume_archive_retirement(manifest, now)
        return (), (), (), (), ()

    async def _resume_archive_retirement(
        self,
        manifest: ConversationSummaryRetirementManifest,
        now: datetime,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """严格按下线、History、子摘要、Archive 的阶段幂等续跑。"""

        current = manifest
        address = current.address
        use_store = self.compactor.use_store
        if use_store is None:
            raise ConversationLifecycleError("retiring Archive lacks a Summary use store")
        if current.phase is ConversationSummaryRetirementPhase.RETIRING:
            states = use_store.read_many((current.archive.reference,))
            use_state = states[0] if states else None
            claimed = (
                use_state is not None
                and use_state.version == current.expected_use_version + 1
                and use_state.retiring_at is not None
            )
            if not claimed:
                if use_state is None or use_state.version != current.expected_use_version:
                    self.retirement_store.complete(current)
                    return (), (), (), (), ()
                use_store.claim_retirement(
                    current.archive.reference,
                    expected_version=current.expected_use_version,
                    claimed_at=now,
                )
            await self.summary_vector_index.synchronize(
                address,
                removed_references=(current.archive.reference,),
            )
            current = self.retirement_store.advance(
                current,
                ConversationSummaryRetirementPhase.INDEX_REMOVED,
                updated_at=now,
            )

        purged: tuple[str, ...] = ()
        released: tuple[str, ...] = ()
        if current.phase is ConversationSummaryRetirementPhase.INDEX_REMOVED:
            journal_state = self.journal.read_state(address)
            if journal_state.released_through < current.end_sequence:
                segment_sources = {
                    item.reference.summary_id: item.digest for item in current.segments
                }
                active = tuple(
                    segment
                    for segment in self.journal.list_history(address)
                    if current.start_sequence <= segment.start_sequence
                    and segment.end_sequence <= current.end_sequence
                )[: self.summary_config.cleanup_batch_size]
                for segment in active:
                    expected_digest = segment_sources.get(segment.segment_id)
                    summary = self.segment_store.try_read_by_id(address, segment.segment_id)
                    if (
                        expected_digest is None
                        or summary is None
                        or summary.digest != expected_digest
                    ):
                        raise ConversationLifecycleError(
                            "retiring Archive does not match retained History Segment"
                        )
                    summary.require_matches_source(segment)
                    if not self._workflow_is_committed(address, segment):
                        return (), (), (), (), ()
                if not active:
                    raise ConversationLifecycleError("retiring Archive History source is missing")
                released = self.journal.release_history_prefix(address, active)
                purged = self.journal.purge_released_history(
                    address,
                    max_items=self.summary_config.cleanup_batch_size,
                )
                journal_state = self.journal.read_state(address)
                if journal_state.released_through < current.end_sequence:
                    return purged, released, (), (), ()
            current = self.retirement_store.advance(
                current,
                ConversationSummaryRetirementPhase.HISTORY_RELEASED,
                updated_at=now,
            )

        deleted_segments: list[str] = []
        deleted_ranges: list[str] = []
        deleted_archives: list[str] = []
        if current.phase is ConversationSummaryRetirementPhase.HISTORY_RELEASED:
            for source in current.segments:
                existing = self.segment_store.try_read_by_id(address, source.reference.summary_id)
                if existing is not None and existing.digest != source.digest:
                    raise ConversationLifecycleError("retiring Segment Summary digest changed")
                if existing is not None and self.segment_store.delete_by_id(
                    address,
                    source.reference.summary_id,
                ):
                    deleted_segments.append(source.reference.summary_id)
            for source in current.ranges:
                range_existing = self.range_store.try_read(
                    address,
                    ConversationRangeSummaryStage.RANGE,
                    source.reference.summary_id,
                )
                if range_existing is not None and range_existing.digest != source.digest:
                    raise ConversationLifecycleError("retiring Range Summary digest changed")
                if range_existing is not None and self.range_store.delete(
                    address,
                    ConversationRangeSummaryStage.RANGE,
                    source.reference.summary_id,
                ):
                    deleted_ranges.append(source.reference.summary_id)
            archive = self.range_store.try_read(
                address,
                ConversationRangeSummaryStage.ARCHIVE,
                current.archive.reference.summary_id,
            )
            if archive is not None and archive.digest != current.archive.digest:
                raise ConversationLifecycleError("retiring Archive Summary digest changed")
            if archive is not None and self.range_store.delete(
                address,
                ConversationRangeSummaryStage.ARCHIVE,
                current.archive.reference.summary_id,
            ):
                deleted_archives.append(current.archive.reference.summary_id)
            use_store.delete_coverage(
                address,
                start_sequence=current.start_sequence,
                end_sequence=current.end_sequence,
            )
            current = self.retirement_store.advance(
                current,
                ConversationSummaryRetirementPhase.SOURCES_DELETED,
                updated_at=now,
            )
        if current.phase is ConversationSummaryRetirementPhase.SOURCES_DELETED:
            self.retirement_store.complete(current)
        return (
            purged,
            released,
            tuple(deleted_segments),
            tuple(deleted_ranges),
            tuple(deleted_archives),
        )

    def _archive_sources(
        self,
        address: ConversationAddress,
        archive: ConversationRangeSummary,
    ) -> tuple[tuple[ConversationRangeSummary, ...], tuple[ConversationSegmentSummary, ...]]:
        ranges: list[ConversationRangeSummary] = []
        segments: list[ConversationSegmentSummary] = []
        seen_segments: set[str] = set()
        for range_reference in archive.source_refs:
            if range_reference.kind is not ConversationSummarySourceKind.RANGE:
                raise ConversationLifecycleError("Archive Summary contains a non-Range source")
            source_range = self.range_store.try_read(
                address,
                ConversationRangeSummaryStage.RANGE,
                range_reference.summary_id,
            )
            if source_range is None or source_range.digest != range_reference.digest:
                raise ConversationLifecycleError("Archive Summary Range recovery source is missing or changed")
            ranges.append(source_range)
            for segment_reference in source_range.source_refs:
                if segment_reference.kind is not ConversationSummarySourceKind.SEGMENT:
                    raise ConversationLifecycleError("Range Summary contains a non-Segment source")
                if segment_reference.summary_id in seen_segments:
                    raise ConversationLifecycleError("Archive Summary source chain repeats one Segment")
                segment_summary = self.segment_store.try_read_by_id(address, segment_reference.summary_id)
                if segment_summary is None or segment_summary.digest != segment_reference.digest:
                    raise ConversationLifecycleError(
                        "Archive Summary Segment recovery source is missing or changed"
                    )
                seen_segments.add(segment_reference.summary_id)
                segments.append(segment_summary)
        ordered_ranges = tuple(sorted(ranges, key=lambda item: item.start_sequence))
        ordered_segments = tuple(sorted(segments, key=lambda item: item.start_sequence))
        return ordered_ranges, ordered_segments

    def _workflow_is_committed(
        self,
        address: ConversationAddress,
        segment: ConversationSegment,
    ) -> bool:
        job = self.jobs.try_read_source(address, segment.segment_id, segment.digest)
        if job is None:
            raise ConversationLifecycleError("History Segment has no durable MemoryJob")
        if job.status is not MemoryJobStatus.COMMITTED:
            return False
        source = MemoryChangeSource.from_job(job)
        receipt = self.receipts.try_read(source)
        if receipt is None:
            raise ConversationLifecycleError("COMMITTED MemoryJob has no durable change receipt")
        if receipt.state is not MemoryChangeReceiptState.COMMITTED:
            return False
        transaction = self.transaction_journal.try_read(job.transaction_id)
        if transaction is None:
            return True
        if transaction.state is MemoryTransactionJournalState.PREPARED:
            return False
        if transaction.state is MemoryTransactionJournalState.ROLLED_BACK:
            raise ConversationLifecycleError("COMMITTED MemoryJob points to a rolled-back transaction")
        return True

    def _delete_expired_committed_jobs(
        self,
        address: ConversationAddress,
        now: datetime,
    ) -> tuple[int, ...]:
        """只清理已释放来源、已完成回执且不存在恢复日志的过期 Job。"""

        cutoff = now - timedelta(days=self.workflow_config.committed_job_retention_days)
        released_through = self.journal.read_state(address).released_through
        deleted: list[int] = []
        for job in self.jobs.list_for_conversation(address):
            if job.status is not MemoryJobStatus.COMMITTED or job.updated_at > cutoff:
                continue
            if not self._source_history_is_released(job.segment_id, released_through):
                continue
            source = MemoryChangeSource.from_job(job)
            receipt = self.receipts.try_read(source)
            if receipt is None:
                raise ConversationLifecycleError("COMMITTED MemoryJob has no durable change receipt")
            if receipt.state is not MemoryChangeReceiptState.COMMITTED:
                raise ConversationLifecycleError("COMMITTED MemoryJob points to an unfinished change receipt")
            self._discard_terminal_journal(job.transaction_id)
            if self.jobs.discard_committed(job):
                deleted.append(job.memory_sequence)
            if len(deleted) >= self.workflow_config.cleanup_batch_size:
                break
        return tuple(deleted)

    def _delete_expired_committed_receipts(
        self,
        address: ConversationAddress,
        now: datetime,
    ) -> tuple[str, ...]:
        """在 Job 已清理且来源 History 已释放后删除过期的技术审计回执。"""

        cutoff = now - timedelta(days=self.workflow_config.committed_receipt_retention_days)
        released_through = self.journal.read_state(address).released_through
        high_watermark = self.jobs.high_watermark()
        deleted: list[str] = []
        for receipt in self.receipts.list_for_conversation(address):
            if receipt.state is not MemoryChangeReceiptState.COMMITTED:
                continue
            assert receipt.committed_at is not None
            if receipt.committed_at > cutoff:
                continue
            source = receipt.source
            if source.memory_sequence > high_watermark:
                raise ConversationLifecycleError("change receipt sequence exceeds the durable high-watermark")
            if not self._source_history_is_released(source.segment_id, released_through):
                continue
            job = self.jobs.try_read_source(
                address,
                source.segment_id,
                source.source_segment_digest,
            )
            if job is not None:
                continue
            self._discard_terminal_journal(source.transaction_id)
            if self.receipts.discard_committed(receipt):
                deleted.append(source.receipt_id)
            if len(deleted) >= self.workflow_config.cleanup_batch_size:
                break
        return tuple(deleted)

    def _discard_terminal_journal(self, transaction_id: str) -> None:
        journal = self.transaction_journal.try_read(transaction_id)
        if journal is None:
            return
        if journal.state is MemoryTransactionJournalState.PREPARED:
            raise ConversationLifecycleError("terminal workflow still has a PREPARED transaction journal")
        if journal.state is MemoryTransactionJournalState.ROLLED_BACK:
            raise ConversationLifecycleError("committed workflow points to a rolled-back transaction journal")
        self.transaction_journal.discard_terminal(transaction_id)

    @staticmethod
    def _source_history_is_released(segment_id: str, released_through: int) -> bool:
        _start, end = ConversationLayout.segment_range(segment_id)
        return end <= released_through

def _utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("lifecycle now must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("lifecycle now must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "ConversationLifecycleError",
    "ConversationLifecycleMaintenanceResult",
    "ConversationLifecycleManager",
    "MemoryWorkflowLifecycleConfig",
]
