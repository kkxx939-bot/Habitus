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
    ConversationSummaryStore,
)
from memory.conversation.indexing import PersistentConversationSummaryVectorIndex
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
    ConversationSummarySourceKind,
    ConversationSummarySourceRef,
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
    """先生成不可变父摘要，再按最旧连续前缀统一释放原文和被覆盖摘要。"""

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
        self.summary_config = resolved_summary_config
        self.workflow_config = workflow_config or MemoryWorkflowLifecycleConfig()

    async def maintain_once(
        self,
        address: ConversationAddress,
        *,
        now: datetime | None = None,
    ) -> ConversationLifecycleMaintenanceResult:
        """每次至多生成一个父摘要，并有界清理一批已经安全被覆盖的来源。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        current_time = _utc_datetime(now or datetime.now(timezone.utc))
        compaction = await self.compactor.compact_once(address, now=current_time)
        await self.summary_vector_index.synchronize(address)
        purged = self.journal.purge_released_history(
            address,
            max_items=self.summary_config.cleanup_batch_size,
        )
        released: tuple[str, ...] = ()
        deleted_segments: tuple[str, ...] = ()
        deleted_ranges: tuple[str, ...] = ()
        if self.summary_config.enabled:
            ranges = self.range_store.list(address, ConversationRangeSummaryStage.RANGE)
            archives = self.range_store.list(address, ConversationRangeSummaryStage.ARCHIVE)
            segment_parents = self._parent_map(
                ranges,
                expected_kind=ConversationSummarySourceKind.SEGMENT,
            )
            archive_parents = self._parent_map(
                archives,
                expected_kind=ConversationSummarySourceKind.RANGE,
            )
            releasable = self._releasable_history_prefix(
                address,
                ranges=segment_parents,
                now=current_time,
                limit=self.summary_config.cleanup_batch_size,
            )
            released = self.journal.release_history_prefix(address, releasable) if releasable else ()
            state = self.journal.read_state(address)
            deleted_segments = self._delete_superseded_segment_summaries(
                address,
                ranges=ranges,
                released_through=state.released_through,
                now=current_time,
            )
            deleted_ranges = self._delete_superseded_ranges(
                address,
                archives=archives,
                archive_parents=archive_parents,
                released_through=state.released_through,
                now=current_time,
            )
        deleted_jobs = self._delete_expired_committed_jobs(address, current_time)
        deleted_receipts = self._delete_expired_committed_receipts(address, current_time)
        return ConversationLifecycleMaintenanceResult(
            compaction=compaction,
            summary_indexed=True,
            purged_history_segment_ids=purged,
            released_history_segment_ids=released,
            deleted_segment_summary_ids=deleted_segments,
            deleted_range_summary_ids=deleted_ranges,
            deleted_memory_job_sequences=deleted_jobs,
            deleted_memory_receipt_ids=deleted_receipts,
        )

    def _releasable_history_prefix(
        self,
        address: ConversationAddress,
        *,
        ranges: dict[str, tuple[ConversationRangeSummary, ConversationSummarySourceRef]],
        now: datetime,
        limit: int,
    ) -> tuple[ConversationSegment, ...]:
        candidates: list[ConversationSegment] = []
        safe_count = 0
        max_wait_cutoff = now - timedelta(days=self.summary_config.segment_to_range.max_wait_days)
        for segment in self.journal.list_history(address):
            summary = self.segment_store.try_read(address, segment)
            if summary is None:
                raise ConversationLifecycleError("retained History Segment has no valid one-to-one Segment Summary")
            parent_entry = ranges.get(summary.segment_id)
            if parent_entry is None:
                if summary.ended_at > max_wait_cutoff:
                    break
            else:
                parent, reference = parent_entry
                if reference.digest != summary.digest:
                    raise ConversationLifecycleError("Range Summary source digest does not match Segment Summary")
                if not self._parent_is_mature(parent, now):
                    break
            if not self._workflow_is_committed(address, segment):
                break
            candidates.append(segment)
            if not segment.ends_mid_turn:
                safe_count = len(candidates)
            if safe_count >= limit:
                break
        return tuple(candidates[:safe_count])

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

    def _delete_superseded_segment_summaries(
        self,
        address: ConversationAddress,
        *,
        ranges: tuple[ConversationRangeSummary, ...],
        released_through: int,
        now: datetime,
    ) -> tuple[str, ...]:
        deleted: list[str] = []
        for parent in ranges:
            if not self._parent_is_mature(parent, now):
                continue
            for reference in parent.source_refs:
                if reference.end_sequence > released_through:
                    continue
                source = self.segment_store.try_read_by_id(address, reference.summary_id)
                if source is None:
                    continue
                if source.digest != reference.digest:
                    raise ConversationLifecycleError("Segment Summary digest changed before cleanup")
                if self.segment_store.delete_by_id(address, reference.summary_id):
                    deleted.append(reference.summary_id)
                if len(deleted) >= self.summary_config.cleanup_batch_size:
                    return tuple(deleted)
        return tuple(deleted)

    def _delete_superseded_ranges(
        self,
        address: ConversationAddress,
        *,
        archives: tuple[ConversationRangeSummary, ...],
        archive_parents: dict[str, tuple[ConversationRangeSummary, ConversationSummarySourceRef]],
        released_through: int,
        now: datetime,
    ) -> tuple[str, ...]:
        deleted: list[str] = []
        for range_id, (parent, reference) in sorted(
            archive_parents.items(),
            key=lambda item: item[1][1].start_sequence,
        ):
            if parent not in archives or not self._parent_is_mature(parent, now):
                continue
            source = self.range_store.try_read(
                address,
                ConversationRangeSummaryStage.RANGE,
                range_id,
            )
            if source is None:
                continue
            if source.digest != reference.digest:
                raise ConversationLifecycleError("Range Summary digest changed before cleanup")
            if any(
                child.end_sequence > released_through
                or self.segment_store.try_read_by_id(address, child.summary_id) is not None
                for child in source.source_refs
            ):
                continue
            if self.range_store.delete(address, ConversationRangeSummaryStage.RANGE, range_id):
                deleted.append(range_id)
            if len(deleted) >= self.summary_config.cleanup_batch_size:
                break
        return tuple(deleted)

    def _parent_is_mature(self, parent: ConversationRangeSummary, now: datetime) -> bool:
        cutoff = now - timedelta(days=self.summary_config.superseded_source_retention_days)
        return parent.generated_at <= cutoff

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

    @staticmethod
    def _parent_map(
        parents: tuple[ConversationRangeSummary, ...],
        *,
        expected_kind: ConversationSummarySourceKind,
    ) -> dict[str, tuple[ConversationRangeSummary, ConversationSummarySourceRef]]:
        result: dict[str, tuple[ConversationRangeSummary, ConversationSummarySourceRef]] = {}
        for parent in parents:
            for reference in parent.source_refs:
                if reference.kind is not expected_kind:
                    raise ConversationLifecycleError("summary parent contains an invalid source kind")
                if reference.summary_id in result:
                    raise ConversationLifecycleError("one summary source is covered by multiple parents")
                result[reference.summary_id] = (parent, reference)
        return result


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
