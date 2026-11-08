"""COMMITTED 记忆工作流的终态判定，以及过期 Job、回执与终态事务日志的有界清理。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from habitus.memory.conversation import ConversationAddress, ConversationLayout, ConversationMessageJournal
from habitus.memory.editor import MemoryTransactionJournal, MemoryTransactionJournalState
from habitus.memory.workflow.jobs import MemoryJobStatus, MemoryJobStore
from habitus.memory.workflow.lifecycle_errors import ConversationLifecycleError
from habitus.memory.workflow.receipt import (
    MemoryChangeReceiptState,
    MemoryChangeReceiptStore,
    MemoryChangeSource,
)
from habitus.pre.conversation import ConversationSegment

if TYPE_CHECKING:
    from habitus.memory.workflow.lifecycle import MemoryWorkflowLifecycleConfig


class CommittedWorkflowCleanup:
    """只认 Job、回执与事务日志都已终态的工作流；过期清理按配置批次有界执行。"""

    def __init__(
        self,
        journal: ConversationMessageJournal,
        jobs: MemoryJobStore,
        receipts: MemoryChangeReceiptStore,
        transaction_journal: MemoryTransactionJournal,
        config: MemoryWorkflowLifecycleConfig,
    ) -> None:
        self.journal = journal
        self.jobs = jobs
        self.receipts = receipts
        self.transaction_journal = transaction_journal
        self.config = config

    def is_committed(
        self,
        address: ConversationAddress,
        segment: ConversationSegment,
    ) -> bool:
        """History 来源的记忆工作流是否已完整终态；未完成返回 False，状态矛盾明确失败。"""

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

    def delete_expired_jobs(
        self,
        address: ConversationAddress,
        now: datetime,
    ) -> tuple[int, ...]:
        """只清理已释放来源、已完成回执且不存在恢复日志的过期 Job。"""

        cutoff = now - timedelta(days=self.config.committed_job_retention_days)
        released_through = self.journal.read_state(address).released_through
        deleted: list[int] = []
        for job in self.jobs.list_for_conversation(address):
            if job.status is not MemoryJobStatus.COMMITTED or job.updated_at > cutoff:
                continue
            if not self.source_history_is_released(job.segment_id, released_through):
                continue
            source = MemoryChangeSource.from_job(job)
            receipt = self.receipts.try_read(source)
            if receipt is None:
                raise ConversationLifecycleError("COMMITTED MemoryJob has no durable change receipt")
            if receipt.state is not MemoryChangeReceiptState.COMMITTED:
                raise ConversationLifecycleError("COMMITTED MemoryJob points to an unfinished change receipt")
            self.discard_terminal_journal(job.transaction_id)
            if self.jobs.discard_committed(job):
                deleted.append(job.memory_sequence)
            if len(deleted) >= self.config.cleanup_batch_size:
                break
        return tuple(deleted)

    def delete_expired_receipts(
        self,
        address: ConversationAddress,
        now: datetime,
    ) -> tuple[str, ...]:
        """在 Job 已清理且来源 History 已释放后删除过期的技术审计回执。"""

        cutoff = now - timedelta(days=self.config.committed_receipt_retention_days)
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
            if not self.source_history_is_released(source.segment_id, released_through):
                continue
            job = self.jobs.try_read_source(
                address,
                source.segment_id,
                source.source_segment_digest,
            )
            if job is not None:
                continue
            self.discard_terminal_journal(source.transaction_id)
            if self.receipts.discard_committed(receipt):
                deleted.append(source.receipt_id)
            if len(deleted) >= self.config.cleanup_batch_size:
                break
        return tuple(deleted)

    def discard_terminal_journal(self, transaction_id: str) -> None:
        journal = self.transaction_journal.try_read(transaction_id)
        if journal is None:
            return
        if journal.state is MemoryTransactionJournalState.PREPARED:
            raise ConversationLifecycleError("terminal workflow still has a PREPARED transaction journal")
        if journal.state is MemoryTransactionJournalState.ROLLED_BACK:
            raise ConversationLifecycleError("committed workflow points to a rolled-back transaction journal")
        self.transaction_journal.discard_terminal(transaction_id)

    @staticmethod
    def source_history_is_released(segment_id: str, released_through: int) -> bool:
        _start, end = ConversationLayout.segment_range(segment_id)
        return end <= released_through


__all__ = ["CommittedWorkflowCleanup"]
