"""后台任务执行前的 Conversation outbox 恢复。"""

from __future__ import annotations

from habitus.memory.conversation import ConversationAddress, ConversationMessageJournal
from habitus.memory.editor.transaction import MemoryCommitTransaction
from habitus.memory.editor.transaction_log import (
    MemoryTransactionJournalError,
    MemoryTransactionJournalRecord,
    MemoryTransactionJournalState,
)
from habitus.memory.workflow.jobs import (
    MemoryJob,
    MemoryJobError,
    MemoryJobExecutionError,
    MemoryJobNotReadyError,
    MemoryJobStatus,
    MemoryJobStore,
)
from habitus.memory.workflow.receipt import (
    MemoryChangeReceiptState,
    MemoryChangeReceiptStore,
    MemoryChangeSource,
)


class MemoryStagedJobRecovery:
    """幂等补完 STAGED Job 对应的 history 发布与激活。"""

    def __init__(
        self,
        conversations: ConversationMessageJournal,
        jobs: MemoryJobStore,
    ) -> None:
        if not isinstance(conversations, ConversationMessageJournal):
            raise TypeError("conversations must be ConversationMessageJournal")
        if not isinstance(jobs, MemoryJobStore):
            raise TypeError("jobs must be MemoryJobStore")
        self.conversations = conversations
        self.jobs = jobs

    def recover(self, job: MemoryJob) -> MemoryJob:
        """非 STAGED Job 原样返回；STAGED Job 必须发布并验证原始片段。"""

        if not isinstance(job, MemoryJob):
            raise TypeError("job must be MemoryJob")
        if job.status is not MemoryJobStatus.STAGED:
            return job
        job = self.jobs.require_staged_ready(job)
        try:
            address = ConversationAddress(job.conversation_id, job.started_on)
            _start_sequence, end_sequence = self.conversations.layout.segment_range(job.segment_id)
            sealed = self.conversations.seal(address, through_sequence=end_sequence)
            if sealed.segment.segment_id != job.segment_id or sealed.segment.digest != job.source_segment_digest:
                raise MemoryJobError("recovered ConversationSegment does not match its staged MemoryJob")
            return self.jobs.activate(job)
        except MemoryJobNotReadyError:
            raise
        except Exception as exc:
            failed = self.jobs.record_staged_failure(job, exc)
            raise MemoryJobExecutionError(
                "staged memory job could not publish and verify its ConversationSegment",
                job=failed,
            ) from exc


class MemoryJobTransactionRecovery:
    """恢复事务日志并清理尚未进入正常执行的 Receipt 状态。"""

    def __init__(
        self,
        transaction: MemoryCommitTransaction,
        change_receipts: MemoryChangeReceiptStore,
    ) -> None:
        if not isinstance(transaction, MemoryCommitTransaction):
            raise TypeError("transaction must be MemoryCommitTransaction")
        if not isinstance(change_receipts, MemoryChangeReceiptStore):
            raise TypeError("change_receipts must be MemoryChangeReceiptStore")
        if change_receipts.codec is not transaction.tree.document_codec:
            raise ValueError("transaction recovery requires one shared document codec")
        self.transaction = transaction
        self.change_receipts = change_receipts

    def recover_pending(self) -> tuple[str, ...]:
        """恢复所有 PREPARED 事务，同时保留终态日志供 Job 完成后续步骤。"""

        return self.transaction.recover_pending(discard_terminal=False)

    def inspect(self, job: MemoryJob) -> MemoryTransactionJournalRecord | None:
        """返回可继续补完的 COMMITTED 日志；其他可恢复状态清理后返回 None。"""

        if not isinstance(job, MemoryJob):
            raise TypeError("job must be MemoryJob")
        source = MemoryChangeSource.from_job(job)
        journal = self.transaction.journal.try_read(job.transaction_id)
        if journal is not None and journal.state is MemoryTransactionJournalState.COMMITTED:
            return journal
        if journal is not None and journal.state is MemoryTransactionJournalState.ROLLED_BACK:
            self.change_receipts.discard_prepared(source)
            self.transaction.journal.discard_terminal(job.transaction_id)
        elif journal is None:
            current_receipt = self.change_receipts.try_read(source)
            if current_receipt is not None:
                if current_receipt.state is MemoryChangeReceiptState.COMMITTED:
                    raise MemoryJobError("committed change receipt exists without its recoverable transaction journal")
                self.change_receipts.discard_prepared(source)
        return None

    def discard_uncommitted(self, source: MemoryChangeSource) -> None:
        """正常执行失败后只清理尚未对应 COMMITTED 事务的准备态回执。"""

        if not isinstance(source, MemoryChangeSource):
            raise TypeError("source must be MemoryChangeSource")
        journal = self.transaction.journal.try_read(source.transaction_id)
        if journal is not None and journal.state is MemoryTransactionJournalState.COMMITTED:
            return
        current = self.change_receipts.try_read(source)
        if current is not None and current.state is MemoryChangeReceiptState.PREPARED:
            self.change_receipts.discard_prepared(source)
        if journal is not None and journal.state is MemoryTransactionJournalState.ROLLED_BACK:
            self.transaction.journal.discard_terminal(source.transaction_id)

    def discard_terminal(self, transaction_id: str) -> bool:
        """尽力清理已经完成的事务日志，并报告是否成功。"""

        try:
            self.transaction.journal.discard_terminal(transaction_id)
        except MemoryTransactionJournalError:
            return False
        return True


__all__ = ["MemoryJobTransactionRecovery", "MemoryStagedJobRecovery"]
