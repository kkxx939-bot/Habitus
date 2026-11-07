"""后台任务执行前的 Conversation outbox 恢复。"""

from __future__ import annotations

from collections.abc import Callable

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
from habitus.pre.conversation import ConversationSegment


def staged_job_segment(
    conversations: ConversationMessageJournal,
    job: MemoryJob,
    *,
    before_history_publish: Callable[[ConversationSegment], None] | None = None,
) -> ConversationSegment:
    """STAGED Job 原边界上的那一段：还没发布就按原边界封存；已经按原边界发布、后面又封了新段（重试用尽成 FAILED 时
    前台照常往后封），就读回那份 history。读回的不是它（那个边界上没有这一段）照常报错。调用方核对段身份与摘要。"""

    address = ConversationAddress(job.conversation_id, job.started_on)
    _start_sequence, end_sequence = conversations.layout.segment_range(job.segment_id)
    archived = conversations.read_state(address).archived_through_sequence
    if archived is not None and archived > end_sequence:
        return conversations.read_segment(address, job.segment_id)
    return conversations.seal(
        address,
        through_sequence=end_sequence,
        before_history_publish=before_history_publish,
    ).segment


class MemoryStagedJobRecovery:
    """幂等补完 STAGED Job 对应的 history 发布与激活。"""

    def __init__(
        self,
        conversations: ConversationMessageJournal,
        jobs: MemoryJobStore,
    ) -> None:
        self.conversations = conversations
        self.jobs = jobs

    def recover(self, job: MemoryJob) -> MemoryJob:
        """非 STAGED Job 原样返回；STAGED Job 必须发布并验证原始片段。"""

        if job.status is not MemoryJobStatus.STAGED:
            return job
        job = self.jobs.require_staged_ready(job)
        try:
            segment = staged_job_segment(self.conversations, job)
            if segment.segment_id != job.segment_id or segment.digest != job.source_segment_digest:
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
        if change_receipts.codec is not transaction.tree.document_codec:
            raise ValueError("transaction recovery requires one shared document codec")
        self.transaction = transaction
        self.change_receipts = change_receipts

    def recover_pending(self) -> tuple[str, ...]:
        """恢复所有 PREPARED 事务，同时保留终态日志供 Job 完成后续步骤。"""

        return self.transaction.recover_pending(discard_terminal=False)

    def inspect(self, job: MemoryJob) -> MemoryTransactionJournalRecord | None:
        """返回可继续补完的 COMMITTED 日志；其他可恢复状态清理后返回 None。"""

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
        """正常执行失败后只清理确定不会提交的准备态回执。

        COMMITTED 日志与 PREPARED 日志都保留回执：PREPARED 表示 L2 是否已写完尚未确定，
        下一次执行先由 `recover_pending` 判定，提交了就靠这份回执补完，回滚了再由 `inspect` 清理。
        """

        journal = self.transaction.journal.try_read(source.transaction_id)
        if journal is not None and journal.state in {
            MemoryTransactionJournalState.COMMITTED,
            MemoryTransactionJournalState.PREPARED,
        }:
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


__all__ = ["MemoryJobTransactionRecovery", "MemoryStagedJobRecovery", "staged_job_segment"]
