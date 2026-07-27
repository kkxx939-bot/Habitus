"""尚未提交的 MemoryJob 执行阶段。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from memory.conversation import ConversationAddress, ConversationMessageJournal
from memory.conversation.summary import ConversationSummaryService
from memory.editor.engine import MemoryEditor
from memory.editor.transaction import MemoryCommitResult
from memory.editor.transaction_log import (
    MemoryTransactionJournalRecord,
    MemoryTransactionJournalState,
)
from memory.workflow.jobs import (
    MemoryJobError,
    MemoryJobLease,
    MemoryJobStore,
)
from memory.workflow.planning import MemorySegmentProductBuilder
from memory.workflow.receipt import MemoryChangeReceiptStore, MemoryChangeSource


@dataclass(frozen=True)
class MemoryJobCommit:
    """正常执行阶段已经耐久发布的 L2 事务。"""

    commit: MemoryCommitResult
    journal: MemoryTransactionJournalRecord
    summary_generated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.commit, MemoryCommitResult):
            raise TypeError("commit must be MemoryCommitResult")
        if not isinstance(self.journal, MemoryTransactionJournalRecord):
            raise TypeError("journal must be MemoryTransactionJournalRecord")
        if self.journal.state is not MemoryTransactionJournalState.COMMITTED:
            raise ValueError("job commit requires a COMMITTED transaction journal")
        if self.commit.transaction_id != self.journal.transaction_id:
            raise ValueError("commit result and transaction journal identities differ")
        if not isinstance(self.summary_generated, bool):
            raise TypeError("summary_generated must be boolean")


class MemoryJobExecutor:
    """生成 Summary 与编辑计划，并把准备态回执和 L2 事务耐久发布。"""

    def __init__(
        self,
        conversations: ConversationMessageJournal,
        editor: MemoryEditor,
        summary_service: ConversationSummaryService,
        change_receipts: MemoryChangeReceiptStore,
        jobs: MemoryJobStore,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(conversations, ConversationMessageJournal):
            raise TypeError("conversations must be ConversationMessageJournal")
        if not isinstance(editor, MemoryEditor):
            raise TypeError("editor must be MemoryEditor")
        if not isinstance(summary_service, ConversationSummaryService):
            raise TypeError("summary_service must be ConversationSummaryService")
        if not isinstance(change_receipts, MemoryChangeReceiptStore):
            raise TypeError("change_receipts must be MemoryChangeReceiptStore")
        if not isinstance(jobs, MemoryJobStore):
            raise TypeError("jobs must be MemoryJobStore")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if summary_service.store.layout.root != conversations.layout.root:
            raise ValueError("summary service and conversations must use one conversation root")
        if change_receipts.codec is not editor.transaction.tree.document_codec:
            raise ValueError("change receipts and MemoryEditor must share one document codec")
        self.conversations = conversations
        self.editor = editor
        self.summary_service = summary_service
        self.change_receipts = change_receipts
        self.jobs = jobs
        self.segment_products = MemorySegmentProductBuilder(summary_service, editor)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    async def execute(self, lease: MemoryJobLease) -> MemoryJobCommit:
        """只接受当前 claim，成功时保证事务日志已经到达 COMMITTED。"""

        if not isinstance(lease, MemoryJobLease):
            raise TypeError("lease must be a MemoryJobLease")
        job = self.jobs.assert_current(lease)
        address = ConversationAddress(job.conversation_id, job.started_on)
        segment = self.conversations.read_segment(address, job.segment_id)
        if segment.digest != job.source_segment_digest:
            raise MemoryJobError("sealed ConversationSegment digest does not match its MemoryJob")
        summary_existed = self.summary_service.store.try_read(address, segment) is not None
        products = await self.segment_products.build(address, segment)
        products.summary.require_matches_source(segment)
        self.jobs.assert_current(lease)

        source = MemoryChangeSource.from_job(job)
        self.change_receipts.prepare(
            source,
            products.editor_plan,
            timestamp=self._timestamp(),
        )
        self.jobs.assert_current(lease)
        commit = self.editor.commit(
            products.editor_plan,
            transaction_id=job.transaction_id,
            retain_transaction_journal=True,
        )
        self.jobs.assert_current(lease)
        journal = self.editor.transaction.journal.read(job.transaction_id)
        return MemoryJobCommit(
            commit=commit,
            journal=journal,
            summary_generated=not summary_existed,
        )

    def _timestamp(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime):
            raise TypeError("memory workflow clock must return datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory workflow clock must return timezone-aware datetime")
        return value.astimezone(timezone.utc)


__all__ = ["MemoryJobCommit", "MemoryJobExecutor"]
