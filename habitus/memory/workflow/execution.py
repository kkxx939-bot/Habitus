"""尚未提交的 MemoryJob 执行阶段。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from habitus.foundation.observability import NullObserver, Observer, observe_operation
from habitus.memory.conversation import ConversationAddress, ConversationMessageJournal
from habitus.memory.conversation.summary import ConversationSummaryService
from habitus.memory.editor.engine import MemoryEditor
from habitus.memory.editor.extraction import MemoryTurnContext
from habitus.memory.editor.transaction import MemoryCommitResult
from habitus.memory.editor.transaction_log import (
    MemoryTransactionJournalRecord,
    MemoryTransactionJournalState,
)
from habitus.memory.workflow.jobs import (
    MemoryJobError,
    MemoryJobLease,
    MemoryJobStore,
)
from habitus.memory.workflow.planning import MemorySegmentProductBuilder
from habitus.memory.workflow.receipt import MemoryChangeReceiptStore, MemoryChangeSource
from habitus.pre.conversation import ConversationSegment

#: 观察事件文本属性的上限（``foundation.observability.ObservationEvent``）：剔除原因拼起来超出时截断。
_OBSERVED_TEXT_MAX_CHARS = 256


@dataclass(frozen=True)
class MemoryJobCommit:
    """正常执行阶段已经耐久发布的 L2 事务。"""

    commit: MemoryCommitResult
    journal: MemoryTransactionJournalRecord
    summary_generated: bool

    def __post_init__(self) -> None:
        if self.journal.state is not MemoryTransactionJournalState.COMMITTED:
            raise ValueError("job commit requires a COMMITTED transaction journal")
        if self.commit.transaction_id != self.journal.transaction_id:
            raise ValueError("commit result and transaction journal identities differ")


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
        observer: Observer | None = None,
    ) -> None:
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
        self.clock = clock or (lambda: datetime.now(UTC))
        self.observer = observer or NullObserver()

    async def execute(self, lease: MemoryJobLease) -> MemoryJobCommit:
        """只接受当前 claim，成功时保证事务日志已经到达 COMMITTED。"""

        job = self.jobs.assert_current(lease)
        address = ConversationAddress(job.conversation_id, job.started_on)
        with observe_operation(self.observer, "workflow", "segment_read"):
            segment = self.conversations.read_segment(address, job.segment_id)
        if segment.digest != job.source_segment_digest:
            raise MemoryJobError("sealed ConversationSegment digest does not match its MemoryJob")
        editor_segment, turn_context = self._editor_input(address, segment)
        summary_existed = self.summary_service.store.try_read(address, segment) is not None
        with observe_operation(self.observer, "workflow", "memory_plan") as observed:
            products = await self.segment_products.build(
                address,
                segment,
                editor_segment=editor_segment,
                turn_context=turn_context,
            )
            extraction = products.editor_plan.extraction
            if extraction is not None:
                observed["retrieval_ended_at_limit"] = extraction.retrieval_ended_at_limit
                observed["read_only_noop_count"] = len(extraction.mutations.read_only_targets)
                observed["source_rejected_count"] = len(extraction.source_rejections)
                if extraction.source_rejections:
                    reasons = "; ".join(item.describe() for item in extraction.source_rejections)
                    observed["source_rejected_reasons"] = reasons[:_OBSERVED_TEXT_MAX_CHARS]
        products.summary.require_matches_source(segment)
        self.jobs.assert_current(lease)

        source = MemoryChangeSource.from_job(
            job,
            editor_segment=editor_segment,
        )
        with observe_operation(self.observer, "workflow", "receipt_prepare"):
            self.change_receipts.prepare(
                source,
                products.editor_plan,
                timestamp=self._timestamp(),
            )
        self.jobs.assert_current(lease)
        with observe_operation(self.observer, "workflow", "l2_commit"):
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

    def _editor_input(
        self,
        address: ConversationAddress,
        segment: ConversationSegment,
    ) -> tuple[ConversationSegment | None, MemoryTurnContext | None]:
        """这一段的 Job 抽取什么：整轮放得下就照旧等到轮尾一次抽整轮（轮中间的段只写摘要）；放不下就每段抽自己，
        带上这一轮开头的人话与前面各段的段摘要（照 OpenViking ``retention.py`` ``_partial_latest_turn_plan``）。

        判据只看整轮的消息与配置（``MemoryEditor.turn_fits_whole``），同一轮里每一段的 Job 判出来一样：要么只有轮尾那段
        抽整轮，要么每段各抽各的，不漏不重。
        """

        turn = self.conversations.read_editor_turn(address, segment)
        if self.editor.turn_fits_whole(turn.whole):
            return (None if segment.ends_mid_turn else turn.whole), None
        summaries = tuple(
            summary
            for summary in (self.summary_service.store.try_read(address, earlier) for earlier in turn.earlier)
            if summary is not None
        )
        context = MemoryTurnContext(
            opening_prompts=() if turn.source_index == 0 else turn.opening_prompts,
            earlier_summaries=summaries,
            earlier_segment_count=len(turn.earlier),
        )
        return segment, context

    def _timestamp(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory workflow clock must return timezone-aware datetime")
        return value.astimezone(UTC)


__all__ = ["MemoryJobCommit", "MemoryJobExecutor"]
