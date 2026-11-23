"""MemoryJob 卡片写完之前的执行阶段：摘要、记忆计划、写卡片。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.foundation.observability import NullObserver, Observer, observe_operation
from habitus.memory.conversation import ConversationAddress, ConversationMessageJournal
from habitus.memory.conversation.summary import ConversationSummaryService
from habitus.memory.editor.commit import MemoryCommitResult
from habitus.memory.editor.engine import MemoryEditor
from habitus.memory.editor.extraction import MemoryTurnContext
from habitus.memory.workflow.failure import MemoryJobSkipped, skip_reason
from habitus.memory.workflow.jobs import (
    MemoryJob,
    MemoryJobError,
    MemoryJobLease,
    MemoryJobStore,
)
from habitus.memory.workflow.planning import MemorySegmentProductBuilder
from habitus.pre.conversation import ConversationSegment

#: 观察事件文本属性的上限（``foundation.observability.ObservationEvent``）：剔除原因拼起来超出时截断。
_OBSERVED_TEXT_MAX_CHARS = 256


@dataclass(frozen=True)
class MemoryJobWrite:
    """正常执行阶段已经写进记忆树的卡片，以及记下"卡片已写完"之后的任务。"""

    commit: MemoryCommitResult
    job: MemoryJob
    summary_generated: bool

    def __post_init__(self) -> None:
        if not self.job.cards_written:
            raise ValueError("job write requires a job whose cards are written")


class MemoryJobExecutor:
    """生成摘要与编辑计划（摘要生成即落盘），把卡片写进记忆树，再在任务文件里记下"卡片已写完"。"""

    def __init__(
        self,
        conversations: ConversationMessageJournal,
        editor: MemoryEditor,
        summary_service: ConversationSummaryService,
        jobs: MemoryJobStore,
        *,
        observer: Observer | None = None,
    ) -> None:
        if summary_service.store.layout.root != conversations.layout.root:
            raise ValueError("summary service and conversations must use one conversation root")
        self.conversations = conversations
        self.editor = editor
        self.summary_service = summary_service
        self.jobs = jobs
        self.segment_products = MemorySegmentProductBuilder(summary_service, editor)
        self.observer = observer or NullObserver()

    async def execute(self, lease: MemoryJobLease) -> MemoryJobWrite:
        """只接受当前 claim；成功时卡片已全部写进记忆树，任务文件已记下这一阶段与改动的地址。"""

        job = self.jobs.assert_current(lease)
        address = ConversationAddress(job.conversation_id, job.started_on)
        with observe_operation(self.observer, "workflow", "segment_read"):
            segment = self.conversations.read_segment(address, job.segment_id)
        if segment.digest != job.source_segment_digest:
            raise MemoryJobError("sealed ConversationSegment digest does not match its MemoryJob")
        editor_segment, turn_context = self._editor_input(address, segment)
        summary_existed = self.summary_service.store.try_read(address, segment) is not None
        with observe_operation(self.observer, "workflow", "memory_plan") as observed:
            try:
                products = await self.segment_products.build(
                    address,
                    segment,
                    editor_segment=editor_segment,
                    turn_context=turn_context,
                )
            except Exception as exc:
                reason = skip_reason(exc)
                if reason is None:
                    raise
                observed["skipped"] = reason[:_OBSERVED_TEXT_MAX_CHARS]
                raise MemoryJobSkipped(reason) from exc
            extraction = products.editor_plan.extraction
            if extraction is not None:
                observed["retrieval_ended_at_limit"] = extraction.retrieval_ended_at_limit
                observed["source_rejected_count"] = len(extraction.source_rejections)
                if extraction.source_rejections:
                    reasons = "; ".join(item.describe() for item in extraction.source_rejections)
                    observed["source_rejected_reasons"] = reasons[:_OBSERVED_TEXT_MAX_CHARS]
        products.summary.require_matches_source(segment)
        self.jobs.assert_current(lease)

        with observe_operation(self.observer, "workflow", "l2_write"):
            commit = self.editor.commit(products.editor_plan)
        with observe_operation(self.observer, "workflow", "cards_written"):
            written = self.jobs.record_written(
                lease,
                changed_uris=tuple(str(uri) for uri in commit.written_uris),
                deleted_uris=tuple(str(uri) for uri in commit.deleted_uris),
            )
        return MemoryJobWrite(commit=commit, job=written, summary_generated=not summary_existed)

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


__all__ = ["MemoryJobExecutor", "MemoryJobWrite"]
