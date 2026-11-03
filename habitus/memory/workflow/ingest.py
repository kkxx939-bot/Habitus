"""将 Conversation 安全接入有序记忆任务的前台编排。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from habitus.memory.conversation import (
    ConversationAddress,
    ConversationAppendResult,
    ConversationIngressRequest,
    ConversationMessageJournal,
)
from habitus.memory.conversation.retention import (
    ConversationRetentionPlan,
    ConversationRetentionPlanner,
    ConversationToolResultReducer,
)
from habitus.memory.conversation.segmentation import (
    ConversationBoundaryHints,
    ConversationMessageChunker,
)
from habitus.memory.workflow.jobs import MemoryJob, MemoryJobError, MemoryJobStatus, MemoryJobStore
from habitus.memory.workflow.recovery import staged_job_segment
from habitus.pre.conversation import (
    ConversationBatch,
    ConversationMessageRole,
    ConversationSegment,
)


@dataclass(frozen=True)
class ConversationMemoryIngestResult:
    """一次追加后实际排队的 Segment 与最终保留判断。"""

    append: ConversationAppendResult
    jobs: tuple[MemoryJob, ...]
    retention: ConversationRetentionPlan

    def __post_init__(self) -> None:
        if not isinstance(self.append, ConversationAppendResult):
            raise TypeError("append must be ConversationAppendResult")
        if not isinstance(self.jobs, tuple) or any(not isinstance(job, MemoryJob) for job in self.jobs):
            raise TypeError("jobs must contain MemoryJob values")
        if not isinstance(self.retention, ConversationRetentionPlan):
            raise TypeError("retention must be ConversationRetentionPlan")


@dataclass(frozen=True)
class ConversationMemoryFlushResult:
    """一次显式 flush 实际封存的 Segment 任务与最终 live 状态。"""

    jobs: tuple[MemoryJob, ...]
    retention: ConversationRetentionPlan

    def __post_init__(self) -> None:
        if not isinstance(self.jobs, tuple) or any(not isinstance(job, MemoryJob) for job in self.jobs):
            raise TypeError("jobs must contain MemoryJob values")
        if not isinstance(self.retention, ConversationRetentionPlan):
            raise TypeError("retention must be ConversationRetentionPlan")


class ConversationMemoryEnqueuer:
    """把 history 封存结果幂等接入 memory-root 顺序队列。"""

    def __init__(
        self,
        conversations: ConversationMessageJournal,
        jobs: MemoryJobStore,
        retention_planner: ConversationRetentionPlanner | None = None,
        tool_result_reducer: ConversationToolResultReducer | None = None,
        message_chunker: ConversationMessageChunker | None = None,
    ) -> None:
        if not isinstance(conversations, ConversationMessageJournal):
            raise TypeError("conversations must be a ConversationMessageJournal")
        if not isinstance(jobs, MemoryJobStore):
            raise TypeError("jobs must be a MemoryJobStore")
        if retention_planner is not None and not isinstance(
            retention_planner,
            ConversationRetentionPlanner,
        ):
            raise TypeError("retention_planner must be ConversationRetentionPlanner")
        if tool_result_reducer is not None and not isinstance(
            tool_result_reducer,
            ConversationToolResultReducer,
        ):
            raise TypeError("tool_result_reducer must be ConversationToolResultReducer")
        if message_chunker is not None and not isinstance(
            message_chunker,
            ConversationMessageChunker,
        ):
            raise TypeError("message_chunker must be ConversationMessageChunker")
        self.conversations = conversations
        self.jobs = jobs
        self.retention_planner = retention_planner or ConversationRetentionPlanner()
        self.tool_result_reducer = tool_result_reducer or ConversationToolResultReducer(self.retention_planner.config)
        self.message_chunker = message_chunker or ConversationMessageChunker(
            max_message_tokens=self.retention_planner.config.max_segment_tokens,
            token_estimator=self.retention_planner.token_estimator,
        )

    def append_and_maybe_enqueue(
        self,
        address: ConversationAddress,
        batch: ConversationBatch,
        *,
        after_turn: bool = False,
        omit_tool_call_ids: frozenset[str] = frozenset(),
        ingress: ConversationIngressRequest | None = None,
    ) -> ConversationMemoryIngestResult:
        """追加消息；只有 afterTurn 边界才可能自动提交完整旧轮次。"""

        if not isinstance(batch, ConversationBatch):
            raise TypeError("batch must be ConversationBatch")
        if not isinstance(after_turn, bool):
            raise TypeError("after_turn must be boolean")
        if not isinstance(omit_tool_call_ids, frozenset) or any(
            not isinstance(identifier, str) or not identifier for identifier in omit_tool_call_ids
        ):
            raise TypeError("omit_tool_call_ids must be a frozenset of non-empty strings")
        appended = self.append(
            address,
            batch,
            omit_tool_call_ids=omit_tool_call_ids,
            ingress=ingress,
        )
        queued, final_plan = self.enqueue_ready_segments(
            address,
            after_turn=after_turn,
            flush=False,
        )
        return ConversationMemoryIngestResult(
            append=appended,
            jobs=queued,
            retention=final_plan,
        )

    def append(
        self,
        address: ConversationAddress,
        batch: ConversationBatch,
        *,
        omit_tool_call_ids: frozenset[str] = frozenset(),
        ingress: ConversationIngressRequest | None = None,
    ) -> ConversationAppendResult:
        """降载工具结果、切分超长文本、把超过一段上限的工具交换抽到放得下，再写入唯一 live 主链。"""

        if not isinstance(batch, ConversationBatch):
            raise TypeError("batch must be ConversationBatch")
        if not isinstance(omit_tool_call_ids, frozenset) or any(
            not isinstance(identifier, str) or not identifier for identifier in omit_tool_call_ids
        ):
            raise TypeError("omit_tool_call_ids must be a frozenset of non-empty strings")
        reduced = ConversationBatch(
            batch.conversation_id,
            tuple(
                self.tool_result_reducer.reduce(
                    message,
                    force_omit=message.tool_call_id in omit_tool_call_ids,
                )
                if message.role is ConversationMessageRole.TOOL_RESULT
                else message
                for message in batch.messages
            ),
        )
        normalized = self.message_chunker.normalize(reduced)
        fitted = self.tool_result_reducer.fit_exchanges(
            normalized,
            self.retention_planner,
            originals={
                message.message_id: message
                for message in batch.messages
                if message.role in {ConversationMessageRole.TOOL_CALL, ConversationMessageRole.TOOL_RESULT}
            },
            omit_tool_call_ids=omit_tool_call_ids,
        )
        return self.conversations.append(address, fitted, ingress=ingress)

    def flush(self, address: ConversationAddress) -> ConversationMemoryFlushResult:
        """显式提交全部剩余完整轮次，不关闭或冻结 Conversation。"""

        queued, final_plan = self.enqueue_ready_segments(
            address,
            after_turn=False,
            flush=True,
        )
        return ConversationMemoryFlushResult(jobs=queued, retention=final_plan)

    def preview_retention(
        self,
        address: ConversationAddress,
        *,
        after_turn: bool,
        flush: bool = False,
        boundary_hints: ConversationBoundaryHints | None = None,
    ) -> ConversationRetentionPlan:
        """读取当前 live 快照并执行一次无副作用切段规划。"""

        return self._plan(
            address,
            after_turn=after_turn,
            flush=flush,
            drain_pending=False,
            boundary_hints=boundary_hints,
        )

    def enqueue_ready_segments(
        self,
        address: ConversationAddress,
        *,
        after_turn: bool,
        flush: bool,
        boundary_hints: ConversationBoundaryHints | None = None,
    ) -> tuple[tuple[MemoryJob, ...], ConversationRetentionPlan]:
        queued: list[MemoryJob] = []
        final_plan = self._plan(
            address,
            after_turn=after_turn,
            flush=flush,
            drain_pending=False,
            boundary_hints=boundary_hints,
        )
        staged_checked = False
        for _ in range(1_000):
            if not final_plan.should_seal:
                return tuple(queued), final_plan
            if not staged_checked:
                staged_checked = True
                completed, sealed = self._complete_staged(address)
                queued.extend(completed)
                if sealed:
                    final_plan = self._plan(
                        address,
                        after_turn=after_turn,
                        flush=flush,
                        drain_pending=False,
                        boundary_hints=boundary_hints,
                    )
                    continue
            assert final_plan.through_sequence is not None
            queued.append(
                self.seal_and_enqueue(
                    address,
                    through_sequence=final_plan.through_sequence,
                )
            )
            final_plan = self._plan(
                address,
                after_turn=after_turn,
                flush=flush,
                drain_pending=after_turn and not flush,
                boundary_hints=boundary_hints,
            )
        raise MemoryJobError("automatic conversation sealing exceeded its progress bound")

    def _plan(
        self,
        address: ConversationAddress,
        *,
        after_turn: bool,
        flush: bool,
        drain_pending: bool,
        boundary_hints: ConversationBoundaryHints | None,
    ) -> ConversationRetentionPlan:
        live = self.conversations.read_live(address)
        state = self.conversations.read_state(address)
        leading_continuation = bool(
            live is not None
            and state.archived_through_sequence is not None
            and live.start_sequence == state.archived_through_sequence + 1
            and not live.messages[0].starts_turn
        )
        return self.retention_planner.plan(
            live,
            after_turn=after_turn,
            flush=flush,
            drain_pending=drain_pending,
            boundary_hints=boundary_hints,
            leading_continuation=leading_continuation,
        )

    def _complete_staged(self, address: ConversationAddress) -> tuple[tuple[MemoryJob, ...], bool]:
        """按原边界补完本会话在 history 发布或激活前中断的 STAGED Job，新边界只能接在它们之后；返回激活的 Job 与
        这一步有没有封过段。重试用尽、还没被认领过的 FAILED Job（人工 ``retry_failed`` 后回到 STAGED）也先按原边界
        发布 history，但不激活：它照旧等人工处置，重试时 history 已在原边界上，补完能对上。"""

        activated: list[MemoryJob] = []
        sealed = False
        for job in self.jobs.list_for_conversation(address):
            exhausted = job.status is MemoryJobStatus.FAILED and job.claim_generation == 0
            if job.status is not MemoryJobStatus.STAGED and not exhausted:
                continue
            if exhausted:
                _start_sequence, end_sequence = self.conversations.layout.segment_range(job.segment_id)
                archived = self.conversations.read_state(address).archived_through_sequence
                if archived is not None and archived >= end_sequence:
                    continue  # 原边界已发布过：重试时由补完核对
            segment = staged_job_segment(
                self.conversations,
                job,
                before_history_publish=partial(self._require_staged_source, job),
            )
            self._require_staged_source(job, segment)
            sealed = True
            if not exhausted:
                activated.append(self.jobs.activate(job))
        return tuple(activated), sealed

    @staticmethod
    def _require_staged_source(job: MemoryJob, segment: ConversationSegment) -> None:
        if segment.segment_id != job.segment_id or segment.digest != job.source_segment_digest:
            raise MemoryJobError("staged MemoryJob does not match the ConversationSegment at its boundary")

    def seal_and_enqueue(
        self,
        address: ConversationAddress,
        *,
        through_sequence: int,
    ) -> MemoryJob:
        """先建立 STAGED outbox，发布原文后再激活后台任务。"""

        def stage_before_publish(segment: ConversationSegment) -> None:
            self.jobs.stage(address, segment)

        sealed = self.conversations.seal(
            address,
            through_sequence=through_sequence,
            before_history_publish=stage_before_publish,
        )
        staged = self.jobs.stage(address, sealed.segment)
        return self.jobs.activate(staged)


__all__ = [
    "ConversationMemoryEnqueuer",
    "ConversationMemoryFlushResult",
    "ConversationMemoryIngestResult",
]
