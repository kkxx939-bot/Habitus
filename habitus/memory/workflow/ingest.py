"""将 Conversation 安全接入有序记忆任务的前台编排。"""

from __future__ import annotations

from dataclasses import dataclass

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
from habitus.memory.workflow.jobs import MemoryJob, MemoryJobError, MemoryJobStore
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
        """降载工具结果、切分超长文本，再写入唯一 live 主链。"""

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
                    force_summarize=(
                        message.tool_call_id not in omit_tool_call_ids
                        and self.retention_planner.token_estimator(message)
                        > self.retention_planner.config.max_segment_tokens
                    ),
                )
                if message.role is ConversationMessageRole.TOOL_RESULT
                else message
                for message in batch.messages
            ),
        )
        normalized = self.message_chunker.normalize(reduced)
        return self.conversations.append(address, normalized, ingress=ingress)

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

        live = self.conversations.read_live(address)
        state = self.conversations.read_state(address)
        leading_continuation = bool(
            live is not None
            and state.archived_through_sequence is not None
            and live.start_sequence == state.archived_through_sequence + 1
            and (
                live.messages[0].role is not ConversationMessageRole.PROMPT
                or live.messages[0].is_logical_continuation
            )
        )
        return self.retention_planner.plan(
            live,
            after_turn=after_turn,
            flush=flush,
            drain_pending=False,
            boundary_hints=boundary_hints,
            leading_continuation=leading_continuation,
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
        live = self.conversations.read_live(address)
        state = self.conversations.read_state(address)
        leading_continuation = bool(
            live is not None
            and state.archived_through_sequence is not None
            and live.start_sequence == state.archived_through_sequence + 1
            and (
                live.messages[0].role is not ConversationMessageRole.PROMPT
                or live.messages[0].is_logical_continuation
            )
        )
        final_plan = self.retention_planner.plan(
            live,
            after_turn=after_turn,
            flush=flush,
            drain_pending=False,
            boundary_hints=boundary_hints,
            leading_continuation=leading_continuation,
        )
        for _ in range(1_000):
            if not final_plan.should_seal:
                return tuple(queued), final_plan
            assert final_plan.through_sequence is not None
            queued.append(
                self.seal_and_enqueue(
                    address,
                    through_sequence=final_plan.through_sequence,
                )
            )
            live = self.conversations.read_live(address)
            state = self.conversations.read_state(address)
            leading_continuation = bool(
                live is not None
                and state.archived_through_sequence is not None
                and live.start_sequence == state.archived_through_sequence + 1
                and (
                    live.messages[0].role is not ConversationMessageRole.PROMPT
                    or live.messages[0].is_logical_continuation
                )
            )
            final_plan = self.retention_planner.plan(
                live,
                after_turn=after_turn,
                flush=flush,
                drain_pending=after_turn and not flush,
                boundary_hints=boundary_hints,
                leading_continuation=leading_continuation,
            )
        raise MemoryJobError("automatic conversation sealing exceeded its progress bound")

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
