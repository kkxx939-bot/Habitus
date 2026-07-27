"""将 Conversation 安全接入有序记忆任务的前台编排。"""

from __future__ import annotations

from dataclasses import dataclass

from memory.conversation import (
    ConversationAddress,
    ConversationAppendResult,
    ConversationMessageJournal,
)
from memory.conversation.retention import (
    ConversationRetentionPlan,
    ConversationRetentionPlanner,
    ConversationToolResultReducer,
)
from memory.workflow.jobs import MemoryJob, MemoryJobError, MemoryJobStore
from pre.conversation import (
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
        self.conversations = conversations
        self.jobs = jobs
        self.retention_planner = retention_planner or ConversationRetentionPlanner()
        self.tool_result_reducer = tool_result_reducer or ConversationToolResultReducer(self.retention_planner.config)

    def append_and_maybe_enqueue(
        self,
        address: ConversationAddress,
        batch: ConversationBatch,
        *,
        after_turn: bool = False,
        omit_tool_call_ids: frozenset[str] = frozenset(),
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
        normalized = ConversationBatch(
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
        appended = self.conversations.append(address, normalized)
        queued, final_plan = self._enqueue_ready_segments(
            address,
            after_turn=after_turn,
            flush=False,
        )
        return ConversationMemoryIngestResult(
            append=appended,
            jobs=queued,
            retention=final_plan,
        )

    def flush(self, address: ConversationAddress) -> ConversationMemoryFlushResult:
        """显式提交全部剩余完整轮次，不关闭或冻结 Conversation。"""

        queued, final_plan = self._enqueue_ready_segments(
            address,
            after_turn=False,
            flush=True,
        )
        return ConversationMemoryFlushResult(jobs=queued, retention=final_plan)

    def _enqueue_ready_segments(
        self,
        address: ConversationAddress,
        *,
        after_turn: bool,
        flush: bool,
    ) -> tuple[tuple[MemoryJob, ...], ConversationRetentionPlan]:
        queued: list[MemoryJob] = []
        final_plan = self.retention_planner.plan(
            self.conversations.read_live(address),
            after_turn=after_turn,
            flush=flush,
            drain_pending=False,
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
            final_plan = self.retention_planner.plan(
                self.conversations.read_live(address),
                after_turn=after_turn,
                flush=flush,
                drain_pending=after_turn and not flush,
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
