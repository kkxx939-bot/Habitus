"""将 Conversation 安全接入有序记忆任务的前台编排。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

from habitus.foundation.observability import ObservationEvent, ObservationStatus
from habitus.memory.conversation import (
    ConversationAddress,
    ConversationAppendResult,
    ConversationAppendStatus,
    ConversationIngressRequest,
    ConversationMessageJournal,
)
from habitus.memory.conversation.interruption import is_interruption_message
from habitus.memory.conversation.retention import (
    ConversationRetentionPlan,
    ConversationRetentionPlanner,
)
from habitus.memory.conversation.segmentation import ConversationMessageChunker
from habitus.memory.conversation.tool_result_reduction import ConversationToolResultReducer
from habitus.memory.workflow.jobs import MemoryJob, MemoryJobError, MemoryJobStatus, MemoryJobStore
from habitus.memory.workflow.recovery import staged_job_segment
from habitus.pre.conversation import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationSegment,
)


@dataclass(frozen=True)
class ConversationMemoryIngestResult:
    """一次追加后实际排队的 Segment 与最终保留判断。"""

    append: ConversationAppendResult
    jobs: tuple[MemoryJob, ...]
    retention: ConversationRetentionPlan


@dataclass(frozen=True)
class ConversationMemoryFlushResult:
    """一次显式 flush 实际封存的 Segment 任务与最终 live 状态。"""

    jobs: tuple[MemoryJob, ...]
    retention: ConversationRetentionPlan


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

        if any(not identifier for identifier in omit_tool_call_ids):
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
        """顺延撞上系统收尾消息的序号、丢掉迟到的工具结果、降载工具结果、切分超长文本、把超过一段上限的工具交换抽到放得下，
        再写入唯一 live 主链。"""

        if any(not identifier for identifier in omit_tool_call_ids):
            raise TypeError("omit_tool_call_ids must be a frozenset of non-empty strings")
        kept = _after_interruption(self.conversations, address, batch)
        if kept != batch.messages:
            if not kept:
                live = self.conversations.read_live(address)
                next_sequence = (
                    self.conversations.read_state(address).next_sequence if live is None else live.end_sequence + 1
                )
                return ConversationAppendResult(ConversationAppendStatus.UNCHANGED, 0, live, next_sequence)
            batch = ConversationBatch(batch.conversation_id, kept)
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

    def enqueue_ready_segments(
        self,
        address: ConversationAddress,
        *,
        after_turn: bool,
        flush: bool,
    ) -> tuple[tuple[MemoryJob, ...], ConversationRetentionPlan]:
        queued: list[MemoryJob] = []
        final_plan = self._plan(
            address,
            after_turn=after_turn,
            flush=flush,
            drain_pending=False,
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
            )
        raise MemoryJobError("automatic conversation sealing exceeded its progress bound")

    def _plan(
        self,
        address: ConversationAddress,
        *,
        after_turn: bool,
        flush: bool,
        drain_pending: bool,
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


def _after_interruption(
    journal: ConversationMessageJournal, address: ConversationAddress, batch: ConversationBatch
) -> tuple[ConversationMessage, ...]:
    """闲置封存补过收尾消息之后（裁定 107 第 2 条），宿主送来的这一批怎么接上：

    - 开头的序号落在系统收尾消息上（插件编好序号、还没送出，会话就闲置封存了）：整批顺延到那一串收尾消息之后；
    - 系统已经替某个调用补过终态结果、而且补在这一批之前，这一批里它的结果就是迟到的：丢掉、记一条观测，后面的消息
      顺着补上序号（与切分超长消息重排序号同理，插件以服务端回的下一序号为准）。

    两样都只看盘上已经写下、不会再变的事实（收尾消息在哪），同一份投递重放时判出同一批，入口回执对得上。"""

    messages = batch.messages
    live = journal.read_live(address)
    local_calls = {
        item.tool_call_id
        for item in (*messages, *(() if live is None else live.messages))
        if item.role is ConversationMessageRole.TOOL_CALL
    }
    late_candidates = any(
        item.role is ConversationMessageRole.TOOL_RESULT and item.tool_call_id not in local_calls for item in messages
    )
    next_sequence = journal.read_state(address).next_sequence if live is None else live.end_sequence + 1
    if batch.start_sequence >= next_sequence and not late_candidates:
        return messages
    stored = {
        item.sequence: item
        for item in (
            *(item for segment in journal.list_history(address) for item in segment.messages),
            *(() if live is None else live.messages),
        )
    }
    start = batch.start_sequence
    while start in stored and is_interruption_message(stored[start]):
        start += 1
    closed = {
        item.tool_call_id
        for sequence, item in stored.items()
        if sequence < start and item.role is ConversationMessageRole.TOOL_RESULT and is_interruption_message(item)
    }
    kept = [
        item
        for item in messages
        if item.role is not ConversationMessageRole.TOOL_RESULT
        or item.tool_call_id in local_calls
        or item.tool_call_id not in closed
    ]
    dropped = len(messages) - len(kept)
    if dropped:
        journal.path_lock.observer.record(
            ObservationEvent(
                category="conversation",
                operation="late_tool_result_dropped",
                status=ObservationStatus.DEGRADED,
                duration_seconds=0.0,
                attributes={"dropped": dropped},
            )
        )
    return tuple(replace(item, sequence=start + index) for index, item in enumerate(kept))


__all__ = [
    "ConversationMemoryEnqueuer",
    "ConversationMemoryFlushResult",
    "ConversationMemoryIngestResult",
]
