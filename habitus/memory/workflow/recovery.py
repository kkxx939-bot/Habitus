"""后台任务执行前的 Conversation outbox 恢复。"""

from __future__ import annotations

from collections.abc import Callable

from habitus.memory.conversation import ConversationAddress, ConversationMessageJournal
from habitus.memory.workflow.jobs import (
    MemoryJob,
    MemoryJobError,
    MemoryJobExecutionError,
    MemoryJobNotReadyError,
    MemoryJobStatus,
    MemoryJobStore,
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


__all__ = ["MemoryStagedJobRecovery", "staged_job_segment"]
