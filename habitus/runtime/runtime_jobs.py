"""FAILED MemoryJob 的人工运维：重试与显式放弃（裁定 111）。

FAILED 已是终态、不挡后面的任务，处置不要求 Worker 被挡住：重试把它按序号排回队列，由正在跑的单并发 Worker
照常执行（Runtime 在 RUNNING 时顺手唤醒 Worker）；原文已经释放的不能重试。入口仍在 Runtime 门面。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from habitus.foundation.observability import ObservationStatus, bind_observation_context
from habitus.memory.conversation import ConversationAddress
from habitus.memory.workflow import MemoryJob, MemoryJobAbandonment, MemoryJobStatus
from habitus.runtime.components import RuntimeComponents
from habitus.runtime.runtime_state import RuntimeStateError

#: Runtime 的观测出口：(类别, 操作, 状态, 起始单调时刻, 属性)。
Observe = Callable[[str, str, ObservationStatus, float, dict[str, str | int | float | bool]], None]


@dataclass(frozen=True)
class MemoryJobRetryResult:
    """一次人工重试的失败快照、新状态和是否唤醒了正在跑的 Worker。"""

    failed_job: MemoryJob
    reopened_job: MemoryJob
    worker_restarted: bool

    def __post_init__(self) -> None:
        if self.failed_job.status is not MemoryJobStatus.FAILED:
            raise ValueError("failed_job must be a FAILED MemoryJob snapshot")
        if self.reopened_job.status not in {MemoryJobStatus.STAGED, MemoryJobStatus.QUEUED}:
            raise ValueError("reopened_job must be a STAGED or QUEUED MemoryJob")
        if self.failed_job.source_identity != self.reopened_job.source_identity:
            raise ValueError("retried MemoryJob source identity changed")
        if self.reopened_job.attempts != 0 or self.reopened_job.last_error is not None:
            raise ValueError("reopened MemoryJob must reset its retry state")


@dataclass(frozen=True)
class MemoryJobAbandonResult:
    """一次显式永久失败处置及其 Worker 恢复结果。"""

    failed_job: MemoryJob
    abandonment: MemoryJobAbandonment
    worker_restarted: bool


class FailedMemoryJobOperations:
    """``running()`` 由门面给出：Runtime 此刻是否处于 RUNNING（在处置当下读，不在构造时定格）。"""

    def __init__(self, components: RuntimeComponents, *, running: Callable[[], bool], observe: Observe) -> None:
        self.components = components
        self._running = running
        self._observe = observe

    async def retry(self, failed_job: MemoryJob) -> MemoryJobRetryResult:
        """以完整失败快照重新开放一条 FAILED Job；这一段的原文已经释放就拒绝（没有东西可重做）。"""

        started = time.monotonic()
        with bind_observation_context(memory_sequence=failed_job.memory_sequence):
            try:
                # 没封存过的（STAGED 阶段就失败、原文还在 live 里）照常能重试：重开后由 STAGED 恢复补发 History
                address = ConversationAddress(failed_job.conversation_id, failed_job.started_on)
                archived = (
                    await asyncio.to_thread(self.components.conversation.journal.read_state, address)
                ).archived_through_sequence
                retention = self.components.workflow.lifecycle.retention
                if await asyncio.to_thread(retention.history_released, address, failed_job.segment_id, archived):
                    raise RuntimeStateError(
                        "this failed memory job cannot be retried: its conversation history has already been released"
                    )
                reopened = await asyncio.to_thread(self.components.workflow.jobs.retry_failed, failed_job)
                wake = self._running()
                if wake:
                    self.components.workflow.worker.wake()
                result = MemoryJobRetryResult(failed_job=failed_job, reopened_job=reopened, worker_restarted=wake)
            except Exception as exc:
                self._observe(
                    "operations",
                    "retry_job",
                    ObservationStatus.FAILURE,
                    started,
                    {"error_type": type(exc).__name__, "job_status": failed_job.status.value},
                )
                raise
            self._observe(
                "operations",
                "retry_job",
                ObservationStatus.SUCCESS,
                started,
                {"job_status": reopened.status.value},
            )
            return result

    async def abandon(self, failed_job: MemoryJob, *, reason: str) -> MemoryJobAbandonResult:
        """显式放弃一条不再重试的 FAILED Job：写下处置记录、删掉任务；原文照同一条规则到期释放。"""

        abandonment = await asyncio.to_thread(
            self.components.workflow.jobs.abandon_failed,
            failed_job,
            reason=reason,
        )
        return MemoryJobAbandonResult(failed_job, abandonment, False)


__all__ = ["FailedMemoryJobOperations", "MemoryJobAbandonResult", "MemoryJobRetryResult"]
