"""FAILED MemoryJob 的人工运维：重试与显式放弃，带 MemoryWorker 状态门。

Runtime 在 RUNNING 时只允许在 MemoryWorker 已因该 Job 进入 BLOCKED 且无活跃执行后处置，处置完重新启动 Worker；
READY 状态只重新开放 Job，不隐式启动 Worker。入口仍在 Runtime 门面，这里只承载处置本身。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from habitus.foundation.observability import ObservationStatus, bind_observation_context
from habitus.memory.editor import MemoryTransactionJournalState
from habitus.memory.workflow import MemoryChangeSource, MemoryJob, MemoryJobAbandonment, MemoryJobStatus
from habitus.runtime.components import RuntimeComponents
from habitus.runtime.runtime_state import RuntimeStateError
from habitus.runtime.worker import MemoryWorkerState

#: Runtime 的观测出口：(类别, 操作, 状态, 起始单调时刻, 属性)。
Observe = Callable[[str, str, ObservationStatus, float, dict[str, str | int | float | bool]], None]


@dataclass(frozen=True)
class MemoryJobRetryResult:
    """一次人工重试的失败快照、新状态和 Worker 恢复结果。"""

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
        """以完整失败快照重新开放最早 Job，并恢复被它阻塞的常驻 Worker。"""

        started = time.monotonic()
        with bind_observation_context(
            memory_sequence=failed_job.memory_sequence,
            transaction_id=failed_job.transaction_id,
        ):
            try:
                worker = self.components.workflow.worker
                restart_worker = self._running()
                if restart_worker:
                    if worker.state is not MemoryWorkerState.BLOCKED or worker.busy:
                        raise RuntimeStateError(
                            "running runtime can only retry a failed job after its memory worker is blocked and idle"
                        )
                    await worker.wait_stopped()

                reopened = await asyncio.to_thread(
                    self.components.workflow.jobs.retry_failed,
                    failed_job,
                )
                if restart_worker:
                    await worker.start()
                result = MemoryJobRetryResult(
                    failed_job=failed_job,
                    reopened_job=reopened,
                    worker_restarted=restart_worker,
                )
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
        """显式放弃无法修复的最早 Job；有任何未决或已提交副作用时拒绝执行。"""

        source = MemoryChangeSource.from_job(failed_job)
        receipt = await asyncio.to_thread(self.components.workflow.receipts.try_read, source)
        if receipt is not None:
            raise RuntimeStateError("a failed job with a prepared or committed receipt cannot be abandoned")
        journal = await asyncio.to_thread(
            self.components.memory.editor.transaction.journal.try_read,
            failed_job.transaction_id,
        )
        if journal is not None and journal.state is not MemoryTransactionJournalState.ROLLED_BACK:
            raise RuntimeStateError("a failed job with a pending or committed transaction cannot be abandoned")
        worker = self.components.workflow.worker
        restart_worker = self._running()
        if restart_worker:
            if worker.state is not MemoryWorkerState.BLOCKED or worker.busy:
                raise RuntimeStateError(
                    "running runtime can only abandon a failed job after its memory worker is blocked and idle"
                )
            await worker.wait_stopped()
        abandonment = await asyncio.to_thread(
            self.components.workflow.jobs.abandon_failed,
            failed_job,
            reason=reason,
        )
        if restart_worker:
            await worker.start()
        return MemoryJobAbandonResult(failed_job, abandonment, restart_worker)


__all__ = ["FailedMemoryJobOperations", "MemoryJobAbandonResult", "MemoryJobRetryResult"]
