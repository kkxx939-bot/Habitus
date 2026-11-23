"""按 durable lease 执行最早 MemoryJob 的领域 Runner。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.foundation.observability import NullObserver, Observer, bind_observation_context
from habitus.memory.conversation import ConversationMessageJournal
from habitus.memory.conversation.summary import ConversationSummaryService
from habitus.memory.editor import MemoryCommitResult, MemoryEditor
from habitus.memory.semantic import MemorySemanticRefresher
from habitus.memory.workflow.completion import MemoryCommittedJobFinalizer, MemoryRecallStateCleaner
from habitus.memory.workflow.execution import MemoryJobExecutor
from habitus.memory.workflow.failure import (
    MemoryJobFailureKind,
    MemoryJobSkipped,
    memory_job_failure_kind,
    model_wait_seconds,
)
from habitus.memory.workflow.jobs import (
    MemoryJob,
    MemoryJobError,
    MemoryJobExecutionError,
    MemoryJobLease,
    MemoryJobLeaseLostError,
    MemoryJobStatus,
    MemoryJobStore,
)
from habitus.memory.workflow.recovery import MemoryStagedJobRecovery


@dataclass(frozen=True)
class MemoryJobClaim:
    """Worker 已认领的任务，以及过期租约接管时尝试次数是否已经用完。"""

    lease: MemoryJobLease
    expired_attempts_exhausted: bool = False


@dataclass(frozen=True)
class MemoryJobRunResult:
    """一次任务编排实际完成的记忆处理结果。"""

    job: MemoryJob | None
    commit: MemoryCommitResult | None
    #: 卡片在之前某次执行里已经写完，这一次只补了后面的步骤（没有调模型提取）
    recovered: bool = False
    semantic_refreshed: bool = False
    summary_generated: bool = False


class MemoryJobRunner:
    """分离认领与执行，让 Runtime Worker 能独立维持租约心跳。"""

    def __init__(
        self,
        store: MemoryJobStore,
        conversations: ConversationMessageJournal,
        editor: MemoryEditor,
        semantic_refresher: MemorySemanticRefresher,
        summary_service: ConversationSummaryService,
        recall_state_cleaner: MemoryRecallStateCleaner,
        *,
        observer: Observer | None = None,
    ) -> None:
        if store.memory_root != editor.writer.tree.root:
            raise ValueError("MemoryJobStore is bound to another memory root")
        self.store = store
        self.observer = observer or NullObserver()
        self.staged_recovery = MemoryStagedJobRecovery(conversations, store)
        self.executor = MemoryJobExecutor(
            conversations,
            editor,
            summary_service,
            store,
            observer=self.observer,
        )
        self.committed_finalizer = MemoryCommittedJobFinalizer(
            store,
            semantic_refresher,
            recall_state_cleaner,
            observer=self.observer,
        )

    def claim_next(self, worker_id: str) -> MemoryJobClaim | None:
        """认领最早还没到终态的 Job（STAGED 先补完 history 发布）。"""

        job = self.store.oldest_unfinished()
        if job is None:
            return None
        job = self.staged_recovery.recover(job)
        lease = self.store.claim(job, worker_id)
        return MemoryJobClaim(
            lease=lease,
            expired_attempts_exhausted=(
                job.status is MemoryJobStatus.RUNNING and job.attempts >= self.store.config.max_attempts
            ),
        )

    async def run_claimed(self, claim: MemoryJobClaim) -> MemoryJobRunResult:
        """只执行已经由 Worker 认领并持续维持心跳的 Job。

        任务文件里记着"卡片已写完"的，只补刷概览、清冷热状态和标完成，不再调模型提取；没有就整条重做。
        """

        lease = claim.lease
        job = self.store.assert_current(lease)
        with bind_observation_context(memory_sequence=job.memory_sequence, attempt=job.attempts):
            try:
                if job.cards_written:
                    completed = await self.committed_finalizer.finalize(lease)
                    return MemoryJobRunResult(job=completed, commit=None, recovered=True, semantic_refreshed=True)
                if claim.expired_attempts_exhausted:
                    raise MemoryJobError("expired worker lease exhausted attempts before the cards were written")
                execution = await self.executor.execute(lease)
                completed = await self.committed_finalizer.finalize(lease)
                return MemoryJobRunResult(
                    job=completed,
                    commit=execution.commit,
                    semantic_refreshed=True,
                    summary_generated=execution.summary_generated,
                )
            except MemoryJobLeaseLostError:
                raise
            except MemoryJobSkipped as skipped:
                job = await self.committed_finalizer.finalize_skipped(lease, skipped.reason)
                return MemoryJobRunResult(job=job, commit=None)
            except Exception as exc:
                failed = self._record_failure(lease, exc)
                raise MemoryJobExecutionError(
                    f"memory job {failed.memory_sequence} failed with status {failed.status.value}",
                    job=failed,
                ) from exc

    def _record_failure(self, lease: MemoryJobLease, exc: Exception) -> MemoryJob:
        kind = memory_job_failure_kind(exc)
        waits = kind is MemoryJobFailureKind.WAITS_FOR_MODEL
        return self.store.fail(
            lease,
            exc,
            retryable=kind is not MemoryJobFailureKind.PERMANENT,
            waits_for_model=waits,
            wait_seconds=model_wait_seconds(exc) if waits else None,
        )


__all__ = ["MemoryJobClaim", "MemoryJobRunResult", "MemoryJobRunner"]
