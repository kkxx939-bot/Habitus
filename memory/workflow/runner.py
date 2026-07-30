"""按 durable lease 执行最早 MemoryJob 的领域 Runner。"""

from __future__ import annotations

from asyncio import to_thread
from dataclasses import dataclass

from foundation.observability import NullObserver, Observer, bind_observation_context, observe_operation
from memory.conversation import ConversationMessageJournal
from memory.conversation.indexing import PersistentConversationSummaryVectorIndex
from memory.conversation.summary import ConversationSummaryService
from memory.editor import MemoryEditor
from memory.editor.transaction import MemoryCommitResult
from memory.indexing import PersistentMemoryVectorIndex
from memory.semantic import MemorySemanticRefresher
from memory.workflow.completion import MemoryCommittedJobFinalizer
from memory.workflow.execution import MemoryJobExecutor
from memory.workflow.failure import memory_job_failure_is_retryable
from memory.workflow.jobs import (
    MemoryJob,
    MemoryJobError,
    MemoryJobExecutionError,
    MemoryJobLease,
    MemoryJobLeaseLostError,
    MemoryJobStatus,
    MemoryJobStore,
)
from memory.workflow.receipt import (
    MemoryChangeReceipt,
    MemoryChangeReceiptStore,
    MemoryChangeSource,
)
from memory.workflow.recovery import MemoryJobTransactionRecovery, MemoryStagedJobRecovery


@dataclass(frozen=True)
class MemoryJobClaim:
    """Worker 已认领的任务以及是否只能继续已提交事务。"""

    lease: MemoryJobLease
    expired_attempts_exhausted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.lease, MemoryJobLease):
            raise TypeError("lease must be a MemoryJobLease")
        if not isinstance(self.expired_attempts_exhausted, bool):
            raise TypeError("expired_attempts_exhausted must be boolean")


@dataclass(frozen=True)
class MemoryJobRunResult:
    """一次任务编排实际完成的记忆处理结果。"""

    job: MemoryJob | None
    commit: MemoryCommitResult | None
    recovered: bool = False
    semantic_refreshed: bool = False
    vector_indexed: bool = False
    summary_indexed: bool = False
    journal_cleaned: bool = True
    summary_generated: bool = False
    change_receipt: MemoryChangeReceipt | None = None


class MemoryJobRunner:
    """分离认领与执行，让 Runtime Worker 能独立维持租约心跳。"""

    def __init__(
        self,
        store: MemoryJobStore,
        conversations: ConversationMessageJournal,
        editor: MemoryEditor,
        semantic_refresher: MemorySemanticRefresher,
        vector_index: PersistentMemoryVectorIndex,
        summary_service: ConversationSummaryService,
        summary_vector_index: PersistentConversationSummaryVectorIndex,
        change_receipts: MemoryChangeReceiptStore,
        *,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(store, MemoryJobStore):
            raise TypeError("store must be MemoryJobStore")
        if not isinstance(conversations, ConversationMessageJournal):
            raise TypeError("conversations must be ConversationMessageJournal")
        if not isinstance(editor, MemoryEditor):
            raise TypeError("editor must be MemoryEditor")
        if not isinstance(semantic_refresher, MemorySemanticRefresher):
            raise TypeError("semantic_refresher must be MemorySemanticRefresher")
        if not isinstance(vector_index, PersistentMemoryVectorIndex):
            raise TypeError("vector_index must be PersistentMemoryVectorIndex")
        if not isinstance(summary_service, ConversationSummaryService):
            raise TypeError("summary_service must be ConversationSummaryService")
        if not isinstance(summary_vector_index, PersistentConversationSummaryVectorIndex):
            raise TypeError("summary_vector_index must be PersistentConversationSummaryVectorIndex")
        if not isinstance(change_receipts, MemoryChangeReceiptStore):
            raise TypeError("change_receipts must be MemoryChangeReceiptStore")
        if store.memory_root != editor.transaction.tree.root:
            raise ValueError("MemoryJobStore is bound to another memory root")
        if change_receipts.root != store.root:
            raise ValueError("change receipt store and MemoryJobStore must use the same root")
        self.store = store
        self.observer = observer or NullObserver()
        self.staged_recovery = MemoryStagedJobRecovery(conversations, store)
        self.transaction_recovery = MemoryJobTransactionRecovery(
            editor.transaction,
            change_receipts,
        )
        self.executor = MemoryJobExecutor(
            conversations,
            editor,
            summary_service,
            change_receipts,
            store,
            observer=self.observer,
        )
        self.committed_finalizer = MemoryCommittedJobFinalizer(
            store,
            conversations,
            summary_service,
            summary_vector_index,
            change_receipts,
            semantic_refresher,
            vector_index,
            self.transaction_recovery,
            observer=self.observer,
        )

    def claim_next(self, worker_id: str) -> MemoryJobClaim | None:
        """先取得最早 Job 的 durable lease，再接触它的事务恢复状态。"""

        job = self.store.oldest_uncommitted()
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
        """只执行已经由 Worker 认领并持续维持心跳的 Job。"""

        if not isinstance(claim, MemoryJobClaim):
            raise TypeError("claim must be a MemoryJobClaim")
        lease = claim.lease
        job = self.store.assert_current(lease)
        source = MemoryChangeSource.from_job(job)
        observer = getattr(self, "observer", None) or NullObserver()
        with bind_observation_context(
            memory_sequence=job.memory_sequence,
            transaction_id=job.transaction_id,
            attempt=job.attempts,
        ):
            try:
                with observe_operation(observer, "workflow", "transaction_recovery"):
                    await to_thread(self.transaction_recovery.recover_pending)
                self.store.assert_current(lease)
                with observe_operation(observer, "workflow", "transaction_inspection"):
                    journal = await to_thread(self.transaction_recovery.inspect, job)
                if claim.expired_attempts_exhausted and journal is None:
                    raise MemoryJobError("expired worker lease exhausted attempts without a committed transaction")
                if journal is not None:
                    completion = await self.committed_finalizer.finalize(lease, journal)
                    return MemoryJobRunResult(
                        job=completion.job,
                        commit=None,
                        recovered=True,
                        semantic_refreshed=True,
                        vector_indexed=completion.vector_indexed,
                        summary_indexed=completion.summary_indexed,
                        journal_cleaned=completion.journal_cleaned,
                        summary_generated=completion.summary_generated,
                        change_receipt=completion.change_receipt,
                    )

                execution = await self.executor.execute(lease)
                completion = await self.committed_finalizer.finalize(
                    lease,
                    execution.journal,
                    summary_generated=execution.summary_generated,
                )
                return MemoryJobRunResult(
                    job=completion.job,
                    commit=execution.commit,
                    semantic_refreshed=True,
                    vector_indexed=completion.vector_indexed,
                    summary_indexed=completion.summary_indexed,
                    journal_cleaned=completion.journal_cleaned,
                    summary_generated=completion.summary_generated,
                    change_receipt=completion.change_receipt,
                )
            except MemoryJobLeaseLostError:
                raise
            except Exception as exc:
                failed = self.store.fail(
                    lease,
                    exc,
                    retryable=memory_job_failure_is_retryable(exc),
                    before_settlement=lambda: self.transaction_recovery.discard_uncommitted(source),
                )
                raise MemoryJobExecutionError(
                    f"memory job {failed.memory_sequence} failed with status {failed.status.value}",
                    job=failed,
                ) from exc


__all__ = ["MemoryJobClaim", "MemoryJobRunResult", "MemoryJobRunner"]
