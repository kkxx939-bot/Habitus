"""已提交 L2 事务的后续完成阶段。"""

from __future__ import annotations

import time
from asyncio import to_thread
from dataclasses import dataclass
from typing import Protocol

from habitus.foundation.observability import (
    NullObserver,
    ObservationEvent,
    ObservationStatus,
    Observer,
    observe_operation,
)
from habitus.memory.conversation import ConversationAddress, ConversationMessageJournal
from habitus.memory.conversation.indexing import PersistentConversationSummaryVectorIndex
from habitus.memory.conversation.summary import ConversationSummaryService
from habitus.memory.editor.transaction_log import (
    MemoryTransactionJournalRecord,
    MemoryTransactionJournalState,
)
from habitus.memory.indexing import PersistentMemoryVectorIndex
from habitus.memory.semantic import MemorySemanticRefresher, MemorySemanticRefreshResult
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow.jobs import (
    MemoryJob,
    MemoryJobError,
    MemoryJobLease,
    MemoryJobStatus,
    MemoryJobStore,
)
from habitus.memory.workflow.receipt import (
    MemoryChangeReceipt,
    MemoryChangeReceiptState,
    MemoryChangeReceiptStore,
    MemoryChangeSource,
)
from habitus.memory.workflow.recovery import MemoryJobTransactionRecovery


class MemoryRecallStateCleaner(Protocol):
    def forget(self, uris: tuple[MemoryURI, ...]) -> int: ...


@dataclass(frozen=True)
class MemoryJobCompletion:
    """一笔已提交 L2 事务完成全部后续耐久步骤后的结果。"""

    job: MemoryJob
    change_receipt: MemoryChangeReceipt
    summary_generated: bool
    summary_indexed: bool
    vector_indexed: bool
    journal_cleaned: bool

    def __post_init__(self) -> None:
        if not isinstance(self.job, MemoryJob):
            raise TypeError("job must be MemoryJob")
        if self.job.status is not MemoryJobStatus.COMMITTED:
            raise ValueError("completed workflow result requires a COMMITTED MemoryJob")
        if not isinstance(self.change_receipt, MemoryChangeReceipt):
            raise TypeError("change_receipt must be MemoryChangeReceipt")
        if self.change_receipt.state is not MemoryChangeReceiptState.COMMITTED:
            raise ValueError("completed workflow result requires a COMMITTED receipt")
        for name in (
            "summary_generated",
            "summary_indexed",
            "vector_indexed",
            "journal_cleaned",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")


class MemoryCommittedJobFinalizer:
    """补齐 Summary、变更回执、语义层和 Job 终态。"""

    def __init__(
        self,
        store: MemoryJobStore,
        conversations: ConversationMessageJournal,
        summary_service: ConversationSummaryService,
        summary_vector_index: PersistentConversationSummaryVectorIndex,
        change_receipts: MemoryChangeReceiptStore,
        semantic_refresher: MemorySemanticRefresher,
        vector_index: PersistentMemoryVectorIndex,
        transaction_recovery: MemoryJobTransactionRecovery,
        recall_state_cleaner: MemoryRecallStateCleaner,
        *,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(store, MemoryJobStore):
            raise TypeError("store must be MemoryJobStore")
        if not isinstance(conversations, ConversationMessageJournal):
            raise TypeError("conversations must be ConversationMessageJournal")
        if not isinstance(summary_service, ConversationSummaryService):
            raise TypeError("summary_service must be ConversationSummaryService")
        if not isinstance(summary_vector_index, PersistentConversationSummaryVectorIndex):
            raise TypeError("summary_vector_index must be PersistentConversationSummaryVectorIndex")
        if not isinstance(change_receipts, MemoryChangeReceiptStore):
            raise TypeError("change_receipts must be MemoryChangeReceiptStore")
        if not isinstance(semantic_refresher, MemorySemanticRefresher):
            raise TypeError("semantic_refresher must be MemorySemanticRefresher")
        if not isinstance(vector_index, PersistentMemoryVectorIndex):
            raise TypeError("vector_index must be PersistentMemoryVectorIndex")
        if not isinstance(transaction_recovery, MemoryJobTransactionRecovery):
            raise TypeError("transaction_recovery must be MemoryJobTransactionRecovery")
        if not callable(getattr(recall_state_cleaner, "forget", None)):
            raise TypeError("recall_state_cleaner must implement forget")
        if summary_service.store.layout.root != conversations.layout.root:
            raise ValueError("summary service and conversations must use one conversation root")
        if summary_vector_index.journal is not conversations:
            raise ValueError("summary vector index and conversations must use one journal")
        if summary_vector_index.compactor.segment_store is not summary_service.store:
            raise ValueError("summary vector index and summary service must share one store")
        if change_receipts.root != store.root:
            raise ValueError("change receipts and MemoryJobStore must use one workflow root")
        if semantic_refresher.tree.root != store.memory_root:
            raise ValueError("semantic refresher and MemoryJobStore must use one memory root")
        if vector_index.tree.root != store.memory_root:
            raise ValueError("vector index and MemoryJobStore must use one memory root")
        if transaction_recovery.transaction.tree.root != store.memory_root:
            raise ValueError("transaction recovery and MemoryJobStore must use one memory root")
        self.store = store
        self.conversations = conversations
        self.summary_service = summary_service
        self.summary_vector_index = summary_vector_index
        self.change_receipts = change_receipts
        self.semantic_refresher = semantic_refresher
        self.vector_index = vector_index
        self.transaction_recovery = transaction_recovery
        self.recall_state_cleaner = recall_state_cleaner
        self.observer = observer or NullObserver()

    async def finalize(
        self,
        lease: MemoryJobLease,
        journal: MemoryTransactionJournalRecord,
        *,
        summary_generated: bool | None = None,
    ) -> MemoryJobCompletion:
        """对已提交事务执行可幂等重放的后续步骤，并最后结束 Job。"""

        if not isinstance(lease, MemoryJobLease):
            raise TypeError("lease must be a MemoryJobLease")
        job = self.store.assert_current(lease)
        if not isinstance(journal, MemoryTransactionJournalRecord):
            raise TypeError("journal must be MemoryTransactionJournalRecord")
        if journal.state is not MemoryTransactionJournalState.COMMITTED:
            raise MemoryJobError("job finalization requires a COMMITTED transaction journal")
        if journal.transaction_id != job.transaction_id:
            raise MemoryJobError("transaction journal does not belong to the claimed MemoryJob")
        if summary_generated is not None and not isinstance(summary_generated, bool):
            raise TypeError("summary_generated must be boolean or None")

        address = ConversationAddress(job.conversation_id, job.started_on)
        segment = self.conversations.read_segment(address, job.segment_id)
        if segment.digest != job.source_segment_digest:
            raise MemoryJobError("sealed ConversationSegment digest does not match its MemoryJob")
        summary_existed = self.summary_service.store.try_read(address, segment) is not None
        with observe_operation(self.observer, "workflow", "summary_persist"):
            summary = await self.summary_service.get_or_create(address, segment)
        summary.require_matches_source(segment)
        self.store.assert_current(lease)
        resolved_summary_generated = not summary_existed if summary_generated is None else summary_generated
        with observe_operation(self.observer, "workflow", "summary_vector_publish"):
            await self.summary_vector_index.synchronize(
                address,
                checkpoint=job.memory_sequence,
            )
        self.store.assert_current(lease)

        source = MemoryChangeSource.from_job(job)
        with observe_operation(self.observer, "workflow", "receipt_finalize"):
            change_receipt = self.change_receipts.finalize(source, journal)
        self.store.assert_current(lease)
        await self._forget_retired_recall_state(change_receipt.expected_deleted_uris)
        with observe_operation(self.observer, "workflow", "semantic_refresh"):
            semantic_results = await to_thread(self._refresh_semantic_layers, change_receipt)
        self.store.assert_current(lease)
        with observe_operation(self.observer, "workflow", "memory_vector_publish"):
            await self.vector_index.synchronize(
                changed_uris=change_receipt.changed_uris,
                semantic_results=semantic_results,
                checkpoint=change_receipt.source.memory_sequence,
            )
        self.store.assert_current(lease)
        with observe_operation(self.observer, "workflow", "job_complete"):
            committed = self.store.complete(lease)
        with observe_operation(self.observer, "workflow", "transaction_cleanup"):
            journal_cleaned = self.transaction_recovery.discard_terminal(job.transaction_id)
        return MemoryJobCompletion(
            job=committed,
            change_receipt=change_receipt,
            summary_generated=resolved_summary_generated,
            summary_indexed=True,
            vector_indexed=True,
            journal_cleaned=journal_cleaned,
        )

    async def _forget_retired_recall_state(
        self,
        deleted_uris: tuple[MemoryURI, ...],
    ) -> None:
        if not deleted_uris:
            return
        started = time.monotonic()
        try:
            deleted = await to_thread(self.recall_state_cleaner.forget, deleted_uris)
        except Exception as exc:
            self.observer.record(
                ObservationEvent(
                    category="retrieval",
                    operation="recall_lifecycle_cleanup",
                    status=ObservationStatus.DEGRADED,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes={
                        "error_type": type(exc).__name__,
                        "requested": len(deleted_uris),
                    },
                )
            )
            return
        self.observer.record(
            ObservationEvent(
                category="retrieval",
                operation="recall_lifecycle_cleanup",
                status=ObservationStatus.SUCCESS,
                duration_seconds=max(0.0, time.monotonic() - started),
                attributes={"requested": len(deleted_uris), "deleted": deleted},
            )
        )

    def _refresh_semantic_layers(
        self,
        receipt: MemoryChangeReceipt,
    ) -> tuple[MemorySemanticRefreshResult, ...]:
        if not isinstance(receipt, MemoryChangeReceipt):
            raise TypeError("receipt must be MemoryChangeReceipt")
        if receipt.state is not MemoryChangeReceiptState.COMMITTED:
            raise MemoryJobError("semantic refresh requires a COMMITTED memory change receipt")
        addresses = tuple(uri.to_address() for uri in receipt.changed_uris)
        return self.semantic_refresher.refresh_for_many(addresses)


__all__ = ["MemoryCommittedJobFinalizer", "MemoryJobCompletion", "MemoryRecallStateCleaner"]
