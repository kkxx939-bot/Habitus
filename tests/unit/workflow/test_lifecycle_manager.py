"""Conversation、Summary、Job、Receipt 联合生命周期的安全门槛测试。"""

import asyncio
from datetime import date
from pathlib import Path

import pytest

from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationRangeSummaryGenerator,
    ConversationRangeSummaryStore,
    ConversationSummaryCompactionConfig,
    ConversationSummaryCompactor,
    ConversationSummaryStore,
    PersistentConversationSummaryVectorIndex,
)
from memory.editor import MemoryTransactionJournal
from memory.workflow import (
    ConversationLifecycleError,
    ConversationLifecycleManager,
    MemoryChangeReceiptStore,
    MemoryJobStore,
    MemoryWorkflowLifecycleConfig,
)
from pre.conversation import ConversationBatch
from tests.helpers import BASE_TIME, closed_turn, codec


def lifecycle_manager(tmp_path: Path, *, compaction_enabled: bool):
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    segment_store = ConversationSummaryStore(journal.layout)
    range_store = ConversationRangeSummaryStore(journal.layout)
    compaction_config = ConversationSummaryCompactionConfig(enabled=compaction_enabled)
    compactor = ConversationSummaryCompactor(
        journal,
        segment_store,
        range_store,
        object.__new__(ConversationRangeSummaryGenerator),
        config=compaction_config,
    )
    summary_index = object.__new__(PersistentConversationSummaryVectorIndex)
    summary_index.compactor = compactor
    calls = []

    async def synchronize(address, *, checkpoint=None):
        calls.append((address, checkpoint))

    summary_index.synchronize = synchronize
    document_codec = codec()
    jobs = MemoryJobStore(
        tmp_path / "workflow",
        path_lock,
        memory_root=tmp_path / "memory",
    )
    receipts = MemoryChangeReceiptStore(tmp_path / "workflow", document_codec)
    manager = ConversationLifecycleManager(
        compactor,
        journal,
        segment_store,
        range_store,
        summary_index,
        jobs,
        receipts,
        MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec),
        summary_config=compaction_config,
    )
    return manager, calls


def test_empty_lifecycle_cycle_is_idempotent_and_still_reconciles_summary_index(tmp_path: Path) -> None:
    manager, calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))

    result = asyncio.run(manager.maintain_once(address, now=BASE_TIME))

    assert result.compaction.summary is None
    assert result.summary_indexed
    assert calls == [(address, None)]
    assert result.purged_history_segment_ids == ()
    assert result.deleted_memory_job_sequences == ()
    assert result.deleted_memory_receipt_ids == ()


def test_retained_history_without_bound_summary_fails_before_cleanup(tmp_path: Path) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=True)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    manager.journal.append(address, ConversationBatch("conversation-1", closed_turn()))
    segment = manager.journal.seal(address, through_sequence=1).segment
    manager.jobs.activate(manager.jobs.stage(address, segment))

    with pytest.raises(ConversationLifecycleError, match="no valid one-to-one Segment Summary"):
        asyncio.run(manager.maintain_once(address, now=BASE_TIME))


def test_receipt_retention_cannot_be_shorter_than_job_retention() -> None:
    with pytest.raises(ValueError, match="cannot be shorter"):
        MemoryWorkflowLifecycleConfig(
            committed_job_retention_days=30,
            committed_receipt_retention_days=29,
        )

