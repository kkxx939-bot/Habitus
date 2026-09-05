"""Conversation、Summary、Job、Receipt 联合生命周期的安全门槛测试。"""

import asyncio
from datetime import date, timedelta
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationRangeSummaryGenerator,
    ConversationRangeSummaryStore,
    ConversationSummaryCompactionConfig,
    ConversationSummaryCompactor,
    ConversationSummaryStore,
    PersistentConversationSummaryVectorIndex,
    SQLiteConversationSummaryUseStore,
)
from habitus.memory.conversation.indexing import summary_reference
from habitus.memory.editor import MemoryTransactionJournal
from habitus.memory.retrieval import ConversationSearchContextReader
from habitus.memory.workflow import (
    ConversationLifecycleManager,
    MemoryChangeReceiptStore,
    MemoryJobStore,
    MemoryWorkflowLifecycleConfig,
)
from habitus.pre.conversation import (
    ConversationBatch,
    ConversationRangeSummary,
    ConversationRangeSummaryStage,
    ConversationSummarySourceRef,
)
from tests.helpers import BASE_TIME, closed_turn, codec, segment_summary, summary_content
from tests.unit.conversation.test_summary_indexing import Embedder, VectorStore
from tests.unit.retrieval.test_search_service import structured


def lifecycle_manager(
    tmp_path: Path,
    *,
    compaction_enabled: bool,
    compaction_config: ConversationSummaryCompactionConfig | None = None,
    use_store: SQLiteConversationSummaryUseStore | None = None,
):
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    segment_store = ConversationSummaryStore(journal.layout)
    range_store = ConversationRangeSummaryStore(journal.layout)
    compaction_config = compaction_config or ConversationSummaryCompactionConfig(
        enabled=compaction_enabled
    )
    compactor = ConversationSummaryCompactor(
        journal,
        segment_store,
        range_store,
        ConversationRangeSummaryGenerator(
            structured([]),
            compaction_config=compaction_config,
        ),
        use_store=use_store,
        config=compaction_config,
    )
    summary_index = PersistentConversationSummaryVectorIndex(
        journal,
        compactor,
        Embedder(),
        VectorStore(),
        dimension=2,
        embedding_fingerprint="lifecycle-test-v1",
    )
    calls = []

    async def synchronize(address, *, checkpoint=None, removed_references=()):
        calls.append(
            (
                address,
                checkpoint,
                tuple(reference.identity for reference in removed_references),
            )
        )

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
    assert calls == [(address, None, ())]
    assert result.purged_history_segment_ids == ()
    assert result.deleted_memory_job_sequences == ()
    assert result.deleted_memory_receipt_ids == ()


def test_retained_history_without_bound_summary_is_kept_without_ordinary_cleanup(tmp_path: Path) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=True)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    manager.journal.append(address, ConversationBatch("conversation-1", closed_turn()))
    segment = manager.journal.seal(address, through_sequence=1).segment
    manager.jobs.activate(manager.jobs.stage(address, segment))

    result = asyncio.run(manager.maintain_once(address, now=BASE_TIME))
    assert result.released_history_segment_ids == ()
    assert manager.journal.list_history(address) == (segment,)


def test_receipt_retention_cannot_be_shorter_than_job_retention() -> None:
    with pytest.raises(ValueError, match="cannot be shorter"):
        MemoryWorkflowLifecycleConfig(
            committed_job_retention_days=30,
            committed_receipt_retention_days=29,
        )


def test_archive_retirement_waits_for_actual_use_window_and_grace_then_deletes_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_store = SQLiteConversationSummaryUseStore(tmp_path / "workflow" / "summary-use.sqlite3")
    config = ConversationSummaryCompactionConfig(
        enabled=False,
        archive_retire_days=1,
        archive_retire_grace_days=1,
    )
    manager, calls = lifecycle_manager(
        tmp_path,
        compaction_enabled=False,
        compaction_config=config,
        use_store=use_store,
    )
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    summaries = []
    for start in (0, 2, 4, 6):
        manager.journal.append(
            address,
            ConversationBatch("conversation-1", closed_turn(start_sequence=start)),
        )
        source = manager.journal.seal(address, through_sequence=start + 1).segment
        summary = segment_summary(source)
        manager.segment_store.create(address, source, summary)
        summaries.append(summary)

    content = summary_content()

    def parent(stage, sources, generated_at):
        return ConversationRangeSummary(
            conversation_id="conversation-1",
            range_id=f"{sources[0].start_sequence:012d}-{sources[-1].end_sequence:012d}",
            stage=stage,
            source_refs=tuple(ConversationSummarySourceRef.from_summary(item) for item in sources),
            start_sequence=sources[0].start_sequence,
            end_sequence=sources[-1].end_sequence,
            started_at=sources[0].started_at,
            ended_at=sources[-1].ended_at,
            generated_at=generated_at,
            starts_mid_turn=False,
            ends_mid_turn=False,
            **content.to_dict(),
        )

    first_range = parent(ConversationRangeSummaryStage.RANGE, tuple(summaries[:2]), BASE_TIME + timedelta(minutes=1))
    second_range = parent(ConversationRangeSummaryStage.RANGE, tuple(summaries[2:]), BASE_TIME + timedelta(minutes=2))
    manager.range_store.create(address, first_range, tuple(summaries[:2]))
    manager.range_store.create(address, second_range, tuple(summaries[2:]))
    archive = parent(
        ConversationRangeSummaryStage.ARCHIVE,
        (first_range, second_range),
        BASE_TIME + timedelta(minutes=3),
    )
    manager.range_store.create(address, archive, (first_range, second_range))
    archive_reference = summary_reference(address, archive)
    use_store.record_use((archive_reference,), used_at=BASE_TIME + timedelta(days=10))
    monkeypatch.setattr(manager, "_workflow_is_committed", lambda _address, _segment: True)

    protected = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=10, hours=12)))
    assert protected.deleted_archive_summary_ids == ()
    assert use_store.read_many((archive_reference,))[0].retire_candidate_at is None

    candidate = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=12)))
    assert candidate.deleted_archive_summary_ids == ()
    candidate_state = use_store.read_many((archive_reference,))[0]
    assert candidate_state.retire_candidate_at == BASE_TIME + timedelta(days=12)
    manager.retirement_store.prepare(
        address,
        archive,
        (first_range, second_range),
        tuple(summaries),
        expected_use_version=candidate_state.version,
        prepared_at=BASE_TIME + timedelta(days=14),
    )
    search_context = ConversationSearchContextReader(
        manager.journal,
        manager.compactor,
        retirement_filter=manager.retirement_store,
    ).read(address)
    assert archive.range_id not in search_context.summary_context

    delete = manager.range_store.delete

    def interrupted_delete(target_address, stage, range_id):
        if stage is ConversationRangeSummaryStage.ARCHIVE:
            raise RuntimeError("interrupted before Archive delete")
        return delete(target_address, stage, range_id)

    monkeypatch.setattr(manager.range_store, "delete", interrupted_delete)
    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=14)))
    assert manager.journal.list_history(address) == ()
    assert manager.segment_store.list(address) == ()
    assert manager.range_store.list(address, ConversationRangeSummaryStage.RANGE) == ()
    assert manager.range_store.list(address, ConversationRangeSummaryStage.ARCHIVE) == (archive,)
    retiring_state = use_store.read_many((archive_reference,))[0]
    assert retiring_state.retiring_at == BASE_TIME + timedelta(days=14)
    assert manager.retirement_store.for_address(address)
    assert manager.summary_vector_index.sources.active(address) == ()

    monkeypatch.setattr(manager.range_store, "delete", delete)
    compact_once = manager.compactor.compact_once

    async def unexpected_compaction(*_args, **_kwargs):
        raise AssertionError("pending retirement must resume before new Summary compaction")

    monkeypatch.setattr(manager.compactor, "compact_once", unexpected_compaction)
    resumed = asyncio.run(
        manager.maintain_once(address, now=BASE_TIME + timedelta(days=15))
    )
    assert resumed.deleted_archive_summary_ids == (archive.range_id,)
    assert resumed.compaction.reason == "resumed pending Archive retirement before new compaction"
    monkeypatch.setattr(manager.compactor, "compact_once", compact_once)
    retired = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=17)))
    assert retired.deleted_archive_summary_ids == ()
    assert retired.deleted_range_summary_ids == ()
    assert retired.deleted_segment_summary_ids == ()
    assert retired.released_history_segment_ids == ()
    assert retired.purged_history_segment_ids == ()
    assert manager.journal.list_history(address) == ()
    assert manager.segment_store.list(address) == ()
    assert manager.range_store.list(address, ConversationRangeSummaryStage.RANGE) == ()
    assert manager.range_store.list(address, ConversationRangeSummaryStage.ARCHIVE) == ()
    assert use_store.read_many((archive_reference,)) == ()
    assert calls == [
        (address, None, ()),
        (address, None, ()),
        (address, None, (archive_reference.identity,)),
        (address, None, ()),
        (address, None, ()),
    ]
