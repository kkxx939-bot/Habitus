"""Archive 退休清单各相位的安全门与放弃分支。"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from habitus.memory.conversation import (
    ConversationAddress,
    ConversationSummaryRetirementPhase,
    SQLiteConversationSummaryUseStore,
)
from habitus.memory.conversation.indexing import summary_reference
from habitus.memory.workflow import MemoryChangeReceipt, MemoryChangeReceiptState, MemoryChangeSource
from habitus.pre.conversation import (
    ConversationBatch,
    ConversationRangeSummary,
    ConversationRangeSummaryStage,
    ConversationSummarySourceRef,
)
from tests.helpers import BASE_TIME, closed_turn, segment_summary, summary_content
from tests.unit.workflow.test_lifecycle_manager import lifecycle_manager


def _commit_workflow(manager, queued) -> None:
    """把一个 QUEUED Job 推到 COMMITTED，并写入对应的 COMMITTED 变更回执。"""

    job = manager.jobs.complete(manager.jobs.claim(queued, "worker"))
    manager.receipts._create(
        MemoryChangeReceipt(
            source=MemoryChangeSource.from_job(job),
            state=MemoryChangeReceiptState.COMMITTED,
            prepared_at=BASE_TIME,
            committed_at=BASE_TIME,
            expected_created_uris=(),
            expected_updated_uris=(),
            expected_deleted_uris=(),
            unchanged_uris=(),
            prepared_node_changes=(),
            identity_changes=(),
            added_relations=(),
            removed_relations=(),
            node_changes=(),
        )
    )


def _prepared_retirement(tmp_path: Path, *, committed: bool = False):
    """两份 Range、一份 Archive 覆盖四份 History，并为 Archive 准备好退休清单。"""

    use_store = SQLiteConversationSummaryUseStore(tmp_path / "workflow" / "summary-use.sqlite3")
    manager, calls = lifecycle_manager(tmp_path, compaction_enabled=False, use_store=use_store)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    summaries = []
    for start in (0, 2, 4, 6):
        manager.journal.append(address, ConversationBatch("conversation-1", closed_turn(start_sequence=start)))
        source = manager.journal.seal(address, through_sequence=start + 1).segment
        summary = segment_summary(source)
        manager.segment_store.create(address, source, summary)
        queued = manager.jobs.activate(manager.jobs.stage(address, source))
        if committed:
            _commit_workflow(manager, queued)
        summaries.append(summary)

    def parent(stage, sources, minutes):
        return ConversationRangeSummary(
            conversation_id="conversation-1",
            range_id=f"{sources[0].start_sequence:012d}-{sources[-1].end_sequence:012d}",
            stage=stage,
            source_refs=tuple(ConversationSummarySourceRef.from_summary(item) for item in sources),
            start_sequence=sources[0].start_sequence,
            end_sequence=sources[-1].end_sequence,
            started_at=sources[0].started_at,
            ended_at=sources[-1].ended_at,
            generated_at=BASE_TIME + timedelta(minutes=minutes),
            starts_mid_turn=False,
            ends_mid_turn=False,
            **summary_content().to_dict(),
        )

    first = parent(ConversationRangeSummaryStage.RANGE, tuple(summaries[:2]), 1)
    second = parent(ConversationRangeSummaryStage.RANGE, tuple(summaries[2:]), 2)
    manager.range_store.create(address, first, tuple(summaries[:2]))
    manager.range_store.create(address, second, tuple(summaries[2:]))
    archive = parent(ConversationRangeSummaryStage.ARCHIVE, (first, second), 3)
    manager.range_store.create(address, archive, (first, second))
    reference = summary_reference(address, archive)
    use_store.mark_retire_candidate(reference, marked_at=BASE_TIME)
    version = use_store.read_many((reference,))[0].version
    manifest = manager.retirement_store.prepare(
        address,
        archive,
        (first, second),
        tuple(summaries),
        expected_use_version=version,
        prepared_at=BASE_TIME + timedelta(days=1),
    )
    return manager, calls, use_store, address, archive, manifest


def test_history_is_not_released_while_its_memory_workflow_is_uncommitted(tmp_path: Path) -> None:
    manager, calls, _use_store, address, archive, _manifest = _prepared_retirement(tmp_path)
    history = manager.journal.list_history(address)

    result = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=2)))

    assert result.released_history_segment_ids == result.purged_history_segment_ids == ()
    assert result.deleted_archive_summary_ids == ()
    assert calls == [(address, None, (summary_reference(address, archive).identity,)), (address, None, ())]
    (pending,) = manager.retirement_store.for_address(address)
    assert pending.phase is ConversationSummaryRetirementPhase.INDEX_REMOVED
    assert manager.journal.list_history(address) == history
    assert manager.journal.read_state(address).released_through < archive.start_sequence
    assert manager.range_store.list(address, ConversationRangeSummaryStage.ARCHIVE) == (archive,)


def test_retirement_is_abandoned_when_archive_use_changed_after_prepare(tmp_path: Path) -> None:
    manager, calls, use_store, address, archive, _manifest = _prepared_retirement(tmp_path)
    reference = summary_reference(address, archive)
    use_store.record_use((reference,), used_at=BASE_TIME + timedelta(days=1, hours=1))

    result = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=2)))

    assert result.compaction.reason == "resumed pending Archive retirement before new compaction"
    assert result.released_history_segment_ids == result.deleted_archive_summary_ids == ()
    assert calls == [(address, None, ())]
    assert manager.retirement_store.for_address(address) == ()
    assert use_store.read_many((reference,))[0].retiring_at is None
    assert manager.range_store.list(address, ConversationRangeSummaryStage.ARCHIVE) == (archive,)


def test_finished_retirement_completes_manifest_then_expired_workflow_records_are_cleaned(tmp_path: Path) -> None:
    manager, _calls, use_store, address, archive, _manifest = _prepared_retirement(tmp_path, committed=True)
    reference = summary_reference(address, archive)

    retired = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=2)))

    assert retired.deleted_archive_summary_ids == (archive.range_id,)
    assert len(retired.released_history_segment_ids) == 4
    assert manager.retirement_store.for_address(address) == ()
    assert use_store.read_many((reference,)) == ()
    assert retired.deleted_memory_job_sequences == ()
    assert retired.deleted_memory_receipt_ids == ()

    expired = asyncio.run(manager.maintain_once(address, now=datetime.now(UTC) + timedelta(days=400)))

    assert expired.deleted_memory_job_sequences == (1, 2, 3, 4)
    assert len(expired.deleted_memory_receipt_ids) == 4
    assert manager.jobs.list_for_conversation(address) == ()
    assert manager.receipts.list_for_conversation(address) == ()
