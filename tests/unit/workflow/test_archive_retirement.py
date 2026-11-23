"""Archive 退休清单各相位的安全门与放弃分支：原文按段到期各自释放（阶段 3 裁定），退休只确认覆盖范围里一段原文都不剩。"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

from habitus.memory.conversation import (
    ConversationAddress,
    ConversationSummaryRetirementPhase,
    SQLiteConversationSummaryUseStore,
)
from habitus.memory.conversation.summary_reference import summary_reference
from habitus.pre.conversation import (
    ConversationBatch,
    ConversationRangeSummary,
    ConversationRangeSummaryStage,
    ConversationSummarySourceRef,
)
from tests.helpers import BASE_TIME, closed_turn, segment_summary, summary_content
from tests.unit.workflow.test_lifecycle_manager import commit_job, lifecycle_manager


def _prepared_retirement(tmp_path: Path, *, committed: bool = False):
    """两份 Range、一份 Archive 覆盖四份 History，并为 Archive 准备好退休清单。"""

    use_store = SQLiteConversationSummaryUseStore(tmp_path / "workflow" / "summary-use.sqlite3")
    manager = lifecycle_manager(tmp_path, compaction_enabled=False, use_store=use_store)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    summaries = []
    for start in (0, 2, 4, 6):
        manager.journal.append(address, ConversationBatch("conversation-1", closed_turn(start_sequence=start)))
        source = manager.journal.seal(address, through_sequence=start + 1).segment
        summary = segment_summary(source)
        manager.segment_store.create(address, source, summary)
        queued = manager.jobs.activate(manager.jobs.stage(address, source))
        if committed:
            commit_job(manager, queued)
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
    return manager, use_store, address, archive, manifest


def test_claimed_retirement_waits_while_history_in_its_range_is_retained(tmp_path: Path) -> None:
    """认领后确认覆盖范围里的原文都已释放；记忆任务还没到终态、原文还在时停在 RETIRING（已认领），原文原样保留。"""

    manager, use_store, address, archive, _manifest = _prepared_retirement(tmp_path)
    history = manager.journal.list_history(address)

    result = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=2)))

    assert result.released_history_segment_ids == ()
    assert result.deleted_archive_summary_ids == ()
    (pending,) = manager.retirement_store.for_address(address)
    assert pending.phase is ConversationSummaryRetirementPhase.RETIRING
    reference = summary_reference(address, archive)
    assert use_store.read_many((reference,))[0].retiring_at == BASE_TIME + timedelta(days=2)
    assert manager.journal.list_history(address) == history
    assert manager.range_store.list(address, ConversationRangeSummaryStage.ARCHIVE) == (archive,)


def test_retirement_is_abandoned_when_archive_use_changed_after_prepare(tmp_path: Path) -> None:
    manager, use_store, address, archive, _manifest = _prepared_retirement(tmp_path)
    reference = summary_reference(address, archive)
    use_store.record_use((reference,), used_at=BASE_TIME + timedelta(days=1, hours=1))

    result = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=2)))

    assert result.compaction.reason == "resumed pending Archive retirement before new compaction"
    assert result.deleted_archive_summary_ids == ()
    assert manager.retirement_store.for_address(address) == ()
    assert use_store.read_many((reference,))[0].retiring_at is None
    assert manager.range_store.list(address, ConversationRangeSummaryStage.ARCHIVE) == (archive,)


def test_retirement_finishes_after_history_expired_then_expired_jobs_are_cleaned(tmp_path: Path) -> None:
    """任务到终态满 10 天：同一次维护先按段释放原文，再推进退休删掉整条摘要链；任务再过保留期删掉。"""

    manager, use_store, address, archive, _manifest = _prepared_retirement(tmp_path, committed=True)
    reference = summary_reference(address, archive)

    retired = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=11)))

    assert len(retired.released_history_segment_ids) == 4
    assert retired.deleted_archive_summary_ids == (archive.range_id,)
    assert manager.journal.list_history(address) == ()
    assert manager.retirement_store.for_address(address) == ()
    assert use_store.read_many((reference,)) == ()
    assert retired.deleted_memory_job_sequences == ()

    expired = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=31)))

    assert expired.deleted_memory_job_sequences == (1, 2, 3, 4)
    assert manager.jobs.list_for_conversation(address) == ()


def test_archive_with_retained_history_is_not_marked_or_claimed(tmp_path: Path) -> None:
    """覆盖范围里还有没到期的原文：Archive 不进退休候选，免得认领后卡住这个会话的压缩。"""

    manager, use_store, address, archive, manifest = _prepared_retirement(tmp_path)
    manager.retirement_store.complete(manifest)
    reference = summary_reference(address, archive)
    before = use_store.read_many((reference,))[0]

    result = asyncio.run(manager.maintain_once(address, now=BASE_TIME + timedelta(days=4000)))

    assert result.deleted_archive_summary_ids == ()
    assert manager.retirement_store.for_address(address) == ()
    after = use_store.read_many((reference,))[0]
    assert (after.version, after.retiring_at) == (before.version, None)
