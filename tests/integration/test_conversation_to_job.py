"""Conversation 原文进入不可变 History 并生成全局有序 Job 的主链测试。"""

from datetime import date
from pathlib import Path

from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationRetentionPlanner,
    ConversationSegmentationConfig,
)
from memory.workflow import ConversationMemoryEnqueuer
from memory.workflow.jobs import MemoryJobStatus, MemoryJobStore
from pre.conversation import ConversationBatch
from tests.helpers import closed_turn


def build_chain(tmp_path: Path) -> tuple[ConversationMessageJournal, MemoryJobStore, ConversationMemoryEnqueuer]:
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    jobs = MemoryJobStore(
        tmp_path / "workflow",
        path_lock,
        memory_root=tmp_path / "memory",
    )
    planner = ConversationRetentionPlanner(
        ConversationSegmentationConfig(
            commit_token_threshold=1,
            keep_recent_turn_count=1,
            retained_message_token_budget=10_000,
        ),
        token_estimator=lambda _message: 1,
    )
    return journal, jobs, ConversationMemoryEnqueuer(journal, jobs, planner)


def test_after_turn_seals_only_old_complete_turn_and_activates_matching_job(tmp_path: Path) -> None:
    journal, jobs, enqueuer = build_chain(tmp_path)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    batch = ConversationBatch(
        "conversation-1",
        (*closed_turn(start_sequence=0), *closed_turn(start_sequence=2)),
    )

    result = enqueuer.append_and_maybe_enqueue(address, batch, after_turn=True)

    assert len(result.jobs) == 1
    job = result.jobs[0]
    history = journal.list_history(address)
    assert job.status is MemoryJobStatus.QUEUED
    assert job.memory_sequence == 1
    assert job.segment_id == history[0].segment_id
    assert job.source_segment_digest == history[0].digest
    assert tuple(message.sequence for message in journal.read_live(address).messages) == (2, 3)
    assert jobs.oldest_uncommitted() == job


def test_replaying_same_append_does_not_duplicate_history_or_job(tmp_path: Path) -> None:
    journal, jobs, enqueuer = build_chain(tmp_path)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    batch = ConversationBatch(
        "conversation-1",
        (*closed_turn(start_sequence=0), *closed_turn(start_sequence=2)),
    )

    first = enqueuer.append_and_maybe_enqueue(address, batch, after_turn=True)
    replay = enqueuer.append_and_maybe_enqueue(address, batch, after_turn=True)

    assert first.jobs[0] == jobs.oldest_uncommitted()
    assert replay.jobs == ()
    assert len(journal.list_history(address)) == 1
    assert jobs.high_watermark() == 1


def test_flush_seals_remaining_complete_turn_without_closing_conversation(tmp_path: Path) -> None:
    journal, jobs, enqueuer = build_chain(tmp_path)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch(
            "conversation-1",
            (*closed_turn(start_sequence=0), *closed_turn(start_sequence=2)),
        ),
        after_turn=True,
    )

    flushed = enqueuer.flush(address)

    assert len(flushed.jobs) == 1
    assert flushed.jobs[0].memory_sequence == 2
    assert flushed.jobs[0].status is MemoryJobStatus.QUEUED
    assert journal.read_live(address) is None
    assert tuple(segment.segment_id for segment in journal.list_history(address)) == (
        "000000000000-000000000001",
        "000000000002-000000000003",
    )
    assert jobs.high_watermark() == 2

