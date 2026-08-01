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


def test_oversized_turn_uses_regular_contiguous_history_segments_and_jobs(tmp_path: Path) -> None:
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    jobs = MemoryJobStore(
        tmp_path / "workflow",
        path_lock,
        memory_root=tmp_path / "memory",
    )
    config = ConversationSegmentationConfig(
        commit_token_threshold=1,
        keep_recent_turn_count=1,
        retained_message_token_budget=30,
        max_segment_tokens=30,
        max_live_messages=1_000,
        max_live_bytes=1_000_000,
        max_segment_messages=1_000,
        max_segment_bytes=1_000_000,
    )
    planner = ConversationRetentionPlanner(
        config,
        token_estimator=lambda item: len(item.content) if isinstance(item.content, str) else 1,
    )
    enqueuer = ConversationMemoryEnqueuer(journal, jobs, planner)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source_prompt = "A" * 80

    appended = enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-1", closed_turn(prompt=source_prompt)),
        after_turn=True,
    )
    replay = enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-1", closed_turn(prompt=source_prompt)),
        after_turn=True,
    )
    flushed = enqueuer.flush(address)
    history = journal.list_history(address)

    assert len(appended.jobs) == 2
    assert replay.append.appended_count == 0
    assert replay.jobs == ()
    assert len(flushed.jobs) == 1
    assert tuple(item.segment_id for item in history) == (
        "000000000000-000000000000",
        "000000000001-000000000001",
        "000000000002-000000000003",
    )
    assert tuple((item.starts_mid_turn, item.ends_mid_turn) for item in history) == (
        (False, True),
        (True, True),
        (True, False),
    )
    prompt_parts = tuple(
        message
        for segment in history
        for message in segment.messages
        if message.role.value == "prompt"
    )
    assert "".join(str(item.content) for item in prompt_parts) == source_prompt
    assert len({item.logical_message_id for item in prompt_parts}) == 1
    assert journal.read_live(address) is None
    assert jobs.high_watermark() == 3


def test_oversized_logical_message_does_not_split_one_fact_across_memory_jobs(tmp_path: Path) -> None:
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
            retained_message_token_budget=30,
            max_segment_tokens=30,
            max_live_messages=1_000,
            max_live_bytes=1_000_000,
            max_segment_messages=1_000,
            max_segment_bytes=1_000_000,
        ),
        token_estimator=lambda item: len(item.content) if isinstance(item.content, str) else 1,
    )
    enqueuer = ConversationMemoryEnqueuer(journal, jobs, planner)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    fact = "我喜欢蓝色"
    source_prompt = "A" * 28 + fact + "B" * 60

    appended = enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-1", closed_turn(prompt=source_prompt)),
        after_turn=True,
    )
    flushed = enqueuer.flush(address)
    queued = (*appended.jobs, *flushed.jobs)
    editor_segments = tuple(
        journal.read_editor_segment(
            address,
            journal.read_segment(address, job.segment_id),
        )
        for job in queued
    )
    editor_inputs = tuple(
        "".join(
            str(message.content)
            for message in segment.messages
            if isinstance(message.content, str)
        )
        for segment in editor_segments
        if segment is not None
    )

    assert editor_segments[:-1] == (None,) * (len(editor_segments) - 1)
    assert editor_segments[-1] is not None
    assert any(fact in source for source in editor_inputs)
    final_prompt = "".join(
        str(message.content)
        for message in editor_segments[-1].messages
        if message.role.value == "prompt"
    )
    assert final_prompt == source_prompt
