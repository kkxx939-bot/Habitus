"""Conversation 原文进入不可变 History 并生成全局有序 Job 的主链测试。"""

from datetime import date
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationRetentionPlanner,
    ConversationSegmentationConfig,
)
from habitus.memory.workflow import ConversationMemoryEnqueuer
from habitus.memory.workflow.jobs import MemoryJobStatus, MemoryJobStore
from habitus.pre.conversation import ConversationBatch
from tests.helpers import closed_turn
from tests.lock_store import ProcessLocalLockStore


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
        message for segment in history for message in segment.messages if message.role.value == "prompt"
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
    triggers = tuple(journal.read_segment(address, job.segment_id) for job in queued)
    editor_segments = tuple(
        None if trigger.ends_mid_turn else journal.read_editor_turn(address, trigger).whole for trigger in triggers
    )
    editor_inputs = tuple(
        "".join(str(message.content) for message in segment.messages if isinstance(message.content, str))
        for segment in editor_segments
        if segment is not None
    )

    assert editor_segments[:-1] == (None,) * (len(editor_segments) - 1)
    assert editor_segments[-1] is not None
    assert any(fact in source for source in editor_inputs)
    final_prompt = "".join(
        str(message.content) for message in editor_segments[-1].messages if message.role.value == "prompt"
    )
    assert final_prompt == source_prompt


def test_flush_completes_staged_job_left_before_history_publish_before_sealing_a_new_boundary(
    tmp_path: Path,
) -> None:
    journal, jobs, enqueuer = build_chain(tmp_path)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    journal.append(
        address,
        ConversationBatch("conversation-1", (*closed_turn(start_sequence=0), *closed_turn(start_sequence=2))),
    )
    stage = jobs.stage
    failures = [OSError("history write failed after STAGED")]

    def stage_then_fail(address_value, segment):
        staged = stage(address_value, segment)
        if failures:
            raise failures.pop()
        return staged

    jobs.stage = stage_then_fail  # type: ignore[method-assign]
    with pytest.raises(OSError, match="after STAGED"):
        enqueuer.seal_and_enqueue(address, through_sequence=1)
    jobs.stage = stage  # type: ignore[method-assign]
    assert journal.list_history(address) == ()

    result = enqueuer.flush(address)

    history = journal.list_history(address)
    assert tuple(item.segment_id for item in history) == ("000000000000-000000000001", "000000000002-000000000003")
    assert tuple((job.memory_sequence, job.segment_id, job.status) for job in result.jobs) == (
        (1, history[0].segment_id, MemoryJobStatus.QUEUED),
        (2, history[1].segment_id, MemoryJobStatus.QUEUED),
    )
    assert tuple(job.source_segment_digest for job in result.jobs) == tuple(item.digest for item in history)
    assert jobs.claim(jobs.oldest_uncommitted(), "worker").job.memory_sequence == 1


def test_after_turn_activates_staged_job_whose_history_was_published_before_its_activation(
    tmp_path: Path,
) -> None:
    journal, jobs, enqueuer = build_chain(tmp_path)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    journal.append(
        address,
        ConversationBatch("conversation-1", (*closed_turn(start_sequence=0), *closed_turn(start_sequence=2))),
    )
    activate = jobs.activate
    failures = [OSError("process stopped before activation")]

    def fail_once(job):
        if failures:
            raise failures.pop()
        return activate(job)

    jobs.activate = fail_once  # type: ignore[method-assign]
    with pytest.raises(OSError, match="before activation"):
        enqueuer.seal_and_enqueue(address, through_sequence=1)
    jobs.activate = activate  # type: ignore[method-assign]
    assert tuple(item.segment_id for item in journal.list_history(address)) == ("000000000000-000000000001",)

    result = enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch("conversation-1", closed_turn(start_sequence=4)),
        after_turn=True,
    )

    assert tuple((job.memory_sequence, job.segment_id, job.status) for job in result.jobs) == (
        (1, "000000000000-000000000001", MemoryJobStatus.QUEUED),
        (2, "000000000002-000000000003", MemoryJobStatus.QUEUED),
    )
    assert jobs.claim(jobs.oldest_uncommitted(), "worker").job.memory_sequence == 1


def _exhaust_a_staged_job(journal: ConversationMessageJournal, jobs: MemoryJobStore, enqueuer, address) -> None:  # type: ignore[no-untyped-def]
    """封存 0–1 时 history 发布报磁盘满：Job 1 停在 STAGED；之后每次补完都失败，重试用尽成 FAILED（从没被认领过）。"""

    publish = journal._publish_seal

    def disk_full(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError(28, "No space left on device")

    journal._publish_seal = disk_full  # type: ignore[method-assign]
    try:
        with pytest.raises(OSError, match="No space"):
            enqueuer.seal_and_enqueue(address, through_sequence=1)
    finally:
        journal._publish_seal = publish  # type: ignore[method-assign]
    job = jobs.oldest_uncommitted()
    while job.status is MemoryJobStatus.STAGED:
        job = jobs.record_staged_failure(job, OSError(28, "No space left on device"))
    assert job.status is MemoryJobStatus.FAILED and job.claim_generation == 0
    assert journal.list_history(address) == ()


def test_a_staged_job_that_exhausted_its_retries_is_not_bypassed_by_a_new_boundary(tmp_path: Path) -> None:
    """重试用尽的 STAGED Job 等人工处置时，他又说了一轮：前台先按它原来的边界发布 history（Job 照旧 FAILED、不激活），
    新边界接在后面；人工 ``retry_failed`` 之后补完读回那份 history、对上了就激活（U15）。"""

    from habitus.memory.workflow.recovery import MemoryStagedJobRecovery

    journal, jobs, enqueuer = build_chain(tmp_path)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    journal.append(
        address,
        ConversationBatch("conversation-1", (*closed_turn(start_sequence=0), *closed_turn(start_sequence=2))),
    )
    _exhaust_a_staged_job(journal, jobs, enqueuer, address)

    result = enqueuer.append_and_maybe_enqueue(
        address, ConversationBatch("conversation-1", closed_turn(start_sequence=4)), after_turn=True
    )

    history = tuple(item.segment_id for item in journal.list_history(address))
    assert history == ("000000000000-000000000001", "000000000002-000000000003")
    assert tuple((job.memory_sequence, job.segment_id, job.status) for job in result.jobs) == (
        (2, "000000000002-000000000003", MemoryJobStatus.QUEUED),
    )
    failed = jobs.oldest_uncommitted()
    assert (failed.memory_sequence, failed.status) == (1, MemoryJobStatus.FAILED)

    retried = jobs.retry_failed(failed)
    assert retried.status is MemoryJobStatus.STAGED
    recovered = MemoryStagedJobRecovery(journal, jobs).recover(retried)

    assert recovered.status is MemoryJobStatus.QUEUED
    assert recovered.source_segment_digest == journal.list_history(address)[0].digest
    assert jobs.claim(jobs.oldest_uncommitted(), "worker").job.memory_sequence == 1


def test_a_retried_staged_job_whose_history_is_behind_newer_segments_is_activated_by_the_foreground(
    tmp_path: Path,
) -> None:
    """人工重试之后、Worker 补完之前前台又要封存：那份 STAGED Job 的 history 早已发布、后面还有新段，前台读回核对后激活。"""

    journal, jobs, enqueuer = build_chain(tmp_path)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    journal.append(
        address,
        ConversationBatch("conversation-1", (*closed_turn(start_sequence=0), *closed_turn(start_sequence=2))),
    )
    _exhaust_a_staged_job(journal, jobs, enqueuer, address)
    enqueuer.append_and_maybe_enqueue(
        address, ConversationBatch("conversation-1", closed_turn(start_sequence=4)), after_turn=True
    )
    jobs.retry_failed(jobs.oldest_uncommitted())

    result = enqueuer.append_and_maybe_enqueue(
        address, ConversationBatch("conversation-1", closed_turn(start_sequence=6)), after_turn=True
    )

    assert tuple((job.memory_sequence, job.segment_id, job.status) for job in result.jobs) == (
        (1, "000000000000-000000000001", MemoryJobStatus.QUEUED),
        (3, "000000000004-000000000005", MemoryJobStatus.QUEUED),
    )
