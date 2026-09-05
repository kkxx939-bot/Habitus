"""跨 Conversation 全局有序 Job、耐久租约、退避和人工恢复入口测试。"""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.memory.conversation import ConversationAddress
from habitus.memory.workflow.jobs import (
    MemoryJobBlockedError,
    MemoryJobConfig,
    MemoryJobError,
    MemoryJobLeaseLostError,
    MemoryJobNotReadyError,
    MemoryJobStatus,
    MemoryJobStore,
)
from tests.helpers import BASE_TIME, closed_turn, segment


class Clock:
    def __init__(self) -> None:
        self.now = BASE_TIME

    def __call__(self):
        return self.now

    def advance(self, **kwargs: int) -> None:
        self.now += timedelta(**kwargs)


def store(tmp_path: Path, clock: Clock, *, max_attempts: int = 3) -> MemoryJobStore:
    return MemoryJobStore(
        tmp_path / "workflow",
        PathLock(ProcessLocalLockStore()),
        memory_root=tmp_path / "memory",
        config=MemoryJobConfig(
            max_attempts=max_attempts,
            lease_ttl_seconds=3,
            retry_base_delay_seconds=1,
            retry_max_delay_seconds=4,
        ),
        clock=clock,
    )


def source(conversation_id: str, start_sequence: int):
    return segment(
        conversation_id=conversation_id,
        segment_id=f"{start_sequence:012d}-{start_sequence + 1:012d}",
        messages=closed_turn(start_sequence=start_sequence),
    )


def test_stage_is_idempotent_activate_is_separate_and_sequences_are_global(tmp_path: Path) -> None:
    clock = Clock()
    jobs = store(tmp_path, clock)
    first_address = ConversationAddress("conversation-a", date(2026, 7, 1))
    second_address = ConversationAddress("conversation-b", date(2026, 7, 1))
    first_source = source("conversation-a", 0)
    second_source = source("conversation-b", 0)

    staged = jobs.stage(first_address, first_source)
    assert jobs.stage(first_address, first_source) == staged
    second = jobs.stage(second_address, second_source)
    assert (staged.memory_sequence, second.memory_sequence) == (1, 2)
    assert staged.status is MemoryJobStatus.STAGED
    assert jobs.activate(staged).status is MemoryJobStatus.QUEUED
    assert jobs.high_watermark() == 2


def test_oldest_job_blocks_later_conversations_until_committed(tmp_path: Path) -> None:
    clock = Clock()
    jobs = store(tmp_path, clock)
    address_a = ConversationAddress("conversation-a", date(2026, 7, 1))
    address_b = ConversationAddress("conversation-b", date(2026, 7, 1))
    first = jobs.activate(jobs.stage(address_a, source("conversation-a", 0)))
    second = jobs.activate(jobs.stage(address_b, source("conversation-b", 0)))

    with pytest.raises(Exception, match="oldest"):
        jobs.claim(second, "worker-2")
    lease = jobs.claim(first, "worker-1")
    completed = jobs.complete(lease)
    assert completed.status is MemoryJobStatus.COMMITTED
    assert jobs.claim(second, "worker-2").job.memory_sequence == 2


def test_active_lease_excludes_other_worker_and_expired_lease_is_fenced_by_generation(tmp_path: Path) -> None:
    clock = Clock()
    jobs = store(tmp_path, clock)
    address = ConversationAddress("conversation-a", date(2026, 7, 1))
    queued = jobs.activate(jobs.stage(address, source("conversation-a", 0)))
    first = jobs.claim(queued, "worker-1")
    with pytest.raises(MemoryJobBlockedError, match="active worker"):
        jobs.claim(first.job, "worker-2")

    clock.advance(seconds=4)
    replacement = jobs.claim(first.job, "worker-2")
    assert replacement.claim_generation == first.claim_generation + 1
    with pytest.raises(MemoryJobLeaseLostError):
        jobs.assert_current(first)
    assert jobs.assert_current(replacement).worker_id == "worker-2"


def test_retryable_failure_uses_exponential_backoff_then_exhausts_and_manual_retry_reopens(tmp_path: Path) -> None:
    clock = Clock()
    jobs = store(tmp_path, clock, max_attempts=2)
    address = ConversationAddress("conversation-a", date(2026, 7, 1))
    queued = jobs.activate(jobs.stage(address, source("conversation-a", 0)))

    first_lease = jobs.claim(queued, "worker-1")
    retry = jobs.fail(first_lease, RuntimeError("temporary"), retryable=True)
    assert retry.status is MemoryJobStatus.QUEUED
    assert retry.next_attempt_at == clock.now + timedelta(seconds=1)
    with pytest.raises(MemoryJobNotReadyError):
        jobs.claim(retry, "worker-1")

    clock.advance(seconds=1)
    second_lease = jobs.claim(retry, "worker-1")
    failed = jobs.fail(second_lease, RuntimeError("still broken"), retryable=True)
    assert failed.status is MemoryJobStatus.FAILED
    with pytest.raises(MemoryJobBlockedError, match="exhausted"):
        jobs.claim(failed, "worker-1")

    reopened = jobs.retry_failed(failed)
    assert reopened.status is MemoryJobStatus.QUEUED
    assert reopened.attempts == 0
    assert reopened.last_error is None
    assert jobs.claim(reopened, "worker-2").worker_id == "worker-2"


def test_non_retryable_failure_blocks_immediately_and_high_watermark_survives_cleanup(tmp_path: Path) -> None:
    clock = Clock()
    jobs = store(tmp_path, clock)
    address = ConversationAddress("conversation-a", date(2026, 7, 1))
    queued = jobs.activate(jobs.stage(address, source("conversation-a", 0)))
    failed = jobs.fail(jobs.claim(queued, "worker"), ValueError("bad data"), retryable=False)
    assert failed.status is MemoryJobStatus.FAILED
    reopened = jobs.retry_failed(failed)
    committed = jobs.complete(jobs.claim(reopened, "worker"))
    assert jobs.discard_committed(committed)
    assert jobs.high_watermark() == 1
    assert not jobs.discard_committed(committed)


def test_explicit_abandonment_is_durable_idempotent_and_releases_later_sequence(tmp_path: Path) -> None:
    clock = Clock()
    jobs = store(tmp_path, clock)
    first_address = ConversationAddress("conversation-a", date(2026, 7, 1))
    second_address = ConversationAddress("conversation-b", date(2026, 7, 1))
    first = jobs.activate(jobs.stage(first_address, source("conversation-a", 0)))
    second = jobs.activate(jobs.stage(second_address, source("conversation-b", 0)))
    failed = jobs.fail(jobs.claim(first, "worker"), ValueError("permanent"), retryable=False)

    record = jobs.abandon_failed(failed, reason="人工确认来源数据不可恢复")

    assert record.memory_sequence == failed.memory_sequence
    assert record.reason == "人工确认来源数据不可恢复"
    assert jobs.try_read_abandonment(failed) == record
    assert jobs.abandon_failed(failed, reason="重复请求不会覆盖原始决定") == record
    with pytest.raises(MemoryJobBlockedError, match="abandonment decision"):
        jobs.retry_failed(failed)
    assert jobs.oldest_uncommitted() == second
    assert jobs.claim(second, "worker-2").job.memory_sequence == 2


def test_durable_failed_job_without_error_is_rejected_as_corrupt_state(tmp_path: Path) -> None:
    clock = Clock()
    jobs = store(tmp_path, clock)
    address = ConversationAddress("conversation-a", date(2026, 7, 1))
    queued = jobs.activate(jobs.stage(address, source("conversation-a", 0)))
    failed = jobs.fail(jobs.claim(queued, "worker"), ValueError("bad data"), retryable=False)
    assert failed.status is MemoryJobStatus.FAILED

    job_path = next(path for path in jobs.jobs_root.glob("*.json") if path.name != "state.json")
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload["last_error"] = None
    job_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MemoryJobError, match="fields are invalid"):
        jobs.oldest_uncommitted()


def test_retry_delay_is_bounded_exponential() -> None:
    config = MemoryJobConfig(retry_base_delay_seconds=2, retry_max_delay_seconds=5)
    assert [config.retry_delay_seconds(attempt) for attempt in (1, 2, 3, 4)] == [2, 4, 5, 5]
