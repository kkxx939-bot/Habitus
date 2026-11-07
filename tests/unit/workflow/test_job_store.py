"""跨 Conversation 全局有序 Job、耐久租约、退避和人工恢复入口测试。"""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts.path_lock import PathLock
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
from tests.lock_store import ProcessLocalLockStore


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


def test_waiting_for_the_model_uses_its_own_long_backoff_and_never_exhausts_attempts(tmp_path: Path) -> None:
    """模型此刻用不了（余额、鉴权、断网）：长退避排回队列、不计入 max_attempts，停多久都不进 FAILED；模型恢复后
    下一次认领照常跑，别的失败把"等模型"清零、照旧按 max_attempts 收（裁定 80 问题 2）。"""

    clock = Clock()
    jobs = store(tmp_path, clock, max_attempts=2)
    address = ConversationAddress("conversation-a", date(2026, 7, 1))
    job = jobs.activate(jobs.stage(address, source("conversation-a", 0)))

    delays = []
    for _ in range(12):  # 远多于 max_attempts=2
        waiting = jobs.fail(jobs.claim(job, "worker"), RuntimeError("Insufficient Balance"), waits_for_model=True)
        assert waiting.status is MemoryJobStatus.QUEUED and waiting.waiting_for_model
        assert waiting.attempts == 0
        assert waiting.next_attempt_at is not None
        delays.append((waiting.next_attempt_at - clock.now).total_seconds())
        with pytest.raises(MemoryJobNotReadyError):
            jobs.claim(waiting, "worker")
        clock.advance(seconds=int(delays[-1]))
        job = waiting
    assert delays[:4] == [60, 120, 240, 480] and delays[-1] == 86_400
    assert jobs.oldest_uncommitted() == job  # 落盘后读回同一份（含 model_waits）

    # 恢复之后别的失败：等模型清零，照旧按 max_attempts 收
    retried = jobs.fail(jobs.claim(job, "worker"), RuntimeError("temporary"), retryable=True)
    assert retried.status is MemoryJobStatus.QUEUED and retried.model_waits == 0 and retried.attempts == 1
    clock.advance(seconds=1)
    failed = jobs.fail(jobs.claim(retried, "worker"), RuntimeError("still broken"), retryable=True)
    assert failed.status is MemoryJobStatus.FAILED and not failed.waiting_for_model


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("memory_sequence", True, "memory_sequence must be a positive integer"),
        ("memory_sequence", "1", "memory_sequence must be a positive integer"),
        ("conversation_id", 1, "conversation_id must be non-empty normalized text"),
        ("segment_id", None, "segment_id must be non-empty normalized text"),
        ("source_segment_digest", 1, "source_segment_digest must be lowercase SHA-256 text"),
        ("transaction_id", None, "transaction_id must be 32 lowercase hexadecimal characters"),
        ("attempts", 1.0, "memory job attempts must be non-negative"),
        ("claim_id", 5, "running memory job requires a claim_id"),
        ("claim_generation", "1", "memory job claim_generation must be non-negative"),
        ("worker_id", ["worker"], "running memory job requires a normalized worker_id"),
        ("last_error", 3, "memory job last_error is invalid"),
        ("model_waits", False, "memory job model_waits must be non-negative"),
    ],
)
def test_job_file_with_wrongly_typed_field_is_rejected_by_the_decoder(
    tmp_path: Path, field: str, bad: object, message: str
) -> None:
    """磁盘读回的 Job 字段类型错时在解码器就被拒（W16：类型检查在边界，不靠 ``MemoryJob`` 构造器）。"""

    jobs = store(tmp_path, Clock())
    jobs.stage(ConversationAddress("conversation-a", date(2026, 7, 1)), source("conversation-a", 0))
    job_path = next(path for path in jobs.jobs_root.glob("*.json") if path.name != "state.json")
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload[field] = bad
    job_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MemoryJobError, match="fields are invalid") as caught:
        jobs.oldest_uncommitted()
    assert isinstance(caught.value.__cause__, ValueError)
    assert message in str(caught.value.__cause__)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("created_at", 1),
        ("created_at", None),
        ("updated_at", "time"),
        ("updated_at", "2026-07-01T00:00:00"),
        ("started_on", 1),
        ("started_on", None),
        ("lease_expires_at", 0),
        ("next_attempt_at", "time"),
    ],
)
def test_job_file_with_wrongly_typed_time_is_rejected_by_the_decoder(tmp_path: Path, field: str, bad: object) -> None:
    """磁盘读回的 Job 时间与日期字段类型错时在解码器就被拒（W16：``MemoryJob`` 构造器不再查类型）。"""

    jobs = store(tmp_path, Clock())
    jobs.stage(ConversationAddress("conversation-a", date(2026, 7, 1)), source("conversation-a", 0))
    job_path = next(path for path in jobs.jobs_root.glob("*.json") if path.name != "state.json")
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    payload[field] = bad
    job_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MemoryJobError, match="fields are invalid"):
        jobs.oldest_uncommitted()


@pytest.mark.parametrize("bad", [True, "3", 3.0, None])
def test_sequence_state_with_wrongly_typed_high_watermark_is_rejected(tmp_path: Path, bad: object) -> None:
    jobs = store(tmp_path, Clock())
    jobs.initialize()
    jobs.state_path.write_text(
        json.dumps({"schema": "memory_job_sequence_state_v1", "last_allocated_memory_sequence": bad}),
        encoding="utf-8",
    )

    with pytest.raises(MemoryJobError, match="high-watermark must be non-negative"):
        jobs.initialize()


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("memory_sequence", "1", "memory_sequence must be a positive integer"),
        ("conversation_id", None, "conversation_id must be non-empty normalized text"),
        ("segment_id", 1, "segment_id must be non-empty normalized text"),
        ("source_segment_digest", None, "source_segment_digest must be lowercase SHA-256 text"),
        ("transaction_id", 1, "transaction_id must be 32 lowercase hexadecimal characters"),
        ("attempts", True, "attempts must be a positive integer"),
        ("last_error", None, "last_error must be non-empty bounded text"),
        ("reason", ["原因"], "reason must be non-empty bounded text"),
    ],
)
def test_abandonment_file_with_wrongly_typed_field_is_rejected_by_the_decoder(
    tmp_path: Path, field: str, bad: object, message: str
) -> None:
    jobs = store(tmp_path, Clock())
    queued = jobs.activate(
        jobs.stage(ConversationAddress("conversation-a", date(2026, 7, 1)), source("conversation-a", 0))
    )
    failed = jobs.fail(jobs.claim(queued, "worker"), ValueError("permanent"), retryable=False)
    jobs.abandon_failed(failed, reason="人工确认")
    record_path = next(jobs.abandonments_root.glob("*.json"))
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload[field] = bad
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MemoryJobError, match="failed to read memory job abandonment") as caught:
        jobs.try_read_abandonment(failed)
    assert isinstance(caught.value.__cause__, ValueError)
    assert message in str(caught.value.__cause__)
