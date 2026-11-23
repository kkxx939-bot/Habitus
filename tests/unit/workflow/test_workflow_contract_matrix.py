"""耐久 Job、lease、"卡片已写完"阶段与 Memory Consumer 输出的契约矩阵。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from habitus.conversation.source import ConversationSourceError
from habitus.foundation.integrity import canonical_json
from habitus.memory.workflow import (
    ConversationMemoryIngestResult,
    MemoryConversationOutput,
    MemoryConversationOutputStore,
)
from habitus.memory.workflow.jobs import (
    MemoryJob,
    MemoryJobConfig,
    MemoryJobExecutionError,
    MemoryJobLease,
    MemoryJobNotReadyError,
    MemoryJobStatus,
)
from habitus.memory.workflow.jobs.model import MEMORY_JOB_ERROR_MAX_CHARS
from tests.helpers import BASE_TIME
from tests.unit.conversation.source_v2_helpers import NOW, ingest_result
from tests.unit.conversation.source_v2_helpers import source as conversation_source

DIGEST = "a" * 64
CLAIM_ID = "c" * 32


def _job(status: MemoryJobStatus, **overrides: object) -> MemoryJob:
    running = status is MemoryJobStatus.RUNNING
    values: dict[str, object] = {
        "memory_sequence": 1,
        "conversation_id": "conversation-1",
        "started_on": date(2026, 7, 1),
        "segment_id": "000000000000-000000000001",
        "source_segment_digest": DIGEST,
        "status": status,
        "attempts": 1 if status is MemoryJobStatus.FAILED else 0,
        "claim_id": CLAIM_ID if running else None,
        "claim_generation": 1 if running else 0,
        "worker_id": "worker-1" if running else None,
        "lease_expires_at": BASE_TIME + timedelta(minutes=2) if running else None,
        "next_attempt_at": None,
        "last_error": "failed" if status in {MemoryJobStatus.FAILED, MemoryJobStatus.SKIPPED} else None,
        "created_at": BASE_TIME,
        "updated_at": BASE_TIME,
    }
    values.update(overrides)
    return MemoryJob(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("attempts", range(1, 13))
@pytest.mark.parametrize(("base", "maximum"), [(0.1, 1.0), (1.0, 60.0), (2, 5), (10, 10)])
def test_job_retry_backoff_is_exponential_and_capped(attempts: int, base: float, maximum: float) -> None:
    config = MemoryJobConfig(retry_base_delay_seconds=base, retry_max_delay_seconds=maximum)
    assert config.retry_delay_seconds(attempts) == min(maximum, base * (2 ** (attempts - 1)))


@pytest.mark.parametrize("attempts", [0, -1])
def test_job_retry_backoff_rejects_invalid_attempt_count(attempts: int) -> None:
    with pytest.raises(ValueError):
        MemoryJobConfig().retry_delay_seconds(attempts)


@pytest.mark.parametrize("status", tuple(MemoryJobStatus))
def test_job_accepts_each_coherent_durable_status(status: MemoryJobStatus) -> None:
    value = _job(status)
    assert value.status is status
    assert value.source_identity == (
        value.conversation_id,
        value.started_on,
        value.segment_id,
        value.source_segment_digest,
    )


@pytest.mark.parametrize("status", tuple(MemoryJobStatus))
@pytest.mark.parametrize("memory_sequence", [0, -1])
def test_job_rejects_invalid_global_sequence_for_every_status(status: MemoryJobStatus, memory_sequence: int) -> None:
    with pytest.raises(ValueError):
        _job(status, memory_sequence=memory_sequence)


@pytest.mark.parametrize("field", ["conversation_id", "segment_id"])
@pytest.mark.parametrize("value", ["", " ", " leading", "trailing "])
def test_job_rejects_non_normalized_source_identity_text(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.STAGED, **{field: value})


@pytest.mark.parametrize("value", [datetime(2026, 7, 1)])
def test_job_started_on_requires_calendar_date_not_datetime(value: object) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.STAGED, started_on=value)


@pytest.mark.parametrize("field", ["source_segment_digest"])
@pytest.mark.parametrize("value", ["", "0", "G" * 64, "A" * 64, None, 1, True])
def test_job_rejects_invalid_lowercase_hex_identities(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.STAGED, **{field: value})


@pytest.mark.parametrize("attempts", [-1])
def test_job_rejects_invalid_attempt_state(attempts: int) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.QUEUED, attempts=attempts)


@pytest.mark.parametrize("generation", [-1])
def test_job_rejects_invalid_claim_generation(generation: int) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.QUEUED, claim_generation=generation)


@pytest.mark.parametrize("status", [status for status in MemoryJobStatus if status is not MemoryJobStatus.RUNNING])
@pytest.mark.parametrize(
    ("field", "value"),
    [("claim_id", CLAIM_ID), ("worker_id", "worker-1"), ("lease_expires_at", BASE_TIME + timedelta(minutes=1))],
)
def test_non_running_job_rejects_any_lease_ownership(status: MemoryJobStatus, field: str, value: object) -> None:
    with pytest.raises(ValueError, match="non-running"):
        _job(status, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        *((field, value) for field in ("claim_id", "worker_id") for value in (None, "", "invalid worker id!", 0, True)),
        ("lease_expires_at", None),
    ],
)
def test_running_job_requires_complete_normalized_lease_identity(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.RUNNING, **kwargs)


@pytest.mark.parametrize("worker_id", ["w", "worker-1", "worker.one", "worker:one", "A_1", "x" * 128])
def test_worker_id_accepts_exact_operational_character_set(worker_id: str) -> None:
    assert MemoryJob.valid_worker_id(worker_id)
    assert _job(MemoryJobStatus.RUNNING, worker_id=worker_id).worker_id == worker_id


@pytest.mark.parametrize("worker_id", ["", " worker", "worker ", "worker/1", "worker@1", "x" * 129, None, 1, True])
def test_worker_id_rejects_unsafe_or_oversized_values(worker_id: object) -> None:
    assert MemoryJob.valid_worker_id(worker_id) is False
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.RUNNING, worker_id=worker_id)


@pytest.mark.parametrize("status", [MemoryJobStatus.STAGED, MemoryJobStatus.QUEUED])
def test_only_staged_or_queued_jobs_may_carry_retry_schedule(status: MemoryJobStatus) -> None:
    values = {"next_attempt_at": BASE_TIME + timedelta(seconds=1), "attempts": 1, "last_error": "retry"}
    assert _job(status, **values).next_attempt_at is not None
    for other in (MemoryJobStatus.RUNNING, MemoryJobStatus.FAILED, MemoryJobStatus.COMMITTED):
        with pytest.raises(ValueError):
            _job(other, next_attempt_at=BASE_TIME + timedelta(seconds=1))


def test_new_staged_job_cannot_contain_retry_state_but_retried_stage_requires_error() -> None:
    with pytest.raises(ValueError, match="new staged"):
        _job(MemoryJobStatus.STAGED, next_attempt_at=BASE_TIME + timedelta(seconds=1))
    with pytest.raises(ValueError, match="requires its last error"):
        _job(MemoryJobStatus.STAGED, attempts=1)
    assert _job(MemoryJobStatus.STAGED, attempts=1, last_error="retry").last_error == "retry"


@pytest.mark.parametrize("last_error", ["", None, "x" * (MEMORY_JOB_ERROR_MAX_CHARS + 1)])
def test_job_rejects_invalid_error_when_failure_state_requires_it(last_error: str | None) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.FAILED, last_error=last_error)


@pytest.mark.parametrize("field", ["created_at", "updated_at"])
@pytest.mark.parametrize("value", [datetime(2026, 7, 1)])
def test_job_rejects_missing_non_datetime_or_naive_system_timestamps(field: str, value: datetime) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.STAGED, **{field: value})


def test_job_rejects_updated_time_before_creation() -> None:
    with pytest.raises(ValueError, match="precede"):
        _job(MemoryJobStatus.STAGED, updated_at=BASE_TIME - timedelta(seconds=1))


def test_job_lease_exposes_only_running_fencing_identity() -> None:
    running = _job(MemoryJobStatus.RUNNING)
    lease = MemoryJobLease(running)
    assert lease.claim_id == CLAIM_ID
    assert lease.claim_generation == 1
    assert lease.worker_id == "worker-1"
    assert lease.lease_expires_at == running.lease_expires_at
    assert lease.source_identity == running.source_identity
    for status in MemoryJobStatus:
        if status is MemoryJobStatus.RUNNING:
            continue
        with pytest.raises(ValueError, match="RUNNING"):
            MemoryJobLease(_job(status))


@pytest.mark.parametrize("available_at", [BASE_TIME, BASE_TIME + timedelta(days=1)])
def test_not_ready_error_normalizes_and_exposes_available_time(available_at: datetime) -> None:
    error = MemoryJobNotReadyError(available_at)
    assert error.available_at == available_at
    assert available_at.isoformat() in str(error)


@pytest.mark.parametrize("available_at", [datetime(2026, 7, 1)])
def test_not_ready_error_requires_timezone_aware_datetime(available_at: datetime) -> None:
    with pytest.raises(TypeError):
        MemoryJobNotReadyError(available_at)


@pytest.mark.parametrize("job", [None, _job(MemoryJobStatus.FAILED)])
def test_job_execution_error_optionally_carries_failed_job(job: MemoryJob | None) -> None:
    error = MemoryJobExecutionError("failed", job=job)
    assert error.job is job


PREFERENCE_URI = "memory://preferences/回答风格.md"
TOOL_URI = "memory://tools/read_file.md"


@pytest.mark.parametrize("status", [MemoryJobStatus.QUEUED, MemoryJobStatus.RUNNING, MemoryJobStatus.FAILED, MemoryJobStatus.COMMITTED])
def test_written_cards_stage_is_kept_through_retry_and_completion_states(status: MemoryJobStatus) -> None:
    """"卡片已写完"可以跟着任务排回队列、再执行、判失败或完成；地址清单原样保留。"""

    job = _job(status, cards_written=True, changed_uris=(PREFERENCE_URI,), deleted_uris=(TOOL_URI,))
    assert job.cards_written
    assert job.changed_uris == (PREFERENCE_URI,)
    assert job.deleted_uris == (TOOL_URI,)


@pytest.mark.parametrize("status", [MemoryJobStatus.STAGED, MemoryJobStatus.SKIPPED])
def test_staged_or_skipped_job_cannot_have_written_cards(status: MemoryJobStatus) -> None:
    with pytest.raises(ValueError, match="cannot have written cards"):
        _job(status, cards_written=True)


def test_written_uris_require_the_written_stage() -> None:
    with pytest.raises(ValueError, match="before its cards are written"):
        _job(MemoryJobStatus.QUEUED, changed_uris=(PREFERENCE_URI,))


@pytest.mark.parametrize(
    "uris",
    [
        (TOOL_URI, PREFERENCE_URI),  # 没排序
        (PREFERENCE_URI, PREFERENCE_URI),  # 重复
        ("memory://preferences",),  # 目录不是卡片
        ("viking://preferences/回答风格.md",),  # 不是 memory:// 协议
    ],
)
def test_written_uris_must_be_sorted_unique_canonical_card_uris(uris: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.COMMITTED, cards_written=True, changed_uris=uris)


def test_one_uri_cannot_be_both_written_and_deleted() -> None:
    with pytest.raises(ValueError, match="both written and deleted"):
        _job(MemoryJobStatus.COMMITTED, cards_written=True, changed_uris=(PREFERENCE_URI,), deleted_uris=(PREFERENCE_URI,))


@pytest.mark.parametrize("status", [MemoryJobStatus.COMMITTED, MemoryJobStatus.SKIPPED, MemoryJobStatus.FAILED])
def test_committed_skipped_and_failed_are_terminal_and_no_longer_hold_later_sequences(status: MemoryJobStatus) -> None:
    assert status.terminal


@pytest.mark.parametrize("status", [MemoryJobStatus.STAGED, MemoryJobStatus.QUEUED, MemoryJobStatus.RUNNING])
def test_unfinished_statuses_still_hold_later_sequences(status: MemoryJobStatus) -> None:
    assert not status.terminal


_INGEST = ("ingest_result_snapshot",)
BAD_OUTPUT_FIELDS: list[tuple[tuple[str | int, ...], object, type[Exception], str]] = [
    ((*_INGEST, "append", "appended_count"), "1", ValueError, "appended_count must be a non-negative integer"),
    ((*_INGEST, "append", "next_sequence"), True, ValueError, "next_sequence must be a non-negative integer"),
    ((*_INGEST, "jobs", 0, "memory_sequence"), "1", ValueError, "memory_sequence must be a positive integer"),
    ((*_INGEST, "jobs", 0, "conversation_id"), 1, ValueError, "conversation_id must be non-empty normalized"),
    ((*_INGEST, "jobs", 0, "segment_id"), None, ValueError, "segment_id must be non-empty normalized"),
    ((*_INGEST, "jobs", 0, "attempts"), 1.0, ValueError, "memory job attempts must be non-negative"),
    ((*_INGEST, "jobs", 0, "claim_id"), 1, ValueError, "running memory job requires a claim_id"),
    ((*_INGEST, "jobs", 0, "last_error"), 1, ValueError, "memory job last_error is invalid"),
    ((*_INGEST, "retention", "triggered"), "no", TypeError, "retention flags must be boolean"),
    ((*_INGEST, "retention", "pending_tokens"), "1", ValueError, "pending_tokens must be a non-negative integer"),
    ((*_INGEST, "retention", "budget_exceeded"), 0, TypeError, "budget_exceeded must be boolean"),
    ((*_INGEST, "retention", "reason"), None, ValueError, "retention reason must be non-empty text"),
    ((*_INGEST, "retention", "boundary_kind"), 1, ValueError, "boundary_kind must be non-empty text or None"),
    ((*_INGEST, "retention", "through_sequence"), "1", ValueError, "retention boundary must match"),
    ((*_INGEST, "retention", "chunker_version"), 1, ValueError, "chunker_version must be non-empty text"),
    (("output_id",), 1, ConversationSourceError, "memory output_id must be lowercase SHA-256 text"),
    (("processor_fingerprint",), None, ConversationSourceError, "processor_fingerprint must be lowercase SHA-256"),
]


def _memory_output() -> tuple[Any, MemoryConversationOutput]:
    envelope = conversation_source()
    base = ingest_result(envelope)
    result = ConversationMemoryIngestResult(base.append, (_job(MemoryJobStatus.STAGED),), base.retention)
    output = MemoryConversationOutput.create(
        source=envelope, processor_fingerprint="d" * 64, ingest_result=result, recorded_at=NOW
    )
    return envelope, output


def _bad_output_record(path: tuple[str | int, ...], bad: object) -> dict[str, Any]:
    record = _memory_output()[1].to_dict()
    container: Any = record
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = bad
    return record


@pytest.mark.parametrize(("path", "bad", "error", "message"), BAD_OUTPUT_FIELDS)
def test_memory_output_parser_rejects_wrong_types_from_disk(
    path: tuple[str | int, ...], bad: object, error: type[Exception], message: str
) -> None:
    """磁盘读回的 Memory Consumer 输出字段类型错，在解码器构造值对象前就被拒（W16）。"""

    with pytest.raises(error, match=message):
        MemoryConversationOutput.from_dict(_bad_output_record(path, bad))


@pytest.mark.parametrize(("path", "bad", "error", "message"), BAD_OUTPUT_FIELDS[:15:3])
def test_memory_output_store_rejects_wrong_types_read_back_from_disk(
    tmp_path: Path, path: tuple[str | int, ...], bad: object, error: type[Exception], message: str
) -> None:
    """类型错的输出文件经真实 ``MemoryConversationOutputStore.read`` 读回时被拒。"""

    store = MemoryConversationOutputStore(tmp_path, max_files_per_source=8, max_file_bytes=1 << 20)
    envelope, output = _memory_output()
    target = store._path(envelope.source_id, output.output_id)
    target.parent.mkdir(parents=True)
    target.write_text(canonical_json(_bad_output_record(path, bad)) + "\n", encoding="utf-8")
    with pytest.raises(error, match=message):
        store.read(envelope, output.output_id)
