"""验证 HTTP Handler 对 Runtime 结果的语义表达，不重复路由测试。"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from integrations.http import HTTPMemoryJobConflictError, RuntimeHTTPHandlers
from memory.workflow import MemoryJob, MemoryJobStatus
from Runtime import (
    MemoryConsistencySnapshot,
    MemoryConsistencyState,
    Runtime,
)

UTC = timezone.utc


def _runtime(*, adapted_after_turn: bool) -> Runtime:
    runtime = object.__new__(Runtime)
    runtime.append_protocol_conversation = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            adaptation=SimpleNamespace(
                ignored_items=0,
                after_turn=adapted_after_turn,
            ),
            ingest=SimpleNamespace(jobs=()),
        )
    )
    return runtime


def _job(sequence: int, status: MemoryJobStatus, *, updated_offset: int = 0) -> MemoryJob:
    failed = status is MemoryJobStatus.FAILED
    return MemoryJob(
        memory_sequence=sequence,
        conversation_id="conversation-handler",
        started_on=date(2026, 7, 30),
        segment_id=f"segment-{sequence}",
        source_segment_digest=f"{sequence:064x}",
        transaction_id=f"{sequence:032x}",
        status=status,
        attempts=2 if failed else 0,
        claim_id=None,
        claim_generation=1 if failed else 0,
        worker_id=None,
        lease_expires_at=None,
        next_attempt_at=None,
        last_error=(
            "Bearer private-token failed at /Users/alice/private/file.txt"
            if failed
            else None
        ),
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
        updated_at=datetime(2026, 7, 30, tzinfo=UTC) + timedelta(seconds=updated_offset),
    )


def _job_runtime(jobs: tuple[MemoryJob, ...], *, blocked: MemoryJob | None) -> Runtime:
    runtime = object.__new__(Runtime)
    runtime.list_memory_jobs = AsyncMock(return_value=jobs)  # type: ignore[method-assign]
    runtime.failed_memory_job = AsyncMock(return_value=blocked)  # type: ignore[method-assign]

    async def consistency(job: MemoryJob) -> MemoryConsistencySnapshot:
        state = {
            MemoryJobStatus.FAILED: MemoryConsistencyState.FAILED,
            MemoryJobStatus.COMMITTED: MemoryConsistencyState.COMMITTED,
        }.get(job.status, MemoryConsistencyState.PENDING)
        return MemoryConsistencySnapshot(state, job, job, None, None)

    runtime.memory_consistency = consistency  # type: ignore[method-assign]
    return runtime


def test_remember_reports_adapter_boundary_when_caller_does_not_override_it() -> None:
    handler = RuntimeHTTPHandlers(_runtime(adapted_after_turn=True))

    result = asyncio.run(
        handler.remember(
            conversation_id="conversation-handler",
            started_on=date(2026, 7, 30),
            protocol="openai",
            payload={},
            start_sequence=0,
            occurred_at=datetime(2026, 7, 30, tzinfo=UTC),
            after_turn=None,
        )
    )

    assert result["after_turn"] is True


def test_job_queries_apply_reverse_pagination_filtering_and_safe_failure_details() -> None:
    jobs = (
        _job(1, MemoryJobStatus.COMMITTED),
        _job(2, MemoryJobStatus.QUEUED),
        _job(3, MemoryJobStatus.FAILED, updated_offset=3),
    )
    handler = RuntimeHTTPHandlers(_job_runtime(jobs, blocked=jobs[-1]))

    first_page = asyncio.run(
        handler.list_jobs(
            conversation_id="conversation-handler",
            started_on=date(2026, 7, 30),
            limit=2,
        )
    )
    queued = asyncio.run(
        handler.list_jobs(
            conversation_id="conversation-handler",
            started_on=date(2026, 7, 30),
            status=MemoryJobStatus.QUEUED,
            limit=10,
        )
    )
    older = asyncio.run(
        handler.list_jobs(
            conversation_id="conversation-handler",
            started_on=date(2026, 7, 30),
            before_sequence=2,
            limit=10,
        )
    )
    blocked = asyncio.run(handler.blocked_job())
    status = asyncio.run(
        handler.job_status(
            3,
            conversation_id="conversation-handler",
            started_on=date(2026, 7, 30),
        )
    )

    assert [item["memory_sequence"] for item in first_page["jobs"]] == [3, 2]
    assert first_page["next_before_sequence"] == 2
    assert [item["memory_sequence"] for item in queued["jobs"]] == [2]
    assert [item["memory_sequence"] for item in older["jobs"]] == [1]
    assert blocked["job"]["memory_sequence"] == 3  # type: ignore[index]
    assert status["blocking"] is True
    assert status["manual_action_required"] is True
    failure = status["last_failure"]
    assert isinstance(failure, dict)
    assert "private-token" not in failure["message"]
    assert "/Users/alice" not in failure["message"]
    assert "[PATH]" in failure["message"]


def test_retry_requires_exact_current_blocker_version_and_returns_reopened_job() -> None:
    failed = _job(3, MemoryJobStatus.FAILED, updated_offset=3)
    reopened = _job(3, MemoryJobStatus.QUEUED, updated_offset=4)
    runtime = _job_runtime((failed,), blocked=failed)
    runtime.retry_failed_memory_job = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(reopened_job=reopened, worker_restarted=True)
    )
    handler = RuntimeHTTPHandlers(runtime)
    version = handler._job_version(failed)

    with pytest.raises(HTTPMemoryJobConflictError, match="changed"):
        asyncio.run(
            handler.retry_failed_job(
                3,
                conversation_id="conversation-handler",
                started_on=date(2026, 7, 30),
                expected_version="0" * 64,
            )
        )

    result = asyncio.run(
        handler.retry_failed_job(
            3,
            conversation_id="conversation-handler",
            started_on=date(2026, 7, 30),
            expected_version=version,
        )
    )

    assert result["previous"]["job_status"] == "failed"  # type: ignore[index]
    assert result["job"]["job_status"] == "queued"  # type: ignore[index]
    assert result["worker_restarted"] is True
    runtime.retry_failed_memory_job.assert_awaited_once_with(failed)


@pytest.mark.xfail(
    strict=True,
    reason="M2BOS-BUG-HTTP-002: remember response reports adapter boundary instead of explicit override",
)
def test_remember_reports_the_effective_true_override() -> None:
    handler = RuntimeHTTPHandlers(_runtime(adapted_after_turn=False))

    result = asyncio.run(
        handler.remember(
            conversation_id="conversation-handler",
            started_on=date(2026, 7, 30),
            protocol="openai",
            payload={},
            start_sequence=0,
            occurred_at=datetime(2026, 7, 30, tzinfo=UTC),
            after_turn=True,
        )
    )

    assert result["after_turn"] is True


@pytest.mark.xfail(
    strict=True,
    reason="M2BOS-BUG-HTTP-002: remember response reports adapter boundary instead of explicit override",
)
def test_remember_reports_the_effective_false_override() -> None:
    handler = RuntimeHTTPHandlers(_runtime(adapted_after_turn=True))

    result = asyncio.run(
        handler.remember(
            conversation_id="conversation-handler",
            started_on=date(2026, 7, 30),
            protocol="openai",
            payload={},
            start_sequence=0,
            occurred_at=datetime(2026, 7, 30, tzinfo=UTC),
            after_turn=False,
        )
    )

    assert result["after_turn"] is False


@pytest.mark.xfail(
    strict=True,
    reason="M2BOS-BUG-OBS-001: public Job errors do not redact Windows or UNC absolute paths",
)
def test_public_job_failure_redacts_windows_and_unc_absolute_paths() -> None:
    windows = RuntimeHTTPHandlers._sanitize_failure(r"failed at C:\Users\alice\private\secret.txt")
    unc = RuntimeHTTPHandlers._sanitize_failure(r"failed at \\server\private\secret.txt")

    assert "alice" not in windows
    assert "server" not in unc
    assert "[PATH]" in windows
    assert "[PATH]" in unc
