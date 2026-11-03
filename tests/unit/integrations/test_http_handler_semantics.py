"""验证 HTTP Handler 对 Runtime 结果的语义表达，不重复路由测试。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from habitus.integrations.http_api.handlers import HTTPMemoryJobConflictError, RuntimeHTTPHandlers
from habitus.integrations.http_api.schemas import RecallRequest
from habitus.memory.model import MemoryKind
from habitus.memory.workflow import MemoryJob, MemoryJobStatus
from habitus.runtime import (
    MemoryConsistencySnapshot,
    MemoryConsistencyState,
    Runtime,
)

UTC = UTC


def _runtime(*, adapted_after_turn: bool) -> Runtime:
    runtime = object.__new__(Runtime)

    async def append_protocol_conversation(_address: object, **values: object) -> SimpleNamespace:
        override = values["after_turn"]
        return SimpleNamespace(
            adaptation=SimpleNamespace(
                ignored_items=0,
                after_turn=adapted_after_turn,
            ),
            ingest=SimpleNamespace(jobs=()),
            effective_after_turn=(adapted_after_turn if override is None else override),
            next_sequence=2,
        )

    runtime.append_protocol_conversation = AsyncMock(  # type: ignore[method-assign]
        side_effect=append_protocol_conversation
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
        last_error=("Bearer private-token failed at /Users/alice/private/file.txt" if failed else None),
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

    assert result.result.after_turn is True
    assert result.result.next_sequence == 2


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

    assert [item.memory_sequence for item in first_page.jobs] == [3, 2]
    assert first_page.next_before_sequence == 2
    assert [item.memory_sequence for item in queued.jobs] == [2]
    assert [item.memory_sequence for item in older.jobs] == [1]
    assert blocked.job is not None and blocked.job.memory_sequence == 3
    assert status.blocking is True
    assert status.manual_action_required is True
    failure = status.last_failure
    assert failure is not None
    assert "private-token" not in failure.message
    assert "/Users/alice" not in failure.message
    assert "[PATH]" in failure.message


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

    assert result.previous.job_status == "failed"
    assert result.job.job_status == "queued"
    assert result.worker_restarted is True
    runtime.retry_failed_memory_job.assert_awaited_once_with(failed)


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

    assert result.result.after_turn is True


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

    assert result.result.after_turn is False


def test_public_job_failure_redacts_windows_and_unc_absolute_paths() -> None:
    windows = RuntimeHTTPHandlers._sanitize_failure(r"failed at C:\Users\Alice Smith\Private Folder\secret.txt")
    slash_windows = RuntimeHTTPHandlers._sanitize_failure("failed at C:/Users/Alice Smith/Private Folder/secret.txt")
    unc = RuntimeHTTPHandlers._sanitize_failure(r"failed at \\server\Private Share\Alice Smith\secret.txt")
    extended = RuntimeHTTPHandlers._sanitize_failure(r"failed at \\?\C:\Users\Alice Smith\secret.txt")

    assert "Alice Smith" not in windows
    assert "Alice Smith" not in slash_windows
    assert "server" not in unc
    assert "Alice Smith" not in extended
    assert all("[PATH]" in value for value in (windows, slash_windows, unc, extended))


@pytest.mark.parametrize(("ready", "status_code"), [(True, 200), (False, 503)])
def test_readiness_maps_an_unready_runtime_to_503(ready: bool, status_code: int) -> None:
    from habitus.runtime import RuntimeHealthCheck, RuntimeHealthReport, RuntimeHealthStatus

    status = RuntimeHealthStatus.HEALTHY if ready else RuntimeHealthStatus.UNHEALTHY
    runtime = object.__new__(Runtime)
    runtime.health = AsyncMock(  # type: ignore[method-assign]
        return_value=RuntimeHealthReport(
            status=status,
            ready=ready,
            checked_at=datetime(2026, 7, 30, tzinfo=UTC),
            checks=(RuntimeHealthCheck("runtime", status, "ready" if ready else "not started"),),
            recall_ready=ready,
        )
    )

    code, payload = asyncio.run(RuntimeHTTPHandlers(runtime).readiness())

    assert code == status_code
    assert payload.ready is ready


def test_job_list_gives_no_next_page_cursor_when_the_page_holds_every_match() -> None:
    jobs = (_job(1, MemoryJobStatus.COMMITTED), _job(2, MemoryJobStatus.QUEUED))
    handler = RuntimeHTTPHandlers(_job_runtime(jobs, blocked=None))

    for limit in (2, 10):
        page = asyncio.run(
            handler.list_jobs(conversation_id="conversation-handler", started_on=date(2026, 7, 30), limit=limit)
        )
        assert [item.memory_sequence for item in page.jobs] == [2, 1]
        assert page.next_before_sequence is None


def test_retry_refuses_a_failed_job_that_is_not_the_current_blocker() -> None:
    older = _job(2, MemoryJobStatus.FAILED, updated_offset=2)
    blocker = _job(3, MemoryJobStatus.FAILED, updated_offset=3)
    runtime = _job_runtime((older, blocker), blocked=blocker)
    runtime.retry_failed_memory_job = AsyncMock()  # type: ignore[method-assign]
    handler = RuntimeHTTPHandlers(runtime)

    with pytest.raises(HTTPMemoryJobConflictError, match="not the current blocking failed job"):
        asyncio.run(
            handler.retry_failed_job(
                2,
                conversation_id="conversation-handler",
                started_on=date(2026, 7, 30),
                expected_version=handler._job_version(older),
            )
        )
    runtime.retry_failed_memory_job.assert_not_awaited()


def test_job_version_changes_when_only_the_last_error_changes() -> None:
    """只差等长的错误文本也算不同版本（版本看错误内容，不只看长度）。"""

    failed = _job(3, MemoryJobStatus.FAILED)
    assert failed.last_error is not None

    assert RuntimeHTTPHandlers._job_version(failed) != RuntimeHTTPHandlers._job_version(
        replace(failed, last_error=failed.last_error.replace("private", "PRIVATE"))
    )


def test_public_job_failure_is_truncated_to_500_characters() -> None:
    message = RuntimeHTTPHandlers._sanitize_failure("x" * 600)

    assert message == "x" * 500 + "...[truncated]"
    assert RuntimeHTTPHandlers._sanitize_failure("x" * 500) == "x" * 500


def test_recall_request_refuses_duplicate_kinds() -> None:
    assert RecallRequest.model_validate({"query": "q", "kinds": ["event", "tool"]}).kinds == (
        MemoryKind.EVENT,
        MemoryKind.TOOL,
    )
    with pytest.raises(ValidationError, match="kinds must not contain duplicates"):
        RecallRequest.model_validate({"query": "q", "kinds": ["event", "event"]})
