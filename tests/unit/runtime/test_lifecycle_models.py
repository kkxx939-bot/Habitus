"""Runtime 与生命周期 Worker 的可观察状态模型边界测试。"""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import ConversationAddress
from memory.workflow import MemoryJobStatus
from memory.workflow.jobs import MemoryJobStore
from Runtime import (
    LifecycleMaintenanceCycleResult,
    LifecycleMaintenanceFailure,
    MemoryJobRetryResult,
    RuntimeInitialization,
)
from tests.helpers import BASE_TIME, segment


def test_lifecycle_cycle_separates_success_and_failure_within_selected_batch() -> None:
    first = ConversationAddress("a", date(2026, 7, 1))
    second = ConversationAddress("b", date(2026, 7, 1))
    failure = LifecycleMaintenanceFailure(second, "RuntimeError", "临时失败")
    result = LifecycleMaintenanceCycleResult(
        lease_acquired=True,
        started_at=BASE_TIME,
        finished_at=BASE_TIME + timedelta(seconds=1),
        selected_addresses=(first, second),
        maintained_addresses=(first,),
        failures=(failure,),
    )
    assert result.maintained_addresses == (first,)
    assert result.failures == (failure,)

    with pytest.raises(ValueError, match="both succeed and fail"):
        replace(result, maintained_addresses=(first, second))
    with pytest.raises(ValueError, match="skipped lease"):
        replace(result, lease_acquired=False)


def test_runtime_initialization_requires_absolute_root_and_stable_recovery_ids(tmp_path: Path) -> None:
    memory_root = (tmp_path / "memory").resolve()
    result = RuntimeInitialization(memory_root, ("a" * 32,))
    assert result.memory_root == memory_root
    with pytest.raises(ValueError, match="absolute"):
        RuntimeInitialization(Path("relative"), ())
    with pytest.raises(TypeError, match="non-empty"):
        RuntimeInitialization(memory_root, ("",))


def test_failed_job_retry_result_requires_same_source_and_reset_retry_state(tmp_path: Path) -> None:
    jobs = MemoryJobStore(
        tmp_path / "workflow",
        PathLock(ProcessLocalLockStore()),
        memory_root=tmp_path / "memory",
    )
    address = ConversationAddress("conversation-a", date(2026, 7, 1))
    queued = jobs.activate(
        jobs.stage(
            address,
            segment(
                conversation_id="conversation-a",
                segment_id="000000000000-000000000001",
            ),
        )
    )
    failed = jobs.fail(jobs.claim(queued, "worker"), ValueError("bad"), retryable=False)
    reopened = jobs.retry_failed(failed)

    result = MemoryJobRetryResult(failed, reopened, False)
    assert result.failed_job.status is MemoryJobStatus.FAILED
    assert result.reopened_job.status is MemoryJobStatus.QUEUED
    with pytest.raises(ValueError, match="source identity"):
        MemoryJobRetryResult(
            failed,
            replace(reopened, conversation_id="another-conversation"),
            False,
        )
