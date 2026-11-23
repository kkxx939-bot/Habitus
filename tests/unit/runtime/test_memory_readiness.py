"""就绪拆成写入侧与召回侧：记忆 Job 停住不挡召回；等模型恢复与真失败在健康面上分得开（裁定 80 问题 2）。"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import ConversationAddress
from habitus.runtime import RuntimeHealthCheck, RuntimeHealthService, RuntimeHealthStatus, build_runtime
from tests.helpers import closed_turn, segment
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies
from tests.lock_store import ProcessLocalLockStore


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, recall_side: RuntimeHealthStatus):
    providers = runtime_dependencies()
    runtime = build_runtime(
        runtime_config(tmp_path),
        providers=providers,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()

    def lifecycle_storage_check(_self: object, _runtime_state: str) -> RuntimeHealthCheck:
        return RuntimeHealthCheck("lifecycle_storage", recall_side, "stub")

    # 召回侧用生命周期存储这一项做替身；这里只看就绪怎么由各项合成
    monkeypatch.setattr(RuntimeHealthService, "_lifecycle_storage_check", lifecycle_storage_check)
    return runtime, RuntimeHealthService(runtime.components)


def _job(runtime):
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    jobs = runtime.components.workflow.jobs
    return jobs, jobs.activate(
        jobs.stage(address, segment(segment_id="000000000000-000000000001", messages=closed_turn()))
    )


def test_a_job_waiting_for_the_model_blocks_writes_but_not_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, service = _service(tmp_path, monkeypatch, recall_side=RuntimeHealthStatus.HEALTHY)
    jobs, job = _job(runtime)
    jobs.fail(jobs.claim(job, "worker"), RuntimeError("Insufficient Balance"), waits_for_model=True)

    report = asyncio.run(service.report("running"))
    queue = next(check for check in report.checks if check.name == "memory_queue")
    assert queue.status is RuntimeHealthStatus.DEGRADED
    assert queue.detail.startswith("waiting_for_model:1;waits=1;next_attempt_at=")
    assert report.ready is False and report.recall_ready is True


def test_a_failed_job_no_longer_blocks_the_queue_and_is_reported_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 裁定 111：判了失败的任务是终态，不挡后面；健康面单独报失败条数与最早一条，不影响就绪
    runtime, service = _service(tmp_path, monkeypatch, recall_side=RuntimeHealthStatus.HEALTHY)
    jobs, job = _job(runtime)
    jobs.fail(jobs.claim(job, "worker"), ValueError("bad data"), retryable=False)

    report = asyncio.run(service.report("running"))
    queue = next(check for check in report.checks if check.name == "memory_queue")
    failed = next(check for check in report.checks if check.name == "memory_failed_jobs")
    assert queue.detail == "empty"
    assert failed.detail.startswith("failed=1;oldest=1:") and failed.critical is False
    assert report.ready is True and report.recall_ready is True


def test_a_recall_side_problem_makes_recall_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime, service = _service(tmp_path, monkeypatch, recall_side=RuntimeHealthStatus.UNHEALTHY)

    report = asyncio.run(service.report("running"))
    assert report.ready is False and report.recall_ready is False
