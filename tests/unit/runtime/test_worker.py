"""Memory Worker 的启动停止、手动执行、心跳和租约丢失测试。"""

import asyncio
from datetime import date
from pathlib import Path

import pytest

from Config import WorkerConfig
from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import ConversationAddress
from memory.workflow import (
    MemoryJobClaim,
    MemoryJobLeaseLostError,
    MemoryJobRunner,
    MemoryJobRunResult,
)
from memory.workflow.jobs import MemoryJobConfig, MemoryJobStore
from Runtime import MemoryWorker, MemoryWorkerState, MemoryWorkerStateError
from tests.helpers import segment


class LeaseStore:
    def __init__(self, *, fail_renewal: bool = False) -> None:
        self.config = MemoryJobConfig(lease_ttl_seconds=3)
        self.fail_renewal = fail_renewal
        self.renew_calls = 0

    def renew(self, lease):
        self.renew_calls += 1
        if self.fail_renewal:
            raise MemoryJobLeaseLostError("lease lost")
        return lease

    def is_settled(self, _lease) -> bool:
        return False


def claimed_job(tmp_path: Path) -> MemoryJobClaim:
    store = MemoryJobStore(
        tmp_path / "jobs",
        PathLock(ProcessLocalLockStore()),
        memory_root=tmp_path / "memory",
        config=MemoryJobConfig(lease_ttl_seconds=3),
    )
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    queued = store.activate(store.stage(address, segment()))
    return MemoryJobClaim(store.claim(queued, "source-worker"))


def fake_runner(store: LeaseStore, claim: MemoryJobClaim | None, *, delay: float = 0.0) -> MemoryJobRunner:
    runner = object.__new__(MemoryJobRunner)
    runner.store = store
    claimed = False

    def claim_next(_worker_id: str):
        nonlocal claimed
        if claimed:
            return None
        claimed = True
        return claim

    async def run_claimed(current: MemoryJobClaim) -> MemoryJobRunResult:
        await asyncio.sleep(delay)
        return MemoryJobRunResult(job=current.lease.job, commit=None)

    runner.claim_next = claim_next
    runner.run_claimed = run_claimed
    return runner


def test_run_once_without_job_returns_empty_result_and_worker_stays_idle() -> None:
    store = LeaseStore()
    worker = MemoryWorker(
        fake_runner(store, None),
        WorkerConfig(heartbeat_interval_seconds=0.5),
        worker_id="worker-1",
    )

    result = asyncio.run(worker.run_once())

    assert result.job is None
    assert worker.state is MemoryWorkerState.CREATED
    assert not worker.busy


def test_run_once_renews_lease_until_slow_execution_finishes(tmp_path: Path) -> None:
    store = LeaseStore()
    worker = MemoryWorker(
        fake_runner(store, claimed_job(tmp_path), delay=0.04),
        WorkerConfig(heartbeat_interval_seconds=0.01),
        worker_id="worker-1",
    )

    result = asyncio.run(worker.run_once())

    assert result.job is not None
    assert store.renew_calls >= 2
    assert not worker.busy


def test_lost_lease_cancels_execution_and_is_reported(tmp_path: Path) -> None:
    store = LeaseStore(fail_renewal=True)
    worker = MemoryWorker(
        fake_runner(store, claimed_job(tmp_path), delay=1.0),
        WorkerConfig(heartbeat_interval_seconds=0.01),
        worker_id="worker-1",
    )

    with pytest.raises(MemoryJobLeaseLostError):
        asyncio.run(worker.run_once())

    assert store.renew_calls == 1
    assert not worker.busy


def test_background_loop_is_idempotent_blocks_manual_race_and_stops_cleanly() -> None:
    async def scenario() -> None:
        worker = MemoryWorker(
            fake_runner(LeaseStore(), None),
            WorkerConfig(poll_interval_seconds=10, heartbeat_interval_seconds=0.5),
            worker_id="worker-1",
        )
        await worker.start()
        await worker.start()
        assert worker.state is MemoryWorkerState.RUNNING
        with pytest.raises(MemoryWorkerStateError, match="cannot race"):
            await worker.run_once()
        await worker.stop()
        assert worker.state is MemoryWorkerState.STOPPED

    asyncio.run(scenario())


def test_worker_rejects_invalid_identity_and_unsafe_heartbeat_ratio() -> None:
    runner = fake_runner(LeaseStore(), None)
    with pytest.raises(ValueError, match="one third"):
        MemoryWorker(
            runner,
            WorkerConfig(heartbeat_interval_seconds=2),
            worker_id="worker-1",
        )
    with pytest.raises(ValueError, match="normalized"):
        MemoryWorker(
            runner,
            WorkerConfig(heartbeat_interval_seconds=0.5),
            worker_id="bad worker",
        )

