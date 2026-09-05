"""Memory Worker 的启动停止、手动执行、心跳和租约丢失测试。"""

import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from habitus.config import WorkerConfig
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.memory.conversation import ConversationAddress
from habitus.memory.workflow import (
    MemoryJobBlockedError,
    MemoryJobClaim,
    MemoryJobExecutionError,
    MemoryJobLease,
    MemoryJobLeaseLostError,
    MemoryJobNotReadyError,
    MemoryJobRunner,
    MemoryJobRunResult,
)
from habitus.memory.workflow.jobs import MemoryJobConfig, MemoryJobStore
from habitus.runtime import MemoryWorker, MemoryWorkerState, MemoryWorkerStateError
from tests.helpers import segment


class LeaseStore:
    def __init__(
        self,
        *,
        fail_renewal: bool = False,
        renew_error: BaseException | None = None,
        settled: bool = False,
    ) -> None:
        self.config = MemoryJobConfig(lease_ttl_seconds=3)
        self.fail_renewal = fail_renewal
        self.renew_error = renew_error
        self.settled = settled
        self.renew_calls = 0

    def renew(self, lease):
        self.renew_calls += 1
        if self.renew_error is not None:
            raise self.renew_error
        if self.fail_renewal:
            raise MemoryJobLeaseLostError("lease lost")
        return lease

    def is_settled(self, _lease) -> bool:
        return self.settled


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


@pytest.mark.parametrize(
    ("runner", "config", "worker_id", "message"),
    [
        (object(), WorkerConfig(), "worker-1", "runner"),
        (object.__new__(MemoryJobRunner), object(), "worker-1", "config"),
        (object.__new__(MemoryJobRunner), WorkerConfig(), 1, "worker_id"),
    ],
)
def test_worker_constructor_rejects_each_invalid_collaborator(
    runner: object,
    config: object,
    worker_id: object,
    message: str,
) -> None:
    if isinstance(runner, MemoryJobRunner) and not hasattr(runner, "store"):
        runner.store = LeaseStore()
    with pytest.raises(TypeError, match=message):
        MemoryWorker(runner, config, worker_id=worker_id)  # type: ignore[arg-type]


def test_worker_generates_normalized_default_identity() -> None:
    worker = MemoryWorker(
        fake_runner(LeaseStore(), None),
        WorkerConfig(heartbeat_interval_seconds=0.5),
    )
    assert worker.worker_id.startswith("worker-")
    assert " " not in worker.worker_id


def test_stop_before_start_and_repeated_stop_are_idempotent() -> None:
    async def scenario() -> None:
        worker = MemoryWorker(
            fake_runner(LeaseStore(), None),
            WorkerConfig(heartbeat_interval_seconds=0.5),
            worker_id="worker-1",
        )
        await worker.stop()
        await worker.stop()
        assert worker.state is MemoryWorkerState.STOPPED

    asyncio.run(scenario())


def test_start_rejects_stopping_state_and_an_unowned_live_loop() -> None:
    async def scenario() -> None:
        worker = MemoryWorker(
            fake_runner(LeaseStore(), None),
            WorkerConfig(heartbeat_interval_seconds=0.5),
            worker_id="worker-1",
        )
        worker._state = MemoryWorkerState.STOPPING
        with pytest.raises(MemoryWorkerStateError, match="stopping"):
            await worker.start()

        worker._state = MemoryWorkerState.STOPPED
        worker._loop_task = asyncio.create_task(asyncio.sleep(10))
        with pytest.raises(MemoryWorkerStateError, match="live loop"):
            await worker.start()
        worker._loop_task.cancel()
        await asyncio.gather(worker._loop_task, return_exceptions=True)

    asyncio.run(scenario())


def test_stop_cancels_loop_after_shutdown_timeout() -> None:
    async def scenario() -> None:
        worker = MemoryWorker(
            fake_runner(LeaseStore(), None),
            WorkerConfig(heartbeat_interval_seconds=0.5, shutdown_timeout_seconds=0.001),
            worker_id="worker-1",
        )
        worker._state = MemoryWorkerState.RUNNING
        worker._loop_task = asyncio.create_task(asyncio.sleep(10))
        await worker.stop()
        assert worker._loop_task.cancelled()
        assert worker.state is MemoryWorkerState.STOPPED

    asyncio.run(scenario())


def test_wait_stopped_waits_for_existing_task_and_wake_sets_event() -> None:
    async def scenario() -> None:
        worker = MemoryWorker(
            fake_runner(LeaseStore(), None),
            WorkerConfig(heartbeat_interval_seconds=0.5),
            worker_id="worker-1",
        )
        worker._loop_task = asyncio.create_task(asyncio.sleep(0))
        worker.wake()
        assert worker._wake_event.is_set()
        await worker.wait_stopped()
        assert worker._loop_task.done()

    asyncio.run(scenario())


def test_manual_run_rejects_preexisting_active_execution() -> None:
    async def scenario() -> None:
        worker = MemoryWorker(
            fake_runner(LeaseStore(), None),
            WorkerConfig(heartbeat_interval_seconds=0.5),
            worker_id="worker-1",
        )
        worker._active_execution = asyncio.create_task(asyncio.sleep(10))
        with pytest.raises(MemoryWorkerStateError, match="active execution"):
            await worker.run_once()
        worker._active_execution.cancel()
        await asyncio.gather(worker._active_execution, return_exceptions=True)

    asyncio.run(scenario())


def test_lost_lease_after_executor_already_settled_is_not_reported(tmp_path: Path) -> None:
    store = LeaseStore(fail_renewal=True, settled=True)
    worker = MemoryWorker(
        fake_runner(store, claimed_job(tmp_path), delay=0.03),
        WorkerConfig(heartbeat_interval_seconds=0.01),
        worker_id="worker-1",
    )

    result = asyncio.run(worker.run_once())
    assert result.job is not None
    assert store.renew_calls == 1


def test_temporary_renew_timeout_before_expiry_is_retried(tmp_path: Path) -> None:
    store = LeaseStore(renew_error=TimeoutError("temporary"))
    worker = MemoryWorker(
        fake_runner(store, claimed_job(tmp_path), delay=0.03),
        WorkerConfig(heartbeat_interval_seconds=0.01),
        worker_id="worker-1",
    )

    result = asyncio.run(worker.run_once())
    assert result.job is not None
    assert store.renew_calls >= 1


def test_renew_timeout_after_lease_expiry_becomes_lease_loss(tmp_path: Path) -> None:
    original = claimed_job(tmp_path)
    expired_job = replace(
        original.lease.job,
        lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    expired_claim = MemoryJobClaim(MemoryJobLease(expired_job))
    store = LeaseStore(renew_error=TimeoutError("late"))
    worker = MemoryWorker(
        fake_runner(store, expired_claim, delay=1),
        WorkerConfig(heartbeat_interval_seconds=0.01),
        worker_id="worker-1",
    )

    with pytest.raises(MemoryJobLeaseLostError, match="before expiry"):
        asyncio.run(worker.run_once())


def test_unexpected_renew_error_cancels_execution_and_propagates(tmp_path: Path) -> None:
    store = LeaseStore(renew_error=OSError("lock store unavailable"))
    worker = MemoryWorker(
        fake_runner(store, claimed_job(tmp_path), delay=1),
        WorkerConfig(heartbeat_interval_seconds=0.01),
        worker_id="worker-1",
    )

    with pytest.raises(OSError, match="unavailable"):
        asyncio.run(worker.run_once())


def failed_job(tmp_path: Path):
    store = MemoryJobStore(
        tmp_path / "failed-jobs",
        PathLock(ProcessLocalLockStore()),
        memory_root=tmp_path / "failed-memory",
        config=MemoryJobConfig(lease_ttl_seconds=3),
    )
    address = ConversationAddress("conversation-failed", date(2026, 7, 1))
    queued = store.activate(store.stage(address, segment(conversation_id="conversation-failed")))
    lease = store.claim(queued, "source-worker")
    return store.fail(lease, RuntimeError("exhausted"), retryable=False)


def loop_worker(error_factory, *, oldest=None) -> MemoryWorker:
    store = LeaseStore()
    store.oldest_uncommitted = lambda: oldest
    worker = MemoryWorker(
        fake_runner(store, None),
        WorkerConfig(poll_interval_seconds=0.001, heartbeat_interval_seconds=0.5),
        worker_id="worker-1",
    )
    calls = 0

    async def run_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_factory()
        worker._stop_requested.set()
        return MemoryJobRunResult(job=None, commit=None)

    worker._run_once = run_once
    return worker


def test_background_loop_blocks_on_terminal_execution_failure(tmp_path: Path) -> None:
    failed = failed_job(tmp_path)
    worker = loop_worker(
        lambda: MemoryJobExecutionError("failed", job=failed),
    )

    async def scenario() -> None:
        await worker.start()
        await worker.wait_stopped()

    asyncio.run(scenario())
    assert worker.state is MemoryWorkerState.BLOCKED
    assert isinstance(worker.last_error, MemoryJobExecutionError)


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: MemoryJobNotReadyError(datetime.now(UTC)),
        lambda: MemoryJobLeaseLostError("lost"),
        lambda: MemoryJobExecutionError("retrying", job=None),
    ],
)
def test_background_loop_retries_transient_job_states(error_factory) -> None:
    async def scenario() -> MemoryWorker:
        worker = loop_worker(error_factory)
        await worker.start()
        await worker.wait_stopped()
        return worker

    worker = asyncio.run(scenario())
    assert worker.state is MemoryWorkerState.STOPPED
    assert worker.last_error is None


def test_background_loop_blocks_when_queue_head_is_failed(tmp_path: Path) -> None:
    failed = failed_job(tmp_path)
    worker = loop_worker(lambda: MemoryJobBlockedError("blocked"), oldest=failed)

    async def scenario() -> None:
        await worker.start()
        await worker.wait_stopped()

    asyncio.run(scenario())
    assert worker.state is MemoryWorkerState.BLOCKED
    assert isinstance(worker.last_error, MemoryJobBlockedError)


def test_background_loop_retries_nonterminal_queue_block() -> None:
    async def scenario() -> MemoryWorker:
        worker = loop_worker(lambda: MemoryJobBlockedError("held"), oldest=None)
        await worker.start()
        await worker.wait_stopped()
        return worker

    worker = asyncio.run(scenario())
    assert worker.state is MemoryWorkerState.STOPPED
    assert worker.last_error is None


def test_background_loop_records_unexpected_failure() -> None:
    async def scenario() -> MemoryWorker:
        worker = loop_worker(lambda: RuntimeError("unexpected"))
        await worker.start()
        await worker.wait_stopped()
        return worker

    worker = asyncio.run(scenario())
    assert worker.state is MemoryWorkerState.FAILED
    assert isinstance(worker.last_error, RuntimeError)
