"""顶层 Runtime 管理的单 memory-root 常驻 Worker。"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from habitus.config import WorkerConfig
from habitus.foundation.observability import (
    NullObserver,
    ObservationEvent,
    ObservationStatus,
    Observer,
    SpanController,
    bind_observation_context,
)
from habitus.memory.workflow import (
    MemoryJob,
    MemoryJobBlockedError,
    MemoryJobClaim,
    MemoryJobExecutionError,
    MemoryJobLease,
    MemoryJobLeaseLostError,
    MemoryJobNotReadyError,
    MemoryJobRunner,
    MemoryJobRunResult,
    MemoryJobStatus,
)


class MemoryWorkerState(str, Enum):
    """常驻 Worker 的可观察生命周期。"""

    CREATED = "created"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    BLOCKED = "blocked"
    FAILED = "failed"


class MemoryWorkerStateError(RuntimeError):
    """Worker 操作与当前生命周期不相容。"""


class MemoryWorker:
    """单并发认领 Job，并在整个执行期间维持 durable lease 心跳。"""

    def __init__(
        self,
        runner: MemoryJobRunner,
        config: WorkerConfig,
        *,
        worker_id: str | None = None,
        observer: Observer | None = None,
        span_controller: SpanController | None = None,
    ) -> None:
        if not isinstance(runner, MemoryJobRunner):
            raise TypeError("runner must be MemoryJobRunner")
        if not isinstance(config, WorkerConfig):
            raise TypeError("config must be WorkerConfig")
        resolved_worker_id = worker_id or f"worker-{os.getpid()}-{uuid4().hex}"
        if not isinstance(resolved_worker_id, str):
            raise TypeError("worker_id must be text")
        if config.heartbeat_interval_seconds > runner.store.config.lease_ttl_seconds / 3:
            raise ValueError("worker heartbeat must be at most one third of the job lease TTL")
        if not MemoryJob.valid_worker_id(resolved_worker_id):
            raise ValueError("worker_id must be normalized stable text")
        self.runner = runner
        self.config = config
        self.worker_id = resolved_worker_id
        self._state = MemoryWorkerState.CREATED
        self._loop_task: asyncio.Task[None] | None = None
        self._active_execution: asyncio.Task[MemoryJobRunResult] | None = None
        self._stop_requested = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._last_error: BaseException | None = None
        self.observer = observer or NullObserver()
        self.span_controller = span_controller

    @property
    def state(self) -> MemoryWorkerState:
        return self._state

    @property
    def running(self) -> bool:
        return self._state is MemoryWorkerState.RUNNING

    @property
    def busy(self) -> bool:
        return self._active_execution is not None and not self._active_execution.done()

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    async def start(self) -> None:
        """幂等启动轮询；调用方必须已经完成 Runtime 初始化与恢复。"""

        if self._state is MemoryWorkerState.RUNNING:
            return
        if self._state is MemoryWorkerState.STOPPING:
            raise MemoryWorkerStateError("stopping worker cannot be started")
        if self._loop_task is not None and not self._loop_task.done():
            raise MemoryWorkerStateError("worker already owns a live loop task")
        self._stop_requested.clear()
        self._wake_event.clear()
        self._last_error = None
        self._state = MemoryWorkerState.RUNNING
        self._loop_task = asyncio.create_task(
            self._run_loop(),
            name=f"habitus-memory-worker:{self.worker_id}",
        )
        await asyncio.sleep(0)

    async def stop(self) -> None:
        """先停止新认领，再在心跳继续运行时有界排空当前 Job。"""

        if self._state is MemoryWorkerState.CREATED:
            self._state = MemoryWorkerState.STOPPED
            return
        task = self._loop_task
        if task is None or task.done():
            self._state = MemoryWorkerState.STOPPED
            return
        self._state = MemoryWorkerState.STOPPING
        self._stop_requested.set()
        self._wake_event.set()
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.config.shutdown_timeout_seconds,
            )
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        finally:
            self._state = MemoryWorkerState.STOPPED

    async def wait_stopped(self) -> None:
        """等待当前常驻循环退出；不会主动请求停止。"""

        task = self._loop_task
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task

    def wake(self) -> None:
        """提示轮询器尽快检查新 Job；未调用时仍会按配置轮询。"""

        self._wake_event.set()

    async def run_once(self) -> MemoryJobRunResult:
        """在没有常驻循环时处理一项，并同样维持租约心跳。"""

        if self._loop_task is not None and not self._loop_task.done():
            raise MemoryWorkerStateError("manual run_once cannot race the worker loop")
        if self.busy:
            raise MemoryWorkerStateError("worker already has an active execution")
        return await self._run_once()

    async def _run_loop(self) -> None:
        try:
            while not self._stop_requested.is_set():
                try:
                    result = await self._run_once()
                except MemoryJobNotReadyError as exc:
                    await self._wait_until(exc.available_at)
                    continue
                except MemoryJobLeaseLostError:
                    self._observe("job_lease", ObservationStatus.DEGRADED, {"error_type": "MemoryJobLeaseLostError"})
                    await self._wait(self.config.poll_interval_seconds)
                    continue
                except MemoryJobExecutionError as exc:
                    if exc.job is not None and exc.job.status is MemoryJobStatus.FAILED:
                        self._last_error = exc
                        self._state = MemoryWorkerState.BLOCKED
                        return
                    await self._wait(self.config.poll_interval_seconds)
                    continue
                except MemoryJobBlockedError as exc:
                    oldest = await asyncio.to_thread(self.runner.store.oldest_uncommitted)
                    if oldest is not None and oldest.status is MemoryJobStatus.FAILED:
                        self._observe(
                            "job_queue",
                            ObservationStatus.DEGRADED,
                            {"error_type": type(exc).__name__, "memory_sequence": oldest.memory_sequence},
                        )
                        self._last_error = exc
                        self._state = MemoryWorkerState.BLOCKED
                        return
                    await self._wait(self.config.poll_interval_seconds)
                    continue
                if result.job is None:
                    await self._wait(self.config.poll_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = exc
            self._state = MemoryWorkerState.FAILED
            self._observe("worker_loop", ObservationStatus.FAILURE, {"error_type": type(exc).__name__})
        finally:
            active = self._active_execution
            if active is not None and not active.done():
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
            self._active_execution = None
            if self._state in {MemoryWorkerState.RUNNING, MemoryWorkerState.STOPPING}:
                self._state = MemoryWorkerState.STOPPED

    async def _run_once(self) -> MemoryJobRunResult:
        claim = await asyncio.to_thread(self.runner.claim_next, self.worker_id)
        if claim is None:
            return MemoryJobRunResult(job=None, commit=None)
        job = claim.lease.job
        started = time.monotonic()
        with bind_observation_context(
            memory_sequence=job.memory_sequence,
            transaction_id=job.transaction_id,
            worker_id=self.worker_id,
            attempt=job.attempts,
        ):
            span = (
                self.span_controller.start_span(
                    "workflow",
                    "memory_job",
                    attributes={"job_status": job.status.value},
                )
                if self.span_controller is not None
                else _NullSpan()
            )
            try:
                with span:
                    result = await self._execute_claim(claim)
            except asyncio.CancelledError:
                self._observe(
                    "job_execution",
                    ObservationStatus.DEGRADED,
                    {"error_type": "CancelledError", "job_status": job.status.value},
                    started=started,
                )
                raise
            except Exception as exc:
                failed_job = exc.job if isinstance(exc, MemoryJobExecutionError) else None
                self._observe(
                    "job_execution",
                    ObservationStatus.FAILURE,
                    {
                        "error_type": type(exc).__name__,
                        "job_status": job.status.value if failed_job is None else failed_job.status.value,
                    },
                    started=started,
                )
                raise
            self._observe(
                "job_execution",
                ObservationStatus.SUCCESS,
                {"job_status": "unknown" if result.job is None else result.job.status.value},
                started=started,
            )
            return result

    async def _execute_claim(self, claim: MemoryJobClaim) -> MemoryJobRunResult:
        execution = asyncio.create_task(
            self.runner.run_claimed(claim),
            name=(f"habitus-memory-job:{claim.lease.job.memory_sequence}:{claim.lease.claim_generation}"),
        )
        self._active_execution = execution
        heartbeat_stop = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat(claim.lease, execution, heartbeat_stop),
            name=(f"habitus-memory-heartbeat:{claim.lease.job.memory_sequence}:{claim.lease.claim_generation}"),
        )
        try:
            done, _pending = await asyncio.wait(
                {execution, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat in done:
                heartbeat_error = heartbeat.exception()
                if heartbeat_error is not None:
                    execution.cancel()
                    await asyncio.gather(execution, return_exceptions=True)
                    raise heartbeat_error
            return await execution
        finally:
            heartbeat_stop.set()
            if not heartbeat.done():
                heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            self._active_execution = None

    async def _heartbeat(
        self,
        lease: MemoryJobLease,
        execution: asyncio.Task[MemoryJobRunResult],
        stop: asyncio.Event,
    ) -> None:
        current_lease = lease
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=self.config.heartbeat_interval_seconds,
                )
                return
            except TimeoutError:
                try:
                    current_lease = await asyncio.to_thread(
                        self.runner.store.renew,
                        current_lease,
                    )
                except MemoryJobLeaseLostError:
                    if await asyncio.to_thread(
                        self.runner.store.is_settled,
                        current_lease,
                    ):
                        return
                    execution.cancel()
                    raise
                except TimeoutError as exc:
                    if current_lease.lease_expires_at > datetime.now(UTC):
                        continue
                    execution.cancel()
                    raise MemoryJobLeaseLostError("memory job lease could not be renewed before expiry") from exc
                except Exception:
                    execution.cancel()
                    raise

    async def _wait_until(self, available_at: datetime) -> None:
        remaining = (available_at.astimezone(UTC) - datetime.now(UTC)).total_seconds()
        await self._wait(max(self.config.poll_interval_seconds, remaining))

    async def _wait(self, delay_seconds: float) -> None:
        if self._stop_requested.is_set():
            return
        self._wake_event.clear()
        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=delay_seconds)
        except TimeoutError:
            pass

    def _observe(
        self,
        operation: str,
        status: ObservationStatus,
        attributes: dict[str, str | int | float | bool],
        *,
        started: float | None = None,
    ) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category="workflow",
                    operation=operation,
                    status=status,
                    duration_seconds=(0.0 if started is None else max(0.0, time.monotonic() - started)),
                    attributes=attributes,
                )
            )
        except Exception:
            pass


class _NullSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        return None


__all__ = [
    "MemoryWorker",
    "MemoryWorkerState",
    "MemoryWorkerStateError",
]
