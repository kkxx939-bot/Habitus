"""顶层 Runtime 管理的生命周期维护 Worker：只管调度、全局 lease、心跳与参与者之间的失败隔离。

具体放什么由参与者决定（``infrastructure.lifecycle.LifecycleParticipant``）：memory 的会话链与 L2 维护
（``runtime/lifecycle_memory.py``）、行为侧 / 语义树 / 预测层 / 会话源的释放，都挂在同一个 Worker 上。
一个参与者抛错只记进它自己的结果，不打断其余参与者；租约丢失则整轮立即停止。
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from habitus.config import LifecycleWorkerConfig
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.infrastructure.lifecycle import (
    CapacitySnapshot,
    CapacitySnapshotStore,
    LifecycleParticipant,
    ParticipantReport,
)
from habitus.infrastructure.store.contracts import LockStore, LockToken

_MESSAGE_LIMIT = 1_000


class LifecycleWorkerState(str, Enum):
    """生命周期维护 Worker 的可观察运行状态。"""

    CREATED = "created"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class LifecycleWorkerStateError(RuntimeError):
    """LifecycleWorker 操作与当前状态不相容。"""


class LifecycleWorkerLeaseLostError(RuntimeError):
    """本轮全局维护 lease 已被其他 Runtime 接管。"""


@dataclass(frozen=True)
class LifecycleParticipantOutcome:
    """一个参与者这一轮的结果：要么有报告，要么有失败。"""

    name: str
    report: ParticipantReport | None = None
    error_type: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("participant name must be non-empty text")
        failed = self.error_type is not None
        if failed == (self.report is not None):
            raise ValueError("a participant outcome carries either a report or a failure")
        if failed and (not self.error_type or not self.message or len(self.message) > _MESSAGE_LIMIT):
            raise ValueError("participant failure needs a bounded error type and message")

    @property
    def failed(self) -> bool:
        return self.error_type is not None


@dataclass(frozen=True)
class LifecycleMaintenanceCycleResult:
    """一次全局维护周期：拿没拿到 lease，以及每个参与者的结果。"""

    lease_acquired: bool
    started_at: datetime
    finished_at: datetime
    outcomes: tuple[LifecycleParticipantOutcome, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.lease_acquired, bool):
            raise TypeError("lease_acquired must be boolean")
        for name in ("started_at", "finished_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("maintenance cycle cannot finish before it starts")
        if not isinstance(self.outcomes, tuple) or any(
            not isinstance(item, LifecycleParticipantOutcome) for item in self.outcomes
        ):
            raise TypeError("outcomes must contain LifecycleParticipantOutcome values")
        names = [item.name for item in self.outcomes]
        if len(names) != len(set(names)):
            raise ValueError("participant names must be unique within one cycle")
        if not self.lease_acquired and self.outcomes:
            raise ValueError("a skipped lease cycle cannot contain maintenance results")

    @property
    def failures(self) -> tuple[LifecycleParticipantOutcome, ...]:
        return tuple(item for item in self.outcomes if item.failed)

    @property
    def signals(self) -> tuple[str, ...]:
        """各参与者这一轮报的、要人看的情况（"参与者名: 内容"）。"""

        return tuple(
            f"{item.name}: {signal}"
            for item in self.outcomes
            if item.report is not None
            for signal in item.report.signals
        )


class _CycleContext:
    """交给参与者的一轮上下文。``assert_owned`` 是同步的：参与者在线程里调，或用 ``asyncio.to_thread`` 包一层。"""

    def __init__(self, *, now: datetime, token: LockToken, lock_store: LockStore) -> None:
        self._now = now
        self._token = token
        self._lock_store = lock_store

    @property
    def now(self) -> datetime:
        return self._now

    def assert_owned(self) -> None:
        try:
            self._lock_store.assert_owned(self._token)
        except TimeoutError as exc:
            raise LifecycleWorkerLeaseLostError("lifecycle lease is no longer owned") from exc

    def fence_token(self) -> LockToken:
        return self._token


class LifecycleWorker:
    """按周期取得全局 lease，依次运行各参与者。"""

    def __init__(
        self,
        participants: Sequence[LifecycleParticipant],
        config: LifecycleWorkerConfig,
        *,
        lock_store: LockStore,
        lease_key: str,
        capacity: CapacitySnapshotStore | None = None,
        worker_id: str | None = None,
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(config, LifecycleWorkerConfig):
            raise TypeError("config must be LifecycleWorkerConfig")
        resolved = tuple(participants)
        names = [participant.name for participant in resolved]
        if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
            raise ValueError("participants must have unique non-empty names")
        if not isinstance(lease_key, str) or not lease_key:
            raise ValueError("lease_key must be non-empty text")
        resolved_worker_id = worker_id or f"lifecycle-{os.getpid()}-{uuid4().hex}"
        if (
            not isinstance(resolved_worker_id, str)
            or not resolved_worker_id
            or any(ord(character) < 32 for character in resolved_worker_id)
        ):
            raise ValueError("worker_id must be non-empty normalized text")
        self._participants = resolved
        self.config = config
        self.lock_store = lock_store
        self.lock_key = lease_key
        self.capacity = capacity
        self.worker_id = resolved_worker_id
        self.observer = observer or NullObserver()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._state = LifecycleWorkerState.CREATED
        self._loop_task: asyncio.Task[None] | None = None
        self._active_cycle: asyncio.Task[LifecycleMaintenanceCycleResult] | None = None
        self._stop_requested = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._last_cycle: LifecycleMaintenanceCycleResult | None = None
        self._last_error: BaseException | None = None

    @property
    def participants(self) -> tuple[LifecycleParticipant, ...]:
        return self._participants

    @property
    def state(self) -> LifecycleWorkerState:
        return self._state

    @property
    def running(self) -> bool:
        return self._state is LifecycleWorkerState.RUNNING

    @property
    def busy(self) -> bool:
        return self._active_cycle is not None and not self._active_cycle.done()

    @property
    def last_cycle(self) -> LifecycleMaintenanceCycleResult | None:
        return self._last_cycle

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    async def start(self) -> None:
        """幂等启动后台循环；首次维护在一个完整周期后执行。"""

        if self._state is LifecycleWorkerState.RUNNING:
            return
        if self._state is LifecycleWorkerState.STOPPING:
            raise LifecycleWorkerStateError("stopping lifecycle worker cannot be started")
        if self._loop_task is not None and not self._loop_task.done():
            raise LifecycleWorkerStateError("lifecycle worker already owns a live loop task")
        self._stop_requested.clear()
        self._wake_event.clear()
        self._last_error = None
        self._state = LifecycleWorkerState.RUNNING
        self._loop_task = asyncio.create_task(self._run_loop(), name=f"habitus-lifecycle-worker:{self.worker_id}")
        await asyncio.sleep(0)

    async def stop(self) -> None:
        """停止新周期，并有界等待当前维护操作安全结束。"""

        if self._state is LifecycleWorkerState.CREATED:
            self._state = LifecycleWorkerState.STOPPED
            return
        task = self._loop_task
        if task is None or task.done():
            self._state = LifecycleWorkerState.STOPPED
            return
        self._state = LifecycleWorkerState.STOPPING
        self._stop_requested.set()
        self._wake_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self.config.shutdown_timeout_seconds)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        finally:
            self._state = LifecycleWorkerState.STOPPED

    async def run_once(self) -> LifecycleMaintenanceCycleResult:
        """在后台循环停止时显式运行一个全局维护周期。"""

        if self._loop_task is not None and not self._loop_task.done():
            raise LifecycleWorkerStateError("manual lifecycle run cannot race the background loop")
        if self.busy:
            raise LifecycleWorkerStateError("lifecycle worker already has an active cycle")
        result = await self._run_once()
        self._last_cycle = result
        self._last_error = None
        self._observe_cycle(result)
        return result

    def wake(self) -> None:
        """提示后台循环尽快执行一次维护。"""

        self._wake_event.set()

    async def _run_loop(self) -> None:
        try:
            while not self._stop_requested.is_set():
                await self._wait(self.config.maintenance_interval_seconds)
                if self._stop_requested.is_set():
                    break
                try:
                    result = await self._run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._last_error = exc
                    self._observe_failure(exc)
                    continue
                self._last_cycle = result
                self._last_error = None
                self._observe_cycle(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = exc
            self._state = LifecycleWorkerState.FAILED
            self._observe_failure(exc)
        finally:
            active = self._active_cycle
            if active is not None and not active.done():
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
            self._active_cycle = None
            if self._state in {LifecycleWorkerState.RUNNING, LifecycleWorkerState.STOPPING}:
                self._state = LifecycleWorkerState.STOPPED

    async def _run_once(self) -> LifecycleMaintenanceCycleResult:
        started_at = self._clock()
        try:
            token = await asyncio.to_thread(self.lock_store.acquire, self.lock_key, self.config.lease_ttl_seconds)
        except TimeoutError:
            return LifecycleMaintenanceCycleResult(
                lease_acquired=False, started_at=started_at, finished_at=self._clock()
            )

        execution = asyncio.create_task(
            self._maintain_all(token, started_at), name=f"habitus-lifecycle-cycle:{self.worker_id}"
        )
        self._active_cycle = execution
        heartbeat_stop = threading.Event()
        loop = asyncio.get_running_loop()
        heartbeat = asyncio.create_task(
            asyncio.to_thread(self._heartbeat_blocking, token, execution, heartbeat_stop, loop),
            name=f"habitus-lifecycle-heartbeat:{self.worker_id}",
        )
        body_error: BaseException | None = None
        try:
            done, _pending = await asyncio.wait({execution, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                heartbeat_error = heartbeat.exception()
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                if heartbeat_error is not None:
                    raise heartbeat_error
                raise LifecycleWorkerLeaseLostError("lifecycle heartbeat stopped unexpectedly")
            return await execution
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            heartbeat_stop.set()
            await asyncio.gather(heartbeat, return_exceptions=True)
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            self._active_cycle = None
            try:
                await asyncio.to_thread(self.lock_store.release, token)
            except Exception:
                if body_error is None:
                    raise

    async def _maintain_all(self, token: LockToken, started_at: datetime) -> LifecycleMaintenanceCycleResult:
        context = _CycleContext(now=started_at, token=token, lock_store=self.lock_store)
        outcomes = []
        for participant in self._participants:
            await asyncio.to_thread(context.assert_owned)
            outcomes.append(await self._maintain_one(participant, context))
        await asyncio.to_thread(context.assert_owned)
        return LifecycleMaintenanceCycleResult(
            lease_acquired=True, started_at=started_at, finished_at=self._clock(), outcomes=tuple(outcomes)
        )

    async def _maintain_one(
        self, participant: LifecycleParticipant, context: _CycleContext
    ) -> LifecycleParticipantOutcome:
        try:
            report = await participant.maintain(context)
        except (asyncio.CancelledError, LifecycleWorkerLeaseLostError):
            raise  # 租约丢了整轮停，不能当成单个参与者的失败接着跑下一个
        except Exception as exc:
            message = (str(exc).strip() or type(exc).__name__)[:_MESSAGE_LIMIT]
            return LifecycleParticipantOutcome(participant.name, error_type=type(exc).__name__, message=message)
        if self.capacity is not None and report.readings:
            snapshot = CapacitySnapshot(participant.name, context.now, report.readings)
            await asyncio.to_thread(self.capacity.publish, snapshot)
        return LifecycleParticipantOutcome(participant.name, report=report)

    def _heartbeat_blocking(
        self,
        token: LockToken,
        execution: asyncio.Task[LifecycleMaintenanceCycleResult],
        stop: threading.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """在独立线程续租，避免同步文件或 SQLite 操作阻塞事件循环时丢 lease。"""

        while not stop.wait(timeout=self.config.heartbeat_interval_seconds):
            try:
                self.lock_store.renew(token, self.config.lease_ttl_seconds)
            except TimeoutError as exc:
                loop.call_soon_threadsafe(execution.cancel)
                raise LifecycleWorkerLeaseLostError("lifecycle lease could not be renewed") from exc
            except Exception:
                loop.call_soon_threadsafe(execution.cancel)
                raise

    async def _wait(self, delay_seconds: float) -> None:
        if self._stop_requested.is_set():
            return
        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=delay_seconds)
        except TimeoutError:
            pass
        finally:
            self._wake_event.clear()

    def _observe_cycle(self, result: LifecycleMaintenanceCycleResult) -> None:
        self.observer.record(
            ObservationEvent(
                category="lifecycle",
                operation="maintenance_cycle",
                status=ObservationStatus.DEGRADED if result.failures or result.signals else ObservationStatus.SUCCESS,
                duration_seconds=max(0.0, (result.finished_at - result.started_at).total_seconds()),
                attributes={
                    "lease_acquired": result.lease_acquired,
                    "participants": len(result.outcomes),
                    "failures": len(result.failures),
                    "signals": len(result.signals),
                    "released": sum(item.report.released for item in result.outcomes if item.report is not None),
                },
            )
        )

    def _observe_failure(self, error: BaseException) -> None:
        self.observer.record(
            ObservationEvent(
                category="lifecycle",
                operation="maintenance_cycle",
                status=ObservationStatus.FAILURE,
                duration_seconds=0.0,
                attributes={"error_type": type(error).__name__},
            )
        )


__all__ = [
    "LifecycleMaintenanceCycleResult",
    "LifecycleParticipantOutcome",
    "LifecycleWorker",
    "LifecycleWorkerLeaseLostError",
    "LifecycleWorkerState",
    "LifecycleWorkerStateError",
]
