"""顶层 Runtime 管理的 Conversation 生命周期维护 Worker。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from bisect import bisect_right
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from uuid import uuid4

from Config import ConversationLifecycleConfig
from foundation.integrity import canonical_json
from foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from infrastructure.store.contracts import LockToken
from infrastructure.store.filesystem import atomic_replace_bytes, read_regular_bytes
from memory.compaction import MemoryLifecycleMaintenanceResult, MemoryLifecycleManager
from memory.conversation import ConversationAddress, ConversationMessageJournal
from memory.workflow import ConversationLifecycleManager

_LIFECYCLE_CURSOR_SCHEMA = "conversation_lifecycle_cursor_v1"


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
class LifecycleMaintenanceFailure:
    """单个 Conversation 维护失败的有界可观察信息。"""

    address: ConversationAddress
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        for name in ("error_type", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be non-empty text")
        if len(self.message) > 1_000:
            raise ValueError("maintenance failure message exceeds its bound")


@dataclass(frozen=True)
class LifecycleMaintenanceCycleResult:
    """一次全局维护周期的选取、成功和失败结果。"""

    lease_acquired: bool
    started_at: datetime
    finished_at: datetime
    selected_addresses: tuple[ConversationAddress, ...]
    maintained_addresses: tuple[ConversationAddress, ...]
    failures: tuple[LifecycleMaintenanceFailure, ...]
    memory_maintenance: MemoryLifecycleMaintenanceResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lease_acquired, bool):
            raise TypeError("lease_acquired must be boolean")
        for name in ("started_at", "finished_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("maintenance cycle cannot finish before it starts")
        for name in ("selected_addresses", "maintained_addresses"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(value, ConversationAddress) for value in values):
                raise TypeError(f"{name} must contain ConversationAddress values")
        if not isinstance(self.failures, tuple) or any(
            not isinstance(value, LifecycleMaintenanceFailure) for value in self.failures
        ):
            raise TypeError("failures must contain LifecycleMaintenanceFailure values")
        selected = set(self.selected_addresses)
        if any(value not in selected for value in self.maintained_addresses):
            raise ValueError("maintained addresses must come from the selected batch")
        if any(value.address not in selected for value in self.failures):
            raise ValueError("failed addresses must come from the selected batch")
        if set(self.maintained_addresses) & {value.address for value in self.failures}:
            raise ValueError("one address cannot both succeed and fail in one cycle")
        if not self.lease_acquired and (self.selected_addresses or self.maintained_addresses or self.failures):
            raise ValueError("a skipped lease cycle cannot contain maintenance results")
        if self.memory_maintenance is not None and not isinstance(
            self.memory_maintenance,
            MemoryLifecycleMaintenanceResult,
        ):
            raise TypeError("memory_maintenance must be MemoryLifecycleMaintenanceResult or None")
        if not self.lease_acquired and self.memory_maintenance is not None:
            raise ValueError("a skipped lease cycle cannot maintain L2 memory")


class _LifecycleCursorStore:
    """在 Workflow 根内保存跨进程轮转所需的最小耐久游标。"""

    def __init__(self, manager: ConversationLifecycleManager) -> None:
        self.root = manager.jobs.root
        self.path = self.root / "lifecycle" / "state.json"
        self.max_bytes = manager.jobs.config.max_file_bytes

    def read(self) -> tuple[date, str] | None:
        try:
            encoded = read_regular_bytes(
                self.path,
                artifact_root=self.root,
                max_bytes=self.max_bytes,
            )
        except FileNotFoundError:
            return None
        try:
            raw = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise LifecycleWorkerStateError("lifecycle cursor state is invalid JSON") from exc
        expected = {"schema", "started_on", "conversation_id"}
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise LifecycleWorkerStateError("lifecycle cursor state has an invalid shape")
        if raw["schema"] != _LIFECYCLE_CURSOR_SCHEMA:
            raise LifecycleWorkerStateError("lifecycle cursor state has an unsupported schema")
        started_on_value = raw["started_on"]
        conversation_id = raw["conversation_id"]
        if not isinstance(started_on_value, str) or not isinstance(conversation_id, str):
            raise LifecycleWorkerStateError("lifecycle cursor state contains invalid field types")
        try:
            address = ConversationAddress(
                conversation_id=conversation_id,
                started_on=date.fromisoformat(started_on_value),
            )
        except ValueError as exc:
            raise LifecycleWorkerStateError("lifecycle cursor state is invalid") from exc
        if encoded != self._encode(address):
            raise LifecycleWorkerStateError("lifecycle cursor state is not canonically encoded")
        return address.started_on, address.conversation_id

    def write(self, address: ConversationAddress) -> None:
        if not isinstance(address, ConversationAddress):
            raise TypeError("lifecycle cursor address must be ConversationAddress")
        encoded = self._encode(address)
        if len(encoded) > self.max_bytes:
            raise LifecycleWorkerStateError("lifecycle cursor state exceeds its configured bound")
        atomic_replace_bytes(
            self.path,
            encoded,
            artifact_root=self.root,
        )

    @staticmethod
    def _encode(address: ConversationAddress) -> bytes:
        return (
            canonical_json(
                {
                    "schema": _LIFECYCLE_CURSOR_SCHEMA,
                    "started_on": address.started_on.isoformat(),
                    "conversation_id": address.conversation_id,
                }
            )
            + "\n"
        ).encode("utf-8")


class LifecycleWorker:
    """按周期取得全局 lease，并有界轮转维护全部 Conversation。"""

    def __init__(
        self,
        manager: ConversationLifecycleManager,
        config: ConversationLifecycleConfig,
        *,
        worker_id: str | None = None,
        memory_manager: MemoryLifecycleManager | None = None,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(manager, ConversationLifecycleManager):
            raise TypeError("manager must be ConversationLifecycleManager")
        if not isinstance(config, ConversationLifecycleConfig):
            raise TypeError("config must be ConversationLifecycleConfig")
        resolved_worker_id = worker_id or f"lifecycle-{os.getpid()}-{uuid4().hex}"
        if memory_manager is not None and not isinstance(memory_manager, MemoryLifecycleManager):
            raise TypeError("memory_manager must be MemoryLifecycleManager")
        if (
            not isinstance(resolved_worker_id, str)
            or not resolved_worker_id
            or any(ord(character) < 32 for character in resolved_worker_id)
        ):
            raise ValueError("worker_id must be non-empty normalized text")
        self.manager = manager
        self.journal: ConversationMessageJournal = manager.journal
        self.config = config
        self.worker_id = resolved_worker_id
        self.memory_manager = memory_manager
        root_digest = hashlib.sha256(str(self.journal.layout.root).encode("utf-8")).hexdigest()[:24]
        self.lock_key = f"runtime:lifecycle:{root_digest}"
        self._cursor_store = _LifecycleCursorStore(manager)
        self._state = LifecycleWorkerState.CREATED
        self._loop_task: asyncio.Task[None] | None = None
        self._active_cycle: asyncio.Task[LifecycleMaintenanceCycleResult] | None = None
        self._stop_requested = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._last_cycle: LifecycleMaintenanceCycleResult | None = None
        self._last_error: BaseException | None = None
        self.observer = observer or NullObserver()

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
        self._loop_task = asyncio.create_task(
            self._run_loop(),
            name=f"habitus-lifecycle-worker:{self.worker_id}",
        )
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
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.config.shutdown_timeout_seconds,
            )
        except asyncio.TimeoutError:
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
        started_at = datetime.now(timezone.utc)
        lock_store = self.journal.path_lock.lock_store
        try:
            token = await asyncio.to_thread(
                lock_store.acquire,
                self.lock_key,
                self.config.lease_ttl_seconds,
            )
        except TimeoutError:
            finished_at = datetime.now(timezone.utc)
            return LifecycleMaintenanceCycleResult(
                lease_acquired=False,
                started_at=started_at,
                finished_at=finished_at,
                selected_addresses=(),
                maintained_addresses=(),
                failures=(),
            )

        execution = asyncio.create_task(
            self._maintain_batch(token, started_at),
            name=f"habitus-lifecycle-cycle:{self.worker_id}",
        )
        self._active_cycle = execution
        heartbeat_stop = threading.Event()
        loop = asyncio.get_running_loop()
        heartbeat = asyncio.create_task(
            asyncio.to_thread(
                self._heartbeat_blocking,
                token,
                execution,
                heartbeat_stop,
                loop,
            ),
            name=f"habitus-lifecycle-heartbeat:{self.worker_id}",
        )
        body_error: BaseException | None = None
        try:
            done, _pending = await asyncio.wait(
                {execution, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
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
                await asyncio.to_thread(lock_store.release, token)
            except Exception:
                if body_error is None:
                    raise

    async def _maintain_batch(
        self,
        token: LockToken,
        started_at: datetime,
    ) -> LifecycleMaintenanceCycleResult:
        addresses = await asyncio.to_thread(self.journal.list_addresses)
        cursor_key = await asyncio.to_thread(self._cursor_store.read)
        selected = self._select_batch(addresses, cursor_key)
        maintained: list[ConversationAddress] = []
        failures: list[LifecycleMaintenanceFailure] = []
        lock_store = self.journal.path_lock.lock_store
        for address in selected:
            await asyncio.to_thread(lock_store.assert_owned, token)
            try:
                await self.manager.maintain_once(address)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures.append(self._failure(address, exc))
            else:
                maintained.append(address)
            finally:
                await asyncio.to_thread(self._write_cursor_fenced, token, address)
        await asyncio.to_thread(lock_store.assert_owned, token)
        memory_maintenance = None
        if self.memory_manager is not None:
            memory_maintenance = await self.memory_manager.maintain()
            await asyncio.to_thread(lock_store.assert_owned, token)
        return LifecycleMaintenanceCycleResult(
            lease_acquired=True,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            selected_addresses=selected,
            maintained_addresses=tuple(maintained),
            failures=tuple(failures),
            memory_maintenance=memory_maintenance,
        )

    def _write_cursor_fenced(
        self,
        token: LockToken,
        address: ConversationAddress,
    ) -> None:
        """只允许仍持有本轮全局 lease 的维护者推进耐久游标。"""

        lock_store = self.journal.path_lock.lock_store
        with lock_store.fenced((token,), ttl_seconds=self.config.lease_ttl_seconds):
            self._cursor_store.write(address)

    def _heartbeat_blocking(
        self,
        token: LockToken,
        execution: asyncio.Task[LifecycleMaintenanceCycleResult],
        stop: threading.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """在独立线程续租，避免同步文件或 SQLite 操作阻塞事件循环时丢 lease。"""

        lock_store = self.journal.path_lock.lock_store
        while not stop.wait(timeout=self.config.heartbeat_interval_seconds):
            try:
                lock_store.renew(token, self.config.lease_ttl_seconds)
            except TimeoutError as exc:
                loop.call_soon_threadsafe(execution.cancel)
                raise LifecycleWorkerLeaseLostError("lifecycle lease could not be renewed") from exc
            except Exception:
                loop.call_soon_threadsafe(execution.cancel)
                raise

    def _select_batch(
        self,
        addresses: tuple[ConversationAddress, ...],
        cursor_key: tuple[date, str] | None,
    ) -> tuple[ConversationAddress, ...]:
        if not addresses:
            return ()
        limit = min(self.config.max_conversations_per_cycle, len(addresses))
        keys = tuple((address.started_on, address.conversation_id) for address in addresses)
        start = 0 if cursor_key is None else bisect_right(keys, cursor_key)
        ordered = (*addresses[start:], *addresses[:start])
        return tuple(ordered[:limit])

    @staticmethod
    def _failure(
        address: ConversationAddress,
        error: Exception,
    ) -> LifecycleMaintenanceFailure:
        message = str(error).strip() or error.__class__.__name__
        return LifecycleMaintenanceFailure(
            address=address,
            error_type=error.__class__.__name__,
            message=message[:1_000],
        )

    async def _wait(self, delay_seconds: float) -> None:
        if self._stop_requested.is_set():
            return
        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=delay_seconds)
        except asyncio.TimeoutError:
            pass
        finally:
            self._wake_event.clear()

    def _observe_cycle(self, result: LifecycleMaintenanceCycleResult) -> None:
        memory_failures = (
            0
            if result.memory_maintenance is None
            else len(result.memory_maintenance.failures)
        )
        self.observer.record(
            ObservationEvent(
                category="lifecycle",
                operation="maintenance_cycle",
                status=(
                    ObservationStatus.DEGRADED
                    if result.failures or memory_failures
                    else ObservationStatus.SUCCESS
                ),
                duration_seconds=max(0.0, (result.finished_at - result.started_at).total_seconds()),
                attributes={
                    "lease_acquired": result.lease_acquired,
                    "selected": len(result.selected_addresses),
                    "maintained": len(result.maintained_addresses),
                    "failures": len(result.failures),
                    "memory_failures": memory_failures,
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
    "LifecycleMaintenanceFailure",
    "LifecycleWorker",
    "LifecycleWorkerLeaseLostError",
    "LifecycleWorkerState",
    "LifecycleWorkerStateError",
]
