"""Runtime 健康、就绪和依赖状态的只读应用服务。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from memory.workflow import MemoryJobStatus
from Runtime.components import RuntimeComponents
from Runtime.lifecycle import LifecycleWorkerState
from Runtime.worker import MemoryWorkerState


class RuntimeHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass(frozen=True)
class RuntimeHealthCheck:
    name: str
    status: RuntimeHealthStatus
    detail: str
    critical: bool = True


@dataclass(frozen=True)
class RuntimeHealthReport:
    status: RuntimeHealthStatus
    ready: bool
    checked_at: datetime
    checks: tuple[RuntimeHealthCheck, ...]


class RuntimeHealthService:
    """聚合现有组件状态，不触发索引重建或业务写入。"""

    def __init__(self, components: RuntimeComponents) -> None:
        if not isinstance(components, RuntimeComponents):
            raise TypeError("components must be RuntimeComponents")
        self.components = components

    async def report(self, runtime_state: str, *, deep: bool = False) -> RuntimeHealthReport:
        if not isinstance(runtime_state, str) or not runtime_state:
            raise ValueError("runtime_state must be non-empty text")
        if not isinstance(deep, bool):
            raise TypeError("deep must be boolean")
        checks = [
            self._runtime_check(runtime_state),
            self._lifecycle_storage_check(runtime_state),
            self._memory_worker_check(),
            self._lifecycle_check(),
            self._observability_check(),
        ]
        checks.append(await self._queue_check())
        checks.extend(await asyncio.gather(self._vector_check("memory_vector", self.components.memory.vector_index.store), self._vector_check("summary_vector", self.components.conversation.summary_vector_index.store)))
        if deep:
            checks.extend(
                await asyncio.gather(
                    self._chat_check(),
                    self._index_consistency_check(
                        "memory_index_consistency",
                        self.components.memory.vector_index,
                    ),
                    self._index_consistency_check(
                        "summary_index_consistency",
                        self.components.conversation.summary_vector_index,
                    ),
                )
            )
        critical_unhealthy = any(
            check.critical and check.status in {RuntimeHealthStatus.UNHEALTHY, RuntimeHealthStatus.CLOSED}
            for check in checks
        )
        critical_degraded = any(
            check.critical and check.status is RuntimeHealthStatus.DEGRADED for check in checks
        )
        if runtime_state == "closed":
            status = RuntimeHealthStatus.CLOSED
        elif critical_unhealthy:
            status = RuntimeHealthStatus.UNHEALTHY
        elif critical_degraded:
            status = RuntimeHealthStatus.DEGRADED
        else:
            status = RuntimeHealthStatus.HEALTHY
        ready = runtime_state in {"ready", "running"} and not critical_unhealthy and not critical_degraded
        return RuntimeHealthReport(status, ready, datetime.now(timezone.utc), tuple(checks))

    @staticmethod
    def _runtime_check(state: str) -> RuntimeHealthCheck:
        if state == "closed":
            return RuntimeHealthCheck("runtime", RuntimeHealthStatus.CLOSED, state)
        if state == "created":
            return RuntimeHealthCheck("runtime", RuntimeHealthStatus.DEGRADED, "not_initialized")
        return RuntimeHealthCheck("runtime", RuntimeHealthStatus.HEALTHY, state)

    def _memory_worker_check(self) -> RuntimeHealthCheck:
        worker = self.components.workflow.worker
        if worker.state is MemoryWorkerState.FAILED:
            return RuntimeHealthCheck("memory_worker", RuntimeHealthStatus.UNHEALTHY, self._error(worker.last_error))
        if worker.state is MemoryWorkerState.BLOCKED:
            return RuntimeHealthCheck("memory_worker", RuntimeHealthStatus.DEGRADED, self._error(worker.last_error))
        return RuntimeHealthCheck("memory_worker", RuntimeHealthStatus.HEALTHY, worker.state.value)

    def _lifecycle_check(self) -> RuntimeHealthCheck:
        worker = self.components.workflow.lifecycle_worker
        if worker.state is LifecycleWorkerState.FAILED:
            return RuntimeHealthCheck("lifecycle_worker", RuntimeHealthStatus.UNHEALTHY, self._error(worker.last_error))
        if worker.last_error is not None:
            return RuntimeHealthCheck("lifecycle_worker", RuntimeHealthStatus.DEGRADED, self._error(worker.last_error), critical=False)
        return RuntimeHealthCheck("lifecycle_worker", RuntimeHealthStatus.HEALTHY, worker.state.value, critical=False)

    def _lifecycle_storage_check(self, runtime_state: str) -> RuntimeHealthCheck:
        recall_store = self.components.memory.search.recall_lifecycle.store
        summary_use = self.components.conversation.summary_use
        if runtime_state == "created":
            return RuntimeHealthCheck(
                "lifecycle_storage",
                RuntimeHealthStatus.DEGRADED,
                "not_initialized",
            )
        if not getattr(recall_store, "initialized", False) or not summary_use.initialized:
            return RuntimeHealthCheck(
                "lifecycle_storage",
                RuntimeHealthStatus.UNHEALTHY,
                "not_initialized",
            )
        try:
            pending_l2 = len(self.components.memory.lifecycle.operation_store.pending())
            pending_summaries = len(
                self.components.workflow.lifecycle.retirement_store.pending()
            )
        except Exception as exc:
            return RuntimeHealthCheck(
                "lifecycle_storage",
                RuntimeHealthStatus.UNHEALTHY,
                type(exc).__name__,
            )
        return RuntimeHealthCheck(
            "lifecycle_storage",
            RuntimeHealthStatus.HEALTHY,
            f"l2_pending={pending_l2};summary_pending={pending_summaries}",
        )

    def _observability_check(self) -> RuntimeHealthCheck:
        manager = self.components.infrastructure.managed_observability
        if manager is None:
            return RuntimeHealthCheck(
                "observability",
                RuntimeHealthStatus.DEGRADED,
                "not_configured",
                critical=False,
            )
        status, detail = manager.health()
        resolved = RuntimeHealthStatus.HEALTHY if status == "healthy" else RuntimeHealthStatus.DEGRADED
        return RuntimeHealthCheck("observability", resolved, detail, critical=False)

    async def _queue_check(self) -> RuntimeHealthCheck:
        try:
            job = await asyncio.to_thread(self.components.workflow.jobs.oldest_uncommitted)
        except Exception as exc:
            return RuntimeHealthCheck("memory_queue", RuntimeHealthStatus.UNHEALTHY, type(exc).__name__)
        if job is None:
            self.components.infrastructure.observability.set_gauge("memory_queue_blocked", 0)
            return RuntimeHealthCheck("memory_queue", RuntimeHealthStatus.HEALTHY, "empty")
        if job.status is MemoryJobStatus.FAILED:
            self.components.infrastructure.observability.set_gauge("memory_queue_blocked", 1)
            return RuntimeHealthCheck("memory_queue", RuntimeHealthStatus.DEGRADED, f"failed:{job.memory_sequence}")
        self.components.infrastructure.observability.set_gauge("memory_queue_blocked", 0)
        return RuntimeHealthCheck("memory_queue", RuntimeHealthStatus.HEALTHY, f"{job.status.value}:{job.memory_sequence}")

    @staticmethod
    async def _vector_check(name: str, store: object) -> RuntimeHealthCheck:
        try:
            state = await store.state()  # type: ignore[attr-defined]
        except Exception as exc:
            return RuntimeHealthCheck(name, RuntimeHealthStatus.UNHEALTHY, type(exc).__name__)
        if state is None:
            return RuntimeHealthCheck(name, RuntimeHealthStatus.DEGRADED, "not_published")
        if not state.ready:
            return RuntimeHealthCheck(name, RuntimeHealthStatus.DEGRADED, "not_ready")
        return RuntimeHealthCheck(name, RuntimeHealthStatus.HEALTHY, f"generation={state.generation};records={state.record_count}")

    async def _chat_check(self) -> RuntimeHealthCheck:
        result = await asyncio.to_thread(self.components.models.chat.health_check)
        if result.get("ok") is True:
            return RuntimeHealthCheck("chat_model", RuntimeHealthStatus.HEALTHY, "available", critical=False)
        return RuntimeHealthCheck(
            "chat_model",
            RuntimeHealthStatus.DEGRADED,
            str(result.get("error_code", "unavailable"))[:256],
            critical=False,
        )

    @staticmethod
    async def _index_consistency_check(name: str, index: object) -> RuntimeHealthCheck:
        """执行无副作用的真相源/派生索引审计，不在健康检查中重建。"""

        try:
            state = await index.store.state()  # type: ignore[attr-defined]
            if state is None:
                return RuntimeHealthCheck(name, RuntimeHealthStatus.DEGRADED, "not_published")
            if not state.ready:
                return RuntimeHealthCheck(name, RuntimeHealthStatus.DEGRADED, "not_ready")
            report = await index.audit_consistency()  # type: ignore[attr-defined]
        except Exception as exc:
            return RuntimeHealthCheck(name, RuntimeHealthStatus.UNHEALTHY, type(exc).__name__)
        if report.ok:
            return RuntimeHealthCheck(
                name,
                RuntimeHealthStatus.HEALTHY,
                f"expected={report.expected_count};indexed={report.indexed_count}",
            )
        return RuntimeHealthCheck(
            name,
            RuntimeHealthStatus.UNHEALTHY,
            (
                f"missing={len(report.missing_identities)};"
                f"stale={len(report.stale_identities)};"
                f"orphan={len(report.orphan_identities)}"
            ),
        )

    @staticmethod
    def _error(error: BaseException | None) -> str:
        return "none" if error is None else type(error).__name__


__all__ = [
    "RuntimeHealthCheck",
    "RuntimeHealthReport",
    "RuntimeHealthService",
    "RuntimeHealthStatus",
]
