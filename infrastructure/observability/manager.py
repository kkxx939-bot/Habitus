"""管理结构化日志、最小审计与可选 OTLP 的故障隔离生命周期。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from Config.observability import ObservabilityConfig
from foundation.observability import ObservationEvent
from infrastructure.observability.audit import AuditRecord, AuditStore
from infrastructure.observability.logging import StructuredLogObserver
from infrastructure.observability.otel import OpenTelemetryBackend


class ManagedObservability:
    """可观测后端是非关键旁路；任何后端失败都不改变记忆事务结果。"""

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        workflow_root: Path,
        tracing_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(config, ObservabilityConfig):
            raise TypeError("config must be ObservabilityConfig")
        self.config = config
        self.logging = StructuredLogObserver(enabled=config.logging.enabled)
        self.audit = (
            AuditStore(
                workflow_root / "observability" / "audit.sqlite3",
                retention_days=config.audit.retention_days,
                max_records=config.audit.max_records,
            )
            if config.audit.enabled
            else None
        )
        self.otel = (
            OpenTelemetryBackend(config.tracing, headers=tracing_headers)
            if config.tracing.enabled
            else None
        )
        self._initialized = False
        self._degraded_reason: str | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        failures: list[str] = []
        if self.audit is not None:
            try:
                self.audit.initialize()
            except Exception as exc:
                failures.append(f"audit:{type(exc).__name__}")
        if self.otel is not None:
            try:
                self.otel.initialize()
            except Exception as exc:
                failures.append(f"otel:{type(exc).__name__}")
        self._degraded_reason = ";".join(failures) or None
        self._initialized = True

    def record(self, event: ObservationEvent) -> None:
        if not isinstance(event, ObservationEvent):
            raise TypeError("event must be ObservationEvent")
        backends = (self.logging,) if not self._initialized else (self.logging, self.audit, self.otel)
        for backend in backends:
            if backend is None:
                continue
            try:
                backend.record(event)
            except Exception as exc:
                self._degraded_reason = f"{type(backend).__name__}:{type(exc).__name__}"

    @contextmanager
    def start_span(
        self,
        category: str,
        operation: str,
        *,
        attributes: Mapping[str, str | int | float | bool] | None = None,
        traceparent: str | None = None,
    ) -> Iterator[None]:
        if self.otel is None or not self.otel.initialized:
            yield
            return
        with self.otel.start_span(
            category,
            operation,
            attributes=attributes,
            traceparent=traceparent,
        ):
            yield

    def health(self) -> tuple[str, str]:
        if not self._initialized:
            return "degraded", "not_initialized"
        if self._degraded_reason is not None:
            return "degraded", self._degraded_reason
        return "healthy", "available"

    def recent_audit(self, *, limit: int = 100) -> tuple[AuditRecord, ...]:
        if self.audit is None:
            return ()
        return self.audit.recent(limit=limit)

    def close(self) -> None:
        if self.otel is None or not self.otel.initialized:
            return
        try:
            self.otel.shutdown()
        except Exception as exc:
            self._degraded_reason = f"otel_shutdown:{type(exc).__name__}"


__all__ = ["ManagedObservability"]
