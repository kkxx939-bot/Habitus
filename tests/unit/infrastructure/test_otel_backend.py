"""用协议级替身验证 OTel 指标、Span 和关闭语义，不连接外部 Collector。"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import ModuleType

import pytest

from habitus.config.observability import ObservabilityTracingConfig
from habitus.foundation.observability import ObservationEvent, ObservationStatus, bind_observation_context
from habitus.infrastructure.observability.otel import OpenTelemetryBackend

UTC = UTC


class Instrument:
    def __init__(self) -> None:
        self.values: list[tuple[float, dict[str, str]]] = []

    def add(self, value: float, attributes: dict[str, str]) -> None:
        self.values.append((value, attributes))

    def set(self, value: float, attributes: dict[str, str]) -> None:
        self.values.append((value, attributes))

    def record(self, value: float, attributes: dict[str, str]) -> None:
        self.values.append((value, attributes))


class Meter:
    def __init__(self) -> None:
        self.created: dict[tuple[str, str], Instrument] = {}

    def create_counter(self, name: str) -> Instrument:
        return self._create(name, "counter")

    def create_gauge(self, name: str) -> Instrument:
        return self._create(name, "gauge")

    def create_histogram(self, name: str, *, unit: str) -> Instrument:
        assert unit == "s"
        return self._create(name, "histogram")

    def _create(self, name: str, kind: str) -> Instrument:
        instrument = Instrument()
        self.created[(name, kind)] = instrument
        return instrument


class Span:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object], int]] = []
        self.attributes: dict[str, str] = {}
        self.statuses: list[object] = []

    def is_recording(self) -> bool:
        return True

    def add_event(self, name: str, *, attributes: dict[str, object], timestamp: int) -> None:
        self.events.append((name, attributes, timestamp))

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[name] = value

    def set_status(self, status: object) -> None:
        self.statuses.append(status)


class Tracer:
    def __init__(self) -> None:
        self.span = Span()
        self.calls: list[dict[str, object]] = []

    @contextmanager
    def start_as_current_span(self, name: str, **values: object):
        self.calls.append({"name": name, **values})
        yield self.span


def _event() -> ObservationEvent:
    with bind_observation_context(request_id="otel-request", memory_sequence=3, transaction_id="otel-tx"):
        return ObservationEvent(
            category="http",
            operation="request",
            status=ObservationStatus.SUCCESS,
            duration_seconds=0.25,
            occurred_at=datetime(2026, 7, 30, tzinfo=UTC),
            attributes={
                "http_method": "POST",
                "http_route": "/api/v1/memory/recall",
                "http_status_code": 200,
                "http_status_class": "2xx",
            },
        )


def test_uninitialized_backend_is_a_noop_for_record_and_span() -> None:
    backend = OpenTelemetryBackend(ObservabilityTracingConfig())

    backend.record(_event())
    with backend.start_span("http", "request"):
        pass

    assert backend.initialized is False
    assert backend._instruments == {}


def test_backend_accepts_yaml_resolved_headers_without_exposing_values_in_repr() -> None:
    backend = OpenTelemetryBackend(
        ObservabilityTracingConfig(credential_ref="otel"),
        headers={"authorization": "Bearer private-token"},
    )

    assert backend._headers == {"authorization": "Bearer private-token"}
    assert "private-token" not in repr(backend)


def test_initialized_backend_projects_metrics_and_safe_trace_event(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OpenTelemetryBackend(ObservabilityTracingConfig())
    meter = Meter()
    tracer = Tracer()
    current_span = Span()
    backend._initialized = True
    backend._meter = meter
    backend._tracer = tracer
    package = ModuleType("opentelemetry")
    trace_module = ModuleType("opentelemetry.trace")

    class StatusCode:
        ERROR = "error"

    class Status:
        def __init__(self, status_code: str) -> None:
            self.status_code = status_code

    trace_module.get_current_span = lambda: current_span  # type: ignore[attr-defined]
    trace_module.Status = Status  # type: ignore[attr-defined]
    trace_module.StatusCode = StatusCode  # type: ignore[attr-defined]
    package.trace = trace_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry", package)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_module)

    event = _event()
    backend.record(event)
    backend.record(event)
    backend.record(replace(event, status=ObservationStatus.FAILURE))

    assert len(meter.created) == 4
    assert all(len(instrument.values) == 3 for instrument in meter.created.values())
    assert current_span.events[0][0] == "http.request"
    trace_attributes = current_span.events[0][1]
    assert trace_attributes["habitus.request_id"] == "otel-request"
    assert trace_attributes["habitus.memory_sequence"] == 3
    assert "content" not in trace_attributes
    assert len(current_span.statuses) == 1
    assert current_span.statuses[0].status_code == "error"  # type: ignore[attr-defined]


def test_span_keeps_bounded_attributes_and_marks_error_type() -> None:
    backend = OpenTelemetryBackend(ObservabilityTracingConfig())
    tracer = Tracer()
    backend._initialized = True
    backend._tracer = tracer

    with pytest.raises(ValueError, match="boom"):
        with backend.start_span(
            "memory",
            "commit",
            attributes={"stage": "l2_commit", "ignored": object()},  # type: ignore[dict-item]
        ):
            raise ValueError("boom")

    assert tracer.calls[0]["name"] == "memory.commit"
    assert tracer.calls[0]["attributes"] == {"habitus.stage": "l2_commit"}
    assert tracer.span.attributes == {"habitus.error_type": "ValueError"}
    assert len(tracer.span.statuses) == 1


def test_shutdown_attempts_both_providers_and_reports_first_failure() -> None:
    class Provider:
        def __init__(self, error: Exception | None = None) -> None:
            self.error = error
            self.calls = 0

        def shutdown(self) -> None:
            self.calls += 1
            if self.error is not None:
                raise self.error

    backend = OpenTelemetryBackend(ObservabilityTracingConfig())
    meter = Provider(RuntimeError("meter failed"))
    tracer = Provider(ValueError("tracer failed"))
    backend._initialized = True
    backend._meter_provider = meter
    backend._tracer_provider = tracer

    with pytest.raises(RuntimeError, match="meter failed"):
        backend.shutdown()

    assert meter.calls == 1
    assert tracer.calls == 1
    assert backend.initialized is False
