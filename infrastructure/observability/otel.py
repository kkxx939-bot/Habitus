"""可选 OpenTelemetry OTLP 后端；未安装 SDK 时由管理器降级。"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import import_module
from typing import Any

from Config.observability import ObservabilityTracingConfig
from foundation.observability import ObservationEvent, project_metric_updates


class OpenTelemetryBackend:
    """拥有自己的 Provider，不覆盖嵌入式宿主进程的全局 Provider。"""

    def __init__(self, config: ObservabilityTracingConfig) -> None:
        if not isinstance(config, ObservabilityTracingConfig):
            raise TypeError("config must be ObservabilityTracingConfig")
        self.config = config
        self._tracer: Any = None
        self._meter: Any = None
        self._tracer_provider: Any = None
        self._meter_provider: Any = None
        self._instruments: dict[tuple[str, str], Any] = {}
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        headers = _headers_from_environment(self.config.headers_env)
        protocol_module = "http" if self.config.protocol == "http" else "grpc"
        metric_module = import_module(
            f"opentelemetry.exporter.otlp.proto.{protocol_module}.metric_exporter"
        )
        trace_module = import_module(
            f"opentelemetry.exporter.otlp.proto.{protocol_module}.trace_exporter"
        )
        metric_exporter_type: Any = metric_module.OTLPMetricExporter
        span_exporter_type: Any = trace_module.OTLPSpanExporter

        resource = Resource.create({"service.name": self.config.service_name})
        trace_endpoint = self.config.endpoint
        metric_endpoint = self.config.endpoint
        if self.config.protocol == "http":
            base_endpoint = self.config.endpoint.rstrip("/")
            trace_endpoint = f"{base_endpoint}/v1/traces"
            metric_endpoint = f"{base_endpoint}/v1/metrics"
        common_kwargs: dict[str, object] = {"headers": headers}
        if self.config.protocol == "grpc":
            common_kwargs["insecure"] = self.config.insecure
        span_exporter = span_exporter_type(endpoint=trace_endpoint, **common_kwargs)
        metric_exporter = metric_exporter_type(endpoint=metric_endpoint, **common_kwargs)
        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=int(float(self.config.export_interval_seconds) * 1000),
        )
        self._meter_provider = MeterProvider(resource=resource, metric_readers=(reader,))
        self._tracer = self._tracer_provider.get_tracer("m2bos")
        self._meter = self._meter_provider.get_meter("m2bos")
        self._initialized = True

    def record(self, event: ObservationEvent) -> None:
        if not self._initialized:
            return
        for update in project_metric_updates(event):
            instrument = self._instrument(update.name, update.kind)
            attributes = dict(update.labels)
            if update.kind == "counter":
                instrument.add(update.value, attributes)
            elif update.kind == "gauge":
                instrument.set(update.value, attributes)
            else:
                instrument.record(update.value, attributes)
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.add_event(
                f"{event.category}.{event.operation}",
                attributes=_trace_attributes(event),
                timestamp=int(event.occurred_at.timestamp() * 1_000_000_000),
            )

    @contextmanager
    def start_span(
        self,
        category: str,
        operation: str,
        *,
        attributes: Mapping[str, str | int | float | bool] | None = None,
        traceparent: str | None = None,
    ) -> Iterator[None]:
        if not self._initialized:
            yield
            return
        safe = {
            f"m2bos.{key}": value
            for key, value in (attributes or {}).items()
            if isinstance(value, str | int | float | bool) and len(key) <= 64
        }
        parent_context = None
        if traceparent is not None:
            from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

            parent_context = TraceContextTextMapPropagator().extract({"traceparent": traceparent})
        with self._tracer.start_as_current_span(
            f"{category}.{operation}",
            context=parent_context,
            attributes=safe,
            record_exception=False,
        ) as span:
            try:
                yield
            except BaseException as exc:
                span.set_attribute("m2bos.error_type", type(exc).__name__)
                raise

    def shutdown(self) -> None:
        first_error: BaseException | None = None
        for provider in (self._meter_provider, self._tracer_provider):
            if provider is None:
                continue
            try:
                provider.shutdown()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._initialized = False
        if first_error is not None:
            raise first_error

    def _instrument(self, name: str, kind: str):  # noqa: ANN202
        key = (name, kind)
        instrument = self._instruments.get(key)
        if instrument is not None:
            return instrument
        if kind == "counter":
            instrument = self._meter.create_counter(f"m2bos.{name}")
        elif kind == "gauge":
            instrument = self._meter.create_gauge(f"m2bos.{name}")
        else:
            instrument = self._meter.create_histogram(f"m2bos.{name}", unit="s")
        self._instruments[key] = instrument
        return instrument


def _headers_from_environment(name: str) -> dict[str, str]:
    raw = os.environ.get(name, "")
    headers: dict[str, str] = {}
    for item in raw.split(","):
        key, separator, value = item.partition("=")
        if separator and key.strip() and value.strip():
            headers[key.strip()] = value.strip()
    return headers


def _trace_attributes(event: ObservationEvent) -> dict[str, str | int | float | bool]:
    result: dict[str, str | int | float | bool] = {
        "m2bos.category": event.category,
        "m2bos.operation": event.operation,
        "m2bos.status": event.status.value,
        "m2bos.duration_seconds": event.duration_seconds,
    }
    for name in ("request_id", "memory_sequence", "transaction_id", "worker_id", "attempt"):
        value = getattr(event.context, name)
        if value is not None:
            result[f"m2bos.{name}"] = value
    for key, value in event.attributes.items():
        if key in {
            "provider",
            "model",
            "http_method",
            "http_route",
            "http_status_code",
            "error_code",
            "error_type",
            "retryable",
            "retry_count",
            "stage",
            "job_status",
        }:
            result[f"m2bos.{key}"] = value
    return result


__all__ = ["OpenTelemetryBackend"]
