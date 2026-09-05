"""不绑定 Prometheus 或 OpenTelemetry SDK 的进程内观察契约。"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol

_IDENTITY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_DEFAULT_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_LABEL_ATTRIBUTES = frozenset(
    {
        "provider",
        "model",
        "http_method",
        "http_route",
        "http_status_class",
        "stage",
        "worker_state",
        "job_status",
        # Conversation Source Consumer 的取值由枚举封闭，基数恒定。
        "consumer",
    }
)


class ObservationStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class ObservationContext:
    """只携带关联身份；这些字段绝不成为指标标签。"""

    request_id: str | None = None
    memory_sequence: int | None = None
    transaction_id: str | None = None
    worker_id: str | None = None
    attempt: int | None = None

    def __post_init__(self) -> None:
        for name in ("request_id", "transaction_id", "worker_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or _IDENTITY.fullmatch(value) is None):
                raise ValueError(f"{name} must be a bounded normalized identity or None")
        for name in ("memory_sequence", "attempt"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")


_CURRENT_CONTEXT: ContextVar[ObservationContext | None] = ContextVar(
    "habitus_observation_context",
    default=None,
)


def current_observation_context() -> ObservationContext:
    """返回当前异步调用链的只读关联上下文。"""

    return _CURRENT_CONTEXT.get() or ObservationContext()


@contextmanager
def bind_observation_context(
    *,
    request_id: str | None = None,
    memory_sequence: int | None = None,
    transaction_id: str | None = None,
    worker_id: str | None = None,
    attempt: int | None = None,
) -> Iterator[ObservationContext]:
    """在当前调用链内覆盖显式字段，并在退出时恢复父上下文。"""

    current = current_observation_context()
    context = replace(
        current,
        request_id=current.request_id if request_id is None else request_id,
        memory_sequence=(current.memory_sequence if memory_sequence is None else memory_sequence),
        transaction_id=(current.transaction_id if transaction_id is None else transaction_id),
        worker_id=current.worker_id if worker_id is None else worker_id,
        attempt=current.attempt if attempt is None else attempt,
    )
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_CONTEXT.reset(token)


@dataclass(frozen=True)
class ObservationEvent:
    """一条有界、无原始请求内容的操作观察事件。"""

    category: str
    operation: str
    status: ObservationStatus
    duration_seconds: float
    attributes: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    context: ObservationContext = field(default_factory=current_observation_context)

    def __post_init__(self) -> None:
        for name in ("category", "operation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 128:
                raise ValueError(f"{name} must be non-empty bounded text")
        object.__setattr__(self, "status", ObservationStatus(self.status))
        if isinstance(self.duration_seconds, bool) or not isinstance(self.duration_seconds, int | float):
            raise TypeError("duration_seconds must be numeric")
        duration = float(self.duration_seconds)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_seconds must be finite and non-negative")
        object.__setattr__(self, "duration_seconds", duration)
        if not isinstance(self.attributes, Mapping) or len(self.attributes) > 32:
            raise ValueError("observation attributes must be a bounded mapping")
        attributes: dict[str, str | int | float | bool] = {}
        for key, value in self.attributes.items():
            if not isinstance(key, str) or not key or len(key) > 64:
                raise ValueError("observation attribute names must be bounded text")
            if isinstance(value, str):
                if len(value) > 256:
                    raise ValueError("observation text attributes must be bounded")
            elif isinstance(value, bool):
                pass
            elif isinstance(value, int | float):
                if not math.isfinite(float(value)):
                    raise ValueError("observation numeric attributes must be finite")
            else:
                raise TypeError("observation attributes must contain scalar values")
            attributes[key] = value
        object.__setattr__(self, "attributes", attributes)
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(UTC))
        if not isinstance(self.context, ObservationContext):
            raise TypeError("context must be ObservationContext")


class Observer(Protocol):
    def record(self, event: ObservationEvent) -> None: ...


class SpanController(Protocol):
    """供边界层建立可选根 Span，而不依赖具体追踪 SDK。"""

    def start_span(
        self,
        category: str,
        operation: str,
        *,
        attributes: Mapping[str, str | int | float | bool] | None = None,
        traceparent: str | None = None,
    ) -> AbstractContextManager[None]: ...


class NullObserver:
    def record(self, event: ObservationEvent) -> None:
        if not isinstance(event, ObservationEvent):
            raise TypeError("event must be ObservationEvent")


class CompositeObserver:
    """把同一事件发送给多个独立后端；单个观测后端失败不影响业务。"""

    def __init__(self, *observers: Observer) -> None:
        if not observers or any(not callable(getattr(observer, "record", None)) for observer in observers):
            raise TypeError("observers must implement record")
        self.observers = observers

    def record(self, event: ObservationEvent) -> None:
        if not isinstance(event, ObservationEvent):
            raise TypeError("event must be ObservationEvent")
        for observer in self.observers:
            try:
                observer.record(event)
            except Exception:
                continue


@contextmanager
def observe_operation(
    observer: Observer,
    category: str,
    operation: str,
    *,
    attributes: Mapping[str, str | int | float | bool] | None = None,
) -> Iterator[None]:
    """记录一个同步或异步代码块；观察失败不改变代码块结果。"""

    started = time.monotonic()
    try:
        yield
    except BaseException as exc:
        failure_attributes = dict(attributes or {})
        failure_attributes["error_type"] = type(exc).__name__
        _safe_record(
            observer,
            ObservationEvent(
                category=category,
                operation=operation,
                status=ObservationStatus.FAILURE,
                duration_seconds=max(0.0, time.monotonic() - started),
                attributes=failure_attributes,
            ),
        )
        raise
    else:
        _safe_record(
            observer,
            ObservationEvent(
                category=category,
                operation=operation,
                status=ObservationStatus.SUCCESS,
                duration_seconds=max(0.0, time.monotonic() - started),
                attributes=dict(attributes or {}),
            ),
        )


def _safe_record(observer: Observer, event: ObservationEvent) -> None:
    try:
        observer.record(event)
    except Exception:
        pass


@dataclass(frozen=True)
class ObservabilitySnapshot:
    counters: Mapping[str, int]
    duration_seconds: Mapping[str, float]
    gauges: Mapping[str, float]
    recent_events: tuple[ObservationEvent, ...]
    histogram_counts: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricUpdate:
    """由同一投影规则交给进程内指标和可选 OTel 指标。"""

    name: str
    kind: str
    value: float
    labels: tuple[tuple[str, str], ...] = ()


def project_metric_updates(event: ObservationEvent) -> tuple[MetricUpdate, ...]:
    """把事件投影为小而固定的低基数指标集合。"""

    base_labels = (
        ("category", _label_value(event.category)),
        ("operation", _label_value(event.operation)),
        ("status", event.status.value),
    )
    updates = [
        MetricUpdate("operations_total", "counter", 1.0, base_labels),
        MetricUpdate("operation_duration_seconds", "histogram", event.duration_seconds, base_labels),
    ]
    attributes = event.attributes
    if event.category == "http" and event.operation == "request":
        labels = _selected_labels(attributes, "http_method", "http_route", "http_status_class")
        updates.extend(
            (
                MetricUpdate("http_requests_total", "counter", 1.0, labels),
                MetricUpdate("http_request_duration_seconds", "histogram", event.duration_seconds, labels),
            )
        )
    if event.category == "model":
        labels = _selected_labels(attributes, "provider", "model") + (("operation", _label_value(event.operation)),)
        updates.append(MetricUpdate("model_requests_total", "counter", 1.0, labels + (("status", event.status.value),)))
        retries = _non_negative_number(attributes.get("retry_count"))
        if retries:
            updates.append(MetricUpdate("model_retries_total", "counter", retries, labels))
        for attribute, metric_name in (
            ("input_tokens", "model_input_tokens_total"),
            ("output_tokens", "model_output_tokens_total"),
        ):
            value = _non_negative_number(attributes.get(attribute))
            if value:
                updates.append(MetricUpdate(metric_name, "counter", value, labels))
    if event.category == "retrieval":
        labels = (("operation", _label_value(event.operation)), ("status", event.status.value))
        updates.append(MetricUpdate("retrieval_requests_total", "counter", 1.0, labels))
        degradations = _non_negative_number(attributes.get("degradation_count"))
        if degradations:
            updates.append(MetricUpdate("retrieval_degradations_total", "counter", degradations, labels[:1]))
    if event.category == "lock" and event.status is not ObservationStatus.SUCCESS:
        updates.append(
            MetricUpdate(
                "lock_contention_total",
                "counter",
                1.0,
                (("operation", _label_value(event.operation)), ("status", event.status.value)),
            )
        )
    if event.category == "observability" and event.operation == "snapshot":
        for attribute, metric_name in (
            ("queue_staged", "memory_jobs_staged"),
            ("queue_queued", "memory_jobs_queued"),
            ("queue_running", "memory_jobs_running"),
            ("queue_failed", "memory_jobs_failed"),
            ("queue_committed", "memory_jobs_committed_retained"),
            ("queue_oldest_age_seconds", "memory_job_oldest_age_seconds"),
            ("queue_high_watermark", "memory_job_sequence_high_watermark"),
            ("active_locks", "active_locks"),
            ("hanging_locks", "hanging_locks"),
            ("max_active_lock_age_seconds", "active_lock_max_age_seconds"),
        ):
            value = _non_negative_number(attributes.get(attribute))
            if value is not None:
                updates.append(MetricUpdate(metric_name, "gauge", value))
    return tuple(updates)


class MetricRegistry:
    """线程安全的轻量指标注册表，输出 Prometheus 文本且不依赖 SDK。"""

    def __init__(
        self,
        *,
        max_recent_events: int = 256,
        namespace: str = "habitus",
        duration_buckets: tuple[float, ...] = _DEFAULT_DURATION_BUCKETS,
        enabled: bool = True,
    ) -> None:
        if not 1 <= max_recent_events <= 10_000:
            raise ValueError("max_recent_events must be between 1 and 10000")
        if not isinstance(namespace, str) or not namespace or len(namespace) > 64:
            raise ValueError("namespace must be non-empty bounded text")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be boolean")
        normalized_buckets = tuple(float(value) for value in duration_buckets)
        if (
            not normalized_buckets
            or any(not math.isfinite(value) or value <= 0 for value in normalized_buckets)
            or tuple(sorted(set(normalized_buckets))) != normalized_buckets
        ):
            raise ValueError("duration_buckets must be unique positive ascending numbers")
        self.namespace = _metric_name(namespace)
        self.duration_buckets = normalized_buckets
        self.enabled = enabled
        self._lock = threading.RLock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]], tuple[list[int], float, int]
        ] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._recent: deque[ObservationEvent] = deque(maxlen=max_recent_events)

    def record(self, event: ObservationEvent) -> None:
        if not isinstance(event, ObservationEvent):
            raise TypeError("event must be ObservationEvent")
        if not self.enabled:
            return
        with self._lock:
            self._recent.append(event)
            for update in project_metric_updates(event):
                key = (update.name, update.labels)
                if update.kind == "counter":
                    self._counters[key] = self._counters.get(key, 0.0) + update.value
                elif update.kind == "gauge":
                    self._gauges[key] = update.value
                elif update.kind == "histogram":
                    counts, total, count = self._histograms.get(
                        key,
                        ([0] * len(self.duration_buckets), 0.0, 0),
                    )
                    for index, bucket in enumerate(self.duration_buckets):
                        if update.value <= bucket:
                            counts[index] += 1
                    self._histograms[key] = (counts, total + update.value, count + 1)
                else:  # pragma: no cover - 投影函数只创建三种固定类型
                    raise ValueError("unsupported metric update kind")

    def set_gauge(
        self,
        name: str,
        value: int | float,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("gauge name must be non-empty text")
        if not self.enabled:
            return
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise TypeError("gauge value must be a finite number")
        resolved_labels = _mapping_labels(labels)
        with self._lock:
            self._gauges[(_metric_name(name), resolved_labels)] = float(value)

    def snapshot(self) -> ObservabilitySnapshot:
        with self._lock:
            counters = {
                _snapshot_key(name, labels): int(round(value))
                for (name, labels), value in self._counters.items()
            }
            durations = {
                _snapshot_key(name, labels): total
                for (name, labels), (_counts, total, _count) in self._histograms.items()
            }
            gauges = {
                _snapshot_key(name, labels): value
                for (name, labels), value in self._gauges.items()
            }
            histograms = {
                _snapshot_key(name, labels): tuple(counts)
                for (name, labels), (counts, _total, _count) in self._histograms.items()
            }
            return ObservabilitySnapshot(
                counters=counters,
                duration_seconds=durations,
                gauges=gauges,
                recent_events=tuple(self._recent),
                histogram_counts=histograms,
            )

    def prometheus_text(self, *, namespace: str | None = None) -> str:
        """输出包含 HELP、TYPE 与直方图的 Prometheus exposition 文本。"""

        prefix = self.namespace if namespace is None else _metric_name(namespace)
        lines: list[str] = []
        with self._lock:
            counter_names = sorted({name for name, _labels in self._counters})
            for name in counter_names:
                full_name = f"{prefix}_{_metric_name(name)}"
                lines.extend((f"# HELP {full_name} Habitus bounded observation counter.", f"# TYPE {full_name} counter"))
                for (metric_name, labels), value in sorted(self._counters.items()):
                    if metric_name == name:
                        lines.append(f"{full_name}{_render_labels(labels)} {_format_number(value)}")

            histogram_names = sorted({name for name, _labels in self._histograms})
            for name in histogram_names:
                full_name = f"{prefix}_{_metric_name(name)}"
                lines.extend((f"# HELP {full_name} Habitus bounded operation duration.", f"# TYPE {full_name} histogram"))
                for (metric_name, labels), (counts, total, count) in sorted(self._histograms.items()):
                    if metric_name != name:
                        continue
                    for bucket, bucket_count in zip(self.duration_buckets, counts, strict=True):
                        bucket_labels = labels + (("le", _format_number(bucket)),)
                        lines.append(f"{full_name}_bucket{_render_labels(bucket_labels)} {bucket_count}")
                    lines.append(f'{full_name}_bucket{_render_labels(labels + (("le", "+Inf"),))} {count}')
                    lines.append(f"{full_name}_sum{_render_labels(labels)} {_format_number(total)}")
                    lines.append(f"{full_name}_count{_render_labels(labels)} {count}")

            gauge_names = sorted({name for name, _labels in self._gauges})
            for name in gauge_names:
                full_name = f"{prefix}_{_metric_name(name)}"
                lines.extend((f"# HELP {full_name} Habitus current bounded state.", f"# TYPE {full_name} gauge"))
                for (metric_name, labels), value in sorted(self._gauges.items()):
                    if metric_name == name:
                        lines.append(f"{full_name}{_render_labels(labels)} {_format_number(value)}")
        return "\n".join(lines) + ("\n" if lines else "")


def _selected_labels(
    attributes: Mapping[str, str | int | float | bool],
    *names: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, _label_value(attributes[name]))
        for name in names
        if name in _LABEL_ATTRIBUTES and name in attributes
    )


def _mapping_labels(labels: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if labels is None:
        return ()
    if not isinstance(labels, Mapping) or len(labels) > 8:
        raise ValueError("metric labels must be a bounded mapping")
    result: list[tuple[str, str]] = []
    for name, value in labels.items():
        if name not in _LABEL_ATTRIBUTES:
            raise ValueError("metric label is not in the low-cardinality allowlist")
        result.append((_metric_name(name), _label_value(value)))
    return tuple(sorted(result))


def _non_negative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _label_value(value: object) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        return "unknown"
    return normalized[:128]


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{_metric_name(name)}="{_escape_label(value)}"' for name, value in labels)
    return "{" + rendered + "}"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _snapshot_key(name: str, labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return name
    return f"{name}[{','.join(f'{key}={value}' for key, value in labels)}]"


def _format_number(value: float) -> str:
    return f"{value:.9g}"


def _metric_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_:]", "_", value)
    if not normalized or normalized[0].isdigit():
        normalized = f"metric_{normalized}"
    return normalized.lower()


__all__ = [
    "CompositeObserver",
    "MetricRegistry",
    "MetricUpdate",
    "NullObserver",
    "ObservabilitySnapshot",
    "ObservationContext",
    "ObservationEvent",
    "ObservationStatus",
    "Observer",
    "SpanController",
    "bind_observation_context",
    "current_observation_context",
    "observe_operation",
    "project_metric_updates",
]
