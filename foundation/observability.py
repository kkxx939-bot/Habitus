"""不绑定 Prometheus 或 OpenTelemetry SDK 的进程内观察契约。"""

from __future__ import annotations

import re
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol


class ObservationStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class ObservationEvent:
    """一条有界、无原始请求内容的操作观察事件。"""

    category: str
    operation: str
    status: ObservationStatus
    duration_seconds: float
    attributes: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        for name in ("category", "operation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 128:
                raise ValueError(f"{name} must be non-empty bounded text")
        object.__setattr__(self, "status", ObservationStatus(self.status))
        if isinstance(self.duration_seconds, bool) or not isinstance(self.duration_seconds, int | float):
            raise TypeError("duration_seconds must be numeric")
        if float(self.duration_seconds) < 0:
            raise ValueError("duration_seconds must be non-negative")
        if not isinstance(self.attributes, Mapping) or len(self.attributes) > 32:
            raise ValueError("observation attributes must be a bounded mapping")
        object.__setattr__(self, "attributes", dict(self.attributes))
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(timezone.utc))


class Observer(Protocol):
    def record(self, event: ObservationEvent) -> None: ...


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


@dataclass(frozen=True)
class ObservabilitySnapshot:
    counters: Mapping[str, int]
    duration_seconds: Mapping[str, float]
    gauges: Mapping[str, float]
    recent_events: tuple[ObservationEvent, ...]


class MetricRegistry:
    """线程安全的轻量指标注册表，可被 Prometheus 或 OTel Adapter 拉取。"""

    def __init__(self, *, max_recent_events: int = 256) -> None:
        if not 1 <= max_recent_events <= 10_000:
            raise ValueError("max_recent_events must be between 1 and 10000")
        self._lock = threading.RLock()
        self._counters: dict[str, int] = {}
        self._durations: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._recent: deque[ObservationEvent] = deque(maxlen=max_recent_events)

    def record(self, event: ObservationEvent) -> None:
        if not isinstance(event, ObservationEvent):
            raise TypeError("event must be ObservationEvent")
        key = f"{event.category}.{event.operation}.{event.status.value}"
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
            self._durations[key] = self._durations.get(key, 0.0) + float(event.duration_seconds)
            self._recent.append(event)

    def set_gauge(self, name: str, value: int | float) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("gauge name must be non-empty text")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("gauge value must be numeric")
        with self._lock:
            self._gauges[name] = float(value)

    def snapshot(self) -> ObservabilitySnapshot:
        with self._lock:
            return ObservabilitySnapshot(
                counters=dict(self._counters),
                duration_seconds=dict(self._durations),
                gauges=dict(self._gauges),
                recent_events=tuple(self._recent),
            )

    def prometheus_text(self, *, namespace: str = "m2bos") -> str:
        """输出无依赖的 Prometheus exposition 文本。"""

        prefix = _metric_name(namespace)
        snapshot = self.snapshot()
        lines: list[str] = []
        for key, count in sorted(snapshot.counters.items()):
            lines.append(f"{prefix}_operations_total{{operation=\"{key}\"}} {count}")
        for key, duration in sorted(snapshot.duration_seconds.items()):
            lines.append(f"{prefix}_operation_duration_seconds_sum{{operation=\"{key}\"}} {duration:.9f}")
        for key, gauge in sorted(snapshot.gauges.items()):
            lines.append(f"{prefix}_{_metric_name(key)} {gauge:.9f}")
        return "\n".join(lines) + ("\n" if lines else "")


def _metric_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_:]", "_", value)
    if not normalized or normalized[0].isdigit():
        normalized = f"metric_{normalized}"
    return normalized.lower()


__all__ = [
    "CompositeObserver",
    "MetricRegistry",
    "NullObserver",
    "ObservabilitySnapshot",
    "ObservationEvent",
    "ObservationStatus",
    "Observer",
]
