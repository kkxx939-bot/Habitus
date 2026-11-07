"""不绑定 Prometheus 或 OpenTelemetry SDK 的进程内观察契约。"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol

_IDENTITY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


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
            if value is not None and _IDENTITY.fullmatch(value) is None:
                raise ValueError(f"{name} must be a bounded normalized identity or None")
        for name in ("memory_sequence", "attempt"):
            value = getattr(self, name)
            if value is not None and value < 0:
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
            if not value or len(value) > 128:
                raise ValueError(f"{name} must be non-empty bounded text")
        object.__setattr__(self, "status", ObservationStatus(self.status))
        duration = float(self.duration_seconds)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_seconds must be finite and non-negative")
        object.__setattr__(self, "duration_seconds", duration)
        if len(self.attributes) > 32:
            raise ValueError("observation attributes must be a bounded mapping")
        attributes: dict[str, str | int | float | bool] = {}
        for key, value in self.attributes.items():
            if not key or len(key) > 64:
                raise ValueError("observation attribute names must be bounded text")
            if isinstance(value, str):
                if len(value) > 256:
                    raise ValueError("observation text attributes must be bounded")
            elif isinstance(value, bool):
                pass
            elif isinstance(value, int | float):
                if not math.isfinite(float(value)):
                    raise ValueError("observation numeric attributes must be finite")
            else:  # 观测属性只放有界标量：进审计、指标与日志的数据形状契约，不是类型防御（W16 复核保留）
                raise TypeError("observation attributes must contain scalar values")
            attributes[key] = value
        object.__setattr__(self, "attributes", attributes)
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(UTC))


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
        """丢弃事件。"""


class CompositeObserver:
    """把同一事件发送给多个独立后端；单个观测后端失败不影响业务。"""

    def __init__(self, *observers: Observer) -> None:
        if not observers or any(not callable(getattr(observer, "record", None)) for observer in observers):
            raise TypeError("observers must implement record")
        self.observers = observers

    def record(self, event: ObservationEvent) -> None:
        for observer in self.observers:
            try:
                observer.record(event)
            except Exception:
                # 观测后端彼此独立：一个后端坏了不能拖垮业务，也不能挡住其余后端收到同一事件。
                continue


@contextmanager
def observe_operation(
    observer: Observer,
    category: str,
    operation: str,
    *,
    attributes: Mapping[str, str | int | float | bool] | None = None,
) -> Iterator[dict[str, str | int | float | bool]]:
    """记录一个同步或异步代码块；观察失败不改变代码块结果。

    yield 出的字典供代码块补写只有做完才知道的属性（产出条数等），随事件一起记录。
    """

    started = time.monotonic()
    collected: dict[str, str | int | float | bool] = dict(attributes or {})
    try:
        yield collected
    except BaseException as exc:
        failure_attributes = dict(collected)
        failure_attributes["error_type"] = type(exc).__name__
        _record_outcome(observer, category, operation, ObservationStatus.FAILURE, started, failure_attributes)
        raise
    else:
        _record_outcome(observer, category, operation, ObservationStatus.SUCCESS, started, collected)


def _record_outcome(
    observer: Observer,
    category: str,
    operation: str,
    status: ObservationStatus,
    started: float,
    attributes: Mapping[str, str | int | float | bool],
) -> None:
    """事件构造也在保护之内：代码块补写的属性越界只丢这条观察，不改变代码块结果。"""

    try:
        event = ObservationEvent(
            category=category,
            operation=operation,
            status=status,
            duration_seconds=max(0.0, time.monotonic() - started),
            attributes=dict(attributes),
        )
    except (TypeError, ValueError):
        return
    _safe_record(observer, event)


def record_observation(
    observer: Observer,
    category: str,
    operation: str,
    status: ObservationStatus,
    *,
    started: float | None = None,
    attributes: Mapping[str, str | int | float | bool] | None = None,
) -> None:
    """记一条点事件：``started`` 是 ``time.monotonic()`` 读数，缺省记 0 时长。

    事件构造与记录都在保护之内：观测只是旁路，属性越界或后端失败只丢这条观察，不影响业务调用。
    """

    try:
        observer.record(
            ObservationEvent(
                category=category,
                operation=operation,
                status=status,
                duration_seconds=0.0 if started is None else max(0.0, time.monotonic() - started),
                attributes=dict(attributes or {}),
            )
        )
    except Exception:
        # 观测只是旁路：事件记不下来不能影响业务调用。
        pass


def _safe_record(observer: Observer, event: ObservationEvent) -> None:
    try:
        observer.record(event)
    except Exception:
        # 观测只是旁路：后端写失败不能影响业务调用。
        pass


__all__ = [
    "CompositeObserver",
    "NullObserver",
    "ObservationContext",
    "ObservationEvent",
    "ObservationStatus",
    "Observer",
    "SpanController",
    "bind_observation_context",
    "current_observation_context",
    "observe_operation",
    "record_observation",
]
