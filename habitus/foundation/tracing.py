"""不绑定 OpenTelemetry SDK 的进程内 span：按调用链（``ContextVar``）挂父子，交给绑定的 ``TraceSink`` 落地。

只在有人绑定了 sink（``open_trace``）的调用链里记；没绑定时 ``span`` 是空操作，照样给出一个可写的属性字典，调用方不用分支。
span 的属性可以很细（每一步的输入、输出条数与关键读数）：它们只进 trace，不进指标标签。``asyncio.to_thread`` 会复制上下文，
线程里开的 span 照样挂在发起它的 span 下面。
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

Scalar = str | int | float | bool
_TEXT_LIMIT = 1_000


@dataclass
class SpanRecord:
    """一个 span：名字、父子、起止、属性与结局。属性在 span 结束前都可以补写。"""

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    started_at: datetime
    attributes: dict[str, Scalar] = field(default_factory=dict)
    ended_at: datetime | None = None
    status: str = "ok"
    error_type: str | None = None
    #: 子 span 里的模型调用按（触点, 后端, 模型）累计，结束时由 sink 摊进属性。
    model_calls: dict[tuple[str, str, str], int] = field(default_factory=dict)
    _monotonic: float = field(default_factory=time.monotonic, repr=False)

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or datetime.now(UTC)
        return max(0.0, (end - self.started_at).total_seconds())

    def count_model_call(self, touch: str, provider: str, model: str) -> None:
        key = (touch, provider, model)
        self.model_calls[key] = self.model_calls.get(key, 0) + 1


class TraceSink(Protocol):
    """span 的落地处：开始时（可以进 OTel 的 span）与结束时（写本地文件）各调一次。"""

    def started(self, record: SpanRecord) -> object: ...

    def finished(self, record: SpanRecord, handle: object) -> None: ...


@dataclass(frozen=True)
class _Bound:
    sink: TraceSink
    trace_id: str
    stack: tuple[SpanRecord, ...] = ()


_CURRENT: ContextVar[_Bound | None] = ContextVar("habitus_trace", default=None)


@contextmanager
def open_trace(sink: TraceSink) -> Iterator[None]:
    """在当前调用链里绑定一个 sink；之后的 ``span`` 都交给它，退出时恢复。"""

    token = _CURRENT.set(_Bound(sink=sink, trace_id=uuid.uuid4().hex))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def tracing_active() -> bool:
    return _CURRENT.get() is not None


def current_span() -> SpanRecord | None:
    bound = _CURRENT.get()
    return None if bound is None or not bound.stack else bound.stack[-1]


def ancestors() -> tuple[SpanRecord, ...]:
    """当前 span 与它的各级父 span（从根到叶）。"""

    bound = _CURRENT.get()
    return () if bound is None else bound.stack


@contextmanager
def span(name: str, attributes: Mapping[str, Scalar] | None = None) -> Iterator[MutableMapping[str, Scalar]]:
    """开一个子 span；yield 出的字典供代码块补写只有做完才知道的属性。没绑定 sink 时只给一个字典、什么都不记。"""

    bound = _CURRENT.get()
    collected: dict[str, Scalar] = {}
    set_attributes(collected, attributes or {})
    if bound is None:
        yield collected
        return
    parent = bound.stack[-1] if bound.stack else None
    record = SpanRecord(
        name=name,
        trace_id=bound.trace_id,
        span_id=uuid.uuid4().hex[:16],
        parent_id=None if parent is None else parent.span_id,
        started_at=datetime.now(UTC),
        attributes=collected,
    )
    handle = _safe_start(bound.sink, record)
    token = _CURRENT.set(_Bound(sink=bound.sink, trace_id=bound.trace_id, stack=(*bound.stack, record)))
    try:
        yield collected
    except BaseException as exc:
        record.status = "error"
        record.error_type = type(exc).__name__
        raise
    finally:
        _CURRENT.reset(token)
        record.ended_at = datetime.now(UTC)
        _safe_finish(bound.sink, record, handle)


def annotate(values: Mapping[str, object]) -> None:
    """给当前 span 补属性（没有当前 span 时什么都不做）：深处的步骤不用把读数一层层传上来。"""

    record = current_span()
    if record is not None:
        set_attributes(record.attributes, values)


def record_child(
    name: str, *, started_at: datetime, ended_at: datetime, attributes: Mapping[str, Scalar], error_type: str | None
) -> SpanRecord | None:
    """事后补一个已经结束的子 span（例如一次模型调用：它结束时才有读数）。没有当前 span 时不记。"""

    bound = _CURRENT.get()
    parent = current_span()
    if bound is None or parent is None:
        return None
    collected: dict[str, Scalar] = {}
    set_attributes(collected, attributes)
    record = SpanRecord(
        name=name,
        trace_id=bound.trace_id,
        span_id=uuid.uuid4().hex[:16],
        parent_id=parent.span_id,
        started_at=started_at,
        attributes=collected,
        ended_at=ended_at,
        status="ok" if error_type is None else "error",
        error_type=error_type,
    )
    handle = _safe_start(bound.sink, record)
    _safe_finish(bound.sink, record, handle)
    return record


def set_attributes(target: MutableMapping[str, Scalar], values: Mapping[str, object]) -> None:
    """只收有界标量：文本截断、非有限数丢掉；观测是旁路，不因一个读数越界而抛。"""

    for key, value in values.items():
        if not key or len(key) > 128:
            continue
        if isinstance(value, str):
            target[key] = value[:_TEXT_LIMIT]
        elif isinstance(value, bool | int):
            target[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            target[key] = value


def _safe_start(sink: TraceSink, record: SpanRecord) -> object:
    try:
        return sink.started(record)
    except Exception:
        # 观测只是旁路：sink 坏了不能拖垮业务
        return None


def _safe_finish(sink: TraceSink, record: SpanRecord, handle: object) -> None:
    try:
        sink.finished(record, handle)
    except Exception:
        # 观测只是旁路：sink 坏了不能拖垮业务
        return


__all__ = [
    "Scalar",
    "SpanRecord",
    "TraceSink",
    "ancestors",
    "annotate",
    "current_span",
    "open_trace",
    "record_child",
    "set_attributes",
    "span",
    "tracing_active",
]
