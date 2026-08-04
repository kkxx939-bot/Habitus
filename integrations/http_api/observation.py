"""HTTP 请求的指标、日志事件和可选根 Span。"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from foundation.observability import (
    NullObserver,
    ObservationEvent,
    ObservationStatus,
    Observer,
    SpanController,
)

_TRACEPARENT = re.compile(rb"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_ZERO_TRACE_ID = b"0" * 32
_ZERO_PARENT_ID = b"0" * 16


class HTTPObservationMiddleware:
    """观察 HTTP 状态与耗时，但不负责请求身份或错误渲染。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        observer: Observer | None = None,
        span_controller: SpanController | None = None,
    ) -> None:
        if observer is not None and not callable(getattr(observer, "record", None)):
            raise TypeError("observer must implement record")
        if span_controller is not None and not callable(
            getattr(span_controller, "start_span", None)
        ):
            raise TypeError("span_controller must implement start_span")
        self.app = app
        self.observer = observer or NullObserver()
        self.span_controller = span_controller

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started_at = time.monotonic()
        status_code = 500
        error_type: str | None = None
        method = str(scope.get("method", ""))

        async def observe_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        with _isolated_span(
            self.span_controller,
            category="http",
            operation="request",
            attributes={"http_method": method},
            traceparent=_validated_traceparent(scope),
        ):
            try:
                await self.app(scope, receive, observe_send)
            except BaseException as exc:
                error_type = type(exc).__name__
                raise
            finally:
                self._record(
                    scope,
                    method=method,
                    status_code=status_code,
                    duration_seconds=max(0.0, time.monotonic() - started_at),
                    error_type=error_type,
                )

    def _record(
        self,
        scope: Scope,
        *,
        method: str,
        status_code: int,
        duration_seconds: float,
        error_type: str | None,
    ) -> None:
        attributes: dict[str, str | int] = {
            "http_method": method,
            "http_route": _route_template(scope),
            "http_status_code": status_code,
            "http_status_class": f"{status_code // 100}xx",
        }
        if error_type is not None:
            attributes["error_type"] = error_type
        try:
            self.observer.record(
                ObservationEvent(
                    category="http",
                    operation="request",
                    status=_observation_status(status_code, error_type=error_type),
                    duration_seconds=duration_seconds,
                    attributes=attributes,
                )
            )
        except Exception:
            pass


@contextmanager
def _isolated_span(
    span_controller: SpanController | None,
    *,
    category: str,
    operation: str,
    attributes: Mapping[str, str | int | float | bool],
    traceparent: str | None,
) -> Iterator[None]:
    """隔离 Span 后端故障，同时保留业务异常的原始传播语义。"""

    if span_controller is None:
        yield
        return
    try:
        manager = span_controller.start_span(
            category,
            operation,
            attributes=attributes,
            traceparent=traceparent,
        )
        manager.__enter__()
    except Exception:
        yield
        return
    try:
        yield
    except BaseException as exc:
        try:
            manager.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            pass
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception:
            pass


def _validated_traceparent(scope: Scope) -> str | None:
    values = [value for name, value in scope.get("headers", ()) if name.lower() == b"traceparent"]
    if len(values) != 1:
        return None
    raw = values[0]
    if len(raw) != 55:
        return None
    match = _TRACEPARENT.fullmatch(raw)
    if match is None or match.group(1) == _ZERO_TRACE_ID or match.group(2) == _ZERO_PARENT_ID:
        return None
    return raw.decode("ascii")


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else "/__unmatched__"


def _observation_status(status_code: int, *, error_type: str | None) -> ObservationStatus:
    if error_type is not None or status_code >= 500:
        return ObservationStatus.FAILURE
    if status_code >= 400:
        return ObservationStatus.DEGRADED
    return ObservationStatus.SUCCESS


__all__ = ["HTTPObservationMiddleware"]
