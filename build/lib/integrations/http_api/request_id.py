"""为每个 HTTP 请求建立严格、并发隔离的关联身份。"""

from __future__ import annotations

import re
import time
from uuid import uuid4

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from foundation.observability import (
    NullObserver,
    ObservationEvent,
    ObservationStatus,
    Observer,
    SpanController,
    bind_observation_context,
    current_observation_context,
)
from integrations.http_api.errors import error_response
from integrations.http_api.schemas import HTTPErrorCode

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_HEADER_BYTES = b"x-request-id"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")


def current_request_id() -> str | None:
    """返回当前异步请求的关联身份；非 HTTP 上下文返回 None。"""

    return current_observation_context().request_id


def _resolve_request_id(scope: Scope) -> tuple[str, bool]:
    values = [value for name, value in scope.get("headers", ()) if name.lower() == _REQUEST_ID_HEADER_BYTES]
    invalid = bool(values)
    if len(values) == 1:
        try:
            candidate = values[0].decode("ascii")
        except UnicodeDecodeError:
            pass
        else:
            if _REQUEST_ID.fullmatch(candidate):
                return candidate, False
    return uuid4().hex, invalid


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else "/__unmatched__"


def _traceparent(scope: Scope) -> str | None:
    values = [value for name, value in scope.get("headers", ()) if name.lower() == b"traceparent"]
    if len(values) != 1:
        return None
    try:
        candidate = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    return candidate if _TRACEPARENT.fullmatch(candidate) is not None else None


class RequestIDMiddleware:
    """校验、绑定、回传并记录一个请求 ID；非法原值不会进入日志。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        observer: Observer | None = None,
        span_controller: SpanController | None = None,
    ) -> None:
        self.app = app
        self.observer = observer or NullObserver()
        self.span_controller = span_controller

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started_at = time.perf_counter()
        request_id, invalid = _resolve_request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id
        method = str(scope.get("method", ""))
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != _REQUEST_ID_HEADER_BYTES
                ]
                headers.append((_REQUEST_ID_HEADER_BYTES, request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        with bind_observation_context(request_id=request_id):
            span = (
                self.span_controller.start_span(
                    "http",
                    "request",
                    attributes={"http_method": method},
                    traceparent=_traceparent(scope),
                )
                if self.span_controller is not None
                else _NullSpan()
            )
            try:
                with span:
                    if invalid:
                        request = Request(scope, receive=receive)
                        response = error_response(
                            request,
                            code=HTTPErrorCode.INVALID_ARGUMENT,
                            message=(
                                "X-Request-ID must be supplied once and contain 1-128 "
                                "characters from [A-Za-z0-9._:-]"
                            ),
                            retryable=False,
                        )
                        await response(scope, receive, send_with_request_id)
                    else:
                        await self.app(scope, receive, send_with_request_id)
            finally:
                duration_seconds = max(0.0, time.perf_counter() - started_at)
                if status_code >= 500:
                    status = ObservationStatus.FAILURE
                elif status_code >= 400:
                    status = ObservationStatus.DEGRADED
                else:
                    status = ObservationStatus.SUCCESS
                try:
                    self.observer.record(
                        ObservationEvent(
                            category="http",
                            operation="request",
                            status=status,
                            duration_seconds=duration_seconds,
                            attributes={
                                "http_method": method,
                                "http_route": _route_template(scope),
                                "http_status_code": status_code,
                                "http_status_class": f"{status_code // 100}xx",
                            },
                        )
                    )
                except Exception:
                    pass


class _NullSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        return None


__all__ = ["REQUEST_ID_HEADER", "RequestIDMiddleware", "current_request_id"]
