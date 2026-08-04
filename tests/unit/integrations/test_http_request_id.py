"""本地 HTTP 请求身份与观测边界测试。"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.exceptions import ExceptionMiddleware
from starlette.responses import StreamingResponse

from foundation.observability import ObservationEvent, ObservationStatus, SpanController
from integrations.http_api.errors import install_exception_handlers, unhandled_error
from integrations.http_api.observation import HTTPObservationMiddleware
from integrations.http_api.request_id import REQUEST_ID_HEADER, RequestIDMiddleware, current_request_id
from memory.conversation import ConversationWriteConflictError
from ModelClient import ModelRateLimitError

_SERVER_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


class RecordingSpanController:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @contextmanager
    def start_span(
        self,
        category: str,
        operation: str,
        *,
        attributes: Mapping[str, str | int | float | bool] | None = None,
        traceparent: str | None = None,
    ) -> Iterator[None]:
        self.calls.append(
            {
                "category": category,
                "operation": operation,
                "attributes": dict(attributes or {}),
                "traceparent": traceparent,
            }
        )
        yield


def _app(
    *,
    observer: RecordingObserver | None = None,
    span_controller: SpanController | None = None,
    stream_events: list[str] | None = None,
) -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(ExceptionMiddleware, handlers={Exception: unhandled_error})
    app.add_middleware(
        HTTPObservationMiddleware,
        observer=observer,
        span_controller=span_controller,
    )
    app.add_middleware(RequestIDMiddleware)

    @app.get("/identity")
    async def identity():  # noqa: ANN202
        await asyncio.sleep(0)
        return {"request_id": current_request_id()}

    @app.get("/items/{item_id}")
    async def item(item_id: int):  # noqa: ANN202
        return {"item_id": item_id}

    @app.get("/conflict")
    async def conflict():  # noqa: ANN202
        raise ConversationWriteConflictError("append would create a gap")

    @app.get("/boom")
    async def boom():  # noqa: ANN202
        raise RuntimeError("sensitive internal failure")

    @app.get("/rate-limit")
    async def rate_limit():  # noqa: ANN202
        raise ModelRateLimitError("provider detail", retry_after_seconds=1.2)

    @app.get("/stream")
    async def stream():  # noqa: ANN202
        async def chunks():  # noqa: ANN202
            if stream_events is not None:
                stream_events.append("stream_start")
            yield b"first-"
            await asyncio.sleep(0)
            yield b"second"
            if stream_events is not None:
                stream_events.append("stream_done")

        return StreamingResponse(chunks(), media_type="text/plain")

    return app


def _assert_response_identity(response: httpx.Response) -> str:
    request_id = response.headers[REQUEST_ID_HEADER]
    assert _SERVER_REQUEST_ID.fullmatch(request_id)
    assert response.json()["request_id"] == request_id
    return request_id


def test_service_generates_a_fresh_identity_and_ignores_the_caller_header() -> None:
    with TestClient(_app(), base_url="http://127.0.0.1:8787") as client:
        first = client.get("/identity", headers={REQUEST_ID_HEADER: "caller-owned"})
        second = client.get("/identity")

    first_id = _assert_response_identity(first)
    second_id = _assert_response_identity(second)
    assert first_id != "caller-owned"
    assert first_id != second_id
    assert current_request_id() is None


def test_framework_and_domain_failures_return_the_generated_identity() -> None:
    with TestClient(_app(), base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        validation = client.get("/items/not-an-integer")
        method = client.post("/identity")
        conflict = client.get("/conflict")

    for response in (validation, method, conflict):
        _assert_response_identity(response)
    assert validation.status_code == 400
    assert validation.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert method.status_code == 405
    assert method.json()["error"]["code"] == "METHOD_NOT_ALLOWED"
    assert conflict.status_code == 409
    assert conflict.json()["error"] == {
        "code": "CONFLICT",
        "message": "append would create a gap",
        "retryable": False,
        "details": {"conflict_type": "conversation_write"},
    }


def test_unhandled_and_retryable_failures_keep_the_transport_contract() -> None:
    with TestClient(_app(), base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        unexpected = client.get("/boom")
        retryable = client.get("/rate-limit")

    _assert_response_identity(unexpected)
    assert unexpected.status_code == 500
    assert unexpected.json()["error"] == {
        "code": "INTERNAL",
        "message": "Internal server error",
        "retryable": False,
    }
    _assert_response_identity(retryable)
    assert retryable.status_code == 429
    assert retryable.headers["Retry-After"] == "2"
    assert retryable.json()["error"] == {
        "code": "RESOURCE_EXHAUSTED",
        "message": "Model provider rate limit exceeded",
        "retryable": True,
        "retry_after_seconds": 2,
    }


def test_concurrent_requests_keep_independent_contexts() -> None:
    observer = RecordingObserver()
    app = _app(observer=observer)

    async def scenario() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
            first, second = await asyncio.gather(
                client.get("/identity"),
                client.get("/identity"),
            )
            return first, second

    first, second = asyncio.run(scenario())
    response_ids = {_assert_response_identity(first), _assert_response_identity(second)}
    event_ids = {event.context.request_id for event in observer.events}
    assert len(response_ids) == 2
    assert event_ids == response_ids


def test_http_observation_uses_generated_identity_and_preserves_trace_parent() -> None:
    observer = RecordingObserver()
    spans = RecordingSpanController()
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

    with TestClient(
        _app(observer=observer, span_controller=spans),
        base_url="http://127.0.0.1:8787",
    ) as client:
        response = client.get("/items/7", headers={"traceparent": traceparent})

    request_id = response.headers[REQUEST_ID_HEADER]
    event = next(item for item in observer.events if item.operation == "request")
    assert event.context.request_id == request_id
    assert event.attributes == {
        "http_method": "GET",
        "http_route": "/items/{item_id}",
        "http_status_code": 200,
        "http_status_class": "2xx",
    }
    assert spans.calls == [
        {
            "category": "http",
            "operation": "request",
            "attributes": {"http_method": "GET"},
            "traceparent": traceparent,
        }
    ]


@pytest.mark.parametrize(
    "headers",
    [
        [("traceparent", "invalid")],
        [("traceparent", "00-" + "0" * 32 + "-00f067aa0ba902b7-01")],
        [("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-" + "0" * 16 + "-01")],
        [("traceparent", "00-" + "1" * 200)],
        [
            ("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
            ("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        ],
    ],
)
def test_invalid_duplicate_or_oversized_trace_parent_is_not_forwarded(
    headers: list[tuple[str, str]],
) -> None:
    spans = RecordingSpanController()

    with TestClient(_app(span_controller=spans), base_url="http://127.0.0.1:8787") as client:
        response = client.get("/identity", headers=headers)

    assert response.status_code == 200
    assert spans.calls[0]["traceparent"] is None


@pytest.mark.parametrize("failure_phase", ["create", "enter", "exit"])
def test_span_backend_failure_does_not_change_the_business_response(failure_phase: str) -> None:
    class FaultingManager:
        def __enter__(self) -> None:
            if failure_phase == "enter":
                raise RuntimeError("span enter failed")

        def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
            if failure_phase == "exit":
                raise RuntimeError("span exit failed")

    class FaultingSpanController:
        def start_span(self, *args: object, **kwargs: object) -> FaultingManager:
            if failure_phase == "create":
                raise RuntimeError("span creation failed")
            return FaultingManager()

    with TestClient(
        _app(span_controller=FaultingSpanController()),
        base_url="http://127.0.0.1:8787",
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/identity")

    assert response.status_code == 200
    _assert_response_identity(response)


def test_stream_observation_finishes_after_the_last_response_body() -> None:
    order: list[str] = []

    class OrderedObserver(RecordingObserver):
        def record(self, event: ObservationEvent) -> None:
            super().record(event)
            order.append("record")

    class OrderedSpanController:
        @contextmanager
        def start_span(self, *args: object, **kwargs: object) -> Iterator[None]:
            order.append("span_enter")
            try:
                yield
            finally:
                order.append("span_exit")

    observer = OrderedObserver()
    with TestClient(
        _app(observer=observer, span_controller=OrderedSpanController(), stream_events=order),
        base_url="http://127.0.0.1:8787",
    ) as client:
        response = client.get("/stream")

    assert response.status_code == 200
    assert response.text == "first-second"
    assert order == ["span_enter", "stream_start", "stream_done", "record", "span_exit"]


def test_http_observation_classifies_final_success_degraded_and_failure_statuses() -> None:
    observer = RecordingObserver()
    with TestClient(
        _app(observer=observer),
        base_url="http://127.0.0.1:8787",
        raise_server_exceptions=False,
    ) as client:
        success = client.get("/identity")
        degraded = client.get("/rate-limit")
        failure = client.get("/boom")

    assert [success.status_code, degraded.status_code, failure.status_code] == [200, 429, 500]
    assert [event.status for event in observer.events] == [
        ObservationStatus.SUCCESS,
        ObservationStatus.DEGRADED,
        ObservationStatus.FAILURE,
    ]
