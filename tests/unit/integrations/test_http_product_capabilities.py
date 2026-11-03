"""从远程调用方视角验证 HTTP 产品边界，而不是重复 Runtime 领域测试。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from habitus.config import HTTPAPIConfig
from habitus.foundation.observability import ObservationEvent
from habitus.integrations.http_api import app as app_module
from habitus.integrations.http_api.handlers import RememberOutcome
from habitus.integrations.http_api.schemas import (
    AuditListResult,
    BlockedJobResult,
    CapabilitiesResult,
    ConversationCursorResult,
    FlushResult,
    HealthResult,
    JobListResult,
    JobStatusResult,
    RecallResult,
    RememberResult,
    RetryJobResult,
)
from habitus.memory.workflow import MemoryJob, MemoryJobStatus
from habitus.runtime import Runtime

UTC = UTC
# 分块请求体要读完才往下走：中间件丢了请求体或卡在 receive 上时，测试在这个时限内失败而不是挂住
_BODY_WAIT_SECONDS = 10


def _assert_response_request_id(response: httpx.Response) -> str:
    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 32
    int(request_id, 16)
    assert response.json()["request_id"] == request_id
    return request_id


class CollectingObserver:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def _job(sequence: int = 7, *, status: str = "failed", blocking: bool = True) -> JobStatusResult:
    return JobStatusResult.model_validate(
        {
            "memory_sequence": sequence,
            "conversation_id": "conversation-http",
            "started_on": "2026-07-30",
            "state": status,
            "job_status": status,
            "terminal": status in {"failed", "committed", "abandoned"},
            "attempts": 2,
            "next_attempt_at": None,
            "last_failure": {"message": "model unavailable"} if status == "failed" else None,
            "blocking": blocking,
            "manual_action_required": status == "failed" and blocking,
            "version": "a" * 64,
            "created_at": "2026-07-30T01:00:00Z",
            "updated_at": "2026-07-30T01:01:00Z",
        }
    )


def _accepted_job(conversation_id: str, started_on: date) -> MemoryJob:
    """路由只从受理的 Job 上取观测关联用的事务身份，内部字段不进响应。"""

    moment = datetime(2026, 7, 30, 1, tzinfo=UTC)
    return MemoryJob(
        memory_sequence=7,
        conversation_id=conversation_id,
        started_on=started_on,
        segment_id="internal-segment",
        source_segment_digest="b" * 64,
        transaction_id="7" * 32,
        status=MemoryJobStatus.QUEUED,
        attempts=0,
        claim_id=None,
        claim_generation=0,
        worker_id=None,
        lease_expires_at=None,
        next_attempt_at=None,
        last_error=None,
        created_at=moment,
        updated_at=moment,
    )


class CapabilityHandlers:
    """只替代领域执行，保留真实 FastAPI、Schema 和中间件链。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def capabilities(self) -> CapabilitiesResult:
        self.calls.append(("capabilities", None))
        return CapabilitiesResult(
            api_version="1.0",
            service_version="0.1.0",
            protocols=["openai", "anthropic"],
            features=["recall", "remember", "remember_idempotency_v1"],
        )

    async def remember(self, **values: object) -> RememberOutcome:
        self.calls.append(("remember", values))
        conversation_id, started_on = values["conversation_id"], values["started_on"]
        assert isinstance(conversation_id, str) and isinstance(started_on, date)
        job = _accepted_job(conversation_id, started_on)
        return RememberOutcome(
            result=RememberResult.model_validate(
                {
                    "ignored_items": 0,
                    "after_turn": bool(values["after_turn"]),
                    "next_sequence": 2,
                    "jobs": [
                        {
                            "memory_sequence": job.memory_sequence,
                            "conversation_id": job.conversation_id,
                            "started_on": job.started_on,
                            "status": job.status.value,
                        }
                    ],
                    "consistency": [],
                }
            ),
            accepted_jobs=(job,),
        )

    async def conversation_cursor(self, **values: object) -> ConversationCursorResult:
        self.calls.append(("conversation_cursor", values))
        return ConversationCursorResult(next_sequence=2)

    async def flush(self, **values: object) -> FlushResult:
        self.calls.append(("flush", values))
        return FlushResult(jobs=[], consistency=[])

    async def recall(self, query: str, **values: object) -> RecallResult:
        self.calls.append(("recall", {"query": query, **values}))
        return RecallResult.model_validate(
            {
                "query": query,
                "queries": [query],
                "context": "用户偏好简洁回答。",
                "memories": [
                    {
                        "uri": "memory://preferences/response_style.md",
                        "score": 0.91,
                        "matched_queries": [query],
                    }
                ],
                "summaries": [],
                "degradations": [],
                "budget_exhausted": False,
            }
        )

    async def list_jobs(self, **values: object) -> JobListResult:
        self.calls.append(("list_jobs", values))
        return JobListResult(jobs=[_job()], next_before_sequence=None)

    async def blocked_job(self) -> BlockedJobResult:
        self.calls.append(("blocked_job", None))
        return BlockedJobResult(job=_job())

    async def job_status(self, sequence: int, **values: object) -> JobStatusResult:
        self.calls.append(("job_status", {"sequence": sequence, **values}))
        return _job(sequence)

    async def retry_failed_job(self, sequence: int, **values: object) -> RetryJobResult:
        self.calls.append(("retry_failed_job", {"sequence": sequence, **values}))
        return RetryJobResult(
            previous=_job(sequence),
            job=_job(sequence, status="queued", blocking=False),
            worker_restarted=True,
        )

    async def recent_audit_events(self, *, limit: int) -> AuditListResult:
        self.calls.append(("recent_audit_events", limit))
        return AuditListResult.model_validate(
            {
                "events": [
                    {
                        "audit_id": "c" * 32,
                        "occurred_at": "2026-07-30T01:02:00Z",
                        "category": "operations",
                        "operation": "retry_job",
                        "status": "success",
                        "request_id": "operations-request",
                        "memory_sequence": 7,
                        "attributes": {"job_status": "queued"},
                    }
                ]
            }
        )

    async def health(self, *, deep: bool = False) -> HealthResult:
        self.calls.append(("health", deep))
        return _health(ready=True)

    async def readiness(self) -> tuple[int, HealthResult]:
        self.calls.append(("readiness", None))
        return 200, _health(ready=True)

    async def metrics(self) -> tuple[str, str]:
        self.calls.append(("metrics", None))
        return "habitus_operations_total 3\n", "text/plain; version=0.0.4; charset=utf-8"


def _health(*, ready: bool) -> HealthResult:
    return HealthResult.model_validate(
        {
            "status": "healthy" if ready else "unhealthy",
            "ready": ready,
            "recall_ready": ready,
            "checked_at": datetime(2026, 7, 30, 1, tzinfo=UTC).isoformat(),
            "checks": [
                {
                    "name": "runtime",
                    "status": "healthy" if ready else "unhealthy",
                    "detail": "ready" if ready else "not ready",
                    "critical": True,
                }
            ],
        }
    )


def _runtime(observer: CollectingObserver | None = None) -> Runtime:
    runtime = object.__new__(Runtime)
    runtime.start = AsyncMock()  # type: ignore[method-assign]
    runtime.close = AsyncMock()  # type: ignore[method-assign]
    runtime.components = SimpleNamespace(  # type: ignore[assignment]
        infrastructure=SimpleNamespace(
            observer=observer or CollectingObserver(),
            managed_observability=None,
        )
    )
    return runtime


def _app(monkeypatch: pytest.MonkeyPatch, *, max_bytes: int = 4096):
    handlers = CapabilityHandlers()
    observer = CollectingObserver()
    runtime = _runtime(observer)
    monkeypatch.setattr(app_module, "RuntimeHTTPHandlers", lambda _runtime: handlers)
    app = app_module.create_http_app(
        runtime,
        config=HTTPAPIConfig(max_request_bytes=max_bytes),
    )
    return app, runtime, handlers, observer


def test_memory_routes_expose_public_results_and_preserve_handler_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    app, runtime, handlers, observer = _app(monkeypatch)
    remember_body = {
        "delivery_id": "d" * 64,
        "conversation_id": "conversation-http",
        "started_on": "2026-07-30",
        "protocol": "openai",
        "payload": {"role": "user", "content": "记住我的偏好"},
        "start_sequence": 0,
        "occurred_at": "2026-07-30T01:00:00+00:00",
        "after_turn": True,
    }

    with TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        capabilities = client.get("/api/v1/capabilities")
        remembered = client.post("/api/v1/memory/remember", json=remember_body)
        cursor = client.get(
            "/api/v1/memory/conversations/cursor",
            params={"conversation_id": "conversation-http", "started_on": "2026-07-30"},
        )
        flushed = client.post(
            "/api/v1/memory/flush",
            json={"conversation_id": "conversation-http", "started_on": "2026-07-30"},
        )
        recalled = client.post(
            "/api/v1/memory/recall",
            json={"query": "我喜欢什么回答风格？", "limit": 5},
        )
        jobs = client.get(
            "/api/v1/memory/jobs",
            params={"conversation_id": "conversation-http", "started_on": "2026-07-30"},
        )
        blocked = client.get("/api/v1/memory/jobs/blocked")
        status = client.get(
            "/api/v1/memory/jobs/7",
            params={"conversation_id": "conversation-http", "started_on": "2026-07-30"},
        )

    assert all(
        response.status_code == 200
        for response in (capabilities, remembered, cursor, flushed, recalled, jobs, blocked, status)
    )
    assert capabilities.json()["result"] == {
        "api_version": "1.0",
        "service_version": "0.1.0",
        "protocols": ["openai", "anthropic"],
        "features": ["recall", "remember", "remember_idempotency_v1"],
        "max_request_bytes": 4096,
    }
    public_job = remembered.json()["result"]["jobs"][0]
    assert public_job == {
        "memory_sequence": 7,
        "conversation_id": "conversation-http",
        "started_on": "2026-07-30",
        "status": "queued",
    }
    assert "transaction_id" not in public_job
    assert remembered.json()["result"]["next_sequence"] == 2
    assert cursor.json()["result"] == {"next_sequence": 2}
    assert flushed.json()["result"] == {"jobs": [], "consistency": []}
    assert recalled.json()["result"]["memories"][0]["uri"].startswith("memory://")
    assert jobs.json()["result"]["jobs"][0]["manual_action_required"] is True
    assert blocked.json()["result"]["job"]["blocking"] is True
    assert status.json()["result"]["memory_sequence"] == 7
    remember_call = next(value for name, value in handlers.calls if name == "remember")
    assert isinstance(remember_call, dict)
    assert remember_call["payload"] == remember_body["payload"]
    assert remember_call["after_turn"] is True
    assert remember_call["delivery_id"] == "d" * 64
    assert any(name == "conversation_cursor" for name, _value in handlers.calls)
    assert any(name == "flush" for name, _value in handlers.calls)
    accepted = next(event for event in observer.events if event.operation == "job_accepted")
    assert accepted.context.memory_sequence == 7
    assert accepted.context.transaction_id == "7" * 32
    request_events = [event for event in observer.events if event.operation == "request"]
    assert request_events
    assert all(event.attributes["http_route"] != "/__unmatched__" for event in request_events)
    runtime.start.assert_awaited_once()
    runtime.close.assert_awaited_once()


def test_an_invalid_delivery_identity_is_rejected_before_the_domain_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch)
    body = {
        "delivery_id": "not-a-delivery-id",
        "conversation_id": "delivery",
        "started_on": "2026-08-01",
        "protocol": "codex_rollout",
        "payload": {"records": []},
        "start_sequence": 0,
        "occurred_at": "2026-08-01T00:00:00+00:00",
    }

    with TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        response = client.post("/api/v1/memory/remember", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
    (violation,) = response.json()["error"]["details"]["validation_errors"]
    assert violation["location"] == ["body", "delivery_id"]
    assert not any(name == "remember" for name, _value in handlers.calls)


def test_operations_and_metrics_are_available_on_the_loopback_service(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch)
    params = {"conversation_id": "conversation-http", "started_on": "2026-07-30"}

    with TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        retried = client.post(
            "/api/v1/operations/memory/jobs/7/retry",
            params=params,
            json={"expected_version": "a" * 64},
        )
        audit = client.get("/api/v1/operations/audit", params={"limit": 25})
        metrics = client.get("/metrics")

    assert retried.status_code == 202
    assert retried.json()["result"]["job"]["job_status"] == "queued"
    assert audit.status_code == 200
    assert audit.json()["result"]["events"][0]["operation"] == "retry_job"
    assert metrics.status_code == 200
    assert metrics.text == "habitus_operations_total 3\n"
    retry_call = next(value for name, value in handlers.calls if name == "retry_failed_job")
    assert isinstance(retry_call, dict) and retry_call["expected_version"] == "a" * 64


def test_readiness_keeps_health_snapshot_on_503_and_lifespan_closes_after_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, runtime, handlers, _observer = _app(monkeypatch)
    handlers.readiness = AsyncMock(return_value=(503, _health(ready=False)))

    with TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        response = client.get("/ready", headers={"X-Request-ID": "ready-check"})

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] != "ready-check"
    _assert_response_request_id(response)
    assert response.json()["status"] == "ok"
    assert response.json()["result"]["ready"] is False

    failing_app, failing_runtime, _handlers, _observer = _app(monkeypatch)
    failing_runtime.start.side_effect = RuntimeError("startup failed")
    with pytest.raises(RuntimeError, match="startup failed"):
        with TestClient(failing_app, base_url="http://127.0.0.1:8787"):
            pass
    failing_runtime.close.assert_awaited_once()


def test_declared_oversized_request_is_rejected_before_domain_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch, max_bytes=1024)

    with TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/memory/remember",
            headers={"Content-Length": "1025", "Content-Type": "application/json"},
            content=b"{}",
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    _assert_response_request_id(response)
    assert not any(name == "remember" for name, _value in handlers.calls)


@pytest.mark.parametrize(
    ("headers", "detail"),
    [
        ({"Host": "attacker.example"}, "Host"),
        ({"Origin": "https://attacker.example"}, "Origin"),
        ({"Origin": "null"}, "Origin"),
    ],
)
def test_local_service_rejects_non_loopback_host_and_origin(
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
    detail: str,
) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch)
    with TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        response = client.get("/api/v1/capabilities", headers=headers)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
    _assert_response_request_id(response)
    assert detail in response.json()["error"]["message"]
    assert not handlers.calls


def test_chunked_oversized_request_cannot_bypass_body_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch, max_bytes=1024)
    body = {
        "conversation_id": "conversation-http",
        "started_on": "2026-07-30",
        "protocol": "openai",
        "payload": {"content": "x" * 4096},
        "start_sequence": 0,
        "occurred_at": "2026-07-30T01:00:00+00:00",
        "after_turn": True,
    }
    encoded = json.dumps(body).encode("utf-8")

    async def chunks():
        yield encoded[:1500]
        yield encoded[1500:]

    async def scenario() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
                request = client.build_request(
                    "POST",
                    "/api/v1/memory/remember",
                    headers={"Content-Type": "application/json"},
                    content=chunks(),
                )
                assert "content-length" not in request.headers
                return await client.send(request)

    response = asyncio.run(asyncio.wait_for(scenario(), _BODY_WAIT_SECONDS))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    _assert_response_request_id(response)
    assert not any(name == "remember" for name, _value in handlers.calls)


def test_chunked_body_over_limit_after_a_complete_json_prefix_never_reaches_the_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """分块请求的第一块已是完整 JSON、第二块才超限：必须在路由执行前拒绝，handler 一次都不被调用。"""

    app, _runtime_value, handlers, _observer = _app(monkeypatch, max_bytes=1024)
    encoded = json.dumps(
        {
            "conversation_id": "conversation-http",
            "started_on": "2026-07-30",
            "protocol": "openai",
            "payload": {"content": "short"},
            "start_sequence": 0,
            "occurred_at": "2026-07-30T01:00:00+00:00",
        }
    ).encode("utf-8")
    assert len(encoded) < 1024

    async def chunks():  # noqa: ANN202
        yield encoded
        yield b" " * 1024

    async def scenario() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
                request = client.build_request(
                    "POST",
                    "/api/v1/memory/remember",
                    headers={"Content-Type": "application/json"},
                    content=chunks(),
                )
                assert "content-length" not in request.headers
                return await client.send(request)

    response = asyncio.run(asyncio.wait_for(scenario(), _BODY_WAIT_SECONDS))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    _assert_response_request_id(response)
    assert not any(name == "remember" for name, _value in handlers.calls)


def test_chunked_body_within_limit_is_replayed_to_the_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch, max_bytes=4096)
    encoded = json.dumps(
        {
            "conversation_id": "conversation-http",
            "started_on": "2026-07-30",
            "protocol": "openai",
            "payload": {"content": "short"},
            "start_sequence": 0,
            "occurred_at": "2026-07-30T01:00:00+00:00",
        }
    ).encode("utf-8")

    async def chunks():  # noqa: ANN202
        yield encoded[:10]
        yield encoded[10:]

    async def scenario() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
                request = client.build_request(
                    "POST",
                    "/api/v1/memory/remember",
                    headers={"Content-Type": "application/json"},
                    content=chunks(),
                )
                return await client.send(request)

    response = asyncio.run(asyncio.wait_for(scenario(), _BODY_WAIT_SECONDS))

    assert response.status_code == 200
    remember_call = next(value for name, value in handlers.calls if name == "remember")
    assert isinstance(remember_call, dict) and remember_call["payload"] == {"content": "short"}


@pytest.mark.parametrize(
    "headers",
    [
        [(b"host", b"attacker.example")],
        [(b"host", b"127.0.0.1:8787"), (b"origin", b"https://attacker.example")],
    ],
)
def test_chunked_request_from_a_foreign_host_or_origin_is_refused_before_reading_its_body(
    monkeypatch: pytest.MonkeyPatch,
    headers: list[tuple[bytes, bytes]],
) -> None:
    """不带 Content-Length 的请求：Host / Origin 不合格时一块请求体也不读就拒绝（分块限长先读完整份请求体）。"""

    app, _runtime_value, handlers, _observer = _app(monkeypatch)
    read = 0
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        nonlocal read
        read += 1
        return {"type": "http.request", "body": b"x" * 256, "more_body": read < 10}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/memory/remember",
        "raw_path": b"/api/v1/memory/remember",
        "query_string": b"",
        "headers": [*headers, (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8787),
        "root_path": "",
    }

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            await app(scope, receive, send)

    asyncio.run(asyncio.wait_for(scenario(), _BODY_WAIT_SECONDS))

    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 400
    assert read == 0
    assert not handlers.calls


@pytest.mark.parametrize(
    ("host", "origin", "detail"),
    [
        # 带 userinfo 的 Origin：主机部分是回环也要拒（杀"去掉 userinfo 检查"）
        ("127.0.0.1:8787", "http://user:secret@127.0.0.1:8787", "user information"),
        # 非 HTTP scheme 的回环 Origin（杀"去掉 scheme 检查"）
        ("127.0.0.1:8787", "ftp://127.0.0.1:8787", "loopback HTTP origin"),
        # Host 前后带空白（杀"去掉 Host 空白检查"）：urlsplit 会静默删掉制表符，只有这条检查拦得住
        ("\t127.0.0.1:8787", None, "Host"),
        ("127.0.0.1:8787\t", None, "Host"),
        (" 127.0.0.1:8787", None, "Host"),
    ],
)
def test_local_request_boundary_rejects_userinfo_scheme_and_padded_host(
    host: str,
    origin: str | None,
    detail: str,
) -> None:
    from habitus.integrations.http_api.local_security import local_request_violation

    violation = local_request_violation(host_header=host, origin_header=origin)

    assert violation is not None and detail in violation
    assert local_request_violation(host_header="127.0.0.1:8787", origin_header="http://127.0.0.1:8787") is None


def test_negative_content_length_is_rejected_before_domain_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch)

    with TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/memory/remember",
            headers={"Content-Length": "-1", "Content-Type": "application/json"},
            content=b"",
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Content-Length must not be negative"
    assert not handlers.calls


def test_remember_rejects_an_occurred_at_without_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch)
    body = {
        "conversation_id": "conversation-http",
        "started_on": "2026-07-30",
        "protocol": "openai",
        "payload": {"content": "naive time"},
        "start_sequence": 0,
        "occurred_at": "2026-07-30T01:00:00",
    }

    with TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        response = client.post("/api/v1/memory/remember", json=body)

    assert response.status_code == 400
    (violation,) = response.json()["error"]["details"]["validation_errors"]
    assert violation["location"] == ["body", "occurred_at"]
    assert not any(name == "remember" for name, _value in handlers.calls)
