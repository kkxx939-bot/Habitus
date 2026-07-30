"""从远程调用方视角验证 HTTP 产品边界，而不是重复 Runtime 领域测试。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from Config import HTTPAPIConfig
from foundation.observability import ObservationEvent
from integrations.http_api import app as app_module
from Runtime import Runtime

UTC = timezone.utc
API_KEY = "m" * 32
OPERATIONS_KEY = "o" * 32
AUTH = {"Authorization": f"Bearer {API_KEY}"}
OPERATIONS_AUTH = {"Authorization": f"Bearer {OPERATIONS_KEY}"}


class CollectingObserver:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def _job(sequence: int = 7, *, status: str = "failed", blocking: bool = True) -> dict[str, object]:
    return {
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


class CapabilityHandlers:
    """只替代领域执行，保留真实 FastAPI、Schema、中间件和认证链。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def protocols(self) -> dict[str, object]:
        self.calls.append(("protocols", None))
        return {"protocols": ["openai", "anthropic"]}

    async def remember(self, **values: object) -> dict[str, object]:
        self.calls.append(("remember", values))
        return {
            "ignored_items": 0,
            "after_turn": bool(values["after_turn"]),
            "jobs": [
                {
                    "memory_sequence": 7,
                    "conversation_id": values["conversation_id"],
                    "started_on": str(values["started_on"]),
                    "segment_id": "internal-segment",
                    "source_segment_digest": "b" * 64,
                    "transaction_id": "transaction-http-7",
                    "status": "queued",
                }
            ],
            "consistency": [],
        }

    async def recall(self, query: str, **values: object) -> dict[str, object]:
        self.calls.append(("recall", {"query": query, **values}))
        return {
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

    async def list_jobs(self, **values: object) -> dict[str, object]:
        self.calls.append(("list_jobs", values))
        return {"jobs": [_job()], "next_before_sequence": None}

    async def blocked_job(self) -> dict[str, object]:
        self.calls.append(("blocked_job", None))
        return {"job": _job()}

    async def job_status(self, sequence: int, **values: object) -> dict[str, object]:
        self.calls.append(("job_status", {"sequence": sequence, **values}))
        return _job(sequence)

    async def retry_failed_job(self, sequence: int, **values: object) -> dict[str, object]:
        self.calls.append(("retry_failed_job", {"sequence": sequence, **values}))
        return {
            "previous": _job(sequence),
            "job": _job(sequence, status="queued", blocking=False),
            "worker_restarted": True,
        }

    async def recent_audit_events(self, *, limit: int) -> dict[str, object]:
        self.calls.append(("recent_audit_events", limit))
        return {
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

    async def health(self, *, deep: bool = False) -> dict[str, object]:
        self.calls.append(("health", deep))
        return _health(ready=True)

    async def readiness(self) -> tuple[int, dict[str, object]]:
        self.calls.append(("readiness", None))
        return 200, _health(ready=True)

    async def metrics(self) -> tuple[str, str]:
        self.calls.append(("metrics", None))
        return "m2bos_operations_total 3\n", "text/plain; version=0.0.4; charset=utf-8"


def _health(*, ready: bool) -> dict[str, object]:
    return {
        "status": "healthy" if ready else "unhealthy",
        "ready": ready,
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


def _app(monkeypatch: pytest.MonkeyPatch, *, operations: bool = True, max_bytes: int = 4096):
    handlers = CapabilityHandlers()
    observer = CollectingObserver()
    runtime = _runtime(observer)
    monkeypatch.setattr(app_module, "RuntimeHTTPHandlers", lambda _runtime: handlers)
    app = app_module.create_http_app(
        runtime,
        api_key=API_KEY,
        operations_api_key=OPERATIONS_KEY if operations else None,
        config=HTTPAPIConfig(max_request_bytes=max_bytes),
    )
    return app, runtime, handlers, observer


def test_memory_routes_expose_public_results_and_preserve_handler_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    app, runtime, handlers, observer = _app(monkeypatch)
    remember_body = {
        "conversation_id": "conversation-http",
        "started_on": "2026-07-30",
        "protocol": "openai",
        "payload": {"role": "user", "content": "记住我的偏好"},
        "start_sequence": 0,
        "occurred_at": "2026-07-30T01:00:00+00:00",
        "after_turn": True,
    }

    with TestClient(app, raise_server_exceptions=False) as client:
        protocols = client.get("/api/v1/protocols", headers=AUTH)
        remembered = client.post("/api/v1/memory/remember", headers=AUTH, json=remember_body)
        recalled = client.post(
            "/api/v1/memory/recall",
            headers=AUTH,
            json={"query": "我喜欢什么回答风格？", "limit": 5},
        )
        jobs = client.get(
            "/api/v1/memory/jobs",
            headers=AUTH,
            params={"conversation_id": "conversation-http", "started_on": "2026-07-30"},
        )
        blocked = client.get("/api/v1/memory/jobs/blocked", headers=AUTH)
        status = client.get(
            "/api/v1/memory/jobs/7",
            headers=AUTH,
            params={"conversation_id": "conversation-http", "started_on": "2026-07-30"},
        )

    assert all(response.status_code == 200 for response in (protocols, remembered, recalled, jobs, blocked, status))
    assert protocols.json()["result"] == {"protocols": ["openai", "anthropic"]}
    public_job = remembered.json()["result"]["jobs"][0]
    assert public_job == {
        "memory_sequence": 7,
        "conversation_id": "conversation-http",
        "started_on": "2026-07-30",
        "status": "queued",
    }
    assert "transaction_id" not in public_job
    assert recalled.json()["result"]["memories"][0]["uri"].startswith("memory://")
    assert jobs.json()["result"]["jobs"][0]["manual_action_required"] is True
    assert blocked.json()["result"]["job"]["blocking"] is True
    assert status.json()["result"]["memory_sequence"] == 7
    remember_call = next(value for name, value in handlers.calls if name == "remember")
    assert isinstance(remember_call, dict)
    assert remember_call["payload"] == remember_body["payload"]
    assert remember_call["after_turn"] is True
    assert any(event.operation == "job_accepted" for event in observer.events)
    request_events = [event for event in observer.events if event.operation == "request"]
    assert request_events
    assert all(event.attributes["http_route"] != "/__unmatched__" for event in request_events)
    runtime.start.assert_awaited_once()
    runtime.close.assert_awaited_once()


def test_operations_key_is_isolated_and_state_changing_route_requires_it(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch)
    params = {"conversation_id": "conversation-http", "started_on": "2026-07-30"}

    with TestClient(app, raise_server_exceptions=False) as client:
        operations_on_memory_route = client.get("/api/v1/protocols", headers=OPERATIONS_AUTH)
        main_on_operations_route = client.post(
            "/api/v1/operations/memory/jobs/7/retry",
            headers=AUTH,
            params=params,
            json={"expected_version": "a" * 64},
        )
        retried = client.post(
            "/api/v1/operations/memory/jobs/7/retry",
            headers=OPERATIONS_AUTH,
            params=params,
            json={"expected_version": "a" * 64},
        )
        audit = client.get("/api/v1/operations/audit", headers=OPERATIONS_AUTH, params={"limit": 25})
        metrics_without_auth = client.get("/metrics")
        metrics = client.get("/metrics", headers=AUTH)

    assert operations_on_memory_route.status_code == 401
    assert main_on_operations_route.status_code == 401
    assert retried.status_code == 202
    assert retried.json()["result"]["job"]["job_status"] == "queued"
    assert audit.status_code == 200
    assert audit.json()["result"]["events"][0]["operation"] == "retry_job"
    assert metrics_without_auth.status_code == 401
    assert metrics.status_code == 200
    assert metrics.text == "m2bos_operations_total 3\n"
    retry_call = next(value for name, value in handlers.calls if name == "retry_failed_job")
    assert isinstance(retry_call, dict) and retry_call["expected_version"] == "a" * 64


def test_operations_routes_are_absent_when_no_independent_key_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, _handlers, _observer = _app(monkeypatch, operations=False)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/operations/audit", headers=AUTH)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_readiness_keeps_health_snapshot_on_503_and_lifespan_closes_after_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, runtime, handlers, _observer = _app(monkeypatch)
    handlers.readiness = AsyncMock(return_value=(503, _health(ready=False)))  # type: ignore[method-assign]

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/ready", headers={"X-Request-ID": "ready-check"})

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "ready-check"
    assert response.json()["status"] == "ok"
    assert response.json()["result"]["ready"] is False

    failing_app, failing_runtime, _handlers, _observer = _app(monkeypatch)
    failing_runtime.start.side_effect = RuntimeError("startup failed")
    with pytest.raises(RuntimeError, match="startup failed"):
        with TestClient(failing_app):
            pass
    failing_runtime.close.assert_awaited_once()


def test_declared_oversized_request_is_rejected_before_domain_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch, max_bytes=1024)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/memory/remember",
            headers={**AUTH, "Content-Length": "1025", "Content-Type": "application/json"},
            content=b"{}",
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    assert not any(name == "remember" for name, _value in handlers.calls)


@pytest.mark.xfail(
    strict=True,
    reason="M2BOS-BUG-HTTP-001: chunked request bodies are not counted against max_request_bytes",
)
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
            async with httpx.AsyncClient(transport=transport, base_url="http://m2bos.test") as client:
                request = client.build_request(
                    "POST",
                    "/api/v1/memory/remember",
                    headers={**AUTH, "Content-Type": "application/json"},
                    content=chunks(),
                )
                assert "content-length" not in request.headers
                return await client.send(request)

    response = asyncio.run(scenario())

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    assert not any(name == "remember" for name, _value in handlers.calls)
