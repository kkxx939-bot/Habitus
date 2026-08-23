"""从远程调用方视角验证 HTTP 产品边界，而不是重复 Runtime 领域测试。"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from Config import HTTPAPIConfig
from foundation.observability import ObservationEvent
from integrations.http_api import app as app_module
from integrations.sdk import ConversationRef, HabitusHTTPClient
from Runtime import Runtime

UTC = timezone.utc
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


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


class DelegatingHTTPTransport:
    """证明 SDK 只依赖 request 能力，而非具体 httpx.AsyncClient 类型。"""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: object | None,
        params: Mapping[str, str] | None,
    ) -> httpx.Response:
        return await self.client.request(method, url, headers=headers, json=json, params=params)


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
    """只替代领域执行，保留真实 FastAPI、Schema 和中间件链。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def capabilities(self) -> dict[str, object]:
        self.calls.append(("capabilities", None))
        return {
            "api_version": "1.0",
            "service_version": "0.1.0",
            "protocols": ["openai", "anthropic"],
            "features": ["recall", "remember", "remember_idempotency_v1"],
        }

    async def remember(self, **values: object) -> dict[str, object]:
        self.calls.append(("remember", values))
        return {
            "ignored_items": 0,
            "after_turn": bool(values["after_turn"]),
            "next_sequence": 2,
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

    async def conversation_cursor(self, **values: object) -> dict[str, object]:
        self.calls.append(("conversation_cursor", values))
        return {"next_sequence": 2}

    async def flush(self, **values: object) -> dict[str, object]:
        self.calls.append(("flush", values))
        return {"jobs": [], "consistency": []}

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
        return "habitus_operations_total 3\n", "text/plain; version=0.0.4; charset=utf-8"


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
    assert accepted.context.transaction_id == "transaction-http-7"
    request_events = [event for event in observer.events if event.operation == "request"]
    assert request_events
    assert all(event.attributes["http_route"] != "/__unmatched__" for event in request_events)
    runtime.start.assert_awaited_once()
    runtime.close.assert_awaited_once()


def test_http_client_implements_the_same_agent_memory_port(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _runtime_value, handlers, _observer = _app(monkeypatch)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport) as raw_client:
            client = HabitusHTTPClient(
                "http://127.0.0.1",
                transport=DelegatingHTTPTransport(raw_client),
            )
            conversation = ConversationRef("conversation-http", date(2026, 7, 30))
            capabilities = await client.capabilities()
            remembered = await client.remember(
                conversation,
                protocol="openai_chat_completions",
                payload={"messages": [{"role": "assistant", "content": "好的"}]},
                start_sequence=0,
                occurred_at=datetime(2026, 7, 30, 1, tzinfo=UTC),
                after_turn=True,
                delivery_id="e" * 64,
            )
            cursor = await client.cursor(conversation)
            recalled = await client.recall("回答风格", conversation=conversation)
            flushed = await client.flush(conversation)
            await client.close()

        assert remembered.after_turn is True
        assert capabilities.api_version == "1.0"
        assert capabilities.protocols == ("openai", "anthropic")
        assert remembered.next_sequence == 2
        assert remembered.jobs[0].status == "queued"
        assert cursor == 2
        assert recalled.context == "用户偏好简洁回答。"
        assert flushed.jobs == ()

    asyncio.run(scenario())

    assert any(name == "remember" for name, _value in handlers.calls)
    sdk_remember = next(value for name, value in handlers.calls if name == "remember")
    assert isinstance(sdk_remember, dict)
    assert sdk_remember["delivery_id"] == "e" * 64
    assert any(name == "conversation_cursor" for name, _value in handlers.calls)
    assert any(name == "recall" for name, _value in handlers.calls)
    assert any(name == "flush" for name, _value in handlers.calls)


def test_unauthenticated_sdk_rejects_a_non_loopback_service_url() -> None:
    with pytest.raises(ValueError, match="loopback"):
        HabitusHTTPClient("http://192.168.1.20:8787")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://user:secret@127.0.0.1:8787",
        "http://127.0.0.1:8787/api",
        "http://127.0.0.1:8787;params",
        "http://127.0.0.1:8787?query=1",
        "http://127.0.0.1:8787#fragment",
        "http://127.0.0.1:invalid",
    ],
)
def test_unauthenticated_sdk_accepts_only_a_loopback_origin(base_url: str) -> None:
    with pytest.raises(ValueError):
        HabitusHTTPClient(base_url)


def test_sdk_rejects_an_invalid_delivery_identity_before_network_io() -> None:
    client = HabitusHTTPClient("http://127.0.0.1:8787")

    async def scenario() -> None:
        with pytest.raises(ValueError, match="delivery_id"):
            await client.remember(
                ConversationRef("delivery", date(2026, 8, 1)),
                protocol="codex_rollout",
                payload={"records": []},
                start_sequence=0,
                occurred_at=datetime(2026, 8, 1, tzinfo=UTC),
                delivery_id="not-a-delivery-id",
            )
        await client.close()

    asyncio.run(scenario())


def test_sdk_import_does_not_load_runtime_or_memory_packages() -> None:
    script = (
        "import json, sys; import integrations.sdk; "
        "print(json.dumps({'runtime': 'Runtime' in sys.modules, 'memory': 'memory' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"runtime": False, "memory": False}


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

    response = asyncio.run(scenario())

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    _assert_response_request_id(response)
    assert not any(name == "remember" for name, _value in handlers.calls)
