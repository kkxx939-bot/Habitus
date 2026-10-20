"""提醒出口的取口（方案 32 第 4.1 条）：``POST /api/v1/reminders/claim`` 只调 ``Runtime.claim_reminders``；特性名 ``reminders_v1``；
形状不对、非 loopback、limit 超上限都拒；预测层没开时是空。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from habitus.integrations.http import RuntimeHTTPHandlers
from habitus.integrations.http_api import create_http_app
from habitus.runtime import Runtime
from habitus.runtime.foresight_outlet import OPTIONS, ClaimedReminder

CST = timezone(timedelta(hours=8))
AT = datetime(2026, 10, 22, 15, 52, 10, tzinfo=CST)


def _runtime(claimed: tuple[ClaimedReminder, ...]) -> Runtime:
    runtime = object.__new__(Runtime)
    runtime.claim_reminders = AsyncMock(return_value=claimed)  # type: ignore[method-assign]
    runtime.conversation_protocols = lambda: ("codex_rollout", "claude_code")  # type: ignore[method-assign]
    runtime.start = AsyncMock()  # type: ignore[method-assign]
    runtime.close = AsyncMock()  # type: ignore[method-assign]
    return runtime


def test_the_claim_route_returns_the_outlet_shape_and_advertises_the_feature() -> None:
    claimed = (
        ClaimedReminder(
            "e1",
            "写周报",
            "周四下午你一般会写周报，这周的还没写。",
            OPTIONS,
            AT.replace(hour=18, minute=0, second=0),
            AT + timedelta(minutes=30),
            AT,
        ),
    )
    runtime = _runtime(claimed)
    handlers = RuntimeHTTPHandlers(runtime)
    assert "reminders_v1" in handlers.capabilities()["features"]  # type: ignore[operator]
    app = create_http_app(runtime)
    body = {"host": "codex", "session": "codex-thread-1", "at": AT.isoformat(), "limit": 3}
    with TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        response = client.post("/api/v1/reminders/claim", json=body)
        assert response.status_code == 200, response.text
        (reminder,) = response.json()["result"]["reminders"]
        assert reminder == {
            "request_id": "e1",
            "kind": "写周报",
            "text": "周四下午你一般会写周报，这周的还没写。",
            "options": list(OPTIONS),
            "deadline": "2026-10-22T18:00:00+08:00",
            "lease_until": "2026-10-22T16:22:10+08:00",
            "claimed_at": "2026-10-22T15:52:10+08:00",
        }
        runtime.claim_reminders.assert_awaited_once_with(host="codex", session="codex-thread-1", at=AT, limit=3)  # type: ignore[attr-defined]
        # 形状：宿主名只许小写标识、at 要带时区、limit 有上限、不许多余字段
        for bad in (
            {**body, "host": "Codex Desktop"},
            {**body, "at": "2026-10-22T15:52:10"},
            {**body, "limit": 0},
            {**body, "limit": 21},
            {**body, "extra": 1},
            {**body, "session": " codex-thread-1"},
        ):
            assert client.post("/api/v1/reminders/claim", json=bad).status_code == 400, bad
        # 没取到：空清单（预测层没开也是这个样子）
        runtime.claim_reminders.return_value = ()  # type: ignore[attr-defined]
        assert client.post("/api/v1/reminders/claim", json=body).json()["result"] == {"reminders": []}
    with TestClient(app, base_url="http://example.com", raise_server_exceptions=False) as client:
        assert client.post("/api/v1/reminders/claim", json=body).status_code == 400  # 非 loopback 拒


def test_the_runtime_refuses_a_limit_above_the_configured_cap_and_a_naive_moment() -> None:
    """``Runtime.claim_reminders`` 的闸：limit 不超过 ``outlet_claim_limit``、at 带时区；预测层没装出口时给空。"""

    import asyncio
    from types import SimpleNamespace

    from habitus.config.foresight import ForesightConfig

    runtime = object.__new__(Runtime)
    runtime._state = __import__("habitus.runtime.runtime", fromlist=["RuntimeState"]).RuntimeState.READY  # type: ignore[attr-defined]
    runtime.config = SimpleNamespace(foresight=ForesightConfig(enabled=False))  # type: ignore[assignment]
    runtime.components = SimpleNamespace(foresight=None)  # type: ignore[assignment]
    assert asyncio.run(runtime.claim_reminders(host="codex", session="s", at=AT, limit=3)) == ()
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(runtime.claim_reminders(host="codex", session="s", at=AT, limit=4))
    with pytest.raises(ValueError, match="timezone"):
        asyncio.run(runtime.claim_reminders(host="codex", session="s", at=datetime(2026, 10, 22, 15), limit=1))
    assert datetime.now(UTC).tzinfo is UTC
