"""提醒出口的取口（方案 32 第 4.1 条）：``POST /api/v1/reminders/claim`` 只调 ``Runtime.claim_reminders``；特性名 ``reminders_v1``；
形状不对、非 loopback、limit 超上限都拒；预测层没开时是空、也不公布 ``reminders_v1``。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from habitus.integrations.http_api import create_http_app
from habitus.integrations.http_api.handlers import RuntimeHTTPHandlers
from habitus.runtime import Runtime
from habitus.runtime.foresight_outlet import OPTIONS, ClaimedReminder

CST = timezone(timedelta(hours=8))
AT = datetime(2026, 10, 22, 15, 52, 10, tzinfo=CST)


def _runtime(claimed: tuple[ClaimedReminder, ...], *, enabled: bool = True) -> Runtime:
    runtime = object.__new__(Runtime)
    runtime.reminders_enabled = lambda: enabled  # type: ignore[method-assign]
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
    assert "reminders_v1" in handlers.capabilities().features
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


def test_the_feature_is_advertised_only_while_the_foresight_layer_carries_an_outlet() -> None:
    """裁定 80 第 1 条：预测层没开时取口恒为空，服务不公布 ``reminders_v1``（插件与 ``habitus doctor`` 据此判"提醒没开"）；
    提醒报表 ``reminders_report_v1`` 照常公布。走真的 ``Runtime.reminders_enabled`` 与 HTTP 路由。"""

    from types import SimpleNamespace

    runtime = object.__new__(Runtime)
    runtime.conversation_protocols = lambda: ("codex_rollout",)  # type: ignore[method-assign]
    runtime.start = AsyncMock()  # type: ignore[method-assign]
    runtime.close = AsyncMock()  # type: ignore[method-assign]
    for foresight, advertised in (
        (None, False),
        (SimpleNamespace(outlet=None), False),
        (SimpleNamespace(outlet=object()), True),
    ):
        runtime.components = SimpleNamespace(foresight=foresight)  # type: ignore[assignment]
        assert runtime.reminders_enabled() is advertised
        with TestClient(create_http_app(runtime), base_url="http://127.0.0.1:8787") as client:
            features = client.get("/api/v1/capabilities").json()["result"]["features"]
        assert ("reminders_v1" in features) is advertised
        assert "reminders_report_v1" in features


def test_the_runtime_refuses_a_non_positive_limit_and_a_naive_moment() -> None:
    """``Runtime.claim_reminders`` 的闸：limit 是正整数、at 带时区；预测层没装出口时给空。"""

    import asyncio
    from types import SimpleNamespace

    from habitus.config.foresight import ForesightConfig

    runtime = object.__new__(Runtime)
    runtime._state = __import__("habitus.runtime.runtime", fromlist=["RuntimeState"]).RuntimeState.READY  # type: ignore[attr-defined]
    runtime.config = SimpleNamespace(foresight=ForesightConfig(enabled=False))  # type: ignore[assignment]
    runtime.components = SimpleNamespace(foresight=None)  # type: ignore[assignment]
    assert asyncio.run(runtime.claim_reminders(host="codex", session="s", at=AT, limit=3)) == ()
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="limit"):
            asyncio.run(runtime.claim_reminders(host="codex", session="s", at=AT, limit=bad))
    with pytest.raises(ValueError, match="timezone"):
        asyncio.run(runtime.claim_reminders(host="codex", session="s", at=datetime(2026, 10, 22, 15), limit=1))
    assert datetime.now(UTC).tzinfo is UTC


def test_a_limit_above_the_configured_cap_is_claimed_up_to_the_cap() -> None:
    """T10（裁定 75 第 3 条）：宿主要的条数大于本地服务的 ``outlet_claim_limit``（默认 3）时，按这个上限取、照常 200 返回，不拒收。
    经真实的 HTTP 路由与 ``Runtime.claim_reminders``，出口换成按 limit 给条数的桩。"""

    from types import SimpleNamespace

    from habitus.config.foresight import ForesightConfig

    asked: list[int] = []

    class Outlet:
        def claim(self, host: str, session: str, at: datetime, limit: int) -> tuple[ClaimedReminder, ...]:
            asked.append(limit)
            return tuple(
                ClaimedReminder(f"e{index}", "写周报", "周报还没写。", OPTIONS, AT + timedelta(hours=1), AT + timedelta(minutes=30), AT)
                for index in range(limit)
            )

    runtime = object.__new__(Runtime)
    runtime._state = __import__("habitus.runtime.runtime", fromlist=["RuntimeState"]).RuntimeState.READY  # type: ignore[attr-defined]
    runtime.config = SimpleNamespace(foresight=ForesightConfig())  # type: ignore[assignment]
    runtime.components = SimpleNamespace(foresight=SimpleNamespace(outlet=Outlet()))  # type: ignore[assignment]
    runtime.conversation_protocols = lambda: ("codex_rollout",)  # type: ignore[method-assign]
    runtime.start = AsyncMock()  # type: ignore[method-assign]
    runtime.close = AsyncMock()  # type: ignore[method-assign]
    cap = ForesightConfig().outlet_claim_limit
    with TestClient(create_http_app(runtime), base_url="http://127.0.0.1:8787") as client:
        for limit, expected in ((1, 1), (cap, cap), (cap + 1, cap), (20, cap)):
            body = {"host": "codex", "session": "codex-thread-1", "at": AT.isoformat(), "limit": limit}
            response = client.post("/api/v1/reminders/claim", json=body)
            assert response.status_code == 200, response.text
            assert len(response.json()["result"]["reminders"]) == expected, limit
    assert asked == [1, cap, cap, cap]


def test_the_report_route_lists_each_reminder_stage_and_the_counts() -> None:
    from habitus.runtime.foresight_outlet import Handoff
    from habitus.runtime.foresight_reminders import ReminderDay, ReminderProgress

    deadline = AT.replace(hour=18, minute=0, second=0)
    day = ReminderDay(
        AT.date(),
        (
            ReminderProgress("e1", "写周报", "周报还没写。", deadline, "说了", AT, AT + timedelta(minutes=2), "codex", "正",
                             AT + timedelta(minutes=3), None, 0, (Handoff("codex", "codex-thread-1", AT, "spoken"),)),
            ReminderProgress("e2", "跑备份", "备份还没跑。", deadline, "宿主没取", AT, None, None, None, None, None, None, ()),
            # 段 6 Q15：等取的时候他先做了——「已发生」照样能报出来
            ReminderProgress("e3", "交报销", "报销还没交。", deadline, "已发生", AT, None, None, None, None, "准时", 0, ()),
        ),
    )
    runtime = _runtime(())
    runtime.reminders = AsyncMock(return_value=day)  # type: ignore[method-assign]
    assert "reminders_report_v1" in RuntimeHTTPHandlers(runtime).capabilities().features
    with TestClient(create_http_app(runtime), base_url="http://127.0.0.1:8787") as client:
        response = client.get("/api/v1/reminders", params={"day": "2026-10-22"})
        assert response.status_code == 200, response.text
        result = response.json()["result"]
    assert (result["day"], result["spoken"], result["never_claimed"], result["responded"], result["awaiting"]) == (
        "2026-10-22", 1, 1, 1, 0,
    )
    first = result["reminders"][0]
    assert (first["stage"], first["spoken_via"], first["response"]) == ("说了", "codex", "正")
    assert result["reminders"][2]["stage"] == "已发生"
    assert first["handoffs"] == [{"host": "codex", "session": "codex-thread-1", "claimed_at": AT.isoformat(), "ended": "spoken"}]
    runtime.reminders.assert_awaited_once_with(AT.date())  # type: ignore[attr-defined]
