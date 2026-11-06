"""W1 第 4 项 B2：``conversation_id`` 有 UTF-8 字节上限。

会话号会成为会话目录名（单段）。超长的在 HTTP 入口就以 4xx 拒收、错误体写明上限；绕过 HTTP 直接构造
``ConversationAddress`` 同样被拒。HTTP 与领域用同一个常数。
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from habitus.infrastructure.store.contracts import PathLock
from habitus.integrations.http_api import create_http_app, schemas
from habitus.memory.conversation import ConversationAddress, ConversationLayoutError
from habitus.memory.conversation.layout import MAX_CONVERSATION_ID_UTF8_BYTES
from habitus.pre.conversation import ConversationBatch
from habitus.runtime import build_runtime
from tests.helpers import closed_turn
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies
from tests.lock_store import ProcessLocalLockStore

DAY = date(2026, 8, 7)
LIMIT = MAX_CONVERSATION_ID_UTF8_BYTES
# 六个宿主插件：前缀 + safeIdentifier（只留 [A-Za-z0-9._-]、最多 220 字符），最长的前缀是 "opencode-"
PLUGIN_MAX_BYTES = len("opencode-") + 220


def _id(size: int, unit: str = "a") -> str:
    width = len(unit.encode("utf-8"))
    count, rest = divmod(size, width)
    return unit * count + "b" * rest


def test_the_limit_fits_one_path_segment_covers_every_plugin_and_is_not_below_128() -> None:
    assert 128 <= LIMIT <= 255
    assert PLUGIN_MAX_BYTES <= LIMIT
    # HTTP 层用的就是领域这一个常数，不另写数字
    assert schemas.MAX_CONVERSATION_ID_UTF8_BYTES is MAX_CONVERSATION_ID_UTF8_BYTES


@pytest.mark.parametrize("unit", ["a", "会", "😀"])
def test_conversation_address_accepts_exactly_the_limit_and_rejects_one_byte_more(unit: str) -> None:
    at_limit = _id(LIMIT, unit)
    assert len(at_limit.encode("utf-8")) == LIMIT
    assert ConversationAddress(at_limit, DAY).conversation_id == at_limit
    over = _id(LIMIT + 1, unit)
    with pytest.raises(ConversationLayoutError, match=f"at most {LIMIT} UTF-8 bytes, got {LIMIT + 1}"):
        ConversationAddress(over, DAY)


def test_casefold_growth_of_the_directory_name_is_counted() -> None:
    """会话目录名是归一身份：``İ`` 2 字节 casefold 成 3 字节，原文没超、目录名超了也拒。"""

    raw = "İ" * (LIMIT // 2)
    assert len(raw.encode("utf-8")) <= LIMIT
    with pytest.raises(ConversationLayoutError, match=f"at most {LIMIT} UTF-8 bytes"):
        ConversationAddress(raw, DAY)


def _runtime(tmp_path: Path):
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        runtime_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    return runtime


def test_a_conversation_id_exactly_at_the_limit_is_recorded(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _runtime(tmp_path)
        conversation_id = _id(LIMIT, "会")
        address = ConversationAddress(conversation_id, DAY)
        await runtime.append_conversation(address, ConversationBatch(conversation_id, closed_turn()))
        live = await runtime.read_live_conversation(address)
        assert live is not None and len(live.messages) == 2
        flushed = await runtime.flush_conversation(address)
        assert len(flushed.jobs) == 1
        await runtime.close()

    asyncio.run(scenario())


def _client(tmp_path: Path) -> tuple[TestClient, object]:
    runtime = _runtime(tmp_path)
    runtime.start = AsyncMock()  # type: ignore[method-assign]
    close = runtime.close
    runtime.close = AsyncMock()  # type: ignore[method-assign]
    return TestClient(create_http_app(runtime), base_url="http://127.0.0.1:8787", raise_server_exceptions=False), close


def _remember(conversation_id: str) -> dict[str, object]:
    return {
        "conversation_id": conversation_id,
        "started_on": DAY.isoformat(),
        "protocol": "openai_chat_completions",
        "payload": {"messages": [{"role": "user", "content": "记住"}, {"role": "assistant", "content": "好的"}]},
        "start_sequence": 0,
        "occurred_at": "2026-08-07T01:00:00+00:00",
    }


def test_http_accepts_the_limit_and_rejects_more_at_every_conversation_id_entry(tmp_path: Path) -> None:
    client, close = _client(tmp_path)
    at_limit = _id(LIMIT, "会")
    over = _id(LIMIT + 1, "会")
    params = {"started_on": DAY.isoformat()}
    try:
        with client:
            assert client.post("/api/v1/memory/remember", json=_remember(at_limit)).status_code == 200
            assert (
                client.get("/api/v1/memory/conversations/cursor", params={**params, "conversation_id": at_limit})
            ).status_code == 200
            rejected = [
                client.post("/api/v1/memory/remember", json=_remember(over)),
                client.post("/api/v1/memory/flush", json={"conversation_id": over, **params}),
                client.post("/api/v1/memory/recall", json={"query": "记得什么", "conversation_id": over, **params}),
                client.get("/api/v1/memory/conversations/cursor", params={**params, "conversation_id": over}),
                client.get("/api/v1/memory/jobs", params={**params, "conversation_id": over}),
                client.get("/api/v1/memory/jobs/1", params={**params, "conversation_id": over}),
            ]
    finally:
        asyncio.run(close())  # type: ignore[operator]
    for response in rejected:
        assert response.status_code == 400, (response.request.url, response.text)
        error = response.json()["error"]
        assert error["retryable"] is False
        # 在入口的请求校验这一步拒（不是进了领域再被 ConversationAddress 拦下）
        assert error["message"] == "Request validation failed", response.text
        assert f"at most {LIMIT} UTF-8 bytes, got {LIMIT + 1}" in response.text, response.text


def test_http_counts_casefold_growth_like_the_domain(tmp_path: Path) -> None:
    """``İ`` 2 字节、casefold 后 3 字节：原文正好在上限，目录名超了，入口就拒，与 ``ConversationAddress`` 量法一致。"""

    raw = "İ" * (LIMIT // 2)
    client, close = _client(tmp_path)
    try:
        with client:
            response = client.post("/api/v1/memory/remember", json=_remember(raw))
    finally:
        asyncio.run(close())  # type: ignore[operator]
    assert response.status_code == 400, response.text
    assert response.json()["error"]["message"] == "Request validation failed", response.text
    assert f"at most {LIMIT} UTF-8 bytes, got {LIMIT // 2 * 3}" in response.text, response.text
