"""单独一轮超过请求体上限时的分块上传（``remember_parts_v1``，裁定 72 第 2 条）。

插件把一批 remember 请求体切块按顺序上传，本地服务收齐、核对摘要后拼回原请求体，走与一次送来完全相同的
remember 路径（真实 FastAPI 路由、Schema、中间件与 ``RuntimeHTTPHandlers``，只把 Runtime 的写入换成记下适配结果）。
插件那一半在 ``plugins/memory-plugin-shared/test/request-limit.test.mjs``。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from habitus.config import HTTPAPIConfig
from habitus.integrations.http_api import app as app_module
from habitus.integrations.http_api.remember_parts import (
    RememberPart,
    RememberPartsCapacityError,
    RememberPartsIntegrityError,
    RememberPartsRestartError,
    RememberPartsStaging,
)
from habitus.pre.conversation.adapters import ConversationAdapterContext, ConversationAdapterRegistry
from habitus.runtime import Runtime

MiB = 1024 * 1024
_DELIVERY = "b" * 64
_JSON = {"Content-Type": "application/json"}
_REGISTRY = ConversationAdapterRegistry.with_builtins()


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _recording_runtime(adapted: list[tuple[str | None, object]]) -> Runtime:
    """真 ``RuntimeHTTPHandlers`` 之下的 Runtime：用内置协议适配器适配、记下投递号与适配结果，不落盘。"""

    runtime = object.__new__(Runtime)
    runtime.start = AsyncMock()  # type: ignore[method-assign]
    runtime.close = AsyncMock()  # type: ignore[method-assign]
    runtime.components = SimpleNamespace(  # type: ignore[assignment]
        infrastructure=SimpleNamespace(observer=None, managed_observability=None)
    )

    async def append_protocol_conversation(address: Any, **values: Any) -> object:  # noqa: ANN401
        context = ConversationAdapterContext(address.conversation_id, values["start_sequence"], values["occurred_at"])
        adaptation = _REGISTRY.adapt(values["protocol"], values["payload"], context)
        adapted.append((values["delivery_id"], adaptation))
        return SimpleNamespace(
            adaptation=adaptation,
            ingest=SimpleNamespace(jobs=()),
            effective_after_turn=bool(values["after_turn"]),
            next_sequence=values["start_sequence"] + len(adaptation.batch.messages),
        )

    runtime.append_protocol_conversation = append_protocol_conversation  # type: ignore[method-assign]
    runtime.conversation_protocols = _REGISTRY.protocols  # type: ignore[method-assign]
    runtime.reminders_enabled = lambda: False  # type: ignore[method-assign]
    return runtime


def _client(adapted: list[tuple[str | None, object]], **config: Any) -> TestClient:  # noqa: ANN401
    app = app_module.create_http_app(_recording_runtime(adapted), config=HTTPAPIConfig(**config))
    return TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False)


def _huge_codex_turn() -> dict[str, object]:
    """一轮 40 MiB 以上的纯文字：五个大工具结果，带引号、反斜杠与中文，没有媒体。"""

    records: list[dict[str, object]] = [
        {"type": "event_msg", "payload": {"type": "task_started"}},
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "读几个日志"}]},
        },
    ]
    for index in range(5):
        call_id = f"call-{index}"
        line = f'第 {index} 份日志 "引号" 与 \\ 反斜杠\n'
        records.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": call_id,
                    "arguments": json.dumps({"command": ["cat", f"log-{index}.txt"]}),
                },
            }
        )
        output = (line * (4 * MiB // len(line) + 1))[: 4 * MiB]
        records.append(
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": call_id, "output": output}}
        )
    records.append(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "读完了"}],
            },
        }
    )
    return {"records": records}


def _remember_body(payload: object, *, delivery_id: str = _DELIVERY) -> bytes:
    return json.dumps(
        {
            "delivery_id": delivery_id,
            "conversation_id": "codex-huge-turn",
            "started_on": "2026-10-23",
            "protocol": "codex_rollout",
            "payload": payload,
            "start_sequence": 0,
            "occurred_at": "2026-10-23T08:00:00.000Z",
            "after_turn": True,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _parts(body: bytes, limit: int, *, delivery_id: str = _DELIVERY) -> list[dict[str, object]]:
    """与插件同样的切法：整批字节按 3 的倍数切段、base64 装进 JSON，每块请求体留 512 字节给信封字段。"""

    chunk_bytes = (limit - 512) // 4 * 3
    digest = hashlib.sha256(body).hexdigest()
    chunks = [body[offset : offset + chunk_bytes] for offset in range(0, len(body), chunk_bytes)]
    return [
        {
            "delivery_id": delivery_id,
            "part_index": index,
            "part_count": len(chunks),
            "body_bytes": len(body),
            "body_sha256": digest,
            "data": base64.b64encode(chunk).decode("ascii"),
        }
        for index, chunk in enumerate(chunks)
    ]


def _upload(client: TestClient, parts: list[dict[str, object]]) -> Iterator[Any]:  # noqa: ANN401
    for part in parts:
        response = client.post("/api/v1/memory/remember/parts", content=json.dumps(part), headers=_JSON)
        yield response


@pytest.fixture(scope="module")
def huge_turn() -> tuple[dict[str, object], bytes]:
    payload = _huge_codex_turn()
    body = _remember_body(payload)
    assert len(body) > 40 * MiB
    return payload, body


def test_a_40_mib_turn_uploaded_in_parts_adapts_exactly_like_one_send(
    huge_turn: tuple[dict[str, object], bytes],
) -> None:
    _payload, body = huge_turn
    once: list[tuple[str | None, object]] = []
    with _client(once, max_request_bytes=64 * MiB) as client:
        whole = client.post("/api/v1/memory/remember", content=body, headers=_JSON)
    assert whole.status_code == 200, whole.text[:500]

    chunked: list[tuple[str | None, object]] = []
    parts = _parts(body, 16 * MiB)
    assert len(parts) >= 3
    assert all(len(json.dumps(part)) <= 16 * MiB for part in parts)
    with _client(chunked) as client:
        responses = list(_upload(client, parts))
    assert [response.status_code for response in responses] == [200] * len(parts)
    assert [response.json()["result"]["remember"] for response in responses[:-1]] == [None] * (len(parts) - 1)
    assert [response.json()["result"]["received_parts"] for response in responses] == list(range(1, len(parts) + 1))
    assert responses[-1].json()["result"]["remember"] == whole.json()["result"]

    assert len(once) == len(chunked) == 1
    (once_delivery, once_adapted), (chunked_delivery, chunked_adapted) = once[0], chunked[0]
    assert chunked_delivery == once_delivery == _DELIVERY
    assert chunked_adapted.batch.messages == once_adapted.batch.messages  # type: ignore[attr-defined]
    assert chunked_adapted == once_adapted


def test_a_restart_mid_upload_asks_for_part_0_again_and_the_resend_is_remembered_once(
    huge_turn: tuple[dict[str, object], bytes],
) -> None:
    _payload, body = huge_turn
    parts = _parts(body, 16 * MiB)
    adapted: list[tuple[str | None, object]] = []
    with _client(adapted) as client:
        assert [response.status_code for response in _upload(client, parts[:2])] == [200, 200]
    # 本地服务重启：新进程的暂存是空的，接着送的那一块接不上
    with _client(adapted) as client:
        refused = client.post("/api/v1/memory/remember/parts", content=json.dumps(parts[2]), headers=_JSON)
        assert refused.status_code == 409
        error = refused.json()["error"]
        assert error["code"] == "ABORTED" and error["retryable"] is True
        assert error["details"] == {"conflict_type": "remember_parts_restart"}
        assert adapted == []
        # 插件整批从第 0 块重送
        assert [response.status_code for response in _upload(client, parts)] == [200] * len(parts)
    assert [delivery for delivery, _adaptation in adapted] == [_DELIVERY]


def test_a_failed_part_and_a_resend_from_part_0_leave_one_remember_with_the_same_delivery_id(
    huge_turn: tuple[dict[str, object], bytes],
) -> None:
    _payload, body = huge_turn
    parts = _parts(body, 16 * MiB)
    adapted: list[tuple[str | None, object]] = []
    with _client(adapted) as client:
        list(_upload(client, parts[:2]))  # 第 2 块在路上丢了
        assert [response.status_code for response in _upload(client, parts)] == [200] * len(parts)
        assert [delivery for delivery, _adaptation in adapted] == [_DELIVERY]
        # 最后一块的确认丢了，插件整批重送：同一个投递号再走一次 remember，由 Runtime 的投递幂等不重复记
        assert [response.status_code for response in _upload(client, parts)] == [200] * len(parts)
    assert [delivery for delivery, _adaptation in adapted] == [_DELIVERY, _DELIVERY]


def test_a_tampered_part_is_refused_and_nothing_is_remembered() -> None:
    body = _remember_body({"records": [{"text": "长" * 600}]})
    parts = _parts(body, 1024 + 256)
    assert len(parts) >= 2
    # 改的那几个字节仍是合法 JSON、长度不变：只有摘要能认出它
    chunk = base64.b64decode(str(parts[-1]["data"]))
    altered = chunk.replace(b"T08:00", b"T09:00")
    assert altered != chunk and len(altered) == len(chunk)
    tampered = dict(parts[-1], data=base64.b64encode(altered).decode())
    adapted: list[tuple[str | None, object]] = []
    with _client(adapted, max_request_bytes=1024 + 256) as client:
        responses = list(_upload(client, [*parts[:-1], tampered]))
    assert responses[-1].status_code == 400
    assert responses[-1].json()["error"]["code"] == "INVALID_ARGUMENT"
    assert "body_sha256" in responses[-1].json()["error"]["message"]
    assert adapted == []


def test_a_batch_larger_than_the_staging_limit_is_refused_for_good() -> None:
    body = _remember_body({"records": [{"text": "长" * 3000}]})
    adapted: list[tuple[str | None, object]] = []
    with _client(adapted, max_request_bytes=1024, remember_parts_max_bytes=4096) as client:
        response = client.post(
            "/api/v1/memory/remember/parts", content=json.dumps(_parts(body, 1024)[0]), headers=_JSON
        )
    assert response.status_code == 413
    assert response.json()["error"]["retryable"] is False
    assert "remember_parts_max_bytes" in response.json()["error"]["message"]


def _part(delivery_id: str, index: int, count: int, chunk: bytes, body: bytes) -> RememberPart:
    return RememberPart(
        delivery_id=delivery_id,
        part_index=index,
        part_count=count,
        body_bytes=len(body),
        body_sha256=hashlib.sha256(body).hexdigest(),
        data=base64.b64encode(chunk).decode("ascii"),
    )


def test_staged_parts_expire_after_their_ttl_and_free_their_bytes() -> None:
    asyncio.run(_staged_parts_expire())


async def _staged_parts_expire() -> None:
    clock = _Clock()
    staging = RememberPartsStaging(max_bytes=1024, ttl_seconds=60, clock=clock)
    body = b"0123456789"
    assert await staging.accept(_part("c" * 64, 0, 2, body[:5], body)) is None
    assert staging.staged_bytes == 5 and staging.pending_deliveries == ("c" * 64,)
    clock.now = 61
    with pytest.raises(RememberPartsRestartError):
        await staging.accept(_part("c" * 64, 1, 2, body[5:], body))
    assert staging.staged_bytes == 0 and staging.pending_deliveries == ()
    # 另一个投递号的过期暂存也在下一次收块时清掉
    assert await staging.accept(_part("d" * 64, 0, 2, body[:5], body)) is None
    clock.now = 200
    assert await staging.accept(_part("e" * 64, 0, 2, body[:5], body)) is None
    assert staging.pending_deliveries == ("e" * 64,)
    assert await staging.accept(_part("e" * 64, 1, 2, body[5:], body)) == body
    assert staging.staged_bytes == 0


def test_staging_never_holds_more_than_its_byte_limit() -> None:
    asyncio.run(_staging_is_bounded())


async def _staging_is_bounded() -> None:
    staging = RememberPartsStaging(max_bytes=12, ttl_seconds=60, clock=_Clock())
    body = b"0123456789"
    assert await staging.accept(_part("c" * 64, 0, 2, body[:8], body)) is None
    with pytest.raises(RememberPartsCapacityError):
        await staging.accept(_part("d" * 64, 0, 2, body[:8], body))
    assert staging.staged_bytes == 8 and staging.pending_deliveries == ("c" * 64,)


def test_the_capabilities_declare_remember_parts_and_the_staging_limits_are_configurable() -> None:
    adapted: list[tuple[str | None, object]] = []
    with _client(adapted) as client:
        features = client.get("/api/v1/capabilities").json()["result"]["features"]
    assert "remember_parts_v1" in features
    with pytest.raises(ValueError, match="remember_parts_max_bytes"):
        HTTPAPIConfig(max_request_bytes=4096, remember_parts_max_bytes=2048)
    with pytest.raises(ValueError, match="remember_parts_ttl_seconds"):
        HTTPAPIConfig(remember_parts_ttl_seconds=0)


def _small_codex_turn(text: str) -> dict[str, object]:
    return {
        "records": [
            {"type": "event_msg", "payload": {"type": "task_started"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "好"}],
                },
            },
        ]
    }


def _small_parts(*, delivery_id: str = _DELIVERY, body_delivery_id: str = _DELIVERY) -> list[dict[str, object]]:
    """一轮约 1.5 KB，按 1 KiB 的请求上限切成 4 块。"""

    body = _remember_body(_small_codex_turn("长" * 300), delivery_id=body_delivery_id)
    parts = _parts(body, 1024, delivery_id=delivery_id)
    assert len(parts) == 4 and all(len(json.dumps(part)) <= 1024 for part in parts)
    return parts


def test_a_repeated_part_asks_for_part_0_again_instead_of_failing_the_digest() -> None:
    """同一块送了两遍（插件重试了一块）：接不上，409 ABORTED 可重试、让插件从第 0 块重送，不是 400 摘要不符。"""

    parts = _small_parts()
    adapted: list[tuple[str | None, object]] = []
    with _client(adapted, max_request_bytes=1024, remember_parts_max_bytes=4096) as client:
        responses = list(_upload(client, [parts[0], parts[1], parts[1]]))
        assert [response.status_code for response in responses] == [200, 200, 409]
        error = responses[-1].json()["error"]
        assert error["code"] == "ABORTED" and error["retryable"] is True
        assert error["details"] == {"conflict_type": "remember_parts_restart"}
        assert [response.status_code for response in _upload(client, parts)] == [200] * len(parts)
    assert [delivery for delivery, _adaptation in adapted] == [_DELIVERY]


def test_full_staging_is_retryable_and_leaves_the_other_upload_intact() -> None:
    """暂存被另一个上传占满：429 RESOURCE_EXHAUSTED 可重试（插件稍后重送，不把这一项标成 blocked）。"""

    first = _small_parts()
    second = _small_parts(delivery_id="c" * 64, body_delivery_id="c" * 64)
    adapted: list[tuple[str | None, object]] = []
    with _client(adapted, max_request_bytes=1024, remember_parts_max_bytes=2048) as client:
        assert [response.status_code for response in _upload(client, first[:3])] == [200] * 3
        responses = list(_upload(client, second[:3]))
        assert [response.status_code for response in responses] == [200, 200, 429]
        error = responses[-1].json()["error"]
        assert error["code"] == "RESOURCE_EXHAUSTED" and error["retryable"] is True
        assert error["details"] == {"conflict_type": "remember_parts_capacity"}
        # 被拒的那个上传放掉了暂存，先来的那个照常收齐
        finished = client.post("/api/v1/memory/remember/parts", content=json.dumps(first[3]), headers=_JSON)
    assert finished.status_code == 200, finished.text[:500]
    assert finished.json()["result"]["remember"] is not None
    assert [delivery for delivery, _adaptation in adapted] == [_DELIVERY]


def test_an_assembled_request_with_another_delivery_id_is_refused() -> None:
    parts = _small_parts(body_delivery_id="c" * 64)
    adapted: list[tuple[str | None, object]] = []
    with _client(adapted, max_request_bytes=1024, remember_parts_max_bytes=4096) as client:
        responses = list(_upload(client, parts))
    assert [response.status_code for response in responses] == [200, 200, 200, 400]
    error = responses[-1].json()["error"]
    assert error["code"] == "INVALID_ARGUMENT" and error["retryable"] is False
    assert "delivery_id" in error["message"]
    assert adapted == []


def test_twenty_thousand_small_parts_stage_in_bounded_time() -> None:
    """暂存字节数随收块增减，不每块从头累加：2 万个 16 字节的块在 1 秒内收完（从头累加时约 6 秒）。"""

    asyncio.run(_many_small_parts())


async def _many_small_parts() -> None:
    count, size = 20_000, 16
    body = b"x" * (count * size)
    first = _part("c" * 64, 0, count, b"x" * size, body)
    parts = [RememberPart(**{**first.__dict__, "part_index": index}) for index in range(count)]
    staging = RememberPartsStaging(max_bytes=64 * MiB, ttl_seconds=600, clock=_Clock())
    started = time.perf_counter()
    for part in parts[:-1]:
        assert await staging.accept(part) is None
    assert staging.staged_bytes == (count - 1) * size
    assert await staging.accept(parts[-1]) == body
    assert time.perf_counter() - started < 1.0
    assert staging.staged_bytes == 0


def test_a_part_carrying_more_than_the_declared_body_is_refused_and_released() -> None:
    asyncio.run(_over_declared_part())


async def _over_declared_part() -> None:
    staging = RememberPartsStaging(max_bytes=1024, ttl_seconds=60, clock=_Clock())
    body = b"0123456789"
    with pytest.raises(RememberPartsIntegrityError, match="body_bytes"):
        await staging.accept(_part("c" * 64, 0, 2, body + b"!", body))
    assert staging.staged_bytes == 0 and staging.pending_deliveries == ()
    assert await staging.accept(_part("c" * 64, 0, 2, body[:6], body)) is None
    with pytest.raises(RememberPartsIntegrityError, match="body_bytes"):
        await staging.accept(_part("c" * 64, 1, 2, body[6:] + b"!", body))
    assert staging.staged_bytes == 0 and staging.pending_deliveries == ()


def test_a_skipped_part_or_a_part_of_another_batch_asks_for_part_0_again() -> None:
    asyncio.run(_parts_that_do_not_follow())


async def _parts_that_do_not_follow() -> None:
    staging = RememberPartsStaging(max_bytes=1024, ttl_seconds=60, clock=_Clock())
    body = b"0123456789"
    assert await staging.accept(_part("c" * 64, 0, 3, body[:3], body)) is None
    with pytest.raises(RememberPartsRestartError):
        await staging.accept(_part("c" * 64, 2, 3, body[3:6], body))
    assert staging.staged_bytes == 0 and staging.pending_deliveries == ()
    # 同一个投递号、同样长度但摘要不同的另一批：接不上
    other = b"9876543210"
    assert await staging.accept(_part("c" * 64, 0, 2, body[:5], body)) is None
    with pytest.raises(RememberPartsRestartError):
        await staging.accept(_part("c" * 64, 1, 2, other[5:], other))
    assert staging.staged_bytes == 0 and staging.pending_deliveries == ()


def test_a_resend_from_part_0_replaces_the_stale_staging_instead_of_filling_it() -> None:
    asyncio.run(_resend_from_part_0())


async def _resend_from_part_0() -> None:
    staging = RememberPartsStaging(max_bytes=12, ttl_seconds=60, clock=_Clock())
    body = b"0123456789"
    assert await staging.accept(_part("c" * 64, 0, 2, body[:8], body)) is None
    # 插件整批重送：旧的 8 字节作废，不和新的 8 字节一起算进 12 字节的上限
    assert await staging.accept(_part("c" * 64, 0, 2, body[:8], body)) is None
    assert staging.staged_bytes == 8
    assert await staging.accept(_part("c" * 64, 1, 2, body[8:], body)) == body
    assert staging.staged_bytes == 0


def test_each_part_extends_the_upload_ttl() -> None:
    asyncio.run(_ttl_counts_from_the_last_part())


async def _ttl_counts_from_the_last_part() -> None:
    clock = _Clock()
    staging = RememberPartsStaging(max_bytes=1024, ttl_seconds=60, clock=clock)
    body = b"0123456789"
    assert await staging.accept(_part("c" * 64, 0, 3, body[:3], body)) is None
    clock.now = 50
    assert await staging.accept(_part("c" * 64, 1, 3, body[3:6], body)) is None
    # 离第 0 块已 100 秒、离上一块 50 秒：没过期
    clock.now = 100
    assert await staging.accept(_part("c" * 64, 2, 3, body[6:], body)) == body


@pytest.mark.parametrize("data", ["MDEy!!!!", "MDEy\nMzQ1", "MDEy MzQ1"])
def test_part_data_must_be_strict_base64(data: str) -> None:
    asyncio.run(_strict_base64(data))


async def _strict_base64(data: str) -> None:
    staging = RememberPartsStaging(max_bytes=1024, ttl_seconds=60, clock=_Clock())
    part = _part("c" * 64, 0, 1, b"012345", b"012345")
    with pytest.raises(RememberPartsIntegrityError, match="base64"):
        await staging.accept(RememberPart(**{**part.__dict__, "data": data}))
    assert staging.staged_bytes == 0
