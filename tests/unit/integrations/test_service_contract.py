"""插件与本地服务之间的契约：服务端这一份逐项等于插件测试共用的 fixture。

分块上传（remember_parts_v1）：插件把 fixture 里的请求体切成的块逐字等于 fixture 的 parts，本地服务收下这几块拼回的
就是 fixture 的请求体。

插件那一半在 ``plugins/memory-plugin-shared/test/contract.test.mjs``，对着同一份 fixture 断言插件侧的常量；
任何一侧改了契约而另一侧没改，两条测试里至少有一条失败。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from habitus.config import HTTPAPIConfig
from habitus.foundation.network import is_loopback_host
from habitus.integrations.http_api import app as app_module
from habitus.integrations.http_api.handlers import RememberOutcome, RuntimeHTTPHandlers
from habitus.integrations.http_api.local_security import local_request_violation
from habitus.integrations.http_api.schemas import RememberResult
from habitus.memory.conversation.retention import ConversationToolResultReducer
from habitus.pre.conversation import (
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from habitus.pre.conversation.adapters import ConversationAdapterContext, ConversationAdapterRegistry
from habitus.pre.conversation.media import describe_media_only
from habitus.runtime import Runtime
from habitus.runtime.session_lane import _SUBAGENT_MARK

_FIXTURE = (
    Path(__file__).resolve().parents[3] / "plugins" / "memory-plugin-shared" / "test" / "fixtures" / "service-contract.json"
)
_CONTRACT: dict[str, Any] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
_LOOPBACK_CASES = [(case["host"], case["loopback"]) for case in _CONTRACT["loopback_hosts"]]
# 插件送之前把媒体载荷换成占位（plugins/memory-plugin-shared/lib/media.mjs）：turns 是各宿主一轮剥前、剥后的样子，
# media_only_tool_results 是只有媒体的工具结果剥前、剥后的样子与本地服务存下的 OMITTED 消息（omitted）；插件测试核"剥后"
# 是它的输出、omitted 的字节数与 sha256 是占位上的值按同一口径合成的
_MEDIA_FIXTURE: dict[str, Any] = json.loads((_FIXTURE.parent / "media-placeholder.json").read_text(encoding="utf-8"))
_MEDIA: dict[str, Any] = _MEDIA_FIXTURE["turns"]
_MEDIA_ONLY: list[dict[str, Any]] = _MEDIA_FIXTURE["media_only_tool_results"]


def _served_contract() -> dict[str, Any]:
    """服务真正对外给出的那一份：capabilities 走真 handler，协议表取内置适配器注册表。"""

    registry = ConversationAdapterRegistry.with_builtins()
    runtime = object.__new__(Runtime)
    runtime.conversation_protocols = registry.protocols  # type: ignore[method-assign]
    runtime.reminders_enabled = lambda: True  # type: ignore[method-assign]  # 预测层开着时的全套特性
    capabilities = RuntimeHTTPHandlers(runtime).capabilities()
    return {
        "api_version": capabilities.api_version,
        "features": sorted(capabilities.features),
        "protocols": {
            protocol: {
                "host": registry.host(protocol),
                "question_tools": [tool.tool for tool in registry.question_tools(protocol)],
            }
            for protocol in capabilities.protocols
        },
        "subagent_mark": _SUBAGENT_MARK,
        # 插件拿不到 capabilities 里的 max_request_bytes（老服务）时按这个默认上限拆批
        "default_max_request_bytes": HTTPAPIConfig().max_request_bytes,
    }


def test_the_service_side_of_the_plugin_contract_matches_the_shared_fixture() -> None:
    served = _served_contract()
    expected = {
        key: _CONTRACT[key]
        for key in ("api_version", "features", "protocols", "subagent_mark", "default_max_request_bytes")
    }
    expected["features"] = sorted(expected["features"])
    assert served == expected


def _authority(host: str) -> str:
    return f"[{host}]:8787" if ":" in host else f"{host}:8787"


@pytest.mark.parametrize(("host", "loopback"), _LOOPBACK_CASES)
def test_loopback_judgement_follows_the_shared_table(host: str, loopback: bool) -> None:
    assert is_loopback_host(host) is loopback

    if loopback:
        assert HTTPAPIConfig(host=host).host == host
    else:
        with pytest.raises(ValueError, match="loopback"):
            HTTPAPIConfig(host=host)

    host_violation = local_request_violation(host_header=_authority(host), origin_header=None)
    origin_violation = local_request_violation(host_header="127.0.0.1:8787", origin_header=f"http://{_authority(host)}")
    assert (host_violation is None) is loopback
    assert (origin_violation is None) is loopback


@pytest.mark.parametrize("protocol", sorted(_MEDIA))
def test_the_plugin_media_placeholders_leave_the_stored_conversation_unchanged(protocol: str) -> None:
    """经适配器再经工具结果保留策略：剥前、剥后存下的每条消息整条相同（只有媒体的工具结果存成 OMITTED，``message_id`` 也相同）。"""

    registry = ConversationAdapterRegistry.with_builtins()
    context = ConversationAdapterContext("media-placeholder", 0, datetime(2026, 10, 23, tzinfo=UTC))
    reducer = ConversationToolResultReducer()
    case = _MEDIA[protocol]
    stripped_text = json.dumps(case["stripped"])
    assert "base64" not in stripped_text and "data:image" not in stripped_text

    def stored(records: object) -> tuple[ConversationMessage, ...]:
        adaptation = registry.adapt(protocol, {"records": records}, context)
        return tuple(
            reducer.reduce(message) if message.role is ConversationMessageRole.TOOL_RESULT else message
            for message in adaptation.batch.messages
        )

    assert stored(case["stripped"]) == stored(case["raw"])


def test_the_media_only_fixture_covers_every_placeholder_type_the_plugin_sends() -> None:
    sent = {part["type"] for case in _MEDIA_ONLY for part in case["stripped"]}
    assert sent == {"image", "document", "input_image", "file"}


def _media_only_result(content: object) -> ConversationMessage:
    return ConversationMessage(
        "media-only",
        0,
        ConversationMessageRole.TOOL_RESULT,
        datetime(2026, 10, 23, tzinfo=UTC),
        content,
        tool_call_id="call-1",
        tool_name="Read",
        tool_status=ConversationToolResultStatus.COMPLETED,
        content_mode=ConversationToolResultContentMode.INLINE,
    )


@pytest.mark.parametrize("case", _MEDIA_ONLY, ids=[case["name"] for case in _MEDIA_ONLY])
def test_a_media_only_tool_result_reads_the_same_before_and_after_the_plugin_strips_it(case: dict[str, Any]) -> None:
    described = describe_media_only(case["stripped"])

    assert described is not None
    assert described == describe_media_only(case["raw"])


@pytest.mark.parametrize("case", _MEDIA_ONLY, ids=[case["name"] for case in _MEDIA_ONLY])
def test_a_media_only_tool_result_is_stored_field_for_field_the_same_with_or_without_the_plugin(case: dict[str, Any]) -> None:
    """Q6：经插件（占位）与不经插件（原载荷）存下的 OMITTED 消息逐字段相同，大小与摘要值都指原载荷。"""

    reducer = ConversationToolResultReducer()
    via_plugin = reducer.reduce(_media_only_result(case["stripped"]))
    direct = reducer.reduce(_media_only_result(case["raw"]))

    assert via_plugin.to_dict() == direct.to_dict()
    assert via_plugin.content_mode is ConversationToolResultContentMode.OMITTED
    assert {
        "content": via_plugin.content,
        "original_size_bytes": via_plugin.original_size_bytes,
        "original_sha256": via_plugin.original_sha256,
    } == case["omitted"]


def _claude_code_transcript(tool_result_content: object) -> list[dict[str, Any]]:
    timestamp = "2026-10-23T01:00:00.000Z"
    return [
        {"type": "user", "timestamp": timestamp, "message": {"role": "user", "content": [{"type": "text", "text": "截个图"}]}},
        {
            "type": "assistant",
            "timestamp": timestamp,
            "message": {
                "role": "assistant",
                "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "t1", "name": "Screenshot", "input": {}}],
            },
        },
        {
            "type": "user",
            "timestamp": timestamp,
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_result_content}],
            },
        },
        {
            "type": "assistant",
            "timestamp": timestamp,
            "message": {"role": "assistant", "stop_reason": "end_turn", "content": [{"type": "text", "text": "好了"}]},
        },
    ]


@pytest.mark.parametrize("case", _MEDIA_ONLY, ids=[case["name"] for case in _MEDIA_ONLY])
def test_a_media_only_tool_result_adapts_and_stores_the_same_message_with_or_without_the_plugin(
    case: dict[str, Any],
) -> None:
    """经适配器再经保留策略：原载荷与插件占位得到整条相同的消息，``message_id`` 也相同。"""

    registry = ConversationAdapterRegistry.with_builtins()
    context = ConversationAdapterContext("media-only", 0, datetime(2026, 10, 23, tzinfo=UTC))
    reducer = ConversationToolResultReducer()

    def stored(content: object) -> list[dict[str, object]]:
        batch = registry.adapt("claude_code", {"records": _claude_code_transcript(content)}, context).batch
        return [
            (reducer.reduce(message) if message.role is ConversationMessageRole.TOOL_RESULT else message).to_dict()
            for message in batch.messages
        ]

    direct, via_plugin = stored(case["raw"]), stored(case["stripped"])

    assert via_plugin == direct
    [result] = [message for message in via_plugin if message["role"] == "tool_result"]
    assert result["content_mode"] == "omitted"


@pytest.mark.parametrize(
    "part",
    [
        pytest.param({"type": "image", "omitted_bytes": -5, "omitted_sha256": "a" * 64}, id="negative-size"),
        pytest.param({"type": "image", "omitted_bytes": 10, "omitted_sha256": "not-a-digest"}, id="not-a-digest"),
        pytest.param({"type": "image", "text": "screenshot of the failing login page", "data": "AAAA"}, id="has-text"),
    ],
)
def test_a_part_that_is_not_media_is_kept_inline(part: dict[str, Any]) -> None:
    """占位要有非负字节数与 64 位十六进制摘要；带文字的段不当媒体，内容原样留下。"""

    assert describe_media_only([part]) is None
    original = _media_only_result([part])
    assert ConversationToolResultReducer().reduce(original) == original


@pytest.mark.parametrize("case", _MEDIA_FIXTURE["not_media"], ids=lambda case: case["name"])
def test_the_shared_not_media_cases_are_kept_inline(case: dict[str, Any]) -> None:
    """T21：共享 fixture 的 ``not_media`` 组（没有 type 的 data 对象、没有 type 也没有 mime 字段的 data URL）不当媒体，
    内容原样留下；插件那一半（``stripMedia`` 原样返回）在 ``request-limit.test.mjs``。"""

    content = case["content"]
    assert describe_media_only(content) is None
    original = _media_only_result(content)
    assert ConversationToolResultReducer().reduce(original) == original


def test_a_typed_file_part_with_a_data_url_takes_its_media_type_from_the_url() -> None:
    described = describe_media_only([{"type": "file", "url": "data:image/png;base64,AAAA"}])

    assert described is not None
    assert described.text == "工具结果只有媒体、没有文字，原始载荷未保存：file（image/png，载荷 26 字节）。"


class _RecordingHandlers:
    """只替代领域执行：记下 remember 收到的参数，回一个最小的结果。"""

    def __init__(self) -> None:
        self.remembered: list[dict[str, object]] = []

    async def remember(self, **values: object) -> RememberOutcome:
        self.remembered.append(values)
        return RememberOutcome(
            result=RememberResult(ignored_items=0, after_turn=True, next_sequence=5, jobs=[], consistency=[]),
            accepted_jobs=(),
        )


def test_the_service_assembles_the_fixture_parts_into_the_fixture_remember_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _CONTRACT["remember_parts"]
    assert case["feature"] in _served_contract()["features"]
    handlers = _RecordingHandlers()
    monkeypatch.setattr(app_module, "RuntimeHTTPHandlers", lambda _runtime: handlers)
    runtime = object.__new__(Runtime)
    runtime.start = AsyncMock()  # type: ignore[method-assign]
    runtime.close = AsyncMock()  # type: ignore[method-assign]
    runtime.components = SimpleNamespace(  # type: ignore[assignment]
        infrastructure=SimpleNamespace(observer=None, managed_observability=None)
    )
    app = app_module.create_http_app(runtime, config=HTTPAPIConfig(max_request_bytes=case["max_request_bytes"]))
    with TestClient(app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        responses = [
            client.post(
                case["path"],
                content=json.dumps(part, separators=(",", ":")),
                headers={"Content-Type": "application/json"},
            )
            for part in case["parts"]
        ]
    assert [response.status_code for response in responses] == [200] * len(case["parts"])
    assert responses[-1].json()["result"]["remember"]["next_sequence"] == 5
    sent = json.loads(case["body"])
    [received] = handlers.remembered
    assert received["delivery_id"] == sent["delivery_id"]
    assert received["payload"] == sent["payload"]
    assert received["conversation_id"] == sent["conversation_id"]
    assert received["start_sequence"] == sent["start_sequence"]
    assert received["after_turn"] is sent["after_turn"]
