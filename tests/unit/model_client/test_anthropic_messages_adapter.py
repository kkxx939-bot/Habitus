"""Anthropic Messages 协议适配器的请求编码、响应解码、流式、结构化输出、健康探测与错误映射契约。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from habitus.model_client import (
    ChatClient,
    ChatMessage,
    ChatModelConfig,
    ChatRequest,
    ModelConfigurationError,
    ModelContentSafetyError,
    ModelInputTooLargeError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTransportError,
    ProviderConfig,
    ProviderFactory,
    ReasoningOptions,
    ResponseFormat,
    StructuredChatClient,
    ToolCall,
    ToolDefinition,
)
from habitus.model_client.adapters import register_builtin_adapters
from habitus.model_client.adapters._http import HTTPStatusError
from habitus.model_client.adapters.anthropic_messages import AnthropicMessagesProvider
from habitus.model_client.adapters.anthropic_messages_schema import messages_output_schema
from habitus.model_client.retry import normalize_provider_error

Handler = Callable[[httpx.Request], httpx.Response]


def chat_config(**overrides: object) -> ChatModelConfig:
    route_values: dict[str, object] = {
        "provider": "test-provider",
        "adapter": "anthropic_messages",
        "model": "test-model",
        "base_url": "https://example.com",
        "max_retries": 0,
    }
    model_values: dict[str, object] = {"max_output_tokens": 512}
    for key, value in overrides.items():
        if key in {"max_output_tokens", "structured_output_mode", "reasoning", "context_window_tokens"}:
            model_values[key] = value
        else:
            route_values[key] = value
    return ChatModelConfig(ProviderConfig(**route_values), **model_values)


def request(**overrides: object) -> ChatRequest:
    values: dict[str, object] = {"messages": (ChatMessage(role="user", content="hello"),)}
    values.update(overrides)
    return ChatRequest(**values)


def message_payload(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "served-model",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    value.update(overrides)
    return value


def sync_provider(
    handler: Handler,
    *,
    config: ChatModelConfig | None = None,
    api_key: str = "",
) -> AnthropicMessagesProvider:
    provider = AnthropicMessagesProvider(config or chat_config(), api_key=api_key)
    provider._sync_client.close()
    provider._sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    return provider


def body_of(provider: AnthropicMessagesProvider, logical_request: ChatRequest, *, stream: bool = False) -> dict:
    return json.loads(provider.prepare(logical_request, stream=stream).body)


def sse(*events: dict[str, object]) -> bytes:
    """按 Messages SSE 的真实形状拼出 ``event:`` + ``data:`` 帧。"""

    frames = [f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events]
    return "".join(frames).encode("utf-8")


def _start(input_tokens: int = 4, cache_read: int = 0) -> dict[str, object]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "served-model",
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 1, "cache_read_input_tokens": cache_read},
        },
    }


def _text_block(index: int, *texts: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}
    ]
    events.extend(
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}} for text in texts
    )
    events.append({"type": "content_block_stop", "index": index})
    return events


def _end(stop_reason: str = "end_turn", output_tokens: int = 7) -> list[dict[str, object]]:
    return [
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
        {"type": "message_stop"},
    ]


# ---- 注册与构造 ----


def test_factory_builds_anthropic_messages_chat_provider_from_builtin_registry() -> None:
    factory = ProviderFactory()
    register_builtin_adapters(factory)
    assert factory.registered_adapters("chat") == ("anthropic_messages", "openai_compatible_chat")
    provider = factory.create_chat_provider(chat_config(credential_ref="chat"), api_key="secret")
    try:
        assert isinstance(provider, AnthropicMessagesProvider)
        assert provider.provider_name == "test-provider"
        assert provider.model == "test-model"
        assert provider.capabilities.structured_output_mode == "none"
        assert callable(provider.health_check_async)
    finally:
        asyncio.run(provider.aclose())


@pytest.mark.parametrize(
    ("base_url", "endpoint", "remote"),
    [
        ("https://api.example.com", "https://api.example.com/v1/messages", True),
        ("https://api.example.com/", "https://api.example.com/v1/messages", True),
        ("https://api.example.com/v1", "https://api.example.com/v1/messages", True),
        ("https://api.example.com/v1/", "https://api.example.com/v1/messages", True),
        ("https://gw.example.com/anthropic", "https://gw.example.com/anthropic/v1/messages", True),
        ("https://gw.example.com/anthropic/v1", "https://gw.example.com/anthropic/v1/messages", True),
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000/v1/messages", False),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/messages", False),
    ],
)
def test_endpoint_joins_base_url_without_double_slash_or_missing_version(
    base_url: str, endpoint: str, remote: bool
) -> None:
    seen: list[str] = []

    def handler(item: httpx.Request) -> httpx.Response:
        seen.append(str(item.url))
        return httpx.Response(200, json=message_payload())

    provider = sync_provider(handler, config=chat_config(base_url=base_url))
    try:
        provider.complete(provider.prepare(request(), stream=False))
        assert seen == [endpoint]
        assert provider.is_remote is remote
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"adapter": "openai_compatible_chat"}, "adapter='anthropic_messages'"),
        ({"base_url": ""}, "explicit base_url"),
        ({"max_output_tokens": None}, "max_output_tokens"),
        ({"structured_output_mode": "json_object"}, "json_object"),
        ({"extra_body": {"messages": []}}, "cannot override"),
        ({"extra_body": {"system": "x"}}, "cannot override"),
        ({"extra_body": {"max_tokens": 9}}, "cannot override"),
        ({"extra_body": {"stream": True}}, "cannot override"),
        ({"extra_body": {"tools": []}}, "cannot override"),
        ({"extra_body": {"tool_choice": {}}}, "cannot override"),
        ({"extra_body": {"temperature": 0.2}}, "temperature"),
        ({"extra_body": {"top_p": 0.5}}, "top_p"),
        ({"extra_body": {"top_k": 3}}, "top_k"),
        ({"extra_body": {"output_config": {"format": {}}}}, "output_config.format"),
        ({"extra_body": {"output_config": "high"}}, "output_config must be an object"),
        ({"extra_headers": {"X-Api-Key": "leak"}}, "x-api-key"),
        ({"extra_headers": {"Anthropic-Version": "2023-06-01"}}, "anthropic-version"),
    ],
)
def test_construction_rejects_configuration_the_protocol_cannot_honor(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ModelConfigurationError, match=message):
        AnthropicMessagesProvider(chat_config(**overrides))


# ---- 请求编码 ----


def test_request_uses_messages_headers_and_shape_without_sampling_fields() -> None:
    captured: list[httpx.Request] = []

    def handler(item: httpx.Request) -> httpx.Response:
        captured.append(item)
        return httpx.Response(200, json=message_payload())

    config = chat_config(extra_headers={"anthropic-beta": "some-beta"}, extra_body={"metadata": {"user_id": "u"}})
    provider = sync_provider(handler, config=config, api_key=" secret ")
    logical = request(
        messages=(
            ChatMessage(role="system", content="rule one"),
            ChatMessage(role="developer", content="rule two"),
            ChatMessage(role="user", content="first"),
            ChatMessage(role="user", content="second"),
            ChatMessage(role="assistant", content="answer"),
            ChatMessage(role="user", content="third"),
        ),
        temperature=0.0,
    )
    try:
        provider.complete(provider.prepare(logical, stream=False))
    finally:
        provider._sync_client.close()
    sent = captured[0]
    assert sent.method == "POST"
    assert sent.headers["x-api-key"] == "secret"
    assert sent.headers["anthropic-version"] == "2026-10-21"
    assert sent.headers["anthropic-beta"] == "some-beta"
    assert sent.headers["content-type"] == "application/json"
    assert "authorization" not in sent.headers
    body = json.loads(sent.content)
    assert body == {
        "model": "test-model",
        "max_tokens": 512,
        "metadata": {"user_id": "u"},
        "system": [{"type": "text", "text": "rule one"}, {"type": "text", "text": "rule two"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "third"}]},
        ],
    }


def test_blank_api_key_sends_no_credential_header() -> None:
    captured: list[httpx.Request] = []

    def handler(item: httpx.Request) -> httpx.Response:
        captured.append(item)
        return httpx.Response(200, json=message_payload())

    provider = sync_provider(handler, api_key="  ")
    try:
        provider.complete(provider.prepare(request(), stream=False))
    finally:
        provider._sync_client.close()
    assert "x-api-key" not in captured[0].headers
    assert captured[0].headers["anthropic-version"] == "2026-10-21"


@pytest.mark.parametrize(
    "messages",
    [
        (ChatMessage(role="user", content="u"), ChatMessage(role="system", content="late rule")),
        (
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="u"),
            ChatMessage(role="developer", content="late rule"),
        ),
    ],
)
def test_system_message_after_conversation_starts_is_rejected_not_moved_or_dropped(
    messages: tuple[ChatMessage, ...],
) -> None:
    provider = AnthropicMessagesProvider(chat_config())
    with pytest.raises(ModelConfigurationError, match="system"):
        provider.prepare(request(messages=messages), stream=False)


def test_message_name_has_no_protocol_slot_and_is_rejected() -> None:
    provider = AnthropicMessagesProvider(chat_config())
    with pytest.raises(ModelConfigurationError, match="name"):
        provider.prepare(request(messages=(ChatMessage(role="user", content="u", name="alice"),)), stream=False)


def test_request_output_limit_overrides_route_default() -> None:
    provider = AnthropicMessagesProvider(chat_config())
    assert body_of(provider, request())["max_tokens"] == 512
    prepared = provider.prepare(request(max_output_tokens=64), stream=False)
    assert json.loads(prepared.body)["max_tokens"] == 64
    assert prepared.reserved_output_tokens == 64


def test_stream_flag_is_only_sent_for_stream_requests() -> None:
    provider = AnthropicMessagesProvider(chat_config())
    assert "stream" not in body_of(provider, request())
    assert body_of(provider, request(), stream=True)["stream"] is True


def test_tool_definitions_calls_results_and_choice_use_messages_blocks() -> None:
    provider = AnthropicMessagesProvider(chat_config())
    tool = ToolDefinition(
        name="lookup",
        description="Look something up",
        parameters={"type": "object", "properties": {"q": {"type": "string", "maxLength": 9}}, "required": ["q"]},
        strict=True,
    )
    loose = ToolDefinition(name="peek", description="Peek", parameters={"type": "object", "properties": {}})
    logical = request(
        messages=(
            ChatMessage(role="user", content="find"),
            ChatMessage(
                role="assistant",
                content="checking",
                tool_calls=(
                    ToolCall(id="toolu_1", name="lookup", arguments={"q": "a"}),
                    ToolCall(id="toolu_2", name="peek", arguments={}),
                ),
            ),
            ChatMessage(role="tool", tool_call_id="toolu_1", content="result one"),
            ChatMessage(role="tool", tool_call_id="toolu_2", content="result two"),
        ),
        tools=(tool, loose),
        tool_choice="required",
    )
    body = body_of(provider, logical)
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "find"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "a"}},
                {"type": "tool_use", "id": "toolu_2", "name": "peek", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result one"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "result two"},
            ],
        },
    ]
    strict_tool, loose_tool = body["tools"]
    assert strict_tool["strict"] is True
    assert strict_tool["input_schema"] == {
        "type": "object",
        "properties": {"q": {"type": "string", "description": "{maxLength: 9}"}},
        "required": ["q"],
        "additionalProperties": False,
    }
    assert loose_tool == {"name": "peek", "description": "Peek", "input_schema": {"type": "object", "properties": {}}}
    assert body["tool_choice"] == {"type": "any"}


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ("required", {"type": "any"}),
        ({"type": "function", "function": {"name": "lookup"}}, {"type": "tool", "name": "lookup"}),
        (None, None),
    ],
)
def test_tool_choice_maps_to_messages_vocabulary(choice: object, expected: object) -> None:
    provider = AnthropicMessagesProvider(chat_config())
    tool = ToolDefinition(name="lookup", description="Look", parameters={"type": "object"})
    body = body_of(provider, request(tools=(tool,), tool_choice=choice))
    assert body.get("tool_choice") == expected


@pytest.mark.parametrize("choice", ["sometimes", {"type": "function"}, {"type": "web"}])
def test_unknown_tool_choice_is_rejected(choice: object) -> None:
    provider = AnthropicMessagesProvider(chat_config())
    tool = ToolDefinition(name="lookup", description="Look", parameters={"type": "object"})
    with pytest.raises(ModelConfigurationError, match="tool_choice"):
        provider.prepare(request(tools=(tool,), tool_choice=choice), stream=False)


# ---- 结构化输出与推理 ----

_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string", "minLength": 1}, "count": {"type": "integer", "minimum": 0}},
    "required": ["name", "count"],
}


def test_json_schema_mode_sends_output_config_format_with_protocol_subset() -> None:
    provider = AnthropicMessagesProvider(chat_config(structured_output_mode="json_schema"))
    prepared = provider.prepare(request(response_format=ResponseFormat(name="result", schema=_SCHEMA)), stream=False)
    body = json.loads(prepared.body)
    assert body["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "{minLength: 1}"},
                    "count": {"type": "integer", "description": "{minimum: 0}"},
                },
                "required": ["name", "count"],
                "additionalProperties": False,
            },
        }
    }
    visible = json.loads(prepared.model_visible_body)
    assert visible["output_config"] == body["output_config"]
    assert "model" not in visible and "max_tokens" not in visible


def test_none_mode_sends_no_output_constraint() -> None:
    provider = AnthropicMessagesProvider(chat_config(structured_output_mode="none"))
    body = body_of(provider, request(response_format=ResponseFormat(name="result", schema=_SCHEMA)))
    assert "output_config" not in body


def test_route_output_config_merges_with_format_and_reasoning_effort() -> None:
    provider = AnthropicMessagesProvider(
        chat_config(structured_output_mode="json_schema", reasoning=True, extra_body={"output_config": {"x": 1}})
    )
    body = body_of(
        provider,
        request(
            response_format=ResponseFormat(name="r", schema={"type": "object", "properties": {}}),
            reasoning=ReasoningOptions(effort="high"),
        ),
    )
    assert body["output_config"] == {
        "x": 1,
        "effort": "high",
        "format": {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


@pytest.mark.parametrize(
    ("config", "reasoning", "message"),
    [
        (chat_config(), ReasoningOptions(effort="low"), "does not enable reasoning"),
        (chat_config(reasoning=True), ReasoningOptions(effort="minimal"), "minimal"),
        (
            chat_config(reasoning=True, extra_body={"output_config": {"effort": "max"}}),
            ReasoningOptions(effort="low"),
            "effort",
        ),
    ],
)
def test_reasoning_effort_rejects_what_the_route_cannot_express(
    config: ChatModelConfig, reasoning: ReasoningOptions, message: str
) -> None:
    provider = AnthropicMessagesProvider(config)
    with pytest.raises(ModelConfigurationError, match=message):
        provider.prepare(request(reasoning=reasoning), stream=False)


def test_structured_client_validates_messages_response_against_original_schema() -> None:
    replies = iter(
        [
            {"content": [{"type": "text", "text": '{"name": "", "count": 1}'}]},
            {"content": [{"type": "text", "text": '{"name": "ok", "count": 1}'}]},
        ]
    )
    bodies: list[dict[str, object]] = []

    def handler(item: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(item.content))
        return httpx.Response(200, json=message_payload(**next(replies)))

    provider = sync_provider(handler, config=chat_config(structured_output_mode="json_schema"))
    structured = StructuredChatClient(ChatClient(provider.config, provider))
    schema = {**_SCHEMA, "additionalProperties": False}
    try:
        result = structured.complete_json("give me data", schema=schema, name="result")
    finally:
        provider._sync_client.close()
    # 第一次 name 为空违反原 Schema 的 minLength（只在客户端校验），运行层反馈后第二次合格
    assert result.value == {"name": "ok", "count": 1}
    assert result.validation_attempts == 2
    assert len(bodies) == 2
    assert bodies[0]["system"][0]["type"] == "text"
    assert bodies[1]["messages"][-1]["role"] == "user"


# ---- 响应解码 ----


@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("refusal", "content_filter"),
        ("pause_turn", "length"),
        ("model_context_window_exceeded", "length"),
        ("future_reason", "future_reason"),
    ],
)
def test_complete_merges_blocks_and_maps_stop_reason(stop_reason: str, finish_reason: str) -> None:
    payload = message_payload(
        content=[
            {"type": "thinking", "thinking": "plan", "signature": "sig"},
            {"type": "text", "text": "part one, "},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "part two"},
            {"type": "tool_use", "id": "toolu_9", "name": "lookup", "input": {"q": "x"}},
        ],
        stop_reason=stop_reason,
        usage={
            "input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 20,
            "output_tokens": 7,
        },
    )
    provider = sync_provider(lambda _item: httpx.Response(200, json=payload))
    try:
        response = provider.complete(provider.prepare(request(), stream=False))
    finally:
        provider._sync_client.close()
    assert response.content == "part one, part two"
    assert response.reasoning_content == "plan"
    assert response.tool_calls == (ToolCall(id="toolu_9", name="lookup", arguments={"q": "x"}),)
    assert response.finish_reason == finish_reason
    assert response.model == "served-model"
    assert response.provider == "test-provider"
    assert response.usage.input_tokens == 35
    assert response.usage.cached_input_tokens == 20
    assert response.usage.output_tokens == 7
    assert response.usage.total_tokens == 42


def test_refusal_without_any_output_is_content_safety_not_retryable() -> None:
    payload = message_payload(content=[], stop_reason="refusal")
    provider = sync_provider(lambda _item: httpx.Response(200, json=payload))
    try:
        with pytest.raises(ModelContentSafetyError):
            provider.complete(provider.prepare(request(), stream=False))
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    "payload",
    [
        message_payload(content=[]),
        message_payload(content=[{"type": "thinking", "thinking": "only thoughts"}], stop_reason="max_tokens"),
        message_payload(content="text"),
        message_payload(content=[{"type": "text", "text": 3}]),
        message_payload(content=[{"type": "tool_use", "id": "", "name": "x", "input": {}}]),
        message_payload(content=[{"type": "tool_use", "id": "t", "name": "x", "input": "{}"}]),
        {"type": "error", "error": {"type": "api_error", "message": "boom"}},
    ],
)
def test_complete_rejects_malformed_or_empty_messages(payload: dict[str, object]) -> None:
    provider = sync_provider(lambda _item: httpx.Response(200, json=payload))
    try:
        with pytest.raises(ModelResponseError):
            provider.complete(provider.prepare(request(), stream=False))
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    ("status", "retry_after", "expected_type", "expected_delay"),
    [
        (429, "2.5", ModelRateLimitError, 2.5),
        (500, None, ModelTransportError, None),
        (529, "1", ModelTransportError, 1.0),
    ],
)
def test_http_errors_carry_status_message_and_retry_after_and_are_sent_once(
    status: int, retry_after: str | None, expected_type: type, expected_delay: float | None
) -> None:
    calls = 0

    def handler(_item: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        headers = {} if retry_after is None else {"retry-after": retry_after}
        body = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded now"}}
        return httpx.Response(status, json=body, headers=headers)

    provider = sync_provider(handler)
    try:
        with pytest.raises(HTTPStatusError) as caught:
            provider.complete(provider.prepare(request(), stream=False))
    finally:
        provider._sync_client.close()
    assert calls == 1
    assert caught.value.status_code == status
    assert "Overloaded now" in str(caught.value)
    assert caught.value.retry_after == expected_delay
    normalized = normalize_provider_error(caught.value)
    assert isinstance(normalized, expected_type)
    assert normalized.retryable is True
    assert normalized.retry_after_seconds == expected_delay


def test_response_over_byte_bound_is_rejected() -> None:
    provider = sync_provider(
        lambda _item: httpx.Response(200, content=b"x" * 1025),
        config=chat_config(max_response_bytes=1024),
    )
    try:
        with pytest.raises(ModelResponseError, match="byte bound"):
            provider.complete(provider.prepare(request(), stream=False))
    finally:
        provider._sync_client.close()


# ---- 流式 ----


def _stream(wire: bytes, *, status: int = 200) -> tuple:
    provider = sync_provider(lambda _item: httpx.Response(status, content=wire))
    try:
        return tuple(provider.stream(provider.prepare(request(), stream=True)))
    finally:
        provider._sync_client.close()


def test_stream_decodes_text_thinking_usage_and_single_done() -> None:
    wire = sse(
        _start(input_tokens=4, cache_read=6),
        {"type": "ping"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "s"}},
        {"type": "content_block_stop", "index": 0},
        *_text_block(1, "Hel", "lo"),
        *_end("max_tokens", output_tokens=9),
    )
    events = _stream(wire)
    assert [event.kind for event in events] == ["reasoning_delta", "content_delta", "content_delta", "usage", "done"]
    assert "".join(event.content_delta or "" for event in events) == "Hello"
    usage = events[3].usage
    assert (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (10, 6, 9)
    assert events[-1].finish_reason == "length"


def test_stream_emits_tool_call_identity_then_argument_fragments() -> None:
    wire = sse(
        _start(),
        *_text_block(0, "checking"),
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
        },
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"q": '}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"a"}'}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_2", "name": "peek", "input": {}},
        },
        {"type": "content_block_stop", "index": 2},
        *_end("tool_use"),
    )
    events = _stream(wire)
    tool_events = [event for event in events if event.kind == "tool_call_delta"]
    assert [(e.tool_call_index, e.tool_call_id, e.tool_name, e.tool_arguments_delta) for e in tool_events] == [
        (0, "toolu_1", "lookup", None),
        (0, None, None, '{"q": '),
        (0, None, None, '"a"}'),
        (1, "toolu_2", "peek", None),
    ]
    assert events[-1].kind == "done"
    assert events[-1].finish_reason == "tool_calls"


@pytest.mark.parametrize(
    "wire",
    [
        sse(_start(), *_text_block(0, "partial")),
        sse(_start(), *_text_block(0, "partial"), {"type": "message_stop"}),
        # 有 message_delta（stop_reason）却没有 message_stop：连接在收尾前断开，仍算截断
        sse(_start(), *_text_block(0, "partial"), _end()[0]),
        sse(_start(), *_text_block(0, "x"), *_end(), *_text_block(1, "late")),
        sse(_start(), {"type": "content_block_delta", "index": 3, "delta": {"type": "text_delta", "text": "x"}}),
        # 块已经 content_block_stop 之后又来增量：乱序，拒绝
        sse(
            _start(),
            *_text_block(0, "x"),
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "AFTER_STOP"}},
            *_end(),
        ),
        sse(_start(), *_text_block(0, "x"), {"type": "content_block_stop", "index": 0}, *_end()),
        sse(_start(), *_end()),
        sse(
            _start(),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "t", "name": "n", "input": {}},
            },
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{"}},
            *_end("tool_use"),
        ),
        b"data: not-json\n\n",
        b"data: \xff\n\n",
    ],
)
def test_stream_rejects_truncated_out_of_order_or_malformed_events(wire: bytes) -> None:
    with pytest.raises(ModelResponseError):
        _stream(wire)


def test_stream_error_event_is_classified_like_the_matching_http_status() -> None:
    wire = sse(
        _start(),
        *_text_block(0, "x"),
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    with pytest.raises(HTTPStatusError) as caught:
        _stream(wire)
    assert caught.value.status_code == 529
    assert isinstance(normalize_provider_error(caught.value), ModelTransportError)


def test_stream_refusal_without_output_is_content_safety() -> None:
    with pytest.raises(ModelContentSafetyError):
        _stream(sse(_start(), *_end("refusal")))


def test_stream_http_error_is_raised_before_parsing_events() -> None:
    body = json.dumps({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}).encode()
    with pytest.raises(HTTPStatusError) as caught:
        _stream(body, status=429)
    assert caught.value.status_code == 429


def test_stream_over_byte_bound_is_rejected() -> None:
    wire = sse(_start(), *_text_block(0, "x" * 2000), *_end())
    provider = sync_provider(
        lambda _item: httpx.Response(200, content=wire), config=chat_config(max_response_bytes=1024)
    )
    try:
        with pytest.raises(ModelResponseError, match="byte bound"):
            tuple(provider.stream(provider.prepare(request(), stream=True)))
    finally:
        provider._sync_client.close()


# ---- 健康探测与异步 ----


def test_health_check_sends_one_minimal_messages_call() -> None:
    captured: list[httpx.Request] = []

    def handler(item: httpx.Request) -> httpx.Response:
        captured.append(item)
        return httpx.Response(200, json=message_payload(stop_reason="max_tokens"))

    provider = sync_provider(handler, api_key="k", config=chat_config(extra_body={"metadata": {"a": "b"}}))
    try:
        assert provider.health_check() == {"ok": True, "provider": "test-provider", "model": "test-model"}
    finally:
        provider._sync_client.close()
    assert str(captured[0].url) == "https://example.com/v1/messages"
    assert captured[0].headers["x-api-key"] == "k"
    body = json.loads(captured[0].content)
    assert body["model"] == "test-model"
    assert 0 < body["max_tokens"] <= 16
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "ping"}]}]
    assert "metadata" not in body


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (401, {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}),
        (200, {"type": "not-a-message"}),
    ],
)
def test_health_check_fails_on_error_status_or_non_message_payload(status: int, payload: dict) -> None:
    provider = sync_provider(lambda _item: httpx.Response(status, json=payload))
    try:
        with pytest.raises((HTTPStatusError, ModelResponseError)):
            provider.health_check()
    finally:
        provider._sync_client.close()


def test_async_complete_stream_and_health_share_protocol_semantics() -> None:
    def handler(item: httpx.Request) -> httpx.Response:
        body = json.loads(item.content)
        if body.get("stream"):
            return httpx.Response(200, content=sse(_start(), *_text_block(0, "x"), *_end()))
        return httpx.Response(200, json=message_payload())

    provider = AnthropicMessagesProvider(chat_config())

    async def invoke():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider._async_client = lambda: client
            completed = await provider.complete_async(provider.prepare(request(), stream=False))
            streamed = tuple([event async for event in provider.stream_async(provider.prepare(request(), stream=True))])
            health = await provider.health_check_async()
            return completed, streamed, health

    try:
        completed, streamed, health = asyncio.run(invoke())
    finally:
        provider._sync_client.close()
    assert completed.content == "done"
    assert [event.kind for event in streamed] == ["content_delta", "usage", "done"]
    assert health["ok"] is True


def test_closed_provider_refuses_new_requests() -> None:
    provider = AnthropicMessagesProvider(chat_config())
    asyncio.run(provider.aclose())
    asyncio.run(provider.aclose())
    with pytest.raises(ModelConfigurationError, match="closed"):
        provider.prepare(request(), stream=False)


def test_prepared_stream_mode_must_match_operation() -> None:
    provider = AnthropicMessagesProvider(chat_config())
    try:
        with pytest.raises(ModelConfigurationError, match="stream mode"):
            provider.complete(provider.prepare(request(), stream=True))
    finally:
        provider._sync_client.close()


# ---- Schema 子集转换 ----


def test_schema_subset_moves_unsupported_constraints_into_description_without_mutating_input() -> None:
    original = {
        "title": "Batch",
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}, "minItems": 2, "maxItems": 5},
            "maybe": {"oneOf": [{"type": "string", "pattern": "^a"}, {"type": "null"}]},
            "when": {"type": "string", "format": "date-time"},
            "slug": {"type": "string", "format": "slug", "description": "short id"},
            "first": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"n": {"type": "number", "exclusiveMaximum": 3}},
                "required": ["n"],
                "additionalProperties": False,
            },
            "Ref": {"$ref": "#/$defs/Item", "description": "dropped sibling"},
        },
        "$schema": "https://json-schema.org/draft/2020-12/schema",
    }
    snapshot = json.loads(json.dumps(original))
    converted = messages_output_schema(original)
    assert original == snapshot
    assert converted == {
        "title": "Batch",
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}, "description": "{minItems: 2, maxItems: 5}"},
            "maybe": {"anyOf": [{"type": "string", "description": '{pattern: "^a"}'}, {"type": "null"}]},
            "when": {"type": "string", "format": "date-time"},
            "slug": {"type": "string", "description": 'short id\n\n{format: "slug"}'},
            "first": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"n": {"type": "number", "description": "{exclusiveMaximum: 3}"}},
                "required": ["n"],
                "additionalProperties": False,
            },
            "Ref": {"$ref": "#/$defs/Item"},
        },
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"type": "object", "additionalProperties": {"type": "string"}}, "additionalProperties"),
        ({"type": "object", "additionalProperties": True}, "additionalProperties"),
        ({"type": "object"}, "open object"),
        ({"type": "object", "properties": {"a": {"type": "string"}}, "patternProperties": {"^x": {}}}, "open object"),
        (
            {"type": "object", "properties": {"a": {"type": "string"}}, "propertyNames": {"pattern": "^a"}},
            "open object",
        ),
        ({"type": "object", "properties": {"self": {"$ref": "#"}}}, "recursive"),
        (
            {
                "$ref": "#/$defs/Node",
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {"child": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}},
                    }
                },
            },
            "recursive",
        ),
        ({"type": "object", "properties": {"x": {"$ref": "https://example.com/s.json"}}}, "external"),
        ({"type": "object", "properties": {"x": {"$ref": "#/$defs/Missing"}}}, "unknown"),
        ({"type": "object", "properties": {"x": 3}}, "schema"),
    ],
)
def test_schema_subset_rejects_shapes_the_protocol_cannot_express(schema: dict, message: str) -> None:
    with pytest.raises(ModelConfigurationError, match=message):
        messages_output_schema(schema)


@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (200, message_payload(), {"ok": True, "provider": "test-provider", "model": "test-model"}),
        (
            401,
            {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}},
            {"ok": False, "provider": "test-provider", "model": "test-model", "error_code": "MODEL_AUTHENTICATION"},
        ),
    ],
)
def test_doctor_chat_probe_path_reports_messages_health(status: int, payload: dict, expected: dict) -> None:
    """doctor 的 chat_probe 走 ``ChatClient.health_check_async``：成功给 ok，鉴权失败给稳定错误码。"""

    provider = AnthropicMessagesProvider(chat_config())
    client = ChatClient(provider.config, provider)

    async def invoke() -> dict[str, object]:
        transport = httpx.MockTransport(lambda _item: httpx.Response(status, json=payload))
        async with httpx.AsyncClient(transport=transport) as http:
            provider._async_client = lambda: http
            return await client.health_check_async()

    try:
        assert asyncio.run(invoke()) == expected
    finally:
        provider._sync_client.close()


def test_protocol_context_overflow_message_is_classified_as_input_too_large() -> None:
    """Messages 协议的上下文超长 400 文字是 "prompt is too long"：归到输入过长（不重试、走调用方的降级），不是服务端答坏。"""

    error = HTTPStatusError("prompt is too long: 210000 tokens > 200000 maximum", status_code=400, retry_after=None)
    assert isinstance(normalize_provider_error(error), ModelInputTooLargeError)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (message_payload(stop_reason="  "), "has no stop_reason"),
        (message_payload(stop_reason=1), "has no stop_reason"),
        (
            message_payload(content=[{"type": "tool_use", "id": " ", "name": "x", "input": {}}]),
            "tool_use needs an id and a name",
        ),
        (
            message_payload(content=[{"type": "tool_use", "id": "t", "name": ["x"], "input": {}}]),
            "tool_use needs an id and a name",
        ),
    ],
)
def test_complete_rejects_blank_or_non_text_identity_fields(payload: dict[str, object], message: str) -> None:
    """外部 Messages 响应里空白或非文本的 stop_reason、工具 id/name 在解码处被拒（W16）。"""

    provider = sync_provider(lambda _item: httpx.Response(200, json=payload))
    try:
        with pytest.raises(ModelResponseError, match=message):
            provider.complete(provider.prepare(request(), stream=False))
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("index", [True, "0", -1, 0.0])
def test_stream_rejects_bool_text_or_negative_block_index(index: object) -> None:
    wire = sse(
        _start(),
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
        *_end(),
    )
    with pytest.raises(ModelResponseError, match="content block index must be a non-negative integer"):
        _stream(wire)
