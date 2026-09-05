"""真实 HTTP Adapter 的请求、响应、流式、安全和错误协议矩阵。"""

from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest

from habitus.model_client import (
    ChatCallContext,
    ChatClient,
    ChatMessage,
    ChatModelConfig,
    ChatRequest,
    EmbeddingModelConfig,
    ModelConfigurationError,
    ModelResponseError,
    ProviderConfig,
    ReasoningOptions,
    RerankModelConfig,
    ResponseFormat,
    StructuredChatClient,
    ToolCall,
    ToolDefinition,
)
from habitus.model_client.adapters.ark_multimodal import ArkMultimodalEmbeddingProvider
from habitus.model_client.adapters.openai_compatible_chat import OpenAICompatibleChatProvider
from habitus.model_client.adapters.openai_compatible_rerank import OpenAICompatibleRerankProvider


def chat_config(**overrides: object) -> ChatModelConfig:
    route_values: dict[str, object] = {
        "provider": "test-provider",
        "adapter": "openai_compatible_chat",
        "model": "test-model",
        "base_url": "https://example.com/v1",
        "max_retries": 0,
    }
    model_values: dict[str, object] = {}
    for key, value in overrides.items():
        if key in {"max_output_tokens", "structured_output_mode", "reasoning"}:
            model_values[key] = value
        else:
            route_values[key] = value
    return ChatModelConfig(ProviderConfig(**route_values), **model_values)


def embedding_config(**overrides: object) -> EmbeddingModelConfig:
    route_values: dict[str, object] = {
        "provider": "test-provider",
        "adapter": "ark_multimodal",
        "model": "embedding-model",
        "base_url": "https://example.com/api/v3",
        "max_retries": 0,
    }
    model_values: dict[str, object] = {
        "dimension": 2,
        "input_mode": "multimodal",
    }
    for key, value in overrides.items():
        if key in {
            "dimension",
            "input_mode",
            "max_batch_size",
            "max_input_chars",
            "query_parameters",
            "document_parameters",
        }:
            model_values[key] = value
        else:
            route_values[key] = value
    return EmbeddingModelConfig(ProviderConfig(**route_values), **model_values)


def rerank_config(**overrides: object) -> RerankModelConfig:
    route_values: dict[str, object] = {
        "provider": "test-provider",
        "adapter": "openai_compatible_rerank",
        "model": "rerank-model",
        "base_url": "https://example.com/v1",
        "max_retries": 0,
    }
    route_values.update(overrides)
    return RerankModelConfig(ProviderConfig(**route_values))


def request(**overrides: object) -> ChatRequest:
    values: dict[str, object] = {
        "messages": (ChatMessage(role="user", content="hello"),),
    }
    values.update(overrides)
    return ChatRequest(**values)


def prepared_payload(
    provider: OpenAICompatibleChatProvider,
    logical_request: ChatRequest,
    *,
    stream: bool,
) -> dict[str, object]:
    return json.loads(provider.prepare(logical_request, stream=stream).body)


def complete(
    provider: OpenAICompatibleChatProvider,
    logical_request: ChatRequest,
):
    return provider.complete(provider.prepare(logical_request, stream=False))


def stream(
    provider: OpenAICompatibleChatProvider,
    logical_request: ChatRequest,
):
    return provider.stream(provider.prepare(logical_request, stream=True))


def response_payload(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "model": "served-model",
        "choices": [
            {
                "message": {"content": "done"},
                "finish_reason": "stop",
            }
        ],
    }
    value.update(overrides)
    return value


def sync_provider(
    handler,
    *,
    config: ChatModelConfig | None = None,
    api_key: str = "",
) -> OpenAICompatibleChatProvider:
    provider = OpenAICompatibleChatProvider(config or chat_config(), api_key=api_key)
    provider._sync_client.close()
    provider._sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    return provider


@pytest.mark.parametrize(
    ("base_url", "expected_remote"),
    [
        ("https://example.com/v1", True),
        ("https://api.deepseek.com", True),
        ("http://localhost:8000/v1", False),
        ("http://127.0.0.1:8000/v1", False),
        ("http://[::1]:8000/v1", False),
    ],
)
def test_openai_adapter_derives_endpoint_identity_and_remote_flag(
    base_url: str,
    expected_remote: bool,
) -> None:
    provider = OpenAICompatibleChatProvider(chat_config(base_url=base_url))
    try:
        assert provider._endpoint == f"{base_url}/chat/completions"
        assert provider._models_endpoint == f"{base_url}/models"
        assert provider.is_remote is expected_remote
        assert provider.provider_name == "test-provider"
        assert provider.model == "test-model"
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("invalid", [None, "config", {}, [], 1, True, object()])
def test_openai_adapter_requires_chat_config(invalid: object) -> None:
    with pytest.raises(TypeError):
        OpenAICompatibleChatProvider(invalid)


@pytest.mark.parametrize("adapter", ["test", "ark_multimodal", "openai"])
def test_openai_adapter_rejects_wrong_adapter_identity(adapter: str) -> None:
    with pytest.raises(ModelConfigurationError, match="requires adapter"):
        OpenAICompatibleChatProvider(chat_config(adapter=adapter))


def test_openai_adapter_requires_explicit_base_url() -> None:
    with pytest.raises(ModelConfigurationError, match="explicit base_url"):
        OpenAICompatibleChatProvider(chat_config(base_url=""))


@pytest.mark.parametrize(
    "reserved",
    [
        "max_completion_tokens",
        "max_tokens",
        "messages",
        "model",
        "reasoning_effort",
        "response_format",
        "stream",
        "temperature",
        "tool_choice",
        "tools",
    ],
)
def test_openai_adapter_rejects_extra_body_request_field_override(reserved: str) -> None:
    with pytest.raises((ValueError, ModelConfigurationError), match="cannot override"):
        OpenAICompatibleChatProvider(chat_config(extra_body={reserved: "override"}))


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [("", None), ("   ", None), ("secret", "Bearer secret"), (" secret ", "Bearer secret")],
)
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
def test_openai_adapter_builds_headers_without_leaking_blank_credentials(
    api_key: str,
    expected: str | None,
    accept: str,
) -> None:
    provider = OpenAICompatibleChatProvider(
        chat_config(extra_headers={"X-Trace": "trace-1"}),
        api_key=api_key,
    )
    try:
        headers = provider._headers(accept=accept)
        assert headers["Accept"] == accept
        assert headers["Content-Type"] == "application/json"
        assert headers["X-Trace"] == "trace-1"
        assert headers.get("Authorization") == expected
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    "messages",
    [
        (ChatMessage(role="system", content="system"), ChatMessage(role="user", content="user")),
        (ChatMessage(role="developer", content="developer"), ChatMessage(role="assistant", content="assistant")),
        (ChatMessage(role="tool", content="result", tool_call_id="call-1", name="inspect"),),
        (
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=(ToolCall("call-1", "inspect", {"path": ".", "中文": True}),),
            ),
        ),
    ],
)
def test_openai_payload_preserves_every_message_role_and_tool_binding(
    messages: tuple[ChatMessage, ...],
) -> None:
    provider = OpenAICompatibleChatProvider(chat_config())
    try:
        payload = prepared_payload(provider, ChatRequest(messages=messages), stream=False)
        assert [item["role"] for item in payload["messages"]] == [item.role for item in messages]
        if messages[0].role == "tool":
            assert payload["messages"][0]["tool_call_id"] == "call-1"
            assert payload["messages"][0]["name"] == "inspect"
        if messages[0].tool_calls:
            arguments = payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
            assert json.loads(arguments) == {"path": ".", "中文": True}
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("choice", [None, "auto", "none", "required", {"type": "function"}])
def test_openai_payload_serializes_tool_schema_choice_and_strictness(
    strict: bool,
    choice: object,
) -> None:
    definition = ToolDefinition(
        "inspect",
        "检查工作区",
        {"type": "object", "properties": {"path": {"type": "string"}}},
        strict=strict,
    )
    provider = OpenAICompatibleChatProvider(chat_config())
    try:
        payload = prepared_payload(
            provider,
            request(tools=(definition,), tool_choice=choice),
            stream=False,
        )
        function = payload["tools"][0]["function"]
        assert function["name"] == "inspect"
        assert function["description"] == "检查工作区"
        assert function.get("strict") is (True if strict else None)
        assert payload["tool_choice"] == (choice or "auto")
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("mode", ["none", "json_object", "json_schema"])
def test_openai_payload_adapts_structured_output_to_route_capability(mode: str) -> None:
    provider = OpenAICompatibleChatProvider(chat_config(structured_output_mode=mode))
    format_ = ResponseFormat("memory", {"type": "object"}, strict=True)
    try:
        payload = prepared_payload(provider, request(response_format=format_), stream=False)
        if mode == "none":
            assert "response_format" not in payload
        elif mode == "json_object":
            assert payload["response_format"] == {"type": "json_object"}
        else:
            assert payload["response_format"]["json_schema"] == {
                "name": "memory",
                "strict": True,
                "schema": {"type": "object"},
            }
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_openai_payload_emits_reasoning_effort_and_omits_temperature(effort: str) -> None:
    provider = OpenAICompatibleChatProvider(chat_config(reasoning=True))
    try:
        payload = prepared_payload(
            provider,
            request(temperature=1.5, reasoning=ReasoningOptions(effort)),
            stream=False,
        )
        assert payload["reasoning_effort"] == effort
        assert "temperature" not in payload
    finally:
        provider._sync_client.close()


def test_openai_payload_rejects_reasoning_when_route_does_not_enable_it() -> None:
    provider = OpenAICompatibleChatProvider(chat_config(reasoning=False))
    try:
        with pytest.raises(ModelConfigurationError, match="does not enable reasoning"):
            prepared_payload(
                provider,
                request(reasoning=ReasoningOptions("medium")),
                stream=False,
            )
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    ("request_limit", "config_limit", "expected"),
    [(None, None, None), (None, 128, 128), (64, None, 64), (64, 128, 64)],
)
def test_openai_payload_uses_request_output_limit_before_route_default(
    request_limit: int | None,
    config_limit: int | None,
    expected: int | None,
) -> None:
    provider = OpenAICompatibleChatProvider(chat_config(max_output_tokens=config_limit))
    try:
        payload = prepared_payload(
            provider,
            request(max_output_tokens=request_limit),
            stream=False,
        )
        assert payload.get("max_tokens") == expected
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("stream", [True, False])
def test_openai_payload_preserves_safe_extra_body_and_adds_stream_options_only_for_stream(stream: bool) -> None:
    provider = OpenAICompatibleChatProvider(
        chat_config(extra_body={"top_p": 0.8, "seed": 7, "stream_options": {"include_usage": False}})
    )
    try:
        payload = prepared_payload(provider, request(), stream=stream)
        assert payload["top_p"] == 0.8
        assert payload["seed"] == 7
        if stream:
            assert payload["stream"] is True
            assert payload["stream_options"] == {"include_usage": False}
        else:
            assert "stream" not in payload
    finally:
        provider._sync_client.close()


def test_openai_prepare_freezes_transport_and_model_visible_projections_separately() -> None:
    plain = OpenAICompatibleChatProvider(chat_config())
    with_transport_metadata = OpenAICompatibleChatProvider(
        chat_config(extra_body={"seed": 7, "trace_options": {"request_id": "internal"}})
    )
    logical_request = request()
    try:
        plain_prepared = plain.prepare(logical_request, stream=False)
        metadata_prepared = with_transport_metadata.prepare(logical_request, stream=False)

        assert plain_prepared.model_visible_body == metadata_prepared.model_visible_body
        assert plain_prepared.estimated_input_tokens == metadata_prepared.estimated_input_tokens
        assert plain_prepared.body != metadata_prepared.body
        assert json.loads(metadata_prepared.body)["trace_options"] == {
            "request_id": "internal"
        }
    finally:
        plain._sync_client.close()
        with_transport_metadata._sync_client.close()


@pytest.mark.parametrize("mode", ["none", "json_object", "json_schema"])
def test_structured_prepare_counts_schema_only_where_selected_mode_sends_it(mode: str) -> None:
    provider = OpenAICompatibleChatProvider(chat_config(structured_output_mode=mode))
    structured = StructuredChatClient(ChatClient(provider.config, provider))
    schema = {"type": "object", "description": "unique-schema-marker"}
    try:
        logical_request = structured._prepare("return data", schema=schema, name="result")
        prepared = provider.prepare(logical_request, stream=False)
        visible = json.loads(prepared.model_visible_body)
        messages_text = json.dumps(visible["messages"], ensure_ascii=False)
        response_format_text = json.dumps(
            visible.get("response_format", {}),
            ensure_ascii=False,
        )

        if mode == "json_schema":
            assert "unique-schema-marker" not in messages_text
            assert "unique-schema-marker" in response_format_text
        else:
            assert "unique-schema-marker" in messages_text
            assert "unique-schema-marker" not in response_format_text
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("invalid", [None, "request", {}, [], 1, True, object()])
def test_openai_payload_requires_normalized_chat_request(invalid: object) -> None:
    provider = OpenAICompatibleChatProvider(chat_config())
    try:
        with pytest.raises(TypeError):
            provider.prepare(invalid, stream=False)
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain", "plain"),
        ([{"type": "text", "text": "a"}, {"type": "output_text", "text": "b"}], "ab"),
        ([{"type": "image", "url": "x"}, {"type": "text", "text": "kept"}], "kept"),
    ],
)
def test_openai_complete_normalizes_text_or_text_part_response(content: object, expected: str) -> None:
    provider = sync_provider(
        lambda _request: httpx.Response(
            200,
            json=response_payload(choices=[{"message": {"content": content}, "finish_reason": "stop"}]),
        )
    )
    try:
        response = ChatClient(provider.config, provider).complete(
            request(),
            context=ChatCallContext(prompt_version="prompt-v1"),
        )
        assert response.content == expected
        assert response.model == "served-model"
        assert response.provider == "test-provider"
        assert response.prompt_version == "prompt-v1"
        assert response.latency_ms >= 0
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"prompt_tokens": 2, "completion_tokens": 3},
        {"input_tokens": 2, "output_tokens": 3, "total_tokens": 9},
        {"prompt_tokens": -1, "completion_tokens": True, "total_tokens": "3"},
        {
            "prompt_tokens": 5,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
        {"prompt_tokens": 5, "prompt_cache_hit_tokens": 4},
    ],
)
def test_openai_complete_normalizes_usage_variants_without_type_coercion(usage: dict[str, object]) -> None:
    provider = sync_provider(lambda _request: httpx.Response(200, json=response_payload(usage=usage)))
    try:
        response = complete(provider, request())
        expected_input = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        expected_input = (
            expected_input
            if isinstance(expected_input, int) and not isinstance(expected_input, bool) and expected_input >= 0
            else 0
        )
        expected_output = usage.get("completion_tokens", usage.get("output_tokens", 0))
        expected_output = (
            expected_output
            if isinstance(expected_output, int) and not isinstance(expected_output, bool) and expected_output >= 0
            else 0
        )
        assert response.usage.input_tokens == expected_input
        assert response.usage.output_tokens == expected_output
        assert (
            response.usage.total_tokens >= expected_input + expected_output
            or usage.get("total_tokens") == response.usage.total_tokens
        )
        assert response.usage.details == usage
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("argument_shape", ['{"path":"."}', {"path": "."}, "{}"])
def test_openai_complete_normalizes_tool_call_arguments_from_text_or_mapping(argument_shape: object) -> None:
    payload = response_payload(
        choices=[
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "inspect", "arguments": argument_shape},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    provider = sync_provider(lambda _request: httpx.Response(200, json=payload))
    try:
        response = complete(provider, request())
        assert response.tool_calls[0].id == "call-1"
        assert response.tool_calls[0].name == "inspect"
        assert response.tool_calls[0].arguments == ({"path": "."} if argument_shape != "{}" else {})
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": 1}}]},
        {"choices": [{"message": {"content": []}}]},
        {"choices": [{"message": {"content": None, "tool_calls": "bad"}}]},
        {"choices": [{"message": {"content": None, "tool_calls": [1]}}]},
        {"choices": [{"message": {"content": None, "tool_calls": [{}]}}]},
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"id": "", "function": {"name": "x", "arguments": "{}"}}],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"id": "1", "function": {"name": "", "arguments": "{}"}}],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "[]"}}],
                    }
                }
            ]
        },
    ],
)
def test_openai_complete_rejects_malformed_response_or_tool_calls(payload: dict[str, object]) -> None:
    provider = sync_provider(lambda _request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(ModelResponseError):
            complete(provider, request())
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("body", [b"not-json", b"[]", b"null", b'{"value":NaN}', b"\xff"])
def test_openai_complete_rejects_non_object_malformed_or_non_finite_json(body: bytes) -> None:
    provider = sync_provider(lambda _request: httpx.Response(200, content=body))
    try:
        with pytest.raises(ModelResponseError):
            complete(provider, request())
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    ("status", "body", "retry_after", "expected_message"),
    [
        (400, {"error": {"message": "bad input"}}, None, "bad input"),
        (401, {"message": "bad key"}, None, "bad key"),
        (429, {"error": {"message": "limited"}}, "2.5", "limited"),
        (500, ["not", "object"], "invalid", "HTTP 500"),
        (503, "plain", "-1", "HTTP 503"),
    ],
)
def test_openai_complete_preserves_http_status_message_and_retry_after(
    status: int,
    body: object,
    retry_after: str | None,
    expected_message: str,
) -> None:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    provider = sync_provider(lambda _request: httpx.Response(status, json=body, headers=headers))
    try:
        with pytest.raises(RuntimeError) as caught:
            complete(provider, request())
        assert expected_message in str(caught.value)
        assert caught.value.status_code == status
        if retry_after == "2.5":
            assert caught.value.retry_after == 2.5
        elif retry_after == "-1":
            assert caught.value.retry_after == 0.0
        else:
            assert caught.value.retry_after is None
    finally:
        provider._sync_client.close()


def test_openai_complete_rejects_response_over_byte_bound() -> None:
    provider = sync_provider(
        lambda _request: httpx.Response(200, content=b"x" * 1025),
        config=chat_config(max_response_bytes=1024),
    )
    try:
        with pytest.raises(ModelResponseError, match="byte bound"):
            complete(provider, request())
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    ("wire", "kinds"),
    [
        (
            b'data: {"choices":[{"delta":{"reasoning_content":"r"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}\n\n',
            ["reasoning_delta", "content_delta", "done"],
        ),
        (
            b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n',
            ["content_delta", "usage", "done"],
        ),
        (b'{"choices":[{"delta":{"content":"raw"},"finish_reason":"length"}]}\n\n', ["content_delta", "done"]),
    ],
)
def test_openai_stream_decodes_supported_sse_shapes_and_single_terminal_event(
    wire: bytes,
    kinds: list[str],
) -> None:
    provider = sync_provider(
        lambda _request: httpx.Response(200, content=wire, headers={"content-type": "text/event-stream"})
    )
    try:
        events = tuple(stream(provider, request()))
        assert [event.kind for event in events] == kinds
        assert sum(event.kind == "done" for event in events) == 1
    finally:
        provider._sync_client.close()


def test_openai_stream_preserves_fragmented_tool_call_delta_and_finish_reason() -> None:
    wire = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
        b'"function":{"name":"inspect","arguments":"{\\"path\\":\\""}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":".\\"}"}}]},'
        b'"finish_reason":"tool_calls"}]}\n\ndata: [DONE]\n\n'
    )
    provider = sync_provider(
        lambda _request: httpx.Response(200, content=wire, headers={"content-type": "text/event-stream"})
    )
    try:
        events = tuple(stream(provider, request()))
        assert [event.kind for event in events] == ["tool_call_delta", "tool_call_delta", "done"]
        assert events[0].tool_call_id == "call-1"
        assert events[0].tool_name == "inspect"
        assert events[-1].finish_reason == "tool_calls"
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("wire", [b"data: not-json\n\n", b"data: []\n\n", b"data: \xff\n\n"])
def test_openai_stream_rejects_malformed_or_non_utf8_sse_payload(wire: bytes) -> None:
    provider = sync_provider(
        lambda _request: httpx.Response(200, content=wire, headers={"content-type": "text/event-stream"})
    )
    try:
        with pytest.raises(ModelResponseError):
            tuple(stream(provider, request()))
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize(
    "wire",
    [
        b'data: {"error":{"message":"upstream failed"}}\n\n',
        b'data: {"choices":[]}\n\n',
        b'data: {"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"r"},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"   "},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"ok","tool_calls":[42]},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"ok","tool_calls":"bad"},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"ok","tool_calls":[{"function":42}]},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":123}]}\n\ndata: [DONE]\n\n',
        b'data: {"choices":[{"delta":{"content":"ok","reasoning_content":42},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"ok","tool_calls":[{"index":0,"id":42}]},"finish_reason":"stop"}]}\n\n',
        (
            b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"b"},"finish_reason":null}]}\n\n'
        ),
        (
            b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
        ),
        (
            b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\ndata: {"choices":[{"delta":{"content":"b"},"finish_reason":null}]}\n\n'
        ),
        b'data: {"choices":[{"delta":{"content":"partial output"},"finish_reason":null}]}\n\n',
        (b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1"}]},"finish_reason":"tool_calls"}]}\n\n'),
        (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
            b'"function":{"name":"inspect","arguments":"{\\"x\\":1,\\"x\\":2}"}}]},'
            b'"finish_reason":"tool_calls"}]}\n\n'
        ),
        b": heartbeat\n\ndata: [DONE]\n\n",
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"",
    ],
)
def test_openai_stream_rejects_success_status_without_a_valid_choice(wire: bytes) -> None:
    provider = sync_provider(
        lambda _request: httpx.Response(200, content=wire, headers={"content-type": "text/event-stream"})
    )
    try:
        with pytest.raises(ModelResponseError):
            tuple(stream(provider, request()))
    finally:
        provider._sync_client.close()


def test_openai_stream_rejects_body_over_byte_bound() -> None:
    provider = sync_provider(
        lambda _request: httpx.Response(200, content=b":" + b"x" * 1024),
        config=chat_config(max_response_bytes=1024),
    )
    try:
        with pytest.raises(ModelResponseError, match="stream exceeds"):
            tuple(stream(provider, request()))
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
def test_openai_stream_rejects_http_error_before_parsing_sse(status: int) -> None:
    provider = sync_provider(lambda _request: httpx.Response(status, json={"error": {"message": "stream failed"}}))
    try:
        with pytest.raises(RuntimeError, match="stream failed"):
            tuple(stream(provider, request()))
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("payload", [{}, {"object": "list"}, {"data": []}])
def test_openai_health_check_validates_json_object_and_returns_identity(payload: dict[str, object]) -> None:
    provider = sync_provider(lambda request: httpx.Response(200, json=payload))
    try:
        assert provider.health_check() == {
            "ok": True,
            "provider": "test-provider",
            "model": "test-model",
        }
    finally:
        provider._sync_client.close()


def test_openai_async_complete_and_stream_use_same_protocol_semantics() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=response_payload())
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n',
        )

    provider = OpenAICompatibleChatProvider(chat_config())

    async def invoke():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider._async_client = lambda: client
            completed = await provider.complete_async(provider.prepare(request(), stream=False))
            streamed = tuple(
                [
                    event
                    async for event in provider.stream_async(
                        provider.prepare(request(), stream=True)
                    )
                ]
            )
            return completed, streamed

    try:
        completed, streamed = asyncio.run(invoke())
        assert completed.content == "done"
        assert [event.kind for event in streamed] == ["content_delta", "done"]
    finally:
        provider._sync_client.close()


@pytest.mark.parametrize("invalid", [None, "config", {}, [], 1, True, object()])
def test_ark_adapter_requires_embedding_config(invalid: object) -> None:
    with pytest.raises(TypeError):
        ArkMultimodalEmbeddingProvider(invalid, api_key="secret")


@pytest.mark.parametrize(
    "overrides",
    [
        {"adapter": "other"},
        {"input_mode": "text"},
        {"base_url": ""},
    ],
)
def test_ark_adapter_rejects_incompatible_route(overrides: dict[str, object]) -> None:
    with pytest.raises(ModelConfigurationError):
        ArkMultimodalEmbeddingProvider(embedding_config(**overrides), api_key="secret")


@pytest.mark.parametrize("api_key", ["", " ", "\t", None, 0, True, object()])
def test_ark_adapter_requires_non_empty_text_api_key(api_key: object) -> None:
    with pytest.raises(ModelConfigurationError, match="API key"):
        ArkMultimodalEmbeddingProvider(embedding_config(), api_key=api_key)


@pytest.mark.parametrize("field", ["extra_body", "query_parameters", "document_parameters"])
@pytest.mark.parametrize(
    "reserved",
    ["api_key", "base_url", "encoding_format", "extra_body", "extra_headers", "input", "model", "timeout"],
)
def test_ark_adapter_rejects_reserved_request_field_in_every_parameter_layer(
    field: str,
    reserved: str,
) -> None:
    with pytest.raises((ValueError, ModelConfigurationError), match="cannot override"):
        ArkMultimodalEmbeddingProvider(
            embedding_config(**{field: {reserved: "override"}}),
            api_key="secret",
        )


@pytest.mark.parametrize("is_query", [True, False])
def test_ark_adapter_applies_mode_specific_parameters_and_protected_request_fields(is_query: bool) -> None:
    captured: dict[str, object] = {}
    provider = ArkMultimodalEmbeddingProvider(
        embedding_config(
            extra_body={"dimensions": 2},
            query_parameters={"instruction": "query"},
            document_parameters={"instruction": "document"},
        ),
        api_key=" secret ",
    )

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(http_request.content)
        captured["authorization"] = http_request.headers["Authorization"]
        captured["url"] = str(http_request.url)
        return httpx.Response(200, json={"data": [{"embedding": [3, 4]}]})

    async def invoke():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider._client = lambda: client
            return await provider.embed(" text ", is_query=is_query)

    vector = asyncio.run(invoke())
    assert vector.values == pytest.approx((0.6, 0.8))
    assert captured["payload"]["instruction"] == ("query" if is_query else "document")
    assert captured["payload"]["input"] == [{"type": "text", "text": " text "}]
    assert captured["payload"]["encoding_format"] == "float"
    assert captured["authorization"] == "Bearer secret"
    assert captured["url"].endswith("/embeddings/multimodal")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"data": [{"embedding": [3, 4]}]}, (0.6, 0.8)),
        ({"data": {"embedding": [0, 2]}}, (0.0, 1.0)),
        ({"data": [{"embedding": [3, 4, 99]}]}, (0.6, 0.8)),
    ],
)
def test_ark_adapter_accepts_object_or_single_item_data_and_truncates_declared_dimension(
    payload: dict[str, object],
    expected: tuple[float, ...],
) -> None:
    provider = ArkMultimodalEmbeddingProvider(embedding_config(), api_key="secret")
    assert provider._vector(payload).values == pytest.approx(expected)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": []},
        {"data": [{"embedding": [1, 2]}, {"embedding": [3, 4]}]},
        {"data": None},
        {"data": {}},
        {"data": {"embedding": None}},
        {"data": {"embedding": "1,2"}},
        {"data": {"embedding": [1]}},
        {"data": {"embedding": [0, 0]}},
        {"data": {"embedding": [1, math.nan]}},
        {"data": {"embedding": [1, True]}},
        {"data": {"embedding": [1, "2"]}},
    ],
)
def test_ark_adapter_rejects_malformed_count_shape_dimension_or_numeric_vector(
    payload: dict[str, object],
) -> None:
    provider = ArkMultimodalEmbeddingProvider(embedding_config(), api_key="secret")
    with pytest.raises(ModelResponseError):
        provider._vector(payload)


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b"null",
        b"\xff",
        b'{"data":{"embedding":[1,0]},"data":{"embedding":[0,1]}}',
    ],
)
def test_ark_adapter_rejects_malformed_or_non_object_http_response(body: bytes) -> None:
    provider = ArkMultimodalEmbeddingProvider(embedding_config(), api_key="secret")

    async def invoke():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body))
        ) as client:
            provider._client = lambda: client
            return await provider.embed("text", is_query=True)

    with pytest.raises(ModelResponseError):
        asyncio.run(invoke())


def test_rerank_adapter_rejects_duplicate_json_keys_in_success_response() -> None:
    provider = OpenAICompatibleRerankProvider(rerank_config(), api_key="secret")
    body = b'{"results":[{"index":0,"relevance_score":0.1,"relevance_score":0.9}]}'

    async def invoke():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body))
        ) as client:
            provider._client = lambda: client
            return await provider.rerank("query", ("document",))

    with pytest.raises(ModelResponseError, match="malformed JSON"):
        asyncio.run(invoke())


@pytest.mark.parametrize(
    ("status", "body", "retry_after", "message"),
    [
        (400, {"error": {"message": "bad input"}}, None, "bad input"),
        (401, {"error": {"message": "bad key"}}, None, "bad key"),
        (429, {"error": {"message": "limited"}}, "3", "limited"),
        (500, ["bad"], "invalid", "HTTP 500"),
        (503, "plain", "-1", "HTTP 503"),
    ],
)
def test_ark_adapter_preserves_http_error_status_message_and_retry_after(
    status: int,
    body: object,
    retry_after: str | None,
    message: str,
) -> None:
    provider = ArkMultimodalEmbeddingProvider(embedding_config(), api_key="secret")
    headers = {} if retry_after is None else {"Retry-After": retry_after}

    async def invoke():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(status, json=body, headers=headers))
        ) as client:
            provider._client = lambda: client
            return await provider.embed("text", is_query=True)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(invoke())
    assert caught.value.status_code == status
    assert message in str(caught.value)
    if retry_after == "3":
        assert caught.value.retry_after == 3.0
    elif retry_after == "-1":
        assert caught.value.retry_after == 0.0
    else:
        assert caught.value.retry_after is None


def test_ark_adapter_rejects_response_over_byte_bound() -> None:
    provider = ArkMultimodalEmbeddingProvider(
        embedding_config(max_response_bytes=1024),
        api_key="secret",
    )

    async def invoke():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"x" * 1025))
        ) as client:
            provider._client = lambda: client
            return await provider.embed("text", is_query=True)

    with pytest.raises(ModelResponseError, match="byte bound"):
        asyncio.run(invoke())


def test_ark_adapter_reuses_one_async_client_per_event_loop() -> None:
    provider = ArkMultimodalEmbeddingProvider(embedding_config(), api_key="secret")

    async def invoke() -> bool:
        first = provider._client()
        second = provider._client()
        same = first is second
        await first.aclose()
        return same

    assert asyncio.run(invoke()) is True
