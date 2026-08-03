"""ModelClient 公共值对象、消息协议和供应商路由的系统契约矩阵。"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType

import pytest

from ModelClient import (
    ChatCallContext,
    ChatMessage,
    ChatModelConfig,
    ChatRequest,
    EmbeddingModelConfig,
    EmbeddingVector,
    ModelAuthenticationError,
    ModelClientError,
    ModelConfigurationError,
    ModelContentSafetyError,
    ModelDependencyError,
    ModelInputTooLargeError,
    ModelPermissionError,
    ModelQuotaError,
    ModelRateLimitError,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    ModelStructuredOutputError,
    ModelTransportError,
    PreparedChatRequest,
    ProviderCapabilities,
    ProviderConfig,
    ReasoningOptions,
    RerankModelConfig,
    ResponseFormat,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


def route(**overrides: object) -> ProviderConfig:
    values: dict[str, object] = {
        "provider": "test-provider",
        "adapter": "test-adapter",
        "model": "test-model",
        "base_url": "https://example.com/v1",
        "credential_ref": "test-credential",
    }
    values.update(overrides)
    return ProviderConfig(**values)


def call(**overrides: object) -> ToolCall:
    values: dict[str, object] = {
        "id": "call-1",
        "name": "workspace.inspect",
        "arguments": {"path": "."},
    }
    values.update(overrides)
    return ToolCall(**values)


def tool(**overrides: object) -> ToolDefinition:
    values: dict[str, object] = {
        "name": "workspace.inspect",
        "description": "检查工作区",
        "parameters": {"type": "object"},
    }
    values.update(overrides)
    return ToolDefinition(**values)


def message(**overrides: object) -> ChatMessage:
    values: dict[str, object] = {"role": "user", "content": "hello"}
    values.update(overrides)
    return ChatMessage(**values)


VALID_IDENTIFIERS = (
    "a",
    "A",
    "tool_1",
    "tool-name",
    "workspace.inspect",
    "provider/model:v1",
    "中文工具",
    " name-with-edge-space ",
)

INVALID_EMPTY_TEXT = ("", " ", "\t", "\n", "\r\n", "   \t\n")
NON_TEXT_VALUES = (None, True, False, 0, 1, 1.5, (), [], {}, set(), object())
NON_MAPPING_VALUES = (
    None,
    True,
    False,
    0,
    1.5,
    "object",
    (),
    [],
    set(),
    (("key", "value"),),
    [("key", "value")],
    object(),
)
INVALID_INTEGER_VALUES = (True, False, -1, 1.5, "1", None, (), [], {}, object())


@pytest.mark.parametrize("identifier", VALID_IDENTIFIERS)
def test_tool_call_preserves_every_non_empty_identifier(identifier: str) -> None:
    item = call(id=identifier, name=identifier)
    assert item.id == identifier
    assert item.name == identifier


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"path": "."},
        {"depth": 0, "enabled": False},
        {"nullable": None},
        {"nested": {"items": [1, 2, 3]}},
        MappingProxyType({"query": "memory"}),
    ],
)
def test_tool_call_accepts_json_object_shaped_argument_mappings(arguments: object) -> None:
    item = call(arguments=arguments)
    assert item.arguments == dict(arguments)
    assert isinstance(item.arguments, dict)


def test_tool_call_copies_top_level_argument_mapping() -> None:
    arguments: dict[str, object] = {"path": "."}
    item = call(arguments=arguments)
    arguments["path"] = "changed"
    assert item.arguments == {"path": "."}


@pytest.mark.parametrize("field", ["id", "name"])
@pytest.mark.parametrize("invalid", INVALID_EMPTY_TEXT + NON_TEXT_VALUES)
def test_tool_call_rejects_empty_or_non_text_identity(field: str, invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call(**{field: invalid})


@pytest.mark.parametrize("invalid", NON_MAPPING_VALUES)
def test_tool_call_rejects_non_object_arguments(invalid: object) -> None:
    with pytest.raises(TypeError):
        call(arguments=invalid)


@pytest.mark.parametrize("name", VALID_IDENTIFIERS)
@pytest.mark.parametrize("strict", [True, False])
def test_tool_definition_preserves_name_and_explicit_strictness(name: str, strict: bool) -> None:
    definition = tool(name=name, strict=strict)
    assert definition.name == name
    assert definition.strict is strict


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"type": "object"},
        {"type": "object", "additionalProperties": False},
        {"properties": {"path": {"type": "string"}}},
        MappingProxyType({"required": ["path"]}),
    ],
)
def test_tool_definition_accepts_mapping_schema_and_copies_top_level(parameters: object) -> None:
    definition = tool(parameters=parameters)
    assert definition.parameters == dict(parameters)
    assert isinstance(definition.parameters, dict)


@pytest.mark.parametrize("field", ["name", "description"])
@pytest.mark.parametrize("invalid", INVALID_EMPTY_TEXT + NON_TEXT_VALUES)
def test_tool_definition_rejects_empty_or_non_text_required_text(field: str, invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        tool(**{field: invalid})


@pytest.mark.parametrize("invalid", NON_MAPPING_VALUES)
def test_tool_definition_rejects_non_object_parameters(invalid: object) -> None:
    with pytest.raises(TypeError):
        tool(parameters=invalid)


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", "false", (), [], {}])
def test_tool_definition_rejects_non_boolean_strict(invalid: object) -> None:
    with pytest.raises(TypeError):
        tool(strict=invalid)


@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant"])
@pytest.mark.parametrize("content", ["x", " ", "\n", "中文", "{}", "0", "false"])
def test_text_message_preserves_role_and_exact_content(role: str, content: str) -> None:
    item = ChatMessage(role=role, content=content)
    assert item.role == role
    assert item.content == content


@pytest.mark.parametrize("content", [None, "", "先调用工具"])
@pytest.mark.parametrize("call_count", [1, 2, 4])
def test_assistant_message_supports_content_and_one_or_more_tool_calls(
    content: str | None,
    call_count: int,
) -> None:
    calls = tuple(call(id=f"call-{index}") for index in range(call_count))
    item = ChatMessage(role="assistant", content=content, tool_calls=calls)
    assert item.tool_calls == calls
    assert item.content == content


@pytest.mark.parametrize("content", ["", "ok", "{}", "null", "失败", "\n"])
@pytest.mark.parametrize("name", [None, "workspace.inspect", "工具结果"])
def test_tool_result_requires_binding_and_preserves_even_empty_result(
    content: str,
    name: str | None,
) -> None:
    item = ChatMessage(role="tool", content=content, tool_call_id="call-1", name=name)
    assert item.tool_call_id == "call-1"
    assert item.content == content


@pytest.mark.parametrize(
    "role",
    ["", "USER", "function", "model", "human", "agent", "system ", None, 0, True],
)
def test_chat_message_rejects_every_noncanonical_role(role: object) -> None:
    with pytest.raises(ValueError):
        ChatMessage(role=role, content="hello")


@pytest.mark.parametrize("invalid", [True, False, 0, 1.5, (), [], {}, object()])
def test_chat_message_rejects_non_text_content(invalid: object) -> None:
    with pytest.raises(TypeError):
        message(content=invalid)


@pytest.mark.parametrize("field", ["name", "tool_call_id"])
@pytest.mark.parametrize("invalid", INVALID_EMPTY_TEXT + (True, False, 0, 1.5, (), [], {}, object()))
def test_optional_message_identifiers_reject_empty_or_non_text_values(
    field: str,
    invalid: object,
) -> None:
    kwargs: dict[str, object] = {field: invalid}
    if field == "tool_call_id":
        kwargs.update(role="tool", content="result")
    with pytest.raises((TypeError, ValueError)):
        message(**kwargs)


@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant"])
def test_non_tool_messages_reject_tool_result_binding(role: str) -> None:
    with pytest.raises(ValueError, match="only valid for tool result"):
        ChatMessage(role=role, content="x", tool_call_id="call-1")


@pytest.mark.parametrize("role", ["system", "developer", "user", "tool"])
def test_non_assistant_messages_reject_tool_calls(role: str) -> None:
    kwargs: dict[str, object] = {"role": role, "content": "x", "tool_calls": (call(),)}
    if role == "tool":
        kwargs["tool_call_id"] = "call-1"
    with pytest.raises(ValueError):
        ChatMessage(**kwargs)


@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant"])
def test_message_without_content_or_calls_is_rejected(role: str) -> None:
    with pytest.raises(ValueError, match="require content"):
        ChatMessage(role=role)


@pytest.mark.parametrize("tool_calls", [("not-a-call",), (1,), (object(),), (call(), "bad")])
def test_message_tool_calls_require_normalized_tool_call_values(tool_calls: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChatMessage(role="assistant", tool_calls=tool_calls)


@pytest.mark.parametrize("name", VALID_IDENTIFIERS)
@pytest.mark.parametrize("strict", [True, False])
def test_response_format_accepts_non_empty_schema_and_strictness(name: str, strict: bool) -> None:
    item = ResponseFormat(name=name, schema={"type": "object"}, strict=strict)
    assert item.name == name
    assert item.strict is strict


@pytest.mark.parametrize("invalid", INVALID_EMPTY_TEXT + NON_TEXT_VALUES)
def test_response_format_rejects_empty_or_non_text_name(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ResponseFormat(name=invalid, schema={"type": "object"})


@pytest.mark.parametrize("invalid", NON_MAPPING_VALUES + ({},))
def test_response_format_rejects_empty_or_non_object_schema(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ResponseFormat(name="result", schema=invalid)


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", (), [], {}])
def test_response_format_rejects_non_boolean_strict(invalid: object) -> None:
    with pytest.raises(TypeError):
        ResponseFormat(name="result", schema={"type": "object"}, strict=invalid)


@pytest.mark.parametrize("effort", [None, "minimal", "low", "medium", "high"])
def test_reasoning_options_accept_only_portable_effort_levels(effort: str | None) -> None:
    assert ReasoningOptions(effort=effort).effort == effort


@pytest.mark.parametrize(
    "invalid",
    ["", "none", "very_high", "auto", "MEDIUM", 0, 1, True, False, (), [], {}],
)
def test_reasoning_options_rejects_unknown_or_non_text_effort(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ReasoningOptions(effort=invalid)


@pytest.mark.parametrize(
    "temperature",
    [None, 0, 0.0, 0.01, 0.5, 1, 1.0, 1.99, 2, 2.0],
)
def test_chat_request_accepts_temperature_boundary_values(temperature: float | None) -> None:
    request = ChatRequest(messages=(message(),), temperature=temperature)
    assert request.temperature == temperature


@pytest.mark.parametrize("temperature", [-math.inf, -1, -0.01, 2.01, 3, math.inf, math.nan])
def test_chat_request_rejects_out_of_range_or_non_finite_temperature(temperature: float) -> None:
    with pytest.raises(ValueError):
        ChatRequest(messages=(message(),), temperature=temperature)


@pytest.mark.parametrize("invalid", [True, False, "0", "1", (), [], {}, object()])
def test_chat_request_rejects_non_numeric_temperature(invalid: object) -> None:
    with pytest.raises(TypeError):
        ChatRequest(messages=(message(),), temperature=invalid)


@pytest.mark.parametrize("maximum", [None, 1, 2, 16, 128, 4096, 1_000_000])
def test_chat_request_accepts_positive_optional_output_limit(maximum: int | None) -> None:
    request = ChatRequest(messages=(message(),), max_output_tokens=maximum)
    assert request.max_output_tokens == maximum


@pytest.mark.parametrize("invalid", tuple(value for value in INVALID_INTEGER_VALUES if value is not None))
def test_chat_request_rejects_non_positive_or_non_integer_output_limit(invalid: object) -> None:
    with pytest.raises(ValueError):
        ChatRequest(messages=(message(),), max_output_tokens=invalid)


@pytest.mark.parametrize(
    "messages",
    [(), [], ("hello",), (message(), "bad"), None, "hello", 1, object()],
)
def test_chat_request_requires_non_empty_normalized_message_sequence(messages: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChatRequest(messages=messages)


@pytest.mark.parametrize("choice", ["auto", "none", "required", "workspace.inspect", {"name": "x"}])
def test_chat_request_allows_tool_choice_only_with_declared_tools(choice: object) -> None:
    request = ChatRequest(messages=(message(),), tools=(tool(),), tool_choice=choice)
    assert request.tool_choice == choice


@pytest.mark.parametrize("choice", ["auto", "none", "required", {"name": "x"}])
def test_chat_request_rejects_unbound_tool_choice(choice: object) -> None:
    with pytest.raises(ValueError, match="requires at least one tool"):
        ChatRequest(messages=(message(),), tool_choice=choice)


@pytest.mark.parametrize("invalid", [True, False, 0, 1.5, (), [], object()])
def test_chat_request_rejects_unsupported_tool_choice_shape(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChatRequest(messages=(message(),), tools=(tool(),), tool_choice=invalid)


@pytest.mark.parametrize("invalid", [None, "tool", ("bad",), (tool(), "bad"), (1,), ({},)])
def test_chat_request_tools_require_normalized_tool_definitions(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChatRequest(messages=(message(),), tools=invalid)


@pytest.mark.parametrize(
    "metadata",
    [{}, {"trace": "1"}, {"attempt": 0}, {"enabled": False}, MappingProxyType({"source": "test"})],
)
def test_chat_call_context_copies_internal_metadata_mapping(metadata: object) -> None:
    context = ChatCallContext(metadata=metadata)
    assert context.metadata == dict(metadata)
    assert isinstance(context.metadata, dict)


@pytest.mark.parametrize("invalid", NON_MAPPING_VALUES)
def test_chat_call_context_rejects_non_mapping_metadata(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChatCallContext(metadata=invalid)


@pytest.mark.parametrize(
    "invalid",
    INVALID_EMPTY_TEXT + tuple(value for value in NON_TEXT_VALUES if value is not None),
)
def test_chat_call_context_rejects_invalid_prompt_version(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChatCallContext(prompt_version=invalid)


@pytest.mark.parametrize("invalid", tuple(value for value in INVALID_INTEGER_VALUES if value is not None) + (0,))
def test_chat_call_context_rejects_invalid_input_token_limit(invalid: object) -> None:
    with pytest.raises(ValueError):
        ChatCallContext(input_token_limit=invalid)


def test_prepared_chat_request_binds_body_budget_and_stream_to_logical_request() -> None:
    request = ChatRequest(messages=(message(),), max_output_tokens=64)
    prepared = PreparedChatRequest(request, b'{"wire":true}', b"{}", 64, False)

    assert prepared.request is request
    assert prepared.body == b'{"wire":true}'
    assert prepared.model_visible_body == b"{}"
    assert prepared.estimated_input_tokens > 0
    assert prepared.reserved_output_tokens == 64
    assert prepared.stream is False


@pytest.mark.parametrize("invalid", ["format", {}, [], 1, True, object()])
def test_chat_request_requires_normalized_response_format(invalid: object) -> None:
    with pytest.raises(TypeError):
        ChatRequest(messages=(message(),), response_format=invalid)


@pytest.mark.parametrize("invalid", ["medium", {}, [], 1, True, object()])
def test_chat_request_requires_normalized_reasoning_options(invalid: object) -> None:
    with pytest.raises(TypeError):
        ChatRequest(messages=(message(),), reasoning=invalid)


USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens")


@pytest.mark.parametrize("field", USAGE_FIELDS)
@pytest.mark.parametrize("value", [0, 1, 2, 100, 1_000_000])
def test_token_usage_accepts_non_negative_integer_for_each_counter(field: str, value: int) -> None:
    usage = TokenUsage(**{field: value})
    assert getattr(usage, field) == value


@pytest.mark.parametrize("field", USAGE_FIELDS)
@pytest.mark.parametrize("invalid", INVALID_INTEGER_VALUES)
def test_token_usage_rejects_invalid_counter_type_or_range(field: str, invalid: object) -> None:
    with pytest.raises(ValueError):
        TokenUsage(**{field: invalid})


@pytest.mark.parametrize(
    "details",
    [{}, {"audio_tokens": 2}, {"provider": "test"}, MappingProxyType({"cache_hit": True})],
)
def test_token_usage_copies_provider_details(details: object) -> None:
    usage = TokenUsage(details=details)
    assert usage.details == dict(details)


@pytest.mark.parametrize("invalid", NON_MAPPING_VALUES)
def test_token_usage_rejects_non_mapping_provider_details(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TokenUsage(details=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("content", ["ok", " ", "\n", "0", "false", "中文"])
@pytest.mark.parametrize("finish_reason", ["stop", "length", "tool_calls", "content_filter"])
def test_model_response_preserves_text_and_finish_reason(content: str, finish_reason: str) -> None:
    response = ModelResponse(content, "model", "provider", finish_reason=finish_reason)
    assert response.content == content
    assert response.finish_reason == finish_reason


@pytest.mark.parametrize("content", [None, "", "before tool"])
@pytest.mark.parametrize("call_count", [1, 2, 4])
def test_model_response_accepts_tool_calls_with_optional_content(
    content: str | None,
    call_count: int,
) -> None:
    calls = tuple(call(id=f"call-{index}") for index in range(call_count))
    response = ModelResponse(content, "model", "provider", tool_calls=calls)
    assert response.tool_calls == calls


@pytest.mark.parametrize("field", ["model", "provider", "finish_reason"])
@pytest.mark.parametrize("invalid", INVALID_EMPTY_TEXT + NON_TEXT_VALUES)
def test_model_response_rejects_empty_or_non_text_identity_fields(field: str, invalid: object) -> None:
    values: dict[str, object] = {"content": "ok", "model": "model", "provider": "provider"}
    values[field] = invalid
    with pytest.raises((TypeError, ValueError)):
        ModelResponse(**values)


@pytest.mark.parametrize("invalid", [True, False, 0, 1.5, (), [], {}, object()])
def test_model_response_rejects_non_text_content(invalid: object) -> None:
    with pytest.raises(TypeError):
        ModelResponse(invalid, "model", "provider")


@pytest.mark.parametrize("invalid", [True, False, 0, 1.5, (), [], {}, object()])
def test_model_response_rejects_non_text_reasoning_content(invalid: object) -> None:
    with pytest.raises(TypeError):
        ModelResponse("ok", "model", "provider", reasoning_content=invalid)


@pytest.mark.parametrize("invalid", [None, {}, [], 0, True, object()])
def test_model_response_requires_normalized_token_usage(invalid: object) -> None:
    with pytest.raises(TypeError):
        ModelResponse("ok", "model", "provider", usage=invalid)


@pytest.mark.parametrize("tool_calls", [("bad",), (1,), (object(),), (call(), "bad")])
def test_model_response_tool_calls_require_normalized_values(tool_calls: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ModelResponse(None, "model", "provider", tool_calls=tool_calls)


@pytest.mark.parametrize("raw", [{}, {"id": "response-1"}, MappingProxyType({"choices": []})])
def test_model_response_copies_raw_provider_mapping(raw: object) -> None:
    response = ModelResponse("ok", "model", "provider", raw=raw)
    assert response.raw == dict(raw)


@pytest.mark.parametrize(
    "invalid",
    [True, False, 0, 1.5, "raw", (), [], set(), (("id", "response-1"),), [("id", "response-1")], object()],
)
def test_model_response_rejects_non_mapping_raw(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ModelResponse("ok", "model", "provider", raw=invalid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "content_delta", "content_delta": "x"},
        {"kind": "reasoning_delta", "reasoning_delta": "think"},
        {
            "kind": "tool_call_delta",
            "tool_call_index": 0,
            "tool_call_id": "call-1",
            "tool_name": "inspect",
            "tool_arguments_delta": '{"path"',
        },
        {"kind": "usage", "usage": TokenUsage(total_tokens=3)},
        {"kind": "done", "finish_reason": "stop"},
    ],
)
@pytest.mark.parametrize("raw", [None, {}, {"provider": "test"}])
def test_stream_event_accepts_each_canonical_event_shape(
    kwargs: dict[str, object],
    raw: dict[str, object] | None,
) -> None:
    event = ModelStreamEvent(**kwargs, raw=raw)
    assert event.kind == kwargs["kind"]
    if raw is not None:
        assert event.raw == raw


@pytest.mark.parametrize("kind", ["", "content", "delta", "finish", "error", None, 0, True])
def test_stream_event_rejects_noncanonical_kind(kind: object) -> None:
    with pytest.raises(ValueError):
        ModelStreamEvent(kind=kind)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "content_delta"},
        {"kind": "content_delta", "content_delta": 1},
        {"kind": "reasoning_delta"},
        {"kind": "reasoning_delta", "reasoning_delta": {}},
        {"kind": "tool_call_delta"},
        {"kind": "tool_call_delta", "tool_call_index": -1, "tool_arguments_delta": "{}"},
        {"kind": "usage"},
        {"kind": "usage", "usage": {}},
        {"kind": "done"},
        {"kind": "done", "finish_reason": ""},
        {"kind": "done", "content_delta": "late", "finish_reason": "stop"},
    ],
)
def test_stream_event_rejects_missing_mistyped_or_cross_kind_payload(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ModelStreamEvent(**kwargs)


@pytest.mark.parametrize(
    "invalid",
    [True, False, 0, 1.5, "raw", (), [], set(), (("id", "event-1"),), [("id", "event-1")], object()],
)
def test_stream_event_rejects_non_mapping_raw(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ModelStreamEvent(kind="done", finish_reason="stop", raw=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["none", "json_object", "json_schema"])
@pytest.mark.parametrize("flag", [True, False])
def test_provider_capabilities_accepts_declared_modes_and_boolean_flags(mode: str, flag: bool) -> None:
    capabilities = ProviderCapabilities(
        async_completion=flag,
        streaming=flag,
        tools=flag,
        structured_output_mode=mode,
        reasoning=flag,
    )
    assert capabilities.structured_output_mode == mode


@pytest.mark.parametrize("mode", ["", "schema", "JSON", "JSON_SCHEMA", None, 0, True])
def test_provider_capabilities_rejects_unknown_structured_output_mode(mode: object) -> None:
    with pytest.raises(ValueError):
        ProviderCapabilities(structured_output_mode=mode)


@pytest.mark.parametrize("field", ["async_completion", "streaming", "tools", "reasoning"])
@pytest.mark.parametrize("invalid", [0, 1, "true", None, (), []])
def test_provider_capabilities_rejects_non_boolean_flags(field: str, invalid: object) -> None:
    with pytest.raises(TypeError):
        ProviderCapabilities(**{field: invalid})


@pytest.mark.parametrize("model_type", [ToolCall, ToolDefinition, ChatMessage, ResponseFormat, ReasoningOptions, ChatRequest, TokenUsage, ModelResponse, ModelStreamEvent, ProviderCapabilities])
def test_public_contract_value_objects_are_frozen(model_type: type[object]) -> None:
    assert model_type.__dataclass_params__.frozen is True


def test_frozen_contract_rejects_direct_field_reassignment() -> None:
    item = message()
    with pytest.raises(FrozenInstanceError):
        item.content = "changed"


ERROR_TYPES = (
    ModelClientError,
    ModelConfigurationError,
    ModelDependencyError,
    ModelAuthenticationError,
    ModelPermissionError,
    ModelTransportError,
    ModelRateLimitError,
    ModelQuotaError,
    ModelInputTooLargeError,
    ModelContentSafetyError,
    ModelResponseError,
    ModelStructuredOutputError,
)


@pytest.mark.parametrize("error_type", ERROR_TYPES)
@pytest.mark.parametrize("retry_after", [None, 0.0, 1.5])
def test_model_error_taxonomy_preserves_message_and_retry_hint(
    error_type: type[ModelClientError],
    retry_after: float | None,
) -> None:
    error = error_type("failure", retry_after_seconds=retry_after)
    assert str(error) == "failure"
    assert error.retry_after_seconds == retry_after
    assert isinstance(error.code, str) and error.code


@pytest.mark.parametrize(
    "values",
    [
        [1],
        (1,),
        [3, 4],
        (-3, 4),
        (0.1, 0.2, 0.3),
        (1, 2, 3, 4),
        tuple(range(1, 17)),
        [1e-100, 2e-100],
        [1e100, -1e100],
    ],
)
def test_embedding_vector_normalizes_valid_numeric_sequences(values: list[float] | tuple[float, ...]) -> None:
    vector = EmbeddingVector(values)
    assert vector.dimension == len(values)
    assert math.sqrt(sum(value * value for value in vector.values)) == pytest.approx(1.0)


@pytest.mark.parametrize("invalid", [None, "1,2", 1, 1.5, True, {}, set(), object()])
def test_embedding_vector_requires_list_or_tuple(invalid: object) -> None:
    with pytest.raises(TypeError):
        EmbeddingVector(invalid)


@pytest.mark.parametrize("values", [(), [], (0,), (0, 0), [0.0, -0.0]])
def test_embedding_vector_rejects_empty_or_zero_norm(values: object) -> None:
    with pytest.raises(ValueError):
        EmbeddingVector(values)


@pytest.mark.parametrize(
    "invalid",
    [True, False, None, "1", object(), (), [], {}, set(), complex(1, 2)],
)
def test_embedding_vector_rejects_non_numeric_element(invalid: object) -> None:
    with pytest.raises(TypeError):
        EmbeddingVector((1, invalid))


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_embedding_vector_rejects_non_finite_element(invalid: float) -> None:
    with pytest.raises(ValueError):
        EmbeddingVector((1, invalid))


@pytest.mark.parametrize(
    "provider",
    ["a", "A", "Provider", "provider-1", "provider_1", "provider.one", "x" * 128],
)
@pytest.mark.parametrize("adapter", ["a", "Adapter", "openai_compatible_chat"])
def test_provider_route_normalizes_each_valid_provider_adapter_combination(
    provider: str,
    adapter: str,
) -> None:
    config = route(provider=f" {provider} ", adapter=f" {adapter} ")
    assert config.provider == provider.lower()
    assert config.adapter == adapter.lower()


@pytest.mark.parametrize(
    "field",
    ["provider", "adapter"],
)
@pytest.mark.parametrize(
    "invalid",
    ["", " ", "1provider", "-provider", ".provider", "provider/name", "provider:name", "provider name", "x" * 129],
)
def test_provider_route_rejects_invalid_provider_or_adapter_grammar(field: str, invalid: str) -> None:
    with pytest.raises(ValueError):
        route(**{field: invalid})


@pytest.mark.parametrize(
    "model",
    ["a", "Model", "model-1", "model_1", "model.one", "org/model", "org/model:v1", "x" * 256],
)
def test_provider_route_accepts_model_routing_grammar(model: str) -> None:
    assert route(model=f" {model} ").model == model


@pytest.mark.parametrize(
    "model",
    ["", " ", "-model", ".model", ":model", "model name", "model@version", "中文模型", "x" * 257],
)
def test_provider_route_rejects_invalid_model_routing_grammar(model: str) -> None:
    with pytest.raises(ValueError):
        route(model=model)


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "https://example.com",
        "https://example.com/v1",
        "https://example.com:8443/api",
        "http://localhost:8000/v1",
        "http://127.0.0.1/v1",
        "http://[::1]:8000/v1",
    ],
)
def test_provider_route_accepts_secure_remote_or_loopback_base_url(base_url: str) -> None:
    assert route(base_url=f"{base_url}/" if base_url else "").base_url == base_url


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.com/v1",
        "ws://example.com/v1",
        "http://example.com/v1",
        "https://user@example.com/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?key=secret",
        "https://example.com/v1#fragment",
        "//example.com/v1",
        "/v1",
        "example.com/v1",
    ],
)
def test_provider_route_rejects_insecure_or_non_origin_base_url(base_url: str) -> None:
    with pytest.raises(ValueError):
        route(base_url=base_url)


@pytest.mark.parametrize("name", ["", "provider", "Provider_Key", "provider-key", "provider.key"])
def test_provider_route_accepts_empty_or_valid_credential_reference(name: str) -> None:
    assert route(credential_ref=f" {name} ").credential_ref == name.lower()


@pytest.mark.parametrize("name", ["1key", "bad/name", "bad name", "$api_key", "中文"])
def test_provider_route_rejects_invalid_credential_reference(name: str) -> None:
    with pytest.raises(ValueError):
        route(credential_ref=name)


PROVIDER_INTEGER_BOUNDS = (
    ("max_retries", 0, 10),
    ("max_concurrent", 1, 4096),
    ("max_response_bytes", 1024, 64 * 1024 * 1024),
)


@pytest.mark.parametrize(("field", "minimum", "maximum"), PROVIDER_INTEGER_BOUNDS)
def test_provider_route_accepts_each_integer_boundary(field: str, minimum: int, maximum: int) -> None:
    assert getattr(route(**{field: minimum}), field) == minimum
    assert getattr(route(**{field: maximum}), field) == maximum


@pytest.mark.parametrize(("field", "minimum", "maximum"), PROVIDER_INTEGER_BOUNDS)
@pytest.mark.parametrize("position", ["below", "above"])
def test_provider_route_rejects_each_integer_outside_boundary(
    field: str,
    minimum: int,
    maximum: int,
    position: str,
) -> None:
    invalid = minimum - 1 if position == "below" else maximum + 1
    with pytest.raises(ValueError):
        route(**{field: invalid})


@pytest.mark.parametrize(("field", "_minimum", "_maximum"), PROVIDER_INTEGER_BOUNDS)
@pytest.mark.parametrize("invalid", [True, False, 1.5, "1", None, (), [], {}])
def test_provider_route_rejects_non_integer_operational_limits(
    field: str,
    _minimum: int,
    _maximum: int,
    invalid: object,
) -> None:
    with pytest.raises(ValueError):
        route(**{field: invalid})


PROVIDER_FLOAT_BOUNDS = (
    ("timeout_seconds", 600.0),
    ("retry_base_delay_seconds", 60.0),
    ("retry_max_delay_seconds", 300.0),
)


@pytest.mark.parametrize(("field", "maximum"), PROVIDER_FLOAT_BOUNDS)
@pytest.mark.parametrize("edge", ["small", "maximum"])
def test_provider_route_accepts_each_positive_float_boundary(field: str, maximum: float, edge: str) -> None:
    value = 0.000001 if edge == "small" else maximum
    overrides: dict[str, object] = {field: value}
    if field == "retry_base_delay_seconds":
        overrides["retry_max_delay_seconds"] = max(value, 60.0)
    elif field == "retry_max_delay_seconds":
        overrides["retry_base_delay_seconds"] = min(value, 0.000001)
    assert getattr(route(**overrides), field) == value


@pytest.mark.parametrize(("field", "maximum"), PROVIDER_FLOAT_BOUNDS)
@pytest.mark.parametrize("invalid_kind", ["zero", "negative", "above", "nan", "infinity"])
def test_provider_route_rejects_invalid_numeric_duration(
    field: str,
    maximum: float,
    invalid_kind: str,
) -> None:
    invalid = {
        "zero": 0,
        "negative": -1,
        "above": maximum + 0.1,
        "nan": math.nan,
        "infinity": math.inf,
    }[invalid_kind]
    with pytest.raises(ValueError):
        route(**{field: invalid})


@pytest.mark.parametrize(("field", "_maximum"), PROVIDER_FLOAT_BOUNDS)
@pytest.mark.parametrize("invalid", [True, False, "1", None, (), [], {}])
def test_provider_route_rejects_non_numeric_duration(
    field: str,
    _maximum: float,
    invalid: object,
) -> None:
    with pytest.raises(ValueError):
        route(**{field: invalid})


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Trace": "1"}, {"Content-Type": "application/json"}, MappingProxyType({"X-A": "B"})],
)
def test_provider_route_copies_safe_string_headers(headers: object) -> None:
    config = route(extra_headers=headers)
    assert config.extra_headers == dict(headers)


@pytest.mark.parametrize("name", ["Authorization", "authorization", "AUTHORIZATION", "Proxy-Authorization"])
def test_provider_route_rejects_credential_headers_case_insensitively(name: str) -> None:
    with pytest.raises(ValueError):
        route(extra_headers={name: "secret"})


@pytest.mark.parametrize(
    "headers",
    [None, [], (), "header", {"": "x"}, {" ": "x"}, {1: "x"}, {"X": 1}, {"X": None}],
)
def test_provider_route_rejects_non_string_header_mapping(headers: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        route(extra_headers=headers)


@pytest.mark.parametrize(
    "body",
    [{}, {"seed": 1}, {"flag": False}, {"nested": {"items": [1, None]}}, MappingProxyType({"top_p": 0.9})],
)
def test_provider_route_accepts_json_serializable_extra_body(body: object) -> None:
    config = route(extra_body=body)
    assert config.extra_body == dict(body)


@pytest.mark.parametrize("key", ["provider", "adapter", "model", "base_url", "api_key"])
def test_provider_route_rejects_identity_override_in_extra_body(key: str) -> None:
    with pytest.raises(ValueError, match="route identity"):
        route(extra_body={key: "override"})


@pytest.mark.parametrize(
    "body",
    [None, [], (), "body", {"": 1}, {" ": 1}, {1: "x"}, {"x": object()}, {"x": math.nan}, {"x": {1, 2}}],
)
def test_provider_route_rejects_non_json_extra_body(body: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        route(extra_body=body)


@pytest.mark.parametrize("mode", ["none", "json_object", "json_schema"])
@pytest.mark.parametrize("reasoning", [True, False])
def test_chat_model_config_accepts_supported_mode_and_reasoning(mode: str, reasoning: bool) -> None:
    config = ChatModelConfig(route(), structured_output_mode=mode, reasoning=reasoning)
    assert config.capability == "chat"


@pytest.mark.parametrize("maximum", [None, 1, 63_999])
def test_chat_model_config_accepts_output_limit_boundaries(maximum: int | None) -> None:
    assert ChatModelConfig(route(), max_output_tokens=maximum).max_output_tokens == maximum


@pytest.mark.parametrize("maximum", [1024, 64_000, 10_000_000])
def test_chat_model_config_accepts_context_window_boundaries(maximum: int) -> None:
    assert ChatModelConfig(route(), context_window_tokens=maximum).context_window_tokens == maximum


@pytest.mark.parametrize(
    "invalid",
    tuple(value for value in INVALID_INTEGER_VALUES if value is not None) + (64_000, 10_000_001),
)
def test_chat_model_config_rejects_invalid_output_limit(invalid: object) -> None:
    with pytest.raises(ValueError):
        ChatModelConfig(route(), max_output_tokens=invalid)


@pytest.mark.parametrize("invalid", [None, {}, [], "route", 1, True])
@pytest.mark.parametrize("model_type", [ChatModelConfig, EmbeddingModelConfig, RerankModelConfig])
def test_capability_configs_require_provider_route(model_type: type[object], invalid: object) -> None:
    kwargs: dict[str, object] = {"route": invalid}
    if model_type is EmbeddingModelConfig:
        kwargs["dimension"] = 2
    with pytest.raises(TypeError):
        model_type(**kwargs)


@pytest.mark.parametrize("dimension", [1, 2, 512, 4096, 65_536])
@pytest.mark.parametrize("input_mode", ["text", "multimodal"])
def test_embedding_model_config_accepts_dimension_and_input_mode_boundaries(
    dimension: int,
    input_mode: str,
) -> None:
    config = EmbeddingModelConfig(route(), dimension=dimension, input_mode=input_mode)
    assert config.capability == "embedding"


@pytest.mark.parametrize("dimension", [0, -1, 65_537, True, False, 1.5, "2", None])
def test_embedding_model_config_rejects_invalid_dimension(dimension: object) -> None:
    with pytest.raises(ValueError):
        EmbeddingModelConfig(route(), dimension=dimension)


@pytest.mark.parametrize("mode", ["", "image", "TEXT", None, 0, True])
def test_embedding_model_config_rejects_unknown_input_mode(mode: object) -> None:
    with pytest.raises(ValueError):
        EmbeddingModelConfig(route(), dimension=2, input_mode=mode)


@pytest.mark.parametrize(
    ("field", "minimum", "maximum"),
    [("max_batch_size", 1, 2048), ("max_input_chars", 1, 1_000_000)],
)
@pytest.mark.parametrize("edge", ["minimum", "maximum"])
def test_embedding_model_config_accepts_integer_boundaries(
    field: str,
    minimum: int,
    maximum: int,
    edge: str,
) -> None:
    value = minimum if edge == "minimum" else maximum
    assert getattr(EmbeddingModelConfig(route(), dimension=2, **{field: value}), field) == value


@pytest.mark.parametrize(
    ("field", "minimum", "maximum"),
    [("max_batch_size", 1, 2048), ("max_input_chars", 1, 1_000_000)],
)
@pytest.mark.parametrize("invalid_kind", ["below", "above", "bool", "float", "string", "null"])
def test_embedding_model_config_rejects_invalid_integer_bounds(
    field: str,
    minimum: int,
    maximum: int,
    invalid_kind: str,
) -> None:
    invalid = {
        "below": minimum - 1,
        "above": maximum + 1,
        "bool": True,
        "float": 1.5,
        "string": "1",
        "null": None,
    }[invalid_kind]
    with pytest.raises(ValueError):
        EmbeddingModelConfig(route(), dimension=2, **{field: invalid})


@pytest.mark.parametrize("field", ["query_parameters", "document_parameters"])
@pytest.mark.parametrize(
    "value",
    [{}, {"instruction": "query"}, {"truncate": True}, {"nested": [1, None]}],
)
def test_embedding_model_config_accepts_json_parameter_mappings(
    field: str,
    value: dict[str, object],
) -> None:
    config = EmbeddingModelConfig(route(), dimension=2, **{field: value})
    assert getattr(config, field) == value


@pytest.mark.parametrize("field", ["query_parameters", "document_parameters"])
@pytest.mark.parametrize("invalid", [None, [], (), "params", {"": 1}, {1: "x"}, {"x": object()}])
def test_embedding_model_config_rejects_non_json_parameters(field: str, invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        EmbeddingModelConfig(route(), dimension=2, **{field: invalid})


@pytest.mark.parametrize(
    ("field", "minimum", "maximum"),
    [
        ("max_documents", 1, 2048),
        ("max_query_chars", 1, 1_000_000),
        ("max_document_chars", 1, 1_000_000),
    ],
)
@pytest.mark.parametrize("edge", ["minimum", "maximum"])
def test_rerank_model_config_accepts_integer_boundaries(
    field: str,
    minimum: int,
    maximum: int,
    edge: str,
) -> None:
    value = minimum if edge == "minimum" else maximum
    config = RerankModelConfig(route(), **{field: value})
    assert config.capability == "rerank"
    assert getattr(config, field) == value


@pytest.mark.parametrize(
    ("field", "minimum", "maximum"),
    [
        ("max_documents", 1, 2048),
        ("max_query_chars", 1, 1_000_000),
        ("max_document_chars", 1, 1_000_000),
    ],
)
@pytest.mark.parametrize("invalid_kind", ["below", "above", "bool", "float", "string", "null"])
def test_rerank_model_config_rejects_invalid_integer_bounds(
    field: str,
    minimum: int,
    maximum: int,
    invalid_kind: str,
) -> None:
    invalid = {
        "below": minimum - 1,
        "above": maximum + 1,
        "bool": True,
        "float": 1.5,
        "string": "1",
        "null": None,
    }[invalid_kind]
    with pytest.raises(ValueError):
        RerankModelConfig(route(), **{field: invalid})


@pytest.mark.parametrize("model", [ProviderConfig, ChatModelConfig, EmbeddingModelConfig, RerankModelConfig])
def test_model_configuration_objects_are_frozen_dataclasses(model: type[object]) -> None:
    assert model.__dataclass_params__.frozen is True
    assert fields(model)


def test_model_configuration_rejects_direct_mutation() -> None:
    config = route()
    with pytest.raises(FrozenInstanceError):
        config.model = "changed"


def test_model_configuration_replace_revalidates_all_invariants() -> None:
    config = route()
    assert replace(config, timeout_seconds=1).timeout_seconds == 1.0
    with pytest.raises(ValueError):
        replace(config, timeout_seconds=0)
