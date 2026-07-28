"""结构化输出的语法修复、JSON Schema 和有界重试测试。"""

import asyncio
from dataclasses import dataclass

import pytest

from ModelClient import (
    ChatClient,
    ChatModelConfig,
    ChatRequest,
    ModelResponse,
    ModelStructuredOutputError,
    ProviderCapabilities,
    ProviderConfig,
    StructuredChatClient,
    parse_json_response,
    validate_json_schema,
)
from ModelClient.schema_validation import JSONSchemaValidationError

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "items"],
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "items": {"type": "array", "items": {"type": "string"}},
    },
}


@pytest.mark.parametrize(
    ("source", "mode"),
    [
        ('{"status":"ok","items":[]}', "strict"),
        ('```json\n{"status":"ok","items":[]}\n```', "code_fence"),
        ('result: {"status":"ok","items":[]}', "extracted"),
        ('{"status":"ok","items":[],}', "trailing_comma_repair"),
        ("{'status':'ok','items':[]}", "python_literal_repair"),
    ],
)
def test_json_parser_audits_each_allowed_syntax_path(source: str, mode: str) -> None:
    parsed = parse_json_response(source)
    assert parsed.value == {"status": "ok", "items": []}
    assert parsed.mode == mode


@pytest.mark.parametrize("source", ["", "NaN", "{'items': {1, 2}}", "not-json"])
def test_json_parser_never_invents_semantics_or_accepts_non_json_values(source: str) -> None:
    with pytest.raises(ValueError):
        parse_json_response(source)


def test_json_schema_rejects_unknown_fields_wrong_types_missing_and_invalid_enum() -> None:
    assert validate_json_schema({"status": "ok", "items": ["a"]}, SCHEMA) == {
        "status": "ok",
        "items": ["a"],
    }
    for invalid in (
        {"status": "ok", "items": [], "extra": True},
        {"status": "ok", "items": "not-array"},
        {"status": "bad", "items": []},
        {"status": "ok"},
    ):
        with pytest.raises(JSONSchemaValidationError):
            validate_json_schema(invalid, SCHEMA)


@dataclass
class QueueProvider:
    responses: list[ModelResponse]
    provider_name: str = "test-provider"
    model: str = "test-model"
    is_remote: bool = False
    capabilities: ProviderCapabilities = ProviderCapabilities(structured_output_mode="json_schema")

    def __post_init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def complete_async(self, request: ChatRequest) -> ModelResponse:
        return self.complete(request)

    def stream(self, request: ChatRequest):
        return iter(())

    async def stream_async(self, request: ChatRequest):
        if False:
            yield None

    def health_check(self) -> dict[str, object]:
        return {"ok": True}


def _response(content: str, *, finish_reason: str = "stop") -> ModelResponse:
    return ModelResponse(content, "test-model", "test-provider", finish_reason=finish_reason)


def _structured(provider: QueueProvider, *, retries: int = 1) -> StructuredChatClient:
    config = ChatModelConfig(
        ProviderConfig(
            provider="test-provider",
            adapter="test-adapter",
            model="test-model",
            max_retries=0,
        ),
        structured_output_mode="json_schema",
    )
    return StructuredChatClient(ChatClient(config, provider), validation_retries=retries)


def test_structured_client_returns_valid_value_and_injects_schema_instruction() -> None:
    provider = QueueProvider([_response('{"status":"ok","items":["a"]}')])
    result = _structured(provider).complete_json("return data", schema=SCHEMA)

    assert result.value == {"status": "ok", "items": ["a"]}
    assert result.validation_attempts == 1
    assert provider.requests[0].response_format is not None
    assert provider.requests[0].messages[0].role == "system"


def test_structured_client_retries_with_exact_failure_without_lossy_type_conversion() -> None:
    provider = QueueProvider(
        [
            _response('{"status":"ok","items":"a"}'),
            _response('{"status":"ok","items":["a"]}'),
        ]
    )
    result = _structured(provider).complete_json("return data", schema=SCHEMA)

    assert result.validation_attempts == 2
    correction = provider.requests[1].messages[-1].content or ""
    assert "Do not stringify arrays or objects" in correction
    assert "invent missing facts" in correction


def test_structured_client_stops_after_bound_and_reports_validation_phase() -> None:
    provider = QueueProvider([_response("not-json"), _response("still-not-json")])
    with pytest.raises(ModelStructuredOutputError, match="json_parse validation after 2"):
        _structured(provider).complete_json("return data", schema=SCHEMA)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "safety"])
def test_structured_client_rejects_truncated_or_safety_blocked_responses(finish_reason: str) -> None:
    provider = QueueProvider([_response("{}", finish_reason=finish_reason)])
    with pytest.raises(ModelStructuredOutputError, match="response validation"):
        _structured(provider, retries=0).complete_json("return data", schema=SCHEMA)


def test_structured_async_path_uses_same_validation_and_retry_semantics() -> None:
    provider = QueueProvider([_response('{"status":"ok","items":[]}')])
    result = asyncio.run(_structured(provider).complete_json_async("return data", schema=SCHEMA))
    assert result.value["status"] == "ok"

