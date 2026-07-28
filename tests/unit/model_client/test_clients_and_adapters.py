"""Chat/Embedding 运行层、重试和真实协议 Adapter 测试。"""

import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest

from ModelClient import (
    ChatClient,
    ChatMessage,
    ChatModelConfig,
    ChatRequest,
    EmbeddingClient,
    EmbeddingModelConfig,
    EmbeddingVector,
    ModelAuthenticationError,
    ModelRateLimitError,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    ModelTransportError,
    ProviderCapabilities,
    ProviderConfig,
    ReasoningOptions,
    ResponseFormat,
    ToolCall,
    ToolDefinition,
)
from ModelClient.adapters.ark_multimodal import ArkMultimodalEmbeddingProvider
from ModelClient.adapters.openai_compatible_chat import OpenAICompatibleChatProvider
from ModelClient.retry import normalize_provider_error, retry_delay


def route(adapter: str, **overrides: object) -> ProviderConfig:
    values = {
        "provider": "test-provider",
        "adapter": adapter,
        "model": "test-model",
        "base_url": "https://example.com/v1",
        "max_retries": 1,
        "retry_base_delay_seconds": 0.01,
        "retry_max_delay_seconds": 0.1,
        "max_concurrent": 2,
    }
    values.update(overrides)
    return ProviderConfig(**values)


@dataclass
class FlakyChatProvider:
    failures: int = 0
    invalid: bool = False
    provider_name: str = "test-provider"
    model: str = "test-model"
    is_remote: bool = False
    capabilities: ProviderCapabilities = ProviderCapabilities()

    def __post_init__(self) -> None:
        self.calls = 0

    def _result(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("temporary timeout")
        if self.invalid:
            return "invalid"
        return ModelResponse("ok", self.model, self.provider_name)

    def complete(self, request: ChatRequest):
        return self._result()

    async def complete_async(self, request: ChatRequest):
        return self._result()

    def stream(self, request: ChatRequest):
        if self.failures and self.calls < self.failures:
            self.calls += 1
            raise TimeoutError("temporary timeout")
        self.calls += 1
        yield ModelStreamEvent(kind="content_delta", content_delta="ok")
        yield ModelStreamEvent(kind="done", finish_reason="stop")

    async def stream_async(self, request: ChatRequest):
        for event in self.stream(request):
            yield event

    def health_check(self) -> dict[str, object]:
        if self.failures:
            raise TimeoutError("offline")
        return {"ok": True}


def test_chat_client_applies_default_output_limit_and_retries_only_retryable_failure() -> None:
    provider = FlakyChatProvider(failures=1)
    delays: list[float] = []
    client = ChatClient(
        ChatModelConfig(route("test-adapter"), max_output_tokens=128),
        provider,
        sleep=delays.append,
    )
    response = client.complete("hello")

    assert response.content == "ok"
    assert provider.calls == 2
    assert len(delays) == 1


def test_chat_client_rejects_invalid_normalized_response_and_health_check_returns_stable_error() -> None:
    invalid = ChatClient(ChatModelConfig(route("test-adapter", max_retries=0)), FlakyChatProvider(invalid=True))
    with pytest.raises(ModelResponseError, match="invalid normalized"):
        invalid.complete("hello")

    unhealthy = ChatClient(ChatModelConfig(route("test-adapter")), FlakyChatProvider(failures=1))
    assert unhealthy.health_check() == {
        "ok": False,
        "provider": "test-provider",
        "model": "test-model",
        "error_code": "MODEL_TRANSPORT",
    }


def test_chat_stream_does_not_retry_after_any_event_has_been_emitted() -> None:
    class PartialProvider(FlakyChatProvider):
        def stream(self, request: ChatRequest):
            yield ModelStreamEvent(kind="content_delta", content_delta="partial")
            raise TimeoutError("lost connection")

    client = ChatClient(ChatModelConfig(route("test-adapter", max_retries=3)), PartialProvider())
    stream = client.stream("hello")
    assert next(stream).content_delta == "partial"
    with pytest.raises(ModelTransportError):
        next(stream)


@dataclass
class FlakyEmbeddingProvider:
    failures: int = 0
    wrong_dimension: bool = False
    provider_name: str = "test-provider"
    model: str = "test-model"
    is_remote: bool = False

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    async def embed(self, text: str, *, is_query: bool) -> EmbeddingVector:
        self.calls.append((text, is_query))
        if len(self.calls) <= self.failures:
            raise ConnectionError("temporary")
        return EmbeddingVector((1, 0, 0) if self.wrong_dimension else (1, 0))


def test_embedding_client_trims_input_batches_documents_and_preserves_query_mode() -> None:
    provider = FlakyEmbeddingProvider()
    client = EmbeddingClient(
        EmbeddingModelConfig(route("test-adapter", max_retries=0), dimension=2, max_batch_size=2),
        provider,
    )

    query = asyncio.run(client.embed_query("  query  "))
    documents = asyncio.run(client.embed_documents(("a", "b", "c")))
    assert query.dimension == 2
    assert len(documents) == 3
    assert provider.calls == [("query", True), ("a", False), ("b", False), ("c", False)]


def test_embedding_client_retries_transport_error_and_rejects_wrong_dimension_and_oversize() -> None:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    provider = FlakyEmbeddingProvider(failures=1)
    client = EmbeddingClient(
        EmbeddingModelConfig(route("test-adapter"), dimension=2, max_input_chars=5),
        provider,
        async_sleep=record_sleep,
    )
    assert asyncio.run(client.embed_query("query")).dimension == 2
    assert len(sleeps) == 1
    with pytest.raises(ValueError, match="character bound"):
        asyncio.run(client.embed_query("too-long"))

    wrong = EmbeddingClient(
        EmbeddingModelConfig(route("test-adapter", max_retries=0), dimension=2),
        FlakyEmbeddingProvider(wrong_dimension=True),
    )
    with pytest.raises(ModelResponseError, match="dimension"):
        asyncio.run(wrong.embed_query("query"))


class StatusError(RuntimeError):
    def __init__(self, status_code: int, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (StatusError(401, "bad key"), ModelAuthenticationError),
        (StatusError(429, "too many", 2.0), ModelRateLimitError),
        (StatusError(503, "offline"), ModelTransportError),
        (TimeoutError("timeout"), ModelTransportError),
    ],
)
def test_provider_errors_are_normalized_by_status_and_transport(error: Exception, expected: type[Exception]) -> None:
    assert isinstance(normalize_provider_error(error), expected)


def test_retry_delay_honors_bounded_retry_after_and_deterministic_jitter() -> None:
    error = ModelRateLimitError("limited", retry_after_seconds=100)
    assert retry_delay(0, base_delay=1, max_delay=10, error=error) == 10
    assert retry_delay(
        2,
        base_delay=1,
        max_delay=10,
        error=ModelTransportError("temporary"),
        uniform=lambda low, high: 1.0,
    ) == 4


def test_openai_compatible_adapter_sends_messages_tools_schema_and_normalizes_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "server-model",
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    config = ChatModelConfig(
        route("openai_compatible_chat"),
        structured_output_mode="json_schema",
        reasoning=True,
    )
    provider = OpenAICompatibleChatProvider(config, api_key="secret")
    provider._sync_client.close()
    provider._sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    request = ChatRequest(
        messages=(ChatMessage(role="user", content="run"),),
        tools=(ToolDefinition("inspect", "检查", {"type": "object"}),),
        tool_choice="auto",
        response_format=ResponseFormat("result", {"type": "object"}),
        reasoning=ReasoningOptions("medium"),
    )
    response = provider.complete(request)
    payload = captured["payload"]

    assert response.content == "done"
    assert response.usage.total_tokens == 3
    assert payload["tools"][0]["function"]["name"] == "inspect"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["reasoning_effort"] == "medium"
    assert captured["headers"]["authorization"] == "Bearer secret"


def test_openai_compatible_adapter_normalizes_tool_calls_and_stream_events() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {"name": "inspect", "arguments": '{"path":"."}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                content=(
                    b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            ),
        ]
    )
    provider = OpenAICompatibleChatProvider(ChatModelConfig(route("openai_compatible_chat")), api_key="")
    provider._sync_client.close()
    provider._sync_client = httpx.Client(transport=httpx.MockTransport(lambda request: next(responses)))
    request = ChatRequest(messages=(ChatMessage(role="user", content="run"),))

    response = provider.complete(request)
    events = tuple(provider.stream(request))
    assert response.tool_calls == (ToolCall("call-1", "inspect", {"path": "."}),)
    assert [event.kind for event in events] == ["content_delta", "done"]
    assert events[0].content_delta == "hello"


def test_ark_embedding_adapter_uses_text_part_query_parameters_and_validates_vector() -> None:
    config = EmbeddingModelConfig(
        route("ark_multimodal", max_retries=0),
        dimension=2,
        input_mode="multimodal",
        query_parameters={"instruction": "query"},
        document_parameters={"instruction": "document"},
    )
    provider = ArkMultimodalEmbeddingProvider(config, api_key="secret")
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(200, json={"data": [{"embedding": [3, 4, 99]}]})

    async def invoke() -> tuple[EmbeddingVector, EmbeddingVector]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider._client = lambda: client
            return await provider.embed("query", is_query=True), await provider.embed("document", is_query=False)

    query_vector, document_vector = asyncio.run(invoke())
    assert query_vector.values == pytest.approx((0.6, 0.8))
    assert document_vector.dimension == 2
    assert captured[0]["instruction"] == "query"
    assert captured[1]["instruction"] == "document"
    assert captured[0]["input"] == [{"type": "text", "text": "query"}]


def test_ark_adapter_rejects_wrong_mode_missing_key_and_reserved_parameters() -> None:
    with pytest.raises(Exception, match="multimodal"):
        ArkMultimodalEmbeddingProvider(
            EmbeddingModelConfig(route("ark_multimodal"), dimension=2, input_mode="text"),
            api_key="secret",
        )
    with pytest.raises(Exception, match="API key"):
        ArkMultimodalEmbeddingProvider(
            EmbeddingModelConfig(route("ark_multimodal"), dimension=2, input_mode="multimodal"),
            api_key="",
        )
    with pytest.raises(Exception, match="override"):
        ArkMultimodalEmbeddingProvider(
            EmbeddingModelConfig(
                route("ark_multimodal"),
                dimension=2,
                input_mode="multimodal",
                query_parameters={"model": "override"},
            ),
            api_key="secret",
        )
