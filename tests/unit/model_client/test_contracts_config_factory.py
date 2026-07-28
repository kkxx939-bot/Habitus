"""模型公共契约、供应商路由和显式 Adapter 工厂测试。"""

import asyncio
from dataclasses import dataclass

import pytest

from ModelClient import (
    ChatMessage,
    ChatRequest,
    EmbeddingClient,
    EmbeddingModelConfig,
    EmbeddingVector,
    ModelConfigurationError,
    ModelResponse,
    ProviderCapabilities,
    ProviderConfig,
    ProviderFactory,
    ReasoningOptions,
    RerankModelConfig,
    ResponseFormat,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from ModelClient.adapters import register_builtin_adapters


def route(**overrides: object) -> ProviderConfig:
    values = {
        "provider": "test-provider",
        "adapter": "test-adapter",
        "model": "test-model",
        "base_url": "https://example.com/v1",
        "api_key_env": "TEST_API_KEY",
        "max_retries": 0,
    }
    values.update(overrides)
    return ProviderConfig(**values)


def test_provider_route_normalizes_identity_but_never_embeds_secret_values() -> None:
    config = route(provider=" Test.Provider ", adapter=" Test-Adapter ", base_url="https://example.com/v1/")
    assert config.provider == "test.provider"
    assert config.adapter == "test-adapter"
    assert config.base_url == "https://example.com/v1"
    assert config.api_key_env == "TEST_API_KEY"


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_url": "http://remote.example.com/v1"},
        {"base_url": "https://user:secret@example.com/v1"},
        {"base_url": "https://example.com/v1?token=secret"},
        {"api_key_env": "not-valid-name"},
        {"extra_headers": {"Authorization": "secret"}},
        {"extra_body": {"model": "override"}},
    ],
)
def test_provider_route_rejects_insecure_url_credentials_and_identity_overrides(overrides: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        route(**overrides)


def test_chat_message_strictly_distinguishes_assistant_calls_and_tool_results() -> None:
    call = ToolCall("call-1", "workspace.inspect", {"path": "."})
    assistant = ChatMessage(role="assistant", tool_calls=(call,))
    result = ChatMessage(role="tool", content="ok", tool_call_id="call-1")
    assert assistant.tool_calls == (call,)
    assert result.tool_call_id == "call-1"

    with pytest.raises(ValueError, match="only assistant"):
        ChatMessage(role="user", content="text", tool_calls=(call,))
    with pytest.raises(ValueError, match="require tool_call_id"):
        ChatMessage(role="tool", content="result")


def test_chat_request_rejects_unbound_tool_choice_and_invalid_generation_limits() -> None:
    message = ChatMessage(role="user", content="hello")
    with pytest.raises(ValueError, match="tool_choice"):
        ChatRequest(messages=(message,), tool_choice="auto")
    with pytest.raises(ValueError, match="temperature"):
        ChatRequest(messages=(message,), temperature=3)
    with pytest.raises(ValueError, match="positive"):
        ChatRequest(messages=(message,), max_output_tokens=0)


def test_tool_and_structured_contracts_preserve_json_objects_without_type_coercion() -> None:
    tool = ToolDefinition("search", "检索记忆", {"type": "object"}, strict=True)
    format_ = ResponseFormat("memory_batch", {"type": "object"})
    request = ChatRequest(
        messages=(ChatMessage(role="user", content="search"),),
        tools=(tool,),
        tool_choice="auto",
        response_format=format_,
        reasoning=ReasoningOptions("medium"),
    )
    assert request.tools[0].parameters == {"type": "object"}
    assert request.response_format is format_


def test_usage_response_and_capabilities_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TokenUsage(input_tokens=-1)
    with pytest.raises(ValueError, match="content or tool calls"):
        ModelResponse(None, "model", "provider")
    with pytest.raises(ValueError, match="structured_output_mode"):
        ProviderCapabilities(structured_output_mode="boolean")


def test_embedding_vector_is_finite_nonzero_and_l2_normalized() -> None:
    vector = EmbeddingVector((3, 4))
    assert vector.values == pytest.approx((0.6, 0.8))
    assert vector.dimension == 2
    with pytest.raises(ValueError, match="non-zero"):
        EmbeddingVector((0, 0))
    with pytest.raises(ValueError, match="finite"):
        EmbeddingVector((float("nan"), 1))


@dataclass
class FakeEmbeddingProvider:
    provider_name: str = "test-provider"
    model: str = "test-model"
    is_remote: bool = True

    async def embed(self, text: str, *, is_query: bool) -> EmbeddingVector:
        return EmbeddingVector((1, 0))


@dataclass
class FakeReranker:
    provider_name: str = "test-provider"
    model: str = "test-model"
    is_remote: bool = True

    async def rerank(self, query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(1.0 for _ in documents)


def test_factory_resolves_by_capability_and_adapter_and_wraps_embedding_runtime() -> None:
    factory = ProviderFactory()
    factory.register_adapter("embedding", "test-adapter", lambda context: FakeEmbeddingProvider())
    config = EmbeddingModelConfig(route(), dimension=2)
    embedder = factory.create_embedder(config, environ={"TEST_API_KEY": "secret"})

    assert isinstance(embedder, EmbeddingClient)
    assert asyncio.run(embedder.embed_query("query")).dimension == 2
    assert factory.registered_adapters("embedding") == ("test-adapter",)


def test_factory_does_not_guess_adapter_or_silently_fall_back() -> None:
    factory = ProviderFactory()
    config = EmbeddingModelConfig(route(adapter="unknown"), dimension=2)
    with pytest.raises(ModelConfigurationError, match="not registered"):
        factory.create_embedder(config, environ={"TEST_API_KEY": "secret"})


def test_factory_requires_declared_credential_and_matching_component_identity() -> None:
    factory = ProviderFactory()
    factory.register_adapter("embedding", "test-adapter", lambda context: FakeEmbeddingProvider(model="wrong"))
    config = EmbeddingModelConfig(route(), dimension=2)
    with pytest.raises(ModelConfigurationError, match="credential"):
        factory.create_embedder(config, environ={})
    with pytest.raises(ModelConfigurationError, match="identity"):
        factory.create_embedder(config, environ={"TEST_API_KEY": "secret"})


def test_builtin_registry_has_chat_and_embedding_but_no_fake_reranker() -> None:
    factory = ProviderFactory()
    register_builtin_adapters(factory)
    assert factory.registered_adapters("chat") == ("openai_compatible_chat",)
    assert factory.registered_adapters("embedding") == ("ark_multimodal",)
    assert factory.registered_adapters("rerank") == ()

    rerank = RerankModelConfig(route(adapter="future-rerank"))
    with pytest.raises(ModelConfigurationError, match="not registered"):
        factory.create_reranker(rerank, environ={"TEST_API_KEY": "secret"})


def test_factory_can_accept_a_future_real_rerank_adapter_without_changing_retrieval_contract() -> None:
    factory = ProviderFactory()
    factory.register_adapter("rerank", "test-adapter", lambda context: FakeReranker())
    reranker = factory.create_reranker(RerankModelConfig(route()), environ={"TEST_API_KEY": "secret"})
    assert asyncio.run(reranker.rerank("query", ("a", "b"))) == (1.0, 1.0)

