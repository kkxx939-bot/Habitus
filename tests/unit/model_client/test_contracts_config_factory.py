"""模型公共契约、供应商路由和显式 Adapter 工厂测试。"""

from dataclasses import dataclass

import pytest

from habitus.model_client import (
    ChatMessage,
    ChatModelConfig,
    ChatRequest,
    ModelConfigurationError,
    ModelResponse,
    ProviderBuildContext,
    ProviderCapabilities,
    ProviderConfig,
    ProviderFactory,
    ReasoningOptions,
    ResponseFormat,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from habitus.model_client.adapters import (
    OpenAICompatibleChatProvider,
    register_builtin_adapters,
)


def route(**overrides: object) -> ProviderConfig:
    values = {
        "provider": "test-provider",
        "adapter": "test-adapter",
        "model": "test-model",
        "base_url": "https://example.com/v1",
        "credential_ref": "test-credential",
        "max_retries": 0,
    }
    values.update(overrides)
    return ProviderConfig(**values)


def test_provider_route_normalizes_identity_but_never_embeds_secret_values() -> None:
    config = route(provider=" Test.Provider ", adapter=" Test-Adapter ", base_url="https://example.com/v1/")
    assert config.provider == "test.provider"
    assert config.adapter == "test-adapter"
    assert config.base_url == "https://example.com/v1"
    assert config.credential_ref == "test-credential"


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_url": "http://remote.example.com/v1"},
        {"base_url": "https://user:secret@example.com/v1"},
        {"base_url": "https://example.com/v1?token=secret"},
        {"credential_ref": "not/valid"},
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


@dataclass
class FakeChatProvider:
    """只满足工厂组件校验的假 Chat Provider，不做任何网络调用。"""

    provider_name: str = "test-provider"
    model: str = "test-model"
    is_remote: bool = True

    def prepare(self, request: object, context: object = None) -> object:
        raise NotImplementedError

    def complete(self, prepared: object) -> object:
        raise NotImplementedError

    async def complete_async(self, prepared: object) -> object:
        raise NotImplementedError

    def stream(self, prepared: object) -> object:
        raise NotImplementedError

    async def stream_async(self, prepared: object) -> object:
        raise NotImplementedError

    def health_check(self) -> dict[str, object]:
        return {}

    async def aclose(self) -> None:
        return None


def test_factory_resolves_by_capability_and_adapter() -> None:
    factory = ProviderFactory()
    factory.register_adapter("chat", "test-adapter", lambda context: FakeChatProvider())
    provider = factory.create_chat_provider(ChatModelConfig(route()), api_key="secret")

    assert isinstance(provider, FakeChatProvider)
    assert factory.registered_adapters("chat") == ("test-adapter",)


def test_factory_does_not_guess_adapter_or_silently_fall_back() -> None:
    factory = ProviderFactory()
    config = ChatModelConfig(route(adapter="unknown"))
    with pytest.raises(ModelConfigurationError, match="not registered"):
        factory.create_chat_provider(config, api_key="secret")


def test_factory_requires_declared_credential_and_matching_component_identity() -> None:
    factory = ProviderFactory()
    factory.register_adapter("chat", "test-adapter", lambda context: FakeChatProvider(model="wrong"))
    config = ChatModelConfig(route())
    with pytest.raises(ModelConfigurationError, match="credential"):
        factory.create_chat_provider(config, api_key="")
    with pytest.raises(ModelConfigurationError, match="identity"):
        factory.create_chat_provider(config, api_key="secret")


def test_factory_rejects_a_component_missing_chat_methods() -> None:
    factory = ProviderFactory()

    @dataclass
    class Incomplete:
        provider_name: str = "test-provider"
        model: str = "test-model"

    factory.register_adapter("chat", "test-adapter", lambda context: Incomplete())
    with pytest.raises(ModelConfigurationError, match="missing"):
        factory.create_chat_provider(ChatModelConfig(route()), api_key="secret")


def test_factory_accepts_only_the_chat_capability() -> None:
    factory = ProviderFactory()
    for capability in ("embedding", "rerank"):
        with pytest.raises(ValueError, match="chat"):
            factory.register_adapter(capability, "test-adapter", lambda context: FakeChatProvider())  # type: ignore[arg-type]


def test_builtin_registry_has_only_chat_protocols() -> None:
    factory = ProviderFactory()
    register_builtin_adapters(factory)
    assert factory.registered_adapters("chat") == ("anthropic_messages", "openai_compatible_chat")


def test_builtin_registry_constructs_real_protocol_adapters_without_network_io() -> None:
    factory = ProviderFactory()
    register_builtin_adapters(factory)
    chat = factory.create_chat_provider(ChatModelConfig(route(adapter="openai_compatible_chat")), api_key="secret")
    assert isinstance(chat, OpenAICompatibleChatProvider)


def test_factory_passes_an_adapter_owned_credential_mapping_without_assuming_api_key() -> None:
    factory = ProviderFactory()
    observed: list[ProviderBuildContext] = []

    def build(context: ProviderBuildContext) -> FakeChatProvider:
        observed.append(context)
        return FakeChatProvider()

    factory.register_adapter("chat", "test-adapter", build)

    factory.create_chat_provider(ChatModelConfig(route()), credentials={"token": "secret"})

    assert dict(observed[0].credentials) == {"token": "secret"}
    assert observed[0].api_key == ""
