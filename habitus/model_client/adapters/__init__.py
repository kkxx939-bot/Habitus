"""内置协议适配器的集中注册入口。"""

from typing import TypeVar

from habitus.model_client.adapters.anthropic_messages import AnthropicMessagesProvider
from habitus.model_client.adapters.openai_compatible_chat import OpenAICompatibleChatProvider
from habitus.model_client.adapters.openai_compatible_embedding import OpenAICompatibleEmbeddingProvider
from habitus.model_client.adapters.openai_compatible_rerank import OpenAICompatibleRerankProvider
from habitus.model_client.config import ChatModelConfig, EmbeddingModelConfig, RerankModelConfig
from habitus.model_client.contracts import ModelConfigurationError
from habitus.model_client.factory import ProviderBuildContext, ProviderFactory

_Config = TypeVar("_Config", ChatModelConfig, EmbeddingModelConfig, RerankModelConfig)


def build_openai_compatible_chat_provider(
    context: ProviderBuildContext,
) -> OpenAICompatibleChatProvider:
    """从统一构造上下文创建一个 OpenAI-compatible Chat Provider。"""

    config = _config(context, ChatModelConfig, "openai_compatible_chat")
    return OpenAICompatibleChatProvider(config, api_key=context.api_key, outbound=context.outbound)


def build_anthropic_messages_provider(
    context: ProviderBuildContext,
) -> AnthropicMessagesProvider:
    """从统一构造上下文创建一个 Anthropic Messages Chat Provider。"""

    config = _config(context, ChatModelConfig, "anthropic_messages")
    return AnthropicMessagesProvider(config, api_key=context.api_key, outbound=context.outbound)


def build_openai_compatible_embedding_provider(
    context: ProviderBuildContext,
) -> OpenAICompatibleEmbeddingProvider:
    """从统一构造上下文创建一个 OpenAI-compatible Embedding Provider。"""

    config = _config(context, EmbeddingModelConfig, "openai_compatible_embedding")
    return OpenAICompatibleEmbeddingProvider(config, api_key=context.api_key, outbound=context.outbound)


def build_openai_compatible_rerank_provider(
    context: ProviderBuildContext,
) -> OpenAICompatibleRerankProvider:
    """从统一构造上下文创建一个 OpenAI-compatible Rerank Provider。"""

    config = _config(context, RerankModelConfig, "openai_compatible_rerank")
    return OpenAICompatibleRerankProvider(config, api_key=context.api_key, outbound=context.outbound)


def _config(context: ProviderBuildContext, expected: type[_Config], adapter: str) -> _Config:
    """各 builder 共用的能力配置核对：工厂按 ``(capability, adapter)`` 分发，配错能力的配置在这里报清楚。"""

    config = context.config
    if not isinstance(config, expected):
        raise ModelConfigurationError(f"{adapter} adapter requires {expected.__name__}")
    return config


def register_builtin_adapters(factory: ProviderFactory) -> None:
    """注册 Habitus 当前内置的协议适配器，不执行任何外部调用。"""

    factory.register_adapter(
        "chat",
        "openai_compatible_chat",
        build_openai_compatible_chat_provider,
    )
    factory.register_adapter(
        "chat",
        "anthropic_messages",
        build_anthropic_messages_provider,
    )
    factory.register_adapter(
        "embedding",
        "openai_compatible_embedding",
        build_openai_compatible_embedding_provider,
    )
    factory.register_adapter(
        "rerank",
        "openai_compatible_rerank",
        build_openai_compatible_rerank_provider,
    )


__all__ = [
    "AnthropicMessagesProvider",
    "OpenAICompatibleChatProvider",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleRerankProvider",
    "build_anthropic_messages_provider",
    "build_openai_compatible_chat_provider",
    "build_openai_compatible_embedding_provider",
    "build_openai_compatible_rerank_provider",
    "register_builtin_adapters",
]
