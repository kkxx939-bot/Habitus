"""内置协议适配器的集中注册入口。"""

from ModelClient.adapters.ark_multimodal import ArkMultimodalEmbeddingProvider
from ModelClient.adapters.openai_compatible_chat import OpenAICompatibleChatProvider
from ModelClient.adapters.openai_compatible_rerank import OpenAICompatibleRerankProvider
from ModelClient.config import ChatModelConfig, EmbeddingModelConfig, RerankModelConfig
from ModelClient.contracts import ModelConfigurationError
from ModelClient.factory import ProviderBuildContext, ProviderFactory


def build_ark_multimodal_embedding_provider(
    context: ProviderBuildContext,
) -> ArkMultimodalEmbeddingProvider:
    """从统一构造上下文创建一个方舟图文向量 Provider。"""

    config = context.config
    if not isinstance(config, EmbeddingModelConfig):
        raise ModelConfigurationError("ark_multimodal adapter requires EmbeddingModelConfig")
    return ArkMultimodalEmbeddingProvider(config, api_key=context.api_key)


def build_openai_compatible_chat_provider(
    context: ProviderBuildContext,
) -> OpenAICompatibleChatProvider:
    """从统一构造上下文创建一个 OpenAI-compatible Chat Provider。"""

    config = context.config
    if not isinstance(config, ChatModelConfig):
        raise ModelConfigurationError(
            "openai_compatible_chat adapter requires ChatModelConfig"
        )
    return OpenAICompatibleChatProvider(config, api_key=context.api_key)


def build_openai_compatible_rerank_provider(
    context: ProviderBuildContext,
) -> OpenAICompatibleRerankProvider:
    """从统一构造上下文创建一个 OpenAI-compatible Rerank Provider。"""

    config = context.config
    if not isinstance(config, RerankModelConfig):
        raise ModelConfigurationError(
            "openai_compatible_rerank adapter requires RerankModelConfig"
        )
    return OpenAICompatibleRerankProvider(config, api_key=context.api_key)


def register_builtin_adapters(factory: ProviderFactory) -> None:
    """注册 Habitus 当前内置的协议适配器，不执行任何外部调用。"""

    if not isinstance(factory, ProviderFactory):
        raise TypeError("factory must be a ProviderFactory")
    factory.register_adapter(
        "embedding",
        "ark_multimodal",
        build_ark_multimodal_embedding_provider,
    )
    factory.register_adapter(
        "chat",
        "openai_compatible_chat",
        build_openai_compatible_chat_provider,
    )
    factory.register_adapter(
        "rerank",
        "openai_compatible_rerank",
        build_openai_compatible_rerank_provider,
    )


__all__ = [
    "ArkMultimodalEmbeddingProvider",
    "OpenAICompatibleChatProvider",
    "OpenAICompatibleRerankProvider",
    "build_ark_multimodal_embedding_provider",
    "build_openai_compatible_chat_provider",
    "build_openai_compatible_rerank_provider",
    "register_builtin_adapters",
]
