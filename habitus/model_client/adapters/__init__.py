"""内置协议适配器的集中注册入口。"""

from typing import TypeVar

from habitus.model_client.adapters.anthropic_messages import AnthropicMessagesProvider
from habitus.model_client.adapters.openai_compatible_chat import OpenAICompatibleChatProvider
from habitus.model_client.config import ChatModelConfig
from habitus.model_client.errors import ModelConfigurationError
from habitus.model_client.factory import ProviderBuildContext, ProviderFactory

_Config = TypeVar("_Config", bound=ChatModelConfig)


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


__all__ = [
    "AnthropicMessagesProvider",
    "OpenAICompatibleChatProvider",
    "build_anthropic_messages_provider",
    "build_openai_compatible_chat_provider",
    "register_builtin_adapters",
]
