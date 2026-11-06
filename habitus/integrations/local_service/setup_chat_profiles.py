"""chat 能力的向导 Profile：两种协议各一个不绑厂商的模板，外加一个 DeepSeek 预置。

默认（``cloud_setup.default_cloud_selection``）用 OpenAI 兼容模板；模板里的 ``example.`` 模型名与地址是占位值，
由 doctor 与启动前检查（``setup_placeholders``）报"还是占位值"，交互式向导逼用户填真实值，与 Embedding 同一范式。
"""

from __future__ import annotations

from habitus.config import HabitusConfig
from habitus.integrations.local_service.setup_registry import (
    AdapterProductRegistration,
    SetupProfile,
    _api_key_fields,
    _choice,
    _positive,
    _required,
    _text,
)
from habitus.model_client import ModelConfigurationError
from habitus.model_client.adapters.anthropic_messages import validate_messages_route


def chat_adapter_registrations() -> tuple[AdapterProductRegistration, ...]:
    """chat 两种协议适配器的产品元数据：凭据字段都是 ``api_key``；Anthropic Messages 另有无网络的路由校验。"""

    return (
        AdapterProductRegistration("chat", "openai_compatible_chat", _api_key_fields),
        AdapterProductRegistration(
            "chat", "anthropic_messages", _api_key_fields, validate=_validate_anthropic_messages
        ),
    )


def _validate_anthropic_messages(config: HabitusConfig) -> None:
    """doctor 的 adapter_configuration 与启动前检查走这里：协议做不到的设置（json_object、缺 max_output_tokens、
    采样字段等）在组装运行时之前就报原因，规则与 Provider 构造时同一份。"""

    try:
        validate_messages_route(config.models.chat)
    except ModelConfigurationError as exc:
        raise ValueError(str(exc)) from exc


def chat_profiles() -> tuple[SetupProfile, ...]:
    """按注册顺序返回 chat 的全部 Profile。"""

    return (_deepseek_profile(), _openai_compatible_profile(), _anthropic_messages_profile())


def _deepseek_profile() -> SetupProfile:
    """DeepSeek 预置：真实地址与模型名，只在交互式向导里可选，不是默认。"""

    return SetupProfile(
        "chat.deepseek",
        "chat",
        "DeepSeek (OpenAI-compatible)",
        patch={
            "route": {
                "provider": "deepseek",
                "adapter": "openai_compatible_chat",
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com",
                "credential_ref": "deepseek",
                "extra_body": {},
            },
            "context_window_tokens": 64_000,
            "structured_output_mode": "json_object",
            "reasoning": False,
        },
        match={
            "route.provider": "deepseek",
            "route.adapter": "openai_compatible_chat",
            "route.model": "deepseek-chat",
            "route.base_url": "https://api.deepseek.com",
        },
        replace_paths=("route.extra_body", "max_output_tokens"),
    )


def _openai_compatible_profile() -> SetupProfile:
    """OpenAI Chat Completions 协议的不绑厂商模板（发行默认）。"""

    return SetupProfile(
        "chat.openai_compatible.custom",
        "chat",
        "OpenAI-compatible chat endpoint (POST {base_url}/chat/completions)",
        patch={
            "route": {
                "provider": "custom",
                "adapter": "openai_compatible_chat",
                "model": "example.chat-model",
                "base_url": "https://chat.example.com/v1",
                "credential_ref": "chat",
                "extra_body": {},
            },
            "context_window_tokens": 64_000,
            "structured_output_mode": "json_object",
            "reasoning": False,
        },
        match={"route.adapter": "openai_compatible_chat"},
        fields=(
            _text("provider", "Chat provider identifier", "route.provider", "custom"),
            _required("model", "Chat model", "route.model", "", placeholder_values=("example.chat-model",)),
            _required(
                "base_url",
                "Chat API base URL",
                "route.base_url",
                "",
                placeholder_values=("https://chat.example.com/v1",),
            ),
            _text("credential_ref", "Chat credential reference", "route.credential_ref", "chat"),
            _positive("context_window_tokens", "Chat context window tokens", "context_window_tokens", 64_000),
            _choice(
                "structured_output_mode",
                "Chat structured output",
                "structured_output_mode",
                "json_object",
                (("json_object", "JSON object"), ("json_schema", "JSON schema"), ("none", "Prompt-only / none")),
            ),
            _choice(
                "reasoning",
                "Chat reasoning capability",
                "reasoning",
                False,
                ((False, "Disabled"), (True, "Enabled")),
            ),
        ),
        replace_paths=("route.extra_body",),
    )


def _anthropic_messages_profile() -> SetupProfile:
    """Anthropic Messages 协议的不绑厂商模板；协议要求 max_tokens，所以 max_output_tokens 必填。"""

    return SetupProfile(
        "chat.anthropic_messages.custom",
        "chat",
        "Anthropic Messages endpoint (POST {base_url}/v1/messages)",
        patch={
            "route": {
                "provider": "custom",
                "adapter": "anthropic_messages",
                "model": "example.messages-model",
                "base_url": "https://messages.example.com",
                "credential_ref": "chat",
                "extra_body": {},
            },
            "context_window_tokens": 200_000,
            "max_output_tokens": 16_000,
            "structured_output_mode": "json_schema",
            "reasoning": False,
        },
        match={"route.adapter": "anthropic_messages"},
        fields=(
            _text("provider", "Chat provider identifier", "route.provider", "custom"),
            _required("model", "Chat model", "route.model", "", placeholder_values=("example.messages-model",)),
            _required(
                "base_url",
                "Messages API base URL (for example https://api.anthropic.com)",
                "route.base_url",
                "",
                placeholder_values=("https://messages.example.com",),
            ),
            _text("credential_ref", "Chat credential reference", "route.credential_ref", "chat"),
            _positive("context_window_tokens", "Chat context window tokens", "context_window_tokens", 200_000),
            _positive(
                "max_output_tokens",
                "Chat max output tokens (required by the protocol)",
                "max_output_tokens",
                16_000,
            ),
            _choice(
                "structured_output_mode",
                "Chat structured output",
                "structured_output_mode",
                "json_schema",
                (("json_schema", "JSON schema (output_config.format)"), ("none", "Prompt-only / none")),
            ),
            _choice(
                "reasoning",
                "Chat reasoning effort control",
                "reasoning",
                False,
                ((False, "Disabled"), (True, "Enabled")),
            ),
        ),
        replace_paths=("route.extra_body",),
    )


__all__ = ["chat_adapter_registrations", "chat_profiles"]
