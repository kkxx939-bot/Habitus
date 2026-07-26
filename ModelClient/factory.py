"""按能力和供应商显式注册、构造模型 Provider。"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from ModelClient.client import ChatClient
from ModelClient.config import (
    CapabilityConfig,
    ChatModelConfig,
    EmbeddingModelConfig,
    ModelCapability,
    ProviderConfig,
    RerankModelConfig,
)
from ModelClient.contracts import ChatProvider, ModelConfigurationError
from ModelClient.embedding import Embedder, EmbeddingClient, EmbeddingProvider
from ModelClient.rerank import Reranker

_ADAPTER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ProviderBuildContext:
    """工厂交给 builder 的单次构造上下文；密钥不参与 repr。"""

    config: CapabilityConfig
    api_key: str = field(default="", repr=False)

    @property
    def route(self) -> ProviderConfig:
        return self.config.route


ProviderBuilder = Callable[[ProviderBuildContext], object]


class ProviderFactory:
    """以 `(capability, adapter)` 为唯一键的显式协议适配器工厂。"""

    def __init__(self) -> None:
        self._builders: dict[tuple[ModelCapability, str], ProviderBuilder] = {}

    def register_adapter(
        self,
        capability: ModelCapability,
        adapter: str,
        builder: ProviderBuilder,
    ) -> None:
        key = self._key(capability, adapter)
        if key in self._builders:
            raise ValueError(f"provider adapter is already registered: {key[0]}/{key[1]}")
        if not callable(builder):
            raise TypeError("provider builder must be callable")
        self._builders[key] = builder

    def registered_adapters(self, capability: ModelCapability) -> tuple[str, ...]:
        normalized_capability = self._capability(capability)
        return tuple(
            sorted(
                adapter
                for item_capability, adapter in self._builders
                if item_capability == normalized_capability
            )
        )

    def create(
        self,
        config: CapabilityConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> object:
        if not isinstance(config, ChatModelConfig | EmbeddingModelConfig | RerankModelConfig):
            raise TypeError("config must be a supported capability config")
        key = (config.capability, config.route.adapter)
        builder = self._builders.get(key)
        if builder is None:
            raise ModelConfigurationError(
                f"adapter is not registered for capability: {config.capability}/{config.route.adapter}"
            )
        environment = os.environ if environ is None else environ
        context = ProviderBuildContext(
            config=config,
            api_key=self._credential(config.route, environment),
        )
        component = builder(context)
        self._validate_component(config, component)
        return component

    def create_chat_provider(
        self,
        config: ChatModelConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ChatProvider:
        component = self.create(config, environ=environ)
        return cast(ChatProvider, component)

    def create_chat_client(
        self,
        config: ChatModelConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ChatClient:
        return ChatClient(config, self.create_chat_provider(config, environ=environ))

    def create_embedder(
        self,
        config: EmbeddingModelConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> Embedder:
        provider = cast(EmbeddingProvider, self.create(config, environ=environ))
        return EmbeddingClient(config, provider)

    def create_reranker(
        self,
        config: RerankModelConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> Reranker:
        component = self.create(config, environ=environ)
        return cast(Reranker, component)

    @staticmethod
    def _credential(route: ProviderConfig, environ: Mapping[str, str]) -> str:
        if not route.api_key_env:
            return ""
        value = str(environ.get(route.api_key_env, "")).strip()
        if not value:
            raise ModelConfigurationError(
                f"model credential environment variable is missing: {route.api_key_env}"
            )
        return value

    @staticmethod
    def _validate_component(config: CapabilityConfig, component: object) -> None:
        required_methods: dict[ModelCapability, tuple[str, ...]] = {
            "chat": ("complete", "complete_async", "stream", "stream_async", "health_check"),
            "embedding": ("embed",),
            "rerank": ("rerank",),
        }
        missing = tuple(
            name for name in required_methods[config.capability] if not callable(getattr(component, name, None))
        )
        if missing:
            raise ModelConfigurationError(
                f"provider builder returned an invalid {config.capability} component; missing: {', '.join(missing)}"
            )
        provider_name = str(getattr(component, "provider_name", "")).strip().lower()
        model = str(getattr(component, "model", "")).strip()
        if provider_name != config.route.provider or model != config.route.model:
            raise ModelConfigurationError("provider identity does not match its route config")

    @staticmethod
    def _key(capability: ModelCapability, adapter: str) -> tuple[ModelCapability, str]:
        return ProviderFactory._capability(capability), ProviderFactory._adapter(adapter)

    @staticmethod
    def _capability(value: object) -> ModelCapability:
        if value not in {"chat", "embedding", "rerank"}:
            raise ValueError("model capability must be chat, embedding or rerank")
        return cast(ModelCapability, value)

    @staticmethod
    def _adapter(value: object) -> str:
        adapter = str(value or "").strip().lower()
        if not adapter or not _ADAPTER_NAME.fullmatch(adapter):
            raise ValueError("model adapter must be a non-empty normalized name")
        return adapter


__all__ = ["ProviderBuildContext", "ProviderBuilder", "ProviderFactory"]
