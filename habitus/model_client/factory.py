"""按能力和供应商显式注册、构造模型 Provider。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from habitus.foundation.observability import Observer
from habitus.model_client.client import ChatClient
from habitus.model_client.config import (
    CapabilityConfig,
    ChatModelConfig,
    EmbeddingModelConfig,
    ModelCapability,
    ProviderConfig,
    RerankModelConfig,
)
from habitus.model_client.contracts import ChatProvider, ModelConfigurationError
from habitus.model_client.embedding import Embedder, EmbeddingClient, EmbeddingProvider
from habitus.model_client.rerank import ObservedReranker, RerankClient, Reranker, RerankProvider

_ADAPTER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ProviderBuildContext:
    """工厂交给 builder 的单次构造上下文；密钥不参与 repr。"""

    config: CapabilityConfig
    api_key: str = field(default="", repr=False)
    credentials: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.credentials, Mapping):
            raise TypeError("model credentials must be an object")
        resolved: dict[str, str] = {}
        for raw_name, raw_value in self.credentials.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise TypeError("model credential field names must be non-empty strings")
            if not isinstance(raw_value, str):
                raise TypeError("model credential values must be strings")
            resolved[raw_name.strip().lower()] = raw_value
        if not isinstance(self.api_key, str):
            raise TypeError("model api_key must be a string")
        api_key = self.api_key.strip()
        if api_key:
            existing = resolved.get("api_key")
            if existing is not None and existing != api_key:
                raise ModelConfigurationError("model api_key conflicts with credentials.api_key")
            resolved["api_key"] = api_key
        object.__setattr__(self, "credentials", MappingProxyType(resolved))
        object.__setattr__(self, "api_key", resolved.get("api_key", ""))

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
        api_key: str = "",
        credentials: Mapping[str, str] | None = None,
    ) -> object:
        if not isinstance(config, ChatModelConfig | EmbeddingModelConfig | RerankModelConfig):
            raise TypeError("config must be a supported capability config")
        key = (config.capability, config.route.adapter)
        builder = self._builders.get(key)
        if builder is None:
            raise ModelConfigurationError(
                f"adapter is not registered for capability: {config.capability}/{config.route.adapter}"
            )
        context = ProviderBuildContext(
            config=config,
            credentials=self._credential_mapping(
                config.route,
                api_key=api_key,
                credentials=credentials,
            ),
        )
        component = builder(context)
        self._validate_component(config, component)
        return component

    def create_chat_provider(
        self,
        config: ChatModelConfig,
        *,
        api_key: str = "",
        credentials: Mapping[str, str] | None = None,
    ) -> ChatProvider:
        component = self.create(config, api_key=api_key, credentials=credentials)
        return cast(ChatProvider, component)

    def create_chat_client(
        self,
        config: ChatModelConfig,
        *,
        api_key: str = "",
        credentials: Mapping[str, str] | None = None,
        observer: Observer | None = None,
    ) -> ChatClient:
        return ChatClient(
            config,
            self.create_chat_provider(config, api_key=api_key, credentials=credentials),
            observer=observer,
        )

    def create_embedder(
        self,
        config: EmbeddingModelConfig,
        *,
        api_key: str = "",
        credentials: Mapping[str, str] | None = None,
        observer: Observer | None = None,
    ) -> Embedder:
        provider = cast(
            EmbeddingProvider,
            self.create(config, api_key=api_key, credentials=credentials),
        )
        return EmbeddingClient(config, provider, observer=observer)

    def create_reranker(
        self,
        config: RerankModelConfig,
        *,
        api_key: str = "",
        credentials: Mapping[str, str] | None = None,
        observer: Observer | None = None,
    ) -> Reranker:
        component = self.create(config, api_key=api_key, credentials=credentials)
        provider = cast(RerankProvider, component)
        reranker: Reranker = RerankClient(config, provider)
        return reranker if observer is None else ObservedReranker(reranker, observer=observer)

    @staticmethod
    def _credential(route: ProviderConfig, api_key: str) -> str:
        if not isinstance(api_key, str):
            raise TypeError("model api_key must be a string")
        value = api_key.strip()
        if not route.credential_ref:
            if value:
                raise ModelConfigurationError("model route received an undeclared credential")
            return ""
        if not value:
            raise ModelConfigurationError(
                f"model credential is missing for reference: {route.credential_ref}"
            )
        return value

    @staticmethod
    def _credential_mapping(
        route: ProviderConfig,
        *,
        api_key: str,
        credentials: Mapping[str, str] | None,
    ) -> Mapping[str, str]:
        if credentials is None:
            value = ProviderFactory._credential(route, api_key)
            return {} if not value else {"api_key": value}
        if api_key:
            raise ModelConfigurationError(
                "pass either model credentials or the legacy api_key, not both"
            )
        if not isinstance(credentials, Mapping):
            raise TypeError("model credentials must be an object")
        resolved = dict(credentials)
        if not route.credential_ref:
            if resolved:
                raise ModelConfigurationError("model route received undeclared credentials")
            return {}
        if not any(isinstance(value, str) and value.strip() for value in resolved.values()):
            raise ModelConfigurationError(
                f"model credential is missing for reference: {route.credential_ref}"
            )
        return resolved

    @staticmethod
    def _validate_component(config: CapabilityConfig, component: object) -> None:
        required_methods: dict[ModelCapability, tuple[str, ...]] = {
            "chat": (
                "prepare",
                "complete",
                "complete_async",
                "stream",
                "stream_async",
                "health_check",
                "aclose",
            ),
            "embedding": ("embed", "aclose"),
            "rerank": ("rerank", "aclose"),
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
        if value == "chat":
            return "chat"
        if value == "embedding":
            return "embedding"
        if value == "rerank":
            return "rerank"
        raise ValueError("model capability must be chat, embedding or rerank")

    @staticmethod
    def _adapter(value: object) -> str:
        adapter = str(value or "").strip().lower()
        if not adapter or not _ADAPTER_NAME.fullmatch(adapter):
            raise ValueError("model adapter must be a non-empty normalized name")
        return adapter


__all__ = ["ProviderBuildContext", "ProviderBuilder", "ProviderFactory"]
