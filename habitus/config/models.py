"""模型路由和结构化输出的外部配置分组。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from habitus.config.loader import ConfigError, construct_config, group_fields, require_field_types, required_field
from habitus.model_client import (
    ChatModelConfig,
    EmbeddingModelConfig,
    ProviderConfig,
    RerankModelConfig,
)

_ModelConfigValue = TypeVar("_ModelConfigValue")


@dataclass(frozen=True)
class StructuredOutputConfig:
    """结构化响应的纯语法修复和有界校验重试。"""

    allow_json_repair: bool = True
    validation_retries: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.validation_retries <= 5:
            raise ValueError("validation_retries must be between zero and five")


@dataclass(frozen=True)
class ModelConfig:
    """Chat、Embedding、可选 Rerank 和结构化输出的统一配置。"""

    chat: ChatModelConfig
    embedding: EmbeddingModelConfig
    rerank: RerankModelConfig | None = None
    structured_output: StructuredOutputConfig = field(default_factory=StructuredOutputConfig)

    @classmethod
    def from_mapping(cls, value: object) -> ModelConfig:
        data = group_fields(cls, value, "config.models")
        rerank_value = data.get("rerank")
        return cls(
            chat=_chat_config(
                required_field(data, "chat", path="config.models"),
                "config.models.chat",
            ),
            embedding=_embedding_config(
                required_field(data, "embedding", path="config.models"),
                "config.models.embedding",
            ),
            rerank=(None if rerank_value is None else _rerank_config(rerank_value, "config.models.rerank")),
            structured_output=construct_config(
                StructuredOutputConfig,
                data.get("structured_output", {}),
                "config.models.structured_output",
            ),
        )


def _construct_model_config(
    model_type: type[_ModelConfigValue],
    data: Mapping[str, object],
    path: str,
) -> _ModelConfigValue:
    require_field_types(model_type, data, path=path)
    try:
        return model_type(**dict(data))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid '{path}': {exc}") from exc


def _provider_config(value: object, path: str) -> ProviderConfig:
    data = group_fields(ProviderConfig, value, path)
    for name in ("provider", "adapter", "model", "base_url", "credential_ref"):
        field_value = data.get(name)
        if field_value is not None and not isinstance(field_value, str):
            raise ConfigError(f"'{path}.{name}' must be a string")
    return _construct_model_config(ProviderConfig, data, path)


def _chat_config(value: object, path: str) -> ChatModelConfig:
    data = group_fields(ChatModelConfig, value, path)
    route = _provider_config(
        required_field(data, "route", path=path),
        f"{path}.route",
    )
    return _construct_model_config(ChatModelConfig, {**data, "route": route}, path)


def _embedding_config(value: object, path: str) -> EmbeddingModelConfig:
    data = group_fields(EmbeddingModelConfig, value, path)
    for name in ("query_prefix", "document_prefix"):
        if name in data and not isinstance(data[name], str):  # YAML 里留空是 null，也要拒：不写就是空前缀
            raise ConfigError(f"'{path}.{name}' must be a string")
    route = _provider_config(
        required_field(data, "route", path=path),
        f"{path}.route",
    )
    return _construct_model_config(
        EmbeddingModelConfig,
        {**data, "route": route},
        path,
    )


def _rerank_config(value: object, path: str) -> RerankModelConfig:
    data = group_fields(RerankModelConfig, value, path)
    route = _provider_config(
        required_field(data, "route", path=path),
        f"{path}.route",
    )
    return _construct_model_config(RerankModelConfig, {**data, "route": route}, path)


__all__ = ["ModelConfig", "StructuredOutputConfig"]
