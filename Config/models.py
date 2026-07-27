"""模型路由和结构化输出的外部配置分组。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from Config.loader import ConfigError, construct_config, group_fields, required_field
from ModelClient import (
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
        if not isinstance(self.allow_json_repair, bool):
            raise TypeError("allow_json_repair must be boolean")
        if (
            isinstance(self.validation_retries, bool)
            or not isinstance(self.validation_retries, int)
            or not 0 <= self.validation_retries <= 5
        ):
            raise ValueError("validation_retries must be between zero and five")


@dataclass(frozen=True)
class ModelConfig:
    """Chat、Embedding、可选 Rerank 和结构化输出的统一配置。"""

    chat: ChatModelConfig
    embedding: EmbeddingModelConfig
    rerank: RerankModelConfig | None = None
    structured_output: StructuredOutputConfig = field(default_factory=StructuredOutputConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.chat, ChatModelConfig):
            raise TypeError("models.chat must be ChatModelConfig")
        if not isinstance(self.embedding, EmbeddingModelConfig):
            raise TypeError("models.embedding must be EmbeddingModelConfig")
        if self.rerank is not None and not isinstance(self.rerank, RerankModelConfig):
            raise TypeError("models.rerank must be RerankModelConfig or None")
        if not isinstance(self.structured_output, StructuredOutputConfig):
            raise TypeError("models.structured_output must be StructuredOutputConfig")

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
    try:
        return model_type(**dict(data))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid '{path}': {exc}") from exc


def _provider_config(value: object, path: str) -> ProviderConfig:
    data = group_fields(ProviderConfig, value, path)
    for name in ("provider", "adapter", "model", "base_url", "api_key_env"):
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
