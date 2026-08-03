"""模型供应商公共路由与各能力的严格配置。"""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias
from urllib.parse import urlsplit

ModelCapability = Literal["chat", "embedding", "rerank"]
EmbeddingInputMode = Literal["text", "multimodal"]
ChatStructuredOutputMode = Literal["none", "json_object", "json_schema"]

_PROVIDER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


@dataclass(frozen=True)
class ProviderConfig:
    """分离厂商身份和协议适配器的公共路由；只保存具名凭据引用。"""

    provider: str
    adapter: str
    model: str
    base_url: str = ""
    credential_ref: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 30.0
    max_concurrent: int = 16
    max_response_bytes: int = 8 * 1024 * 1024
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = str(self.provider or "").strip().lower()
        adapter = str(self.adapter or "").strip().lower()
        model = str(self.model or "").strip()
        base_url = str(self.base_url or "").strip().rstrip("/")
        credential_ref = str(self.credential_ref or "").strip().lower()
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "credential_ref", credential_ref)

        if not provider or not _PROVIDER_NAME.fullmatch(provider):
            raise ValueError("model provider must be a non-empty normalized name")
        if not adapter or not _PROVIDER_NAME.fullmatch(adapter):
            raise ValueError("model adapter must be a non-empty normalized name")
        if not model or not _MODEL_NAME.fullmatch(model):
            raise ValueError("model name contains unsupported characters")
        if credential_ref and not _PROVIDER_NAME.fullmatch(credential_ref):
            raise ValueError("model credential_ref must be a normalized credential name")
        if base_url:
            _validate_base_url(base_url)

        timeout = _positive_float(self.timeout_seconds, "model timeout_seconds", maximum=600.0)
        retry_base = _positive_float(
            self.retry_base_delay_seconds,
            "model retry_base_delay_seconds",
            maximum=60.0,
        )
        retry_max = _positive_float(
            self.retry_max_delay_seconds,
            "model retry_max_delay_seconds",
            maximum=300.0,
        )
        if retry_max < retry_base:
            raise ValueError("model retry_max_delay_seconds cannot be below retry_base_delay_seconds")
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "retry_base_delay_seconds", retry_base)
        object.__setattr__(self, "retry_max_delay_seconds", retry_max)

        _bounded_int(self.max_retries, "model max_retries", minimum=0, maximum=10)
        _bounded_int(self.max_concurrent, "model max_concurrent", minimum=1, maximum=4096)
        _bounded_int(
            self.max_response_bytes,
            "model max_response_bytes",
            minimum=1024,
            maximum=64 * 1024 * 1024,
        )

        headers = _string_mapping(self.extra_headers, "model extra_headers")
        if {key.casefold() for key in headers} & {"authorization", "proxy-authorization"}:
            raise ValueError("model extra_headers cannot contain authorization credentials")
        object.__setattr__(self, "extra_headers", headers)
        extra_body = _json_mapping(self.extra_body, "model extra_body")
        reserved = {"adapter", "api_key", "base_url", "model", "provider"} & set(extra_body)
        if reserved:
            raise ValueError(f"model extra_body cannot override route identity: {sorted(reserved)}")
        object.__setattr__(self, "extra_body", extra_body)

@dataclass(frozen=True)
class ChatModelConfig:
    """对话生成能力配置；供应商专用字段只能放入 route.extra_body。"""

    route: ProviderConfig
    context_window_tokens: int = 64_000
    max_output_tokens: int | None = None
    structured_output_mode: ChatStructuredOutputMode = "none"
    reasoning: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.route, ProviderConfig):
            raise TypeError("chat route must be ProviderConfig")
        _bounded_int(
            self.context_window_tokens,
            "chat context_window_tokens",
            minimum=1_024,
            maximum=10_000_000,
        )
        if self.max_output_tokens is not None:
            _bounded_int(
                self.max_output_tokens,
                "chat max_output_tokens",
                minimum=1,
                maximum=10_000_000,
            )
            if self.max_output_tokens >= self.context_window_tokens:
                raise ValueError("chat max_output_tokens must be below context_window_tokens")
        if self.structured_output_mode not in {"none", "json_object", "json_schema"}:
            raise ValueError(
                "chat structured_output_mode must be none, json_object or json_schema"
            )
        if not isinstance(self.reasoning, bool):
            raise TypeError("chat reasoning must be boolean")

    @property
    def capability(self) -> Literal["chat"]:
        return "chat"

@dataclass(frozen=True)
class EmbeddingModelConfig:
    """稠密向量能力配置；不包含任何具体平台或模型枚举。"""

    route: ProviderConfig
    dimension: int
    input_mode: EmbeddingInputMode = "text"
    max_batch_size: int = 32
    max_input_chars: int = 16_000
    query_parameters: Mapping[str, object] = field(default_factory=dict)
    document_parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.route, ProviderConfig):
            raise TypeError("embedding route must be ProviderConfig")
        _bounded_int(self.dimension, "embedding dimension", minimum=1, maximum=65_536)
        if self.input_mode not in {"text", "multimodal"}:
            raise ValueError("embedding input_mode must be text or multimodal")
        _bounded_int(self.max_batch_size, "embedding max_batch_size", minimum=1, maximum=2048)
        _bounded_int(self.max_input_chars, "embedding max_input_chars", minimum=1, maximum=1_000_000)
        object.__setattr__(
            self,
            "query_parameters",
            _json_mapping(self.query_parameters, "embedding query_parameters"),
        )
        object.__setattr__(
            self,
            "document_parameters",
            _json_mapping(self.document_parameters, "embedding document_parameters"),
        )

    @property
    def capability(self) -> Literal["embedding"]:
        return "embedding"

@dataclass(frozen=True)
class RerankModelConfig:
    """文本重排能力配置；具体评分协议由注册的 provider 实现。"""

    route: ProviderConfig
    max_documents: int = 100
    max_query_chars: int = 8_000
    max_document_chars: int = 16_000

    def __post_init__(self) -> None:
        if not isinstance(self.route, ProviderConfig):
            raise TypeError("rerank route must be ProviderConfig")
        _bounded_int(self.max_documents, "rerank max_documents", minimum=1, maximum=2048)
        _bounded_int(self.max_query_chars, "rerank max_query_chars", minimum=1, maximum=1_000_000)
        _bounded_int(
            self.max_document_chars,
            "rerank max_document_chars",
            minimum=1,
            maximum=1_000_000,
        )

    @property
    def capability(self) -> Literal["rerank"]:
        return "rerank"


CapabilityConfig: TypeAlias = ChatModelConfig | EmbeddingModelConfig | RerankModelConfig


def _validate_base_url(base_url: str) -> None:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("model base_url must be one credential-free HTTP(S) origin or API prefix")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("remote model base_url must use HTTPS")


def _is_loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _positive_float(value: object, label: str, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not 0 < number <= maximum:
        raise ValueError(f"{label} must be between zero and {maximum:g}")
    return number


def _bounded_int(value: object, label: str, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")


def _string_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(item, str):
            raise ValueError(f"{label} must contain non-empty string keys and string values")
        result[key] = item
    return result


def _json_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} keys must be non-empty strings")
        result[key] = item
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain JSON-serializable values") from exc
    return result


__all__ = [
    "CapabilityConfig",
    "ChatModelConfig",
    "ChatStructuredOutputMode",
    "EmbeddingInputMode",
    "EmbeddingModelConfig",
    "ModelCapability",
    "ProviderConfig",
    "RerankModelConfig",
]
