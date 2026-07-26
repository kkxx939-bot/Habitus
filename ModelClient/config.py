"""模型供应商公共路由与各能力的严格配置。"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast
from urllib.parse import urlsplit

ModelCapability = Literal["chat", "embedding", "rerank"]
EmbeddingInputMode = Literal["text", "multimodal"]
ChatStructuredOutputMode = Literal["none", "json_object", "json_schema"]

_PROVIDER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ProviderConfig:
    """分离厂商身份和协议适配器的公共路由；只保存密钥环境变量名。"""

    provider: str
    adapter: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
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
        api_key_env = str(self.api_key_env or "").strip()
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "api_key_env", api_key_env)

        if not provider or not _PROVIDER_NAME.fullmatch(provider):
            raise ValueError("model provider must be a non-empty normalized name")
        if not adapter or not _PROVIDER_NAME.fullmatch(adapter):
            raise ValueError("model adapter must be a non-empty normalized name")
        if not model or not _MODEL_NAME.fullmatch(model):
            raise ValueError("model name contains unsupported characters")
        if api_key_env and not _ENV_NAME.fullmatch(api_key_env):
            raise ValueError("model api_key_env must be an environment variable name")
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

    @classmethod
    def from_env(
        cls,
        prefix: str,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ProviderConfig:
        """从统一前缀读取路由元数据；密钥值仍留在环境中。"""

        normalized_prefix = _env_prefix(prefix)
        values = os.environ if environ is None else environ
        return cls(
            provider=values.get(f"{normalized_prefix}_PROVIDER", ""),
            adapter=values.get(f"{normalized_prefix}_ADAPTER", ""),
            model=values.get(f"{normalized_prefix}_MODEL", ""),
            base_url=values.get(f"{normalized_prefix}_BASE_URL", ""),
            api_key_env=values.get(f"{normalized_prefix}_API_KEY_ENV", ""),
            timeout_seconds=_env_float(values, f"{normalized_prefix}_TIMEOUT_SECONDS", 30.0),
            max_retries=_env_int(values, f"{normalized_prefix}_MAX_RETRIES", 2),
            retry_base_delay_seconds=_env_float(
                values,
                f"{normalized_prefix}_RETRY_BASE_DELAY_SECONDS",
                0.5,
            ),
            retry_max_delay_seconds=_env_float(
                values,
                f"{normalized_prefix}_RETRY_MAX_DELAY_SECONDS",
                30.0,
            ),
            max_concurrent=_env_int(values, f"{normalized_prefix}_MAX_CONCURRENT", 16),
            max_response_bytes=_env_int(
                values,
                f"{normalized_prefix}_MAX_RESPONSE_BYTES",
                8 * 1024 * 1024,
            ),
            extra_headers=_env_string_mapping(values, f"{normalized_prefix}_EXTRA_HEADERS_JSON"),
            extra_body=_env_json_mapping(values, f"{normalized_prefix}_EXTRA_BODY_JSON"),
        )


@dataclass(frozen=True)
class ChatModelConfig:
    """对话生成能力配置；供应商专用字段只能放入 route.extra_body。"""

    route: ProviderConfig
    max_output_tokens: int | None = None
    structured_output_mode: ChatStructuredOutputMode = "none"
    reasoning: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.route, ProviderConfig):
            raise TypeError("chat route must be ProviderConfig")
        if self.max_output_tokens is not None:
            _bounded_int(
                self.max_output_tokens,
                "chat max_output_tokens",
                minimum=1,
                maximum=10_000_000,
            )
        if self.structured_output_mode not in {"none", "json_object", "json_schema"}:
            raise ValueError(
                "chat structured_output_mode must be none, json_object or json_schema"
            )
        if not isinstance(self.reasoning, bool):
            raise TypeError("chat reasoning must be boolean")

    @property
    def capability(self) -> Literal["chat"]:
        return "chat"

    @classmethod
    def from_env(
        cls,
        prefix: str = "M2BOS_CHAT",
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ChatModelConfig:
        """加载一条完整对话模型配置，默认使用 m2bOS 的 Chat 前缀。"""

        normalized_prefix = _env_prefix(prefix)
        values = os.environ if environ is None else environ
        mode = values.get(
            f"{normalized_prefix}_STRUCTURED_OUTPUT_MODE",
            "none",
        ).strip().lower()
        return cls(
            route=ProviderConfig.from_env(normalized_prefix, environ=values),
            max_output_tokens=_env_optional_int(
                values,
                f"{normalized_prefix}_MAX_OUTPUT_TOKENS",
            ),
            structured_output_mode=cast(ChatStructuredOutputMode, mode),
            reasoning=_env_bool(values, f"{normalized_prefix}_REASONING", False),
        )


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

    @classmethod
    def from_env(
        cls,
        prefix: str = "M2BOS_EMBEDDING",
        *,
        environ: Mapping[str, str] | None = None,
    ) -> EmbeddingModelConfig:
        """加载一条完整向量配置，默认使用 m2bOS 的 embedding 前缀。"""

        normalized_prefix = _env_prefix(prefix)
        values = os.environ if environ is None else environ
        input_mode = values.get(f"{normalized_prefix}_INPUT_MODE", "text").strip().lower()
        return cls(
            route=ProviderConfig.from_env(normalized_prefix, environ=values),
            dimension=_env_int(values, f"{normalized_prefix}_DIMENSION", 0),
            input_mode=cast(EmbeddingInputMode, input_mode),
            max_batch_size=_env_int(values, f"{normalized_prefix}_MAX_BATCH_SIZE", 32),
            max_input_chars=_env_int(values, f"{normalized_prefix}_MAX_INPUT_CHARS", 16_000),
            query_parameters=_env_json_mapping(
                values,
                f"{normalized_prefix}_QUERY_PARAMETERS_JSON",
            ),
            document_parameters=_env_json_mapping(
                values,
                f"{normalized_prefix}_DOCUMENT_PARAMETERS_JSON",
            ),
        )


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


def _env_prefix(value: object) -> str:
    prefix = str(value or "").strip().upper()
    if not prefix or not _ENV_PREFIX.fullmatch(prefix):
        raise ValueError("model environment prefix must be a normalized environment name")
    return prefix


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _env_optional_int(environ: Mapping[str, str], name: str) -> int | None:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_json_mapping(environ: Mapping[str, str], name: str) -> dict[str, object]:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain an object")
    return dict(value)


def _env_string_mapping(environ: Mapping[str, str], name: str) -> dict[str, str]:
    return _string_mapping(_env_json_mapping(environ, name), name)


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
