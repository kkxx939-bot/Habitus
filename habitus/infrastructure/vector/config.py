"""供应商身份与向量数据库协议分离的严格路由配置。"""

from __future__ import annotations

import ipaddress
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import urlsplit

from habitus.foundation.validation import is_int, is_number

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_COLLECTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_INT_BOUNDS = {
    "max_retries": (0, 10),
    "max_concurrent": (1, 4096),
    "max_response_bytes": (1024, 64 * 1024 * 1024),
}


@dataclass(frozen=True)
class VectorStoreRouteConfig:
    """描述服务来源、协议 Adapter、具名凭据引用和通用运行边界。"""

    provider: str = "local"
    adapter: str = "local_sqlite"
    base_url: str = ""
    credential_ref: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 30.0
    max_concurrent: int = 16
    max_response_bytes: int = 8 * 1024 * 1024
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = _normalized_name(self.provider, "vector store provider")
        adapter = _normalized_name(self.adapter, "vector store adapter")
        base_url = self.base_url.strip().rstrip("/")
        if base_url:
            _validate_base_url(base_url)
        credential_ref = self.credential_ref.strip().lower()
        if credential_ref and _NAME.fullmatch(credential_ref) is None:
            raise ValueError("vector store credential_ref must be a normalized name")
        headers = _header_mapping(self.extra_headers)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "credential_ref", credential_ref)
        object.__setattr__(self, "extra_headers", MappingProxyType(headers))

        timeout = _bounded_float(self.timeout_seconds, "timeout_seconds", 0.001, 600.0)
        retry_base = _bounded_float(
            self.retry_base_delay_seconds,
            "retry_base_delay_seconds",
            0.001,
            60.0,
        )
        retry_max = _bounded_float(
            self.retry_max_delay_seconds,
            "retry_max_delay_seconds",
            0.001,
            300.0,
        )
        if retry_max < retry_base:
            raise ValueError("vector store retry_max_delay_seconds cannot be below retry_base_delay_seconds")
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "retry_base_delay_seconds", retry_base)
        object.__setattr__(self, "retry_max_delay_seconds", retry_max)
        _bounded_int(self.max_retries, "max_retries", *_INT_BOUNDS["max_retries"])
        _bounded_int(self.max_concurrent, "max_concurrent", *_INT_BOUNDS["max_concurrent"])
        _bounded_int(self.max_response_bytes, "max_response_bytes", *_INT_BOUNDS["max_response_bytes"])

    @classmethod
    def from_mapping(cls, value: object) -> VectorStoreRouteConfig:
        data = _strict_dataclass_mapping(cls, value, "vector store route")
        default = cls()
        names_message = "vector store provider and adapter must be strings"
        return cls(
            provider=_text_option(data, "provider", default.provider, names_message),
            adapter=_text_option(data, "adapter", default.adapter, names_message),
            base_url=_text_option(data, "base_url", default.base_url, "vector store base_url must be a string"),
            credential_ref=_text_option(
                data, "credential_ref", default.credential_ref, "vector store credential_ref must be a string"
            ),
            timeout_seconds=_number_option(data, "timeout_seconds", default.timeout_seconds),
            max_retries=_int_option(data, "max_retries", default.max_retries),
            retry_base_delay_seconds=_number_option(data, "retry_base_delay_seconds", default.retry_base_delay_seconds),
            retry_max_delay_seconds=_number_option(data, "retry_max_delay_seconds", default.retry_max_delay_seconds),
            max_concurrent=_int_option(data, "max_concurrent", default.max_concurrent),
            max_response_bytes=_int_option(data, "max_response_bytes", default.max_response_bytes),
            extra_headers=_headers_option(data.get("extra_headers", default.extra_headers)),
        )


@dataclass(frozen=True)
class VectorStoreConfig:
    """选择一个协议路由和逻辑 Collection；厂商字段留给 Adapter 严格解析。"""

    route: VectorStoreRouteConfig = field(default_factory=VectorStoreRouteConfig)
    collection: str = "memory"
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        collection = self.collection.strip()
        if _COLLECTION.fullmatch(collection) is None:
            raise ValueError("vector store collection contains unsupported characters")
        options = _json_mapping(self.options, "vector store options")
        object.__setattr__(self, "collection", collection)
        object.__setattr__(self, "options", MappingProxyType(options))

    @property
    def provider(self) -> str:
        return self.route.provider

    @property
    def adapter(self) -> str:
        return self.route.adapter

    @classmethod
    def from_mapping(cls, value: object) -> VectorStoreConfig:
        data = _strict_dataclass_mapping(cls, value, "vector store")
        route = VectorStoreRouteConfig.from_mapping(data.get("route", {}))
        collection = data.get("collection", "memory")
        if not isinstance(collection, str):
            raise TypeError("vector store collection must be a string")
        options = data.get("options", {})
        if not isinstance(options, Mapping):
            raise TypeError("vector store options must be an object")
        if any(not isinstance(name, str) or not name for name in options):
            raise ValueError("vector store options keys must be non-empty strings")
        return cls(route=route, collection=collection, options=options)


@dataclass(frozen=True)
class VectorStoreRequirements:
    """上层索引交给任意 Adapter 校验的统一容量与向量要求。"""

    dimension: int
    max_records: int
    max_search_hits: int
    max_record_chars: int

    def __post_init__(self) -> None:
        _bounded_int(self.dimension, "requirements.dimension", 1, 65_536)
        _bounded_int(self.max_records, "requirements.max_records", 1, 100_000_000)
        _bounded_int(self.max_search_hits, "requirements.max_search_hits", 1, 1_000_000)
        _bounded_int(self.max_record_chars, "requirements.max_record_chars", 1, 1_000_000)
        if self.max_search_hits > self.max_records:
            raise ValueError("vector store required search hits cannot exceed required records")


def _normalized_name(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if _NAME.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a normalized name")
    return normalized


def _strict_dataclass_mapping(
    model_type: Any,
    value: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    allowed = {item.name for item in fields(model_type)}
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{label} keys must be non-empty strings")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}")
    return dict(cast(Mapping[str, object], value))


def _text_option(data: Mapping[str, object], name: str, default: str, message: str) -> str:
    """YAML 里的文本字段：类型不对按值对象原来的那句报错（照抄 ``__post_init__``）。"""

    value = data.get(name, default)
    if not isinstance(value, str):
        raise TypeError(message)
    return value


def _number_option(data: Mapping[str, object], name: str, default: float) -> float:
    value = data.get(name, default)
    if not is_number(value):
        raise ValueError(f"vector store {name} must be numeric")
    return value


def _int_option(data: Mapping[str, object], name: str, default: int) -> int:
    value = data.get(name, default)
    if not is_int(value):
        minimum, maximum = _INT_BOUNDS[name]
        raise ValueError(f"vector store {name} must be between {minimum} and {maximum}")
    return value


def _headers_option(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("vector store extra_headers must be an object")
    headers: dict[str, str] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not isinstance(item, str):
            raise ValueError("vector store extra_headers must contain string keys and values")
        headers[name] = item
    return headers


def _header_mapping(value: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    seen: set[str] = set()
    for name, item in value.items():
        if not name.strip():
            raise ValueError("vector store extra_headers must contain string keys and values")
        if any(character in name for character in "\r\n:") or "\r" in item or "\n" in item:
            raise ValueError("vector store extra_headers contain invalid HTTP characters")
        if name.casefold() in {"authorization", "proxy-authorization", "host"}:
            raise ValueError("vector store extra_headers cannot contain reserved authentication headers")
        normalized = name.casefold()
        if normalized in seen:
            raise ValueError("vector store extra_headers contain duplicate case-insensitive names")
        seen.add(normalized)
        result[name] = item
    return result


def _json_mapping(value: Mapping[str, object], label: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, item in value.items():
        if not name:
            raise ValueError(f"{label} keys must be non-empty strings")
        result[name] = item
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain JSON-serializable values") from exc
    return result


def _validate_base_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("vector store base_url must be one credential-free HTTP(S) origin or API prefix")
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
        raise ValueError("remote vector store base_url must use HTTPS")


def _is_loopback(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _bounded_float(value: float, name: str, minimum: float, maximum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"vector store {name} must be between {minimum} and {maximum}")
    return number


def _bounded_int(value: int, name: str, minimum: int, maximum: int) -> None:
    if not minimum <= value <= maximum:
        raise ValueError(f"vector store {name} must be between {minimum} and {maximum}")


__all__ = ["VectorStoreConfig", "VectorStoreRequirements", "VectorStoreRouteConfig"]
