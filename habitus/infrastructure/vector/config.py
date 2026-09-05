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

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_COLLECTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")


@dataclass(frozen=True)
class VectorStoreRouteConfig:
    """描述服务来源、协议 Adapter、具名凭据引用和通用运行边界。"""

    provider: str = "volcengine"
    adapter: str = "vikingdb"
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
        if not isinstance(self.provider, str) or not isinstance(self.adapter, str):
            raise TypeError("vector store provider and adapter must be strings")
        if not isinstance(self.base_url, str):
            raise TypeError("vector store base_url must be a string")
        provider = _normalized_name(self.provider, "vector store provider")
        adapter = _normalized_name(self.adapter, "vector store adapter")
        base_url = self.base_url.strip().rstrip("/")
        if base_url:
            _validate_base_url(base_url)
        if not isinstance(self.credential_ref, str):
            raise TypeError("vector store credential_ref must be a string")
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
        _bounded_int(self.max_retries, "max_retries", 0, 10)
        _bounded_int(self.max_concurrent, "max_concurrent", 1, 4096)
        _bounded_int(self.max_response_bytes, "max_response_bytes", 1024, 64 * 1024 * 1024)

    @classmethod
    def from_mapping(cls, value: object) -> VectorStoreRouteConfig:
        data = _strict_dataclass_mapping(cls, value, "vector store route")
        return cls(**cast(dict[str, Any], data))


@dataclass(frozen=True)
class VectorStoreConfig:
    """选择一个协议路由和逻辑 Collection；厂商字段留给 Adapter 严格解析。"""

    route: VectorStoreRouteConfig = field(default_factory=VectorStoreRouteConfig)
    collection: str = "memory"
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.route, VectorStoreRouteConfig):
            raise TypeError("vector store route must be VectorStoreRouteConfig")
        if not isinstance(self.collection, str):
            raise TypeError("vector store collection must be a string")
        collection = self.collection.strip()
        if _COLLECTION.fullmatch(collection) is None:
            raise ValueError("vector store collection contains unsupported characters")
        if not isinstance(self.options, Mapping):
            raise TypeError("vector store options must be an object")
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
        return cls(
            route=route,
            collection=cast(str, data.get("collection", "memory")),
            options=cast(Mapping[str, object], data.get("options", {})),
        )


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


def _normalized_name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
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


def _header_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("vector store extra_headers must be an object")
    result: dict[str, str] = {}
    seen: set[str] = set()
    for name, item in value.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(item, str):
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


def _json_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    result: dict[str, object] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name:
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


def _bounded_float(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"vector store {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"vector store {name} must be between {minimum} and {maximum}")
    return number


def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"vector store {name} must be between {minimum} and {maximum}")


__all__ = ["VectorStoreConfig", "VectorStoreRequirements", "VectorStoreRouteConfig"]
