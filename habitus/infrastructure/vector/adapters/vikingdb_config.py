"""VikingDB 协议的严格资源、认证和容量配置。"""

from __future__ import annotations

import ipaddress
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from habitus.infrastructure.vector.config import VectorStoreRequirements, VectorStoreRouteConfig

VikingDBAuthMode = Literal["api_key", "ak_sk", "private_headers"]
VikingDBSchemaMode = Literal["managed", "precreated"]

_FIELDS = {
    "auth_mode",
    "schema_mode",
    "project_name",
    "index_name",
    "region",
    "console_url",
    "credential_headers",
    "upsert_batch_size",
    "fetch_batch_size",
    "delete_batch_size",
    "search_page_size",
    "scan_page_size",
    "max_search_hits",
    "max_records",
    "index_sync_timeout_seconds",
    "index_sync_poll_interval_seconds",
}
_RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CREDENTIAL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_CREDENTIAL_PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_.-]{0,127})\}")
_REGION = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$")
_PUBLIC_REGIONS = {
    "ap-southeast-1",
    "cn-beijing",
    "cn-guangzhou",
    "cn-shanghai",
}


@dataclass(frozen=True)
class VikingDBVectorStoreConfig:
    """一个 VikingDB Collection 及其公开云或私有部署连接语义。"""

    auth_mode: VikingDBAuthMode = "api_key"
    schema_mode: VikingDBSchemaMode = "precreated"
    project_name: str = "default"
    index_name: str = "default"
    region: str = "cn-beijing"
    console_url: str = ""
    credential_headers: Mapping[str, str] = field(default_factory=dict)
    upsert_batch_size: int = 100
    fetch_batch_size: int = 64
    delete_batch_size: int = 100
    search_page_size: int = 64
    scan_page_size: int = 64
    max_search_hits: int = 10_000
    max_records: int = 1_000_000
    index_sync_timeout_seconds: float = 60.0
    index_sync_poll_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.auth_mode not in {"api_key", "ak_sk", "private_headers"}:
            raise ValueError("vikingdb auth_mode must be api_key, ak_sk or private_headers")
        if self.schema_mode not in {"managed", "precreated"}:
            raise ValueError("vikingdb schema_mode must be managed or precreated")
        if self.schema_mode == "managed" and self.auth_mode != "ak_sk":
            raise ValueError("vikingdb managed schema requires ak_sk authentication")
        for name in ("project_name", "index_name"):
            raw = getattr(self, name)
            if not isinstance(raw, str):
                raise TypeError(f"vikingdb {name} must be a string")
            normalized = raw.strip()
            if _RESOURCE_NAME.fullmatch(normalized) is None:
                raise ValueError(f"vikingdb {name} contains unsupported characters")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.region, str):
            raise TypeError("vikingdb region must be a string")
        region = self.region.strip().lower()
        if self.auth_mode in {"api_key", "ak_sk"} and _REGION.fullmatch(region) is None:
            raise ValueError("public vikingdb authentication requires a normalized region")
        object.__setattr__(self, "region", region)
        if not isinstance(self.console_url, str):
            raise TypeError("vikingdb console_url must be a string")
        console_url = self.console_url.strip().rstrip("/")
        if console_url:
            _validate_origin(console_url, "console_url")
        if self.auth_mode == "ak_sk" and not console_url and region not in _PUBLIC_REGIONS:
            raise ValueError("unknown public region requires an explicit vikingdb console_url")
        if self.auth_mode != "ak_sk" and console_url:
            raise ValueError("vikingdb console_url is only valid for ak_sk authentication")
        object.__setattr__(self, "console_url", console_url)
        object.__setattr__(
            self,
            "credential_headers",
            MappingProxyType(_credential_headers(self.credential_headers)),
        )
        if self.auth_mode != "private_headers" and self.credential_headers:
            raise ValueError("vikingdb credential_headers are only valid for private_headers mode")
        for name, minimum, maximum in (
            ("upsert_batch_size", 1, 100),
            ("fetch_batch_size", 1, 100),
            ("delete_batch_size", 1, 100),
            ("search_page_size", 1, 5000),
            ("scan_page_size", 1, 5000),
            ("max_search_hits", 1, 1_000_000),
            ("max_records", 1, 100_000_000),
        ):
            _bounded_int(getattr(self, name), name, minimum, maximum)
        if self.max_search_hits > self.max_records:
            raise ValueError("vikingdb max_search_hits cannot exceed max_records")
        timeout = _bounded_float(
            self.index_sync_timeout_seconds,
            "index_sync_timeout_seconds",
            0.1,
            3600.0,
        )
        interval = _bounded_float(
            self.index_sync_poll_interval_seconds,
            "index_sync_poll_interval_seconds",
            0.01,
            60.0,
        )
        if interval > timeout:
            raise ValueError("vikingdb index sync poll interval cannot exceed its timeout")
        object.__setattr__(self, "index_sync_timeout_seconds", timeout)
        object.__setattr__(self, "index_sync_poll_interval_seconds", interval)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> VikingDBVectorStoreConfig:
        if not isinstance(value, Mapping):
            raise TypeError("vikingdb vector options must be an object")
        unknown = sorted(set(value) - _FIELDS)
        if unknown:
            raise ValueError(f"vikingdb vector options contain unknown fields: {unknown}")
        return cls(**cast(dict[str, Any], dict(value)))

    def data_url(self, route: VectorStoreRouteConfig) -> str:
        """解析数据面地址；私有化部署必须显式给出地址。"""

        if not isinstance(route, VectorStoreRouteConfig):
            raise TypeError("route must be VectorStoreRouteConfig")
        if route.base_url:
            _validate_origin(route.base_url, "route.base_url")
            return route.base_url
        if self.auth_mode == "private_headers":
            raise ValueError("private vikingdb requires route.base_url")
        if self.region not in _PUBLIC_REGIONS:
            raise ValueError("unknown public region requires an explicit vikingdb route.base_url")
        return f"https://api-vikingdb.vikingdb.{self.region}.volces.com"

    def validate_requirements(
        self,
        requirements: VectorStoreRequirements,
        route: VectorStoreRouteConfig,
    ) -> None:
        """在 Adapter 构造边界校验上层索引容量，不让 Config 根依赖具体厂商。"""

        if not isinstance(requirements, VectorStoreRequirements):
            raise TypeError("requirements must be VectorStoreRequirements")
        if not isinstance(route, VectorStoreRouteConfig):
            raise TypeError("route must be VectorStoreRouteConfig")
        self.data_url(route)
        if requirements.max_records > self.max_records:
            raise ValueError("memory vector max_records cannot exceed vikingdb max_records")
        if requirements.max_search_hits > self.max_search_hits:
            raise ValueError("memory vector max_search_hits cannot exceed vikingdb max_search_hits")
        estimated_record_bytes = (
            requirements.max_record_chars * 4
            + requirements.dimension * 24
            + 8_192
        )
        if (
            max(self.fetch_batch_size, self.search_page_size, self.scan_page_size)
            * estimated_record_bytes
            > route.max_response_bytes
        ):
            raise ValueError("vikingdb read page sizes can exceed route.max_response_bytes")

    def resolved_console_url(self) -> str:
        if self.auth_mode != "ak_sk":
            raise ValueError("only ak_sk authentication has a public console endpoint")
        if self.console_url:
            return self.console_url
        return f"https://vikingdb.{self.region}.volcengineapi.com"


def _credential_headers(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("vikingdb credential_headers must be an object")
    result: dict[str, str] = {}
    seen: set[str] = set()
    for header_name, credential_name in value.items():
        if (
            not isinstance(header_name, str)
            or not header_name.strip()
            or any(character in header_name for character in "\r\n:")
        ):
            raise ValueError("vikingdb credential header names are invalid")
        if not isinstance(credential_name, str) or not credential_name:
            raise ValueError("vikingdb credential header templates must be non-empty strings")
        placeholders = credential_template_names(credential_name)
        if not placeholders:
            raise ValueError("vikingdb credential header templates must reference a credential")
        residual = _CREDENTIAL_PLACEHOLDER.sub("", credential_name)
        if "{" in residual or "}" in residual or "\r" in residual or "\n" in residual:
            raise ValueError("vikingdb credential header template is invalid")
        if header_name.casefold() in {"host", "proxy-authorization"}:
            raise ValueError("vikingdb credential_headers contain a reserved HTTP header")
        normalized_header = header_name.casefold()
        if normalized_header in seen:
            raise ValueError("vikingdb credential_headers contain duplicate header names")
        seen.add(normalized_header)
        result[header_name] = credential_name
    return result


def credential_template_names(value: str) -> tuple[str, ...]:
    """返回 Header 模板引用的规范凭据名。"""

    if not isinstance(value, str):
        raise TypeError("vikingdb credential header template must be a string")
    names = tuple(match.group(1).lower() for match in _CREDENTIAL_PLACEHOLDER.finditer(value))
    for name in names:
        if _CREDENTIAL_NAME.fullmatch(name) is None:
            raise ValueError("vikingdb credential header template contains an invalid name")
    return names


def render_credential_template(value: str, credentials: Mapping[str, str]) -> str:
    """只替换受控凭据占位符，不执行 Python format 表达式。"""

    names = credential_template_names(value)
    missing = sorted(set(names) - set(credentials))
    if missing:
        raise ValueError(f"vikingdb credential header template is missing values: {missing}")
    return _CREDENTIAL_PLACEHOLDER.sub(
        lambda match: credentials[match.group(1).lower()],
        value,
    )


def _validate_origin(value: str, name: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"vikingdb {name} must be one credential-free HTTP(S) origin")
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
        raise ValueError(f"remote vikingdb {name} must use HTTPS")


def _is_loopback(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"vikingdb {name} must be between {minimum} and {maximum}")


def _bounded_float(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"vikingdb {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"vikingdb {name} must be between {minimum} and {maximum}")
    return result


def bounded_retry_after(value: str | None, maximum: float) -> float | None:
    """解析服务端 Retry-After，拒绝非有限值和负值。"""

    if value is None:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if not math.isfinite(result) or result < 0:
        return None
    return min(result, maximum)


__all__ = [
    "VikingDBAuthMode",
    "VikingDBSchemaMode",
    "VikingDBVectorStoreConfig",
    "bounded_retry_after",
    "credential_template_names",
    "render_credential_template",
]
