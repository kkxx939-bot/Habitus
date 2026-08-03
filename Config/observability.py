"""记忆内核可观测旁路的严格配置。"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from Config.loader import construct_config, group_fields

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class ObservabilityMetricsConfig:
    """无外部依赖的进程内指标边界。"""

    enabled: bool = True
    namespace: str = "m2bos"
    max_recent_events: int = 512
    duration_buckets_seconds: tuple[float, ...] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
        if not isinstance(self.namespace, str) or _NAME.fullmatch(self.namespace) is None:
            raise ValueError("namespace must be a normalized bounded name")
        if (
            isinstance(self.max_recent_events, bool)
            or not isinstance(self.max_recent_events, int)
            or not 1 <= self.max_recent_events <= 10_000
        ):
            raise ValueError("max_recent_events must be between 1 and 10000")
        values = self.duration_buckets_seconds
        if isinstance(values, str | bytes) or not isinstance(values, Sequence):
            raise TypeError("duration_buckets_seconds must be a numeric sequence")
        buckets: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("duration buckets must be numeric")
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise ValueError("duration buckets must be finite and positive")
            buckets.append(number)
        normalized = tuple(buckets)
        if not normalized or tuple(sorted(set(normalized))) != normalized or len(normalized) > 32:
            raise ValueError("duration buckets must be unique ascending values with at most 32 entries")
        object.__setattr__(self, "duration_buckets_seconds", normalized)

    @classmethod
    def from_mapping(cls, value: object) -> ObservabilityMetricsConfig:
        return construct_config(cls, value, "config.observability.metrics")


@dataclass(frozen=True)
class ObservabilityLoggingConfig:
    """结构化事件日志配置；日志不记录输入、输出或密钥。"""

    enabled: bool = True
    level: str = "INFO"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
        if self.level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("level must be DEBUG, INFO, WARNING or ERROR")

    @classmethod
    def from_mapping(cls, value: object) -> ObservabilityLoggingConfig:
        return construct_config(cls, value, "config.observability.logging")


@dataclass(frozen=True)
class ObservabilityTracingConfig:
    """可选 OTLP 导出配置；认证头通过统一凭据注册表引用。"""

    enabled: bool = False
    endpoint: str = "http://127.0.0.1:4318"
    protocol: str = "http"
    service_name: str = "m2bos"
    insecure: bool = True
    credential_ref: str = ""
    export_interval_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.insecure, bool):
            raise TypeError("enabled and insecure must be boolean")
        if self.protocol not in {"http", "grpc"}:
            raise ValueError("protocol must be http or grpc")
        if not isinstance(self.endpoint, str) or not self.endpoint or len(self.endpoint) > 2048:
            raise ValueError("endpoint must be non-empty bounded text")
        parsed_endpoint = urlsplit(self.endpoint)
        if (
            parsed_endpoint.scheme not in {"http", "https"}
            or parsed_endpoint.hostname is None
            or parsed_endpoint.username is not None
            or parsed_endpoint.password is not None
            or parsed_endpoint.query
            or parsed_endpoint.fragment
            or parsed_endpoint.path not in {"", "/"}
        ):
            raise ValueError("endpoint must be an HTTP(S) origin without credentials, path, query or fragment")
        if not isinstance(self.service_name, str) or _NAME.fullmatch(self.service_name) is None:
            raise ValueError("service_name must be a normalized bounded name")
        if not isinstance(self.credential_ref, str):
            raise TypeError("credential_ref must be a string")
        credential_ref = self.credential_ref.strip().lower()
        if credential_ref and _NAME.fullmatch(credential_ref) is None:
            raise ValueError("credential_ref must be a normalized credential name")
        object.__setattr__(self, "credential_ref", credential_ref)
        if (
            isinstance(self.export_interval_seconds, bool)
            or not isinstance(self.export_interval_seconds, int | float)
            or not math.isfinite(float(self.export_interval_seconds))
            or not 1.0 <= float(self.export_interval_seconds) <= 300.0
        ):
            raise ValueError("export_interval_seconds must be between 1 and 300")

    @classmethod
    def from_mapping(cls, value: object) -> ObservabilityTracingConfig:
        return construct_config(cls, value, "config.observability.tracing")


@dataclass(frozen=True)
class ObservabilityAuditConfig:
    """仅记录安全与运维动作的最小耐久审计配置。"""

    enabled: bool = True
    retention_days: int = 14
    max_records: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
        if (
            isinstance(self.retention_days, bool)
            or not isinstance(self.retention_days, int)
            or not 1 <= self.retention_days <= 365
        ):
            raise ValueError("retention_days must be between 1 and 365")
        if (
            isinstance(self.max_records, bool)
            or not isinstance(self.max_records, int)
            or not 100 <= self.max_records <= 1_000_000
        ):
            raise ValueError("max_records must be between 100 and 1000000")

    @classmethod
    def from_mapping(cls, value: object) -> ObservabilityAuditConfig:
        return construct_config(cls, value, "config.observability.audit")


@dataclass(frozen=True)
class ObservabilityConfig:
    """把指标、日志、追踪与最小审计保持在记忆主链之外。"""

    metrics: ObservabilityMetricsConfig = field(default_factory=ObservabilityMetricsConfig)
    logging: ObservabilityLoggingConfig = field(default_factory=ObservabilityLoggingConfig)
    tracing: ObservabilityTracingConfig = field(default_factory=ObservabilityTracingConfig)
    audit: ObservabilityAuditConfig = field(default_factory=ObservabilityAuditConfig)
    lock_warning_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name, expected in (
            ("metrics", ObservabilityMetricsConfig),
            ("logging", ObservabilityLoggingConfig),
            ("tracing", ObservabilityTracingConfig),
            ("audit", ObservabilityAuditConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if (
            isinstance(self.lock_warning_seconds, bool)
            or not isinstance(self.lock_warning_seconds, int | float)
            or not math.isfinite(float(self.lock_warning_seconds))
            or not 1.0 <= float(self.lock_warning_seconds) <= 86_400.0
        ):
            raise ValueError("lock_warning_seconds must be between 1 and 86400")

    @classmethod
    def from_mapping(cls, value: object) -> ObservabilityConfig:
        data = group_fields(cls, value, "config.observability")
        return cls(
            metrics=ObservabilityMetricsConfig.from_mapping(data.get("metrics", {})),
            logging=ObservabilityLoggingConfig.from_mapping(data.get("logging", {})),
            tracing=ObservabilityTracingConfig.from_mapping(data.get("tracing", {})),
            audit=ObservabilityAuditConfig.from_mapping(data.get("audit", {})),
            lock_warning_seconds=data.get("lock_warning_seconds", 300.0),  # type: ignore[arg-type]
        )


__all__ = [
    "ObservabilityAuditConfig",
    "ObservabilityConfig",
    "ObservabilityLoggingConfig",
    "ObservabilityMetricsConfig",
    "ObservabilityTracingConfig",
]
