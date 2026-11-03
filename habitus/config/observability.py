"""把 ``observability`` 配置分组严格构造成可观测后端自己的配置值对象。"""

from __future__ import annotations

from habitus.config.loader import construct_config, group_fields
from habitus.infrastructure.observability.config import (
    ObservabilityAuditConfig,
    ObservabilityConfig,
    ObservabilityLoggingConfig,
    ObservabilityMetricsConfig,
    ObservabilityTracingConfig,
)


def observability_config_from_mapping(value: object) -> ObservabilityConfig:
    """未知字段立即失败；各子分组的字段与取值校验归配置值对象自己。"""

    data = group_fields(ObservabilityConfig, value, "config.observability")
    return ObservabilityConfig(
        metrics=construct_config(
            ObservabilityMetricsConfig,
            data.get("metrics", {}),
            "config.observability.metrics",
        ),
        logging=construct_config(
            ObservabilityLoggingConfig,
            data.get("logging", {}),
            "config.observability.logging",
        ),
        tracing=construct_config(
            ObservabilityTracingConfig,
            data.get("tracing", {}),
            "config.observability.tracing",
        ),
        audit=construct_config(
            ObservabilityAuditConfig,
            data.get("audit", {}),
            "config.observability.audit",
        ),
        lock_warning_seconds=data.get("lock_warning_seconds", 300.0),  # type: ignore[arg-type]
    )


__all__ = ["observability_config_from_mapping"]
