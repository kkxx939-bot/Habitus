"""健康检查一项的值对象（``runtime.health`` 与各分项模块共用）。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimeHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


@dataclass(frozen=True)
class RuntimeHealthCheck:
    name: str
    status: RuntimeHealthStatus
    detail: str
    critical: bool = True


__all__ = ["RuntimeHealthCheck", "RuntimeHealthStatus"]
