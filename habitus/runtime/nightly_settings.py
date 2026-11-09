"""夜线 Job 执行器的运维数（``config.scene.nightly_*``，组合根换算过来）。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.foundation.lifecycle.backoff import BackoffPolicy


@dataclass(frozen=True)
class NightlySettings:
    """执行器的运维数（``config.scene.nightly_*``）。"""

    poll_seconds: float = 300.0
    kept_nights: int = 366
    lease_ttl_seconds: float = 120.0
    heartbeat_seconds: float = 30.0
    max_attempts: int = 3
    retry_base_seconds: float = 900.0
    retry_max_seconds: float = 3_600.0

    @property
    def backoff(self) -> BackoffPolicy:
        return BackoffPolicy(self.retry_base_seconds, self.retry_max_seconds)


__all__ = ["NightlySettings"]
