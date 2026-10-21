"""统一生命周期的运维配置（裁定 15：各层生命周期统一设定；裁定 60）。

- ``worker``：LifecycleWorker 的周期、全局 lease、心跳与停机等待。各参与者（memory 会话链与 L2、行为侧、
  语义树、预测层、会话源）共用这一个 Worker，所以这几个数从 ``conversation.lifecycle`` 搬到这里。
- ``retry``：重试账的退避与次数上限，全仓一份。
- 派生物留几代、重试账放弃项留几天（``retry.given_up_days``）、迁移日志轮转份数、容量告警线、物理 lane 上游补发跨度。

"多少次机会算遗忘一半"不在这里：那是预测层读数的尺子，放在 ``foresight``。本模块只有纯标量与校验。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from habitus.config.loader import construct_config, group_fields


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"lifecycle.{name} must be an integer between {minimum} and {maximum}")


def _number(value: object, name: str, *, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 < float(value) <= maximum
    ):
        raise ValueError(f"lifecycle.{name} must be greater than zero and at most {maximum:g}")


@dataclass(frozen=True)
class LifecycleWorkerConfig:
    """LifecycleWorker：每隔多久一轮、全局 lease 多长、多久续一次、停机最多等多久。"""

    maintenance_interval_seconds: int = 3_600
    lease_ttl_seconds: int = 300
    heartbeat_interval_seconds: float = 60.0
    shutdown_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        _integer(self.maintenance_interval_seconds, "worker.maintenance_interval_seconds", minimum=60, maximum=86_400)
        _integer(self.lease_ttl_seconds, "worker.lease_ttl_seconds", minimum=30, maximum=3_600)
        _number(self.heartbeat_interval_seconds, "worker.heartbeat_interval_seconds", maximum=1_200.0)
        _number(self.shutdown_timeout_seconds, "worker.shutdown_timeout_seconds", maximum=3_600.0)
        if self.heartbeat_interval_seconds > self.lease_ttl_seconds / 3:
            raise ValueError("lifecycle.worker heartbeat interval must be at most one third of the lease TTL")

    @classmethod
    def from_mapping(cls, value: Any) -> LifecycleWorkerConfig:
        return construct_config(cls, value, "config.lifecycle.worker")


@dataclass(frozen=True)
class LifecycleRetryConfig:
    """重试账：第 n 次可重试失败后等 base × 2^(n−1) 秒（封顶 cap），第 ``max_attempts`` 次转为放弃；放弃了超过
    ``given_up_days`` 天的条目回收（键里带口径的，口径一变旧键再也不会被清）。"""

    max_attempts: int = 5
    backoff_base_seconds: float = 60.0
    backoff_cap_seconds: float = 86_400.0
    given_up_days: int = 30

    def __post_init__(self) -> None:
        _integer(self.max_attempts, "retry.max_attempts", minimum=1, maximum=100)
        _number(self.backoff_base_seconds, "retry.backoff_base_seconds", maximum=86_400.0)
        _number(self.backoff_cap_seconds, "retry.backoff_cap_seconds", maximum=30 * 86_400.0)
        _integer(self.given_up_days, "retry.given_up_days", minimum=1, maximum=3_650)
        if self.backoff_base_seconds > self.backoff_cap_seconds:
            raise ValueError("lifecycle.retry.backoff_base_seconds must not exceed backoff_cap_seconds")

    @classmethod
    def from_mapping(cls, value: Any) -> LifecycleRetryConfig:
        return construct_config(cls, value, "config.lifecycle.retry")


@dataclass(frozen=True)
class LifecycleConfig:
    worker: LifecycleWorkerConfig = field(default_factory=LifecycleWorkerConfig)
    retry: LifecycleRetryConfig = field(default_factory=LifecycleRetryConfig)
    # 派生物（关系表、机会表、单元历史）当前一代之外留几代备用：读坏时往前退。
    fallback_generations: int = 2
    # 关系迁移日志按大小轮转时留几份（它是关系迁移唯一的持久记录，不删）。
    transitions_rotate_files: int = 4
    # 容量快照里有上限的量，到这个占比报告警（non-critical）。
    capacity_warning_share: float = 0.8
    # 物理 lane 上游最多补发多少天前的单位：段账、时段账按它出账，更早的交付整份拒收。
    physical_max_backfill_days: int = 7
    # 单元的提醒反馈每个行为留最近几条（裁定 66）；更早的、所在那天已整天释放的删掉。不得小于读数窗口 M（组合根核）。
    feedback_kept: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.worker, LifecycleWorkerConfig):
            raise TypeError("lifecycle.worker must be LifecycleWorkerConfig")
        if not isinstance(self.retry, LifecycleRetryConfig):
            raise TypeError("lifecycle.retry must be LifecycleRetryConfig")
        _integer(self.fallback_generations, "fallback_generations", minimum=1, maximum=30)
        _integer(self.transitions_rotate_files, "transitions_rotate_files", minimum=1, maximum=100)
        if (
            isinstance(self.capacity_warning_share, bool)
            or not isinstance(self.capacity_warning_share, int | float)
            or not 0.0 < float(self.capacity_warning_share) < 1.0
        ):
            raise ValueError("lifecycle.capacity_warning_share must lie in (0, 1)")
        _integer(self.physical_max_backfill_days, "physical_max_backfill_days", minimum=1, maximum=365)
        _integer(self.feedback_kept, "feedback_kept", minimum=1, maximum=100_000)

    @classmethod
    def from_mapping(cls, value: Any) -> LifecycleConfig:
        data = dict(group_fields(cls, value, "config.lifecycle"))
        data["worker"] = LifecycleWorkerConfig.from_mapping(data.get("worker", {}))
        data["retry"] = LifecycleRetryConfig.from_mapping(data.get("retry", {}))
        return construct_config(cls, data, "config.lifecycle")


__all__ = ["LifecycleConfig", "LifecycleRetryConfig", "LifecycleWorkerConfig"]
