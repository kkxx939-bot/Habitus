"""耐久 Job 重试、租约与常驻 Worker 配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from Config.loader import construct_config, group_fields
from memory.workflow import (
    MemoryChangeReceiptStoreConfig,
    MemoryJobConfig,
    MemoryWorkflowLifecycleConfig,
)


@dataclass(frozen=True)
class WorkerConfig:
    """常驻 Worker 的轮询、心跳和优雅停止边界。"""

    poll_interval_seconds: float = 0.2
    heartbeat_interval_seconds: float = 10.0
    shutdown_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("poll_interval_seconds", self.poll_interval_seconds, 60.0),
            ("heartbeat_interval_seconds", self.heartbeat_interval_seconds, 1_200.0),
            ("shutdown_timeout_seconds", self.shutdown_timeout_seconds, 3_600.0),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < float(value) <= maximum:
                raise ValueError(f"{name} must be greater than zero and at most {maximum:g}")


@dataclass(frozen=True)
class WorkflowConfig:
    """跨 Conversation 任务与常驻执行器的统一配置。"""

    jobs: MemoryJobConfig = field(default_factory=MemoryJobConfig)
    receipts: MemoryChangeReceiptStoreConfig = field(default_factory=MemoryChangeReceiptStoreConfig)
    lifecycle: MemoryWorkflowLifecycleConfig = field(default_factory=MemoryWorkflowLifecycleConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.jobs, MemoryJobConfig):
            raise TypeError("workflow.jobs must be MemoryJobConfig")
        if not isinstance(self.receipts, MemoryChangeReceiptStoreConfig):
            raise TypeError("workflow.receipts must be MemoryChangeReceiptStoreConfig")
        if not isinstance(self.lifecycle, MemoryWorkflowLifecycleConfig):
            raise TypeError("workflow.lifecycle must be MemoryWorkflowLifecycleConfig")
        if not isinstance(self.worker, WorkerConfig):
            raise TypeError("workflow.worker must be WorkerConfig")

    @classmethod
    def from_mapping(cls, value: object) -> WorkflowConfig:
        data = group_fields(cls, value, "config.workflow")
        return cls(
            jobs=construct_config(
                MemoryJobConfig,
                data.get("jobs", {}),
                "config.workflow.jobs",
            ),
            receipts=construct_config(
                MemoryChangeReceiptStoreConfig,
                data.get("receipts", {}),
                "config.workflow.receipts",
            ),
            lifecycle=construct_config(
                MemoryWorkflowLifecycleConfig,
                data.get("lifecycle", {}),
                "config.workflow.lifecycle",
            ),
            worker=construct_config(
                WorkerConfig,
                data.get("worker", {}),
                "config.workflow.worker",
            ),
        )


__all__ = ["MemoryWorkflowLifecycleConfig", "WorkerConfig", "WorkflowConfig"]
