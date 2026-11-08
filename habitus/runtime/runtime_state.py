"""Runtime 生命周期状态与状态不相容错误：门面与它的协作者（FAILED Job 运维）共用。"""

from __future__ import annotations

from enum import Enum


class RuntimeState(str, Enum):
    """Runtime 的显式初始化、运行、停止和终止状态。"""

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"


class RuntimeStateError(RuntimeError):
    """Runtime 操作与当前生命周期不相容。"""


__all__ = ["RuntimeState", "RuntimeStateError"]
