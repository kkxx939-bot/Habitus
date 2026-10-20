"""生命周期参与者协议：Worker 只管调度、租约、失败隔离，具体放什么由各层的参与者自己做。

协议放在这里而不是 runtime：领域包可以 import infrastructure，不能 import runtime。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from habitus.infrastructure.lifecycle.capacity import CapacityReading
from habitus.infrastructure.store.contracts import LockToken


class LifecycleContext(Protocol):
    """一轮维护的上下文。每做一步有副作用的事之前调 ``assert_owned``：租约丢了要立即停，心跳丢失只会取消协程，
    ``to_thread`` 里正在做的删除不会自己停下。每个参与者的批量上限由它自己的配置给，不经上下文传。"""

    @property
    def now(self) -> datetime: ...

    def assert_owned(self) -> None: ...

    def fence_token(self) -> LockToken: ...


@dataclass(frozen=True)
class ParticipantReport:
    """这一轮放了多少、各个量现在多大、有什么要人看的。"""

    released: int = 0
    readings: tuple[CapacityReading, ...] = ()
    signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.released, bool) or not isinstance(self.released, int) or self.released < 0:
            raise ValueError("released must be a non-negative integer")


class LifecycleParticipant(Protocol):
    @property
    def name(self) -> str: ...

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport: ...


__all__ = ["LifecycleContext", "LifecycleParticipant", "ParticipantReport"]
