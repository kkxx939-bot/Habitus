"""生命周期的基础设施：参与者协议、重试账、隔离区、容量快照。

只依赖 ``foundation`` 与 ``infrastructure.store``（架构测试钉住），领域包可以用，runtime 负责组装。
"""

from habitus.infrastructure.lifecycle.capacity import CapacityReading, CapacitySnapshot, CapacitySnapshotStore
from habitus.infrastructure.lifecycle.participant import LifecycleContext, LifecycleParticipant, ParticipantReport
from habitus.infrastructure.lifecycle.quarantine import Quarantine, QuarantinedItem
from habitus.infrastructure.lifecycle.retry import (
    FailureKind,
    RetryDecision,
    RetryEntry,
    RetryLedger,
    RetryLedgerFullError,
)

__all__ = [
    "CapacityReading",
    "CapacitySnapshot",
    "CapacitySnapshotStore",
    "FailureKind",
    "LifecycleContext",
    "LifecycleParticipant",
    "ParticipantReport",
    "Quarantine",
    "QuarantinedItem",
    "RetryDecision",
    "RetryEntry",
    "RetryLedger",
    "RetryLedgerFullError",
]
