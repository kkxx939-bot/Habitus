"""云侧行为 agent 送入的观测清洗层：协议归一、校验与不可变落盘。

本层只做确定性清洗，不做任何语义判断；事件边界、事件时长、多模态证据合并、观测级去重与
语义字段产出全部属于后续融合层（``TODO(BHV-FUSION-003)``）。
"""

from behavior.observation.adapters import (
    BehaviorObservationAdapter,
    BehaviorObservationAdapterRegistry,
    HabitusBehaviorObservationAdapter,
)
from behavior.observation.config import BehaviorObservationConfig
from behavior.observation.errors import (
    BehaviorObservationError,
    BehaviorObservationLimitError,
    BehaviorObservationProtocolError,
)
from behavior.observation.model import (
    BehaviorObservation,
    BehaviorObservationBatch,
    BehaviorObservationEnvelope,
    BehaviorObservationModality,
)
from behavior.observation.store import BehaviorObservationStore

__all__ = [
    "BehaviorObservation",
    "BehaviorObservationAdapter",
    "BehaviorObservationAdapterRegistry",
    "BehaviorObservationBatch",
    "BehaviorObservationConfig",
    "BehaviorObservationEnvelope",
    "BehaviorObservationError",
    "BehaviorObservationLimitError",
    "BehaviorObservationModality",
    "BehaviorObservationProtocolError",
    "BehaviorObservationStore",
    "HabitusBehaviorObservationAdapter",
]
