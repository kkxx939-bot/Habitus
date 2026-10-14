"""观测：行为判断所依据的凭据——谁、在什么时刻、看到了什么——的形状与不可变落盘。

写它的是融合的两条 lane（``behavior/fusion/lanes``）：每个单位留开始、结束两条凭据，判断的时间由它们派生，
归约落树时读它们，落树之后释放。本层只做确定性的校验与落盘，不做任何语义判断。
"""

from habitus.behavior.observation.config import BehaviorObservationConfig
from habitus.behavior.observation.errors import (
    BehaviorObservationError,
    BehaviorObservationLimitError,
)
from habitus.behavior.observation.model import (
    BehaviorObservation,
    BehaviorObservationBatch,
    BehaviorObservationEnvelope,
    BehaviorObservationModality,
)
from habitus.behavior.observation.store import BehaviorObservationStore

__all__ = [
    "BehaviorObservation",
    "BehaviorObservationBatch",
    "BehaviorObservationConfig",
    "BehaviorObservationEnvelope",
    "BehaviorObservationError",
    "BehaviorObservationLimitError",
    "BehaviorObservationModality",
    "BehaviorObservationStore",
]
