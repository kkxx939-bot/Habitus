"""观测投递的可观测属性：投进来多少、什么来源、上游送得有多晚。"""

from __future__ import annotations

from collections import Counter

from habitus.behavior.observation.model import (
    OBSERVATION_KNOWLEDGE_STATES,
    BehaviorObservationEnvelope,
    BehaviorObservationModality,
)
from habitus.behavior.telemetry import Attributes, seconds_between


def delivery_attributes(envelope: BehaviorObservationEnvelope, stored: BehaviorObservationEnvelope) -> Attributes:
    """``stored`` 是存储层实际保留的那份；与投来的记录不同说明同一投递此前已经落过盘。"""

    observations = envelope.batch.observations
    modalities = Counter(item.modality for item in observations)
    knowledge = Counter(item.knowledge_state for item in observations)
    attributes: Attributes = {
        "observations": len(observations),
        "redelivered": stored.record_digest != envelope.record_digest,
        # 上游从"发生"到"语义可用"的最长延迟：抽帧 + 识别的耗时，决定了融合最早能看到什么。
        "availability_lag_seconds_max": max(
            seconds_between(item.available_at, item.occurred_at) for item in observations
        ),
        # 投递时刻相对最晚可用时刻；负值是允许范围内的时钟偏差。
        "delivery_delay_seconds": round((envelope.recorded_at - envelope.batch.last_available_at).total_seconds(), 3),
    }
    for modality in BehaviorObservationModality:
        attributes[f"modality_{modality.value}"] = modalities.get(modality, 0)
    for state in sorted(OBSERVATION_KNOWLEDGE_STATES):
        attributes[f"knowledge_{state}"] = knowledge.get(state, 0)
    return attributes


__all__ = ["delivery_attributes"]
