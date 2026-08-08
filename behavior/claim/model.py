"""绑定单条 Evidence 后形成的薄不可变 BehaviorClaim。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from behavior._validation import positive_int, utc_text
from behavior.claim.proposal import ClaimKind
from behavior.evidence.content import BehaviorRole
from behavior.evidence.trust import BehaviorSourceTrust

CLAIM_SCHEMA_VERSION = "behavior_claim_v1"
CLAIM_PIPELINE_VERSION = "claim_normalization_pipeline_v1"


class SourceEpistemicClass(str, Enum):
    DIRECT_SOURCE = "DIRECT_SOURCE"
    USER_EXPLICIT = "USER_EXPLICIT"
    SENSOR_INFERRED = "SENSOR_INFERRED"
    MODEL_INFERRED = "MODEL_INFERRED"
    MULTIMODAL_MODEL_INFERRED = "MULTIMODAL_MODEL_INFERRED"


_SOURCE_EPISTEMIC_MAPPING = {
    BehaviorSourceTrust.DIRECT_SYSTEM_LOG: SourceEpistemicClass.DIRECT_SOURCE,
    BehaviorSourceTrust.DIRECT_DEVICE_FACT: SourceEpistemicClass.DIRECT_SOURCE,
    BehaviorSourceTrust.USER_EXPLICIT: SourceEpistemicClass.USER_EXPLICIT,
    BehaviorSourceTrust.SENSOR_INFERRED: SourceEpistemicClass.SENSOR_INFERRED,
    BehaviorSourceTrust.MODEL_INFERRED: SourceEpistemicClass.MODEL_INFERRED,
    BehaviorSourceTrust.MULTIMODAL_MODEL_INFERRED: SourceEpistemicClass.MULTIMODAL_MODEL_INFERRED,
}


def source_epistemic_class(source_trust: BehaviorSourceTrust) -> SourceEpistemicClass:
    return _SOURCE_EPISTEMIC_MAPPING[BehaviorSourceTrust(source_trust)]


class DerivationClass(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    MODEL = "MODEL"


@dataclass(frozen=True, slots=True)
class BehaviorClaim:
    evidence_record_id: str
    evidence_record_digest: str
    subject_role: BehaviorRole
    actor_role: BehaviorRole | None
    time_start: datetime
    time_end: datetime
    time_uncertainty_ms: int
    claim_kind: ClaimKind
    semantic_family: str | None
    predicate: str
    activity: str | None
    phase: str | None
    semantic_payload: Mapping[str, Any]
    human_summary: str | None
    source_epistemic_class: SourceEpistemicClass
    derivation_class: DerivationClass
    source_confidence: float
    normalizer_confidence: float
    effective_confidence: float
    local_alternative_group_id: str | None
    alternative_group_key: str | None
    normalizer_fingerprint: str
    compatibility_policy_digest: str
    binding_policy_digest: str
    confidence_policy_digest: str
    semantic_fingerprint: str
    claim_id: str
    created_at: datetime
    content_digest: str
    schema_version: str = CLAIM_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class BehaviorClaimLedgerEntry:
    sequence: int
    claim: BehaviorClaim

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", positive_int(self.sequence, "claim_sequence"))
        if not isinstance(self.claim, BehaviorClaim):
            raise TypeError("claim must be BehaviorClaim")


def claim_to_dict(value: BehaviorClaim) -> dict[str, Any]:
    return {
        "claim_id": value.claim_id,
        "evidence_record_id": value.evidence_record_id,
        "evidence_record_digest": value.evidence_record_digest,
        "subject_role": value.subject_role.value,
        "actor_role": None if value.actor_role is None else value.actor_role.value,
        "time_start": utc_text(value.time_start),
        "time_end": utc_text(value.time_end),
        "time_uncertainty_ms": value.time_uncertainty_ms,
        "claim_kind": value.claim_kind.value,
        "semantic_family": value.semantic_family,
        "predicate": value.predicate,
        "activity": value.activity,
        "phase": value.phase,
        "semantic_payload": value.semantic_payload,
        "human_summary": value.human_summary,
        "source_epistemic_class": value.source_epistemic_class.value,
        "derivation_class": value.derivation_class.value,
        "source_confidence": value.source_confidence,
        "normalizer_confidence": value.normalizer_confidence,
        "effective_confidence": value.effective_confidence,
        "local_alternative_group_id": value.local_alternative_group_id,
        "alternative_group_key": value.alternative_group_key,
        "normalizer_fingerprint": value.normalizer_fingerprint,
        "compatibility_policy_digest": value.compatibility_policy_digest,
        "binding_policy_digest": value.binding_policy_digest,
        "confidence_policy_digest": value.confidence_policy_digest,
        "semantic_fingerprint": value.semantic_fingerprint,
        "created_at": utc_text(value.created_at),
        "content_digest": value.content_digest,
        "schema_version": value.schema_version,
    }
