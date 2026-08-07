"""绑定单条 Evidence 后形成的不可变原子 BehaviorClaim。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from behavior._validation import (
    claim_semantic_json_snapshot,
    finite_score,
    identifier,
    non_negative_int,
    optional_bounded_text,
    optional_identifier,
    positive_int,
    sha256_digest,
    strict_utc,
    utc_text,
)
from behavior.claim.proposal import ClaimKind
from behavior.evidence.content import BehaviorRole
from behavior.evidence.trust import BehaviorSourceTrust
from foundation.integrity import canonical_digest

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


@dataclass(frozen=True)
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
    created_at: datetime
    claim_id: str = field(init=False)
    semantic_fingerprint: str = field(init=False)
    content_digest: str = field(init=False)
    schema_version: str = field(init=False, default=CLAIM_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        evidence_id = identifier(self.evidence_record_id, "claim.evidence_record_id")
        evidence_digest = sha256_digest(self.evidence_record_digest, "claim.evidence_record_digest")
        subject = BehaviorRole(self.subject_role)
        actor = None if self.actor_role is None else BehaviorRole(self.actor_role)
        start = strict_utc(self.time_start, "claim.time_start")
        end = strict_utc(self.time_end, "claim.time_end")
        if end < start:
            raise ValueError("Claim end cannot precede start")
        uncertainty = non_negative_int(self.time_uncertainty_ms, "claim.time_uncertainty_ms")
        kind = ClaimKind(self.claim_kind)
        family = optional_identifier(self.semantic_family, "claim.semantic_family")
        predicate = identifier(self.predicate, "claim.predicate")
        activity = optional_identifier(self.activity, "claim.activity")
        phase = optional_identifier(self.phase, "claim.phase")
        semantic_payload = claim_semantic_json_snapshot(
            self.semantic_payload,
            "claim.semantic_payload",
            maximum_chars=1_000_000,
            maximum_items=10_000,
            maximum_depth=32,
        )
        summary = optional_bounded_text(self.human_summary, "claim.human_summary", maximum=1_000_000)
        epistemic = SourceEpistemicClass(self.source_epistemic_class)
        derivation = DerivationClass(self.derivation_class)
        source_confidence = finite_score(self.source_confidence, "claim.source_confidence")
        normalizer_confidence = finite_score(
            self.normalizer_confidence,
            "claim.normalizer_confidence",
        )
        effective_confidence = finite_score(self.effective_confidence, "claim.effective_confidence")
        local_group = optional_identifier(
            self.local_alternative_group_id,
            "claim.local_alternative_group_id",
        )
        alternative_key = (
            None
            if self.alternative_group_key is None
            else sha256_digest(self.alternative_group_key, "claim.alternative_group_key")
        )
        if (local_group is None) != (alternative_key is None):
            raise ValueError("local and bound alternative group identities must appear together")
        normalizer_fingerprint = sha256_digest(
            self.normalizer_fingerprint,
            "claim.normalizer_fingerprint",
        )
        compatibility_digest = sha256_digest(
            self.compatibility_policy_digest,
            "claim.compatibility_policy_digest",
        )
        binding_digest = sha256_digest(self.binding_policy_digest, "claim.binding_policy_digest")
        confidence_digest = sha256_digest(
            self.confidence_policy_digest,
            "claim.confidence_policy_digest",
        )
        created_at = strict_utc(self.created_at, "claim.created_at")
        proposal_identity = {
            "activity": activity,
            "claim_kind": kind.value,
            "local_alternative_group_id": local_group,
            "normalizer_confidence": normalizer_confidence,
            "phase": phase,
            "predicate": predicate,
            "semantic_family": family,
            "semantic_payload": semantic_payload,
        }
        claim_id = "claim_" + canonical_digest(
            {
                "alternative_group_key": alternative_key,
                "binding_policy_digest": binding_digest,
                "claim_schema_version": CLAIM_SCHEMA_VERSION,
                "compatibility_policy_digest": compatibility_digest,
                "confidence_policy_digest": confidence_digest,
                "derivation_class": derivation.value,
                "evidence_record_digest": evidence_digest,
                "normalizer_fingerprint": normalizer_fingerprint,
                "proposal": proposal_identity,
            }
        )
        semantic_fingerprint = canonical_digest(
            {
                "activity": activity,
                "actor_role": None if actor is None else actor.value,
                "claim_kind": kind.value,
                "phase": phase,
                "predicate": predicate,
                "semantic_family": family,
                "semantic_payload": semantic_payload,
                "subject_role": subject.value,
            }
        )
        body = {
            "activity": activity,
            "actor_role": None if actor is None else actor.value,
            "alternative_group_key": alternative_key,
            "binding_policy_digest": binding_digest,
            "claim_id": claim_id,
            "claim_kind": kind.value,
            "compatibility_policy_digest": compatibility_digest,
            "confidence_policy_digest": confidence_digest,
            "created_at": utc_text(created_at),
            "derivation_class": derivation.value,
            "effective_confidence": effective_confidence,
            "evidence_record_digest": evidence_digest,
            "evidence_record_id": evidence_id,
            "human_summary": summary,
            "local_alternative_group_id": local_group,
            "normalizer_confidence": normalizer_confidence,
            "normalizer_fingerprint": normalizer_fingerprint,
            "phase": phase,
            "predicate": predicate,
            "schema_version": CLAIM_SCHEMA_VERSION,
            "semantic_family": family,
            "semantic_fingerprint": semantic_fingerprint,
            "semantic_payload": semantic_payload,
            "source_confidence": source_confidence,
            "source_epistemic_class": epistemic.value,
            "subject_role": subject.value,
            "time_end": utc_text(end),
            "time_start": utc_text(start),
            "time_uncertainty_ms": uncertainty,
        }
        object.__setattr__(self, "evidence_record_id", evidence_id)
        object.__setattr__(self, "evidence_record_digest", evidence_digest)
        object.__setattr__(self, "subject_role", subject)
        object.__setattr__(self, "actor_role", actor)
        object.__setattr__(self, "time_start", start)
        object.__setattr__(self, "time_end", end)
        object.__setattr__(self, "time_uncertainty_ms", uncertainty)
        object.__setattr__(self, "claim_kind", kind)
        object.__setattr__(self, "semantic_family", family)
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "activity", activity)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "semantic_payload", semantic_payload)
        object.__setattr__(self, "human_summary", summary)
        object.__setattr__(self, "source_epistemic_class", epistemic)
        object.__setattr__(self, "derivation_class", derivation)
        object.__setattr__(self, "source_confidence", source_confidence)
        object.__setattr__(self, "normalizer_confidence", normalizer_confidence)
        object.__setattr__(self, "effective_confidence", effective_confidence)
        object.__setattr__(self, "local_alternative_group_id", local_group)
        object.__setattr__(self, "alternative_group_key", alternative_key)
        object.__setattr__(self, "normalizer_fingerprint", normalizer_fingerprint)
        object.__setattr__(self, "compatibility_policy_digest", compatibility_digest)
        object.__setattr__(self, "binding_policy_digest", binding_digest)
        object.__setattr__(self, "confidence_policy_digest", confidence_digest)
        object.__setattr__(self, "semantic_fingerprint", semantic_fingerprint)
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "content_digest", canonical_digest(body))


@dataclass(frozen=True)
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


__all__ = [
    "BehaviorClaim",
    "BehaviorClaimLedgerEntry",
    "DerivationClass",
    "SourceEpistemicClass",
    "claim_to_dict",
]
