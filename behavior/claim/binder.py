"""从已持久化 Evidence 与已验证 Proposal 创建 Claim。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from behavior._validation import strict_utc, utc_text
from behavior.claim.compatibility import ClaimCompatibilityPolicy
from behavior.claim.confidence import ClaimConfidencePolicy
from behavior.claim.model import (
    CLAIM_SCHEMA_VERSION,
    BehaviorClaim,
    DerivationClass,
    SourceEpistemicClass,
    source_epistemic_class,
)
from behavior.claim.planner import ClaimNormalizationRoute
from behavior.claim.proposal import ClaimSemanticProposal
from behavior.errors import BehaviorStoreError, ClaimCompatibilityError
from behavior.evidence.content import BehaviorRole
from behavior.evidence.record import BehaviorEvidenceRecord
from behavior.evidence.trust import BehaviorSourceTrust
from foundation.integrity import canonical_digest

CLAIM_BINDING_POLICY_VERSION = "claim_binding_v1"
ALTERNATIVE_GROUP_POLICY_VERSION = "claim_alternative_group_v1"


@dataclass(frozen=True, slots=True)
class ClaimBindingPolicy:
    version: str = CLAIM_BINDING_POLICY_VERSION
    alternative_group_version: str = ALTERNATIVE_GROUP_POLICY_VERSION
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.version != CLAIM_BINDING_POLICY_VERSION:
            raise ValueError("unsupported Claim binding policy version")
        if self.alternative_group_version != ALTERNATIVE_GROUP_POLICY_VERSION:
            raise ValueError("unsupported alternative group policy version")
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "alternative_group_version": self.alternative_group_version,
                    "source_epistemic_mapping": {
                        trust.value: source_epistemic_class(trust).value
                        for trust in BehaviorSourceTrust
                    },
                    "version": self.version,
                }
            ),
        )


class ClaimFactory:
    def __init__(self, *, compatibility: ClaimCompatibilityPolicy | None = None,
                 binding: ClaimBindingPolicy | None = None,
                 confidence: ClaimConfidencePolicy | None = None) -> None:
        self.compatibility = compatibility or ClaimCompatibilityPolicy()
        self.binding = binding or ClaimBindingPolicy()
        self.confidence = confidence or ClaimConfidencePolicy()

    def create(self, record: BehaviorEvidenceRecord, proposal: ClaimSemanticProposal,
               route: ClaimNormalizationRoute, *, created_at: datetime) -> BehaviorClaim:
        derivation = (
            DerivationClass.DETERMINISTIC
            if route.normalizer_kind.value == "DETERMINISTIC"
            else DerivationClass.MODEL
        )
        compatibility = self.compatibility.evaluate(
            record_kind=record.semantic_content.record_kind,
            subject_role=record.semantic_content.subject_role,
            actor_role=record.semantic_content.actor_role,
            normalizer_kind=route.normalizer_kind,
            claim_kind=proposal.claim_kind,
        )
        if not compatibility.allowed:
            raise ClaimCompatibilityError(
                f"Claim compatibility rejected proposal: {compatibility.reason_code}"
            )
        effective = self.confidence.effective(
            record.semantic_content.source_confidence,
            proposal.normalizer_confidence,
            derivation_class=derivation,
        )
        alternative_key = (
            None
            if proposal.local_alternative_group_id is None
            else canonical_digest(
                {
                    "evidence_record_id": record.evidence_record_id,
                    "local_alternative_group_id": proposal.local_alternative_group_id,
                    "normalizer_fingerprint": route.normalizer_fingerprint,
                    "policy_version": self.binding.alternative_group_version,
                }
            )
        )
        content = record.semantic_content
        return self._materialize(
            evidence_record_id=record.evidence_record_id, evidence_record_digest=record.content_digest,
            subject_role=content.subject_role, actor_role=content.actor_role,
            time_start=content.event_time_start, time_end=content.event_time_end,
            time_uncertainty_ms=content.event_time_uncertainty_ms, proposal=proposal,
            source_epistemic=source_epistemic_class(record.source_trust), derivation=derivation,
            source_confidence=content.source_confidence, effective_confidence=effective,
            alternative_group_key=alternative_key, normalizer_fingerprint=route.normalizer_fingerprint,
            compatibility_policy_digest=self.compatibility.digest, binding_policy_digest=self.binding.digest,
            confidence_policy_digest=self.confidence.digest, created_at=created_at)

    @staticmethod
    def restore(stored: BehaviorClaim) -> BehaviorClaim:
        if stored.schema_version != CLAIM_SCHEMA_VERSION:
            raise BehaviorStoreError("Claim schema is incompatible")
        proposal = ClaimSemanticProposal(
            stored.claim_kind, stored.semantic_family, stored.predicate, stored.activity, stored.phase,
            stored.semantic_payload, stored.human_summary, stored.local_alternative_group_id,
            stored.normalizer_confidence,
        )
        restored = ClaimFactory._materialize(
            evidence_record_id=stored.evidence_record_id, evidence_record_digest=stored.evidence_record_digest,
            subject_role=stored.subject_role, actor_role=stored.actor_role, time_start=stored.time_start,
            time_end=stored.time_end, time_uncertainty_ms=stored.time_uncertainty_ms, proposal=proposal,
            source_epistemic=stored.source_epistemic_class, derivation=stored.derivation_class,
            source_confidence=stored.source_confidence, effective_confidence=stored.effective_confidence,
            alternative_group_key=stored.alternative_group_key,
            normalizer_fingerprint=stored.normalizer_fingerprint,
            compatibility_policy_digest=stored.compatibility_policy_digest,
            binding_policy_digest=stored.binding_policy_digest,
            confidence_policy_digest=stored.confidence_policy_digest, created_at=stored.created_at,
        )
        if (restored.claim_id, restored.semantic_fingerprint, restored.content_digest) != (
            stored.claim_id, stored.semantic_fingerprint, stored.content_digest,
        ):
            raise BehaviorStoreError("Claim durable identity or digest has drifted")
        return restored

    @staticmethod
    def _materialize(*, evidence_record_id: str, evidence_record_digest: str,
                     subject_role: BehaviorRole, actor_role: BehaviorRole | None,
                     time_start: datetime, time_end: datetime, time_uncertainty_ms: int,
                     proposal: ClaimSemanticProposal, source_epistemic: SourceEpistemicClass,
                     derivation: DerivationClass, source_confidence: float,
                     effective_confidence: float, alternative_group_key: str | None,
                     normalizer_fingerprint: str, compatibility_policy_digest: str,
                     binding_policy_digest: str, confidence_policy_digest: str,
                     created_at: datetime) -> BehaviorClaim:
        created = strict_utc(created_at, "claim.created_at")
        proposal_identity = {
            "activity": proposal.activity,
            "claim_kind": proposal.claim_kind.value,
            "local_alternative_group_id": proposal.local_alternative_group_id,
            "normalizer_confidence": proposal.normalizer_confidence,
            "phase": proposal.phase,
            "predicate": proposal.predicate,
            "semantic_family": proposal.semantic_family,
            "semantic_payload": proposal.semantic_payload,
        }
        claim_id = "claim_" + canonical_digest(
            {
                "alternative_group_key": alternative_group_key,
                "binding_policy_digest": binding_policy_digest,
                "claim_schema_version": CLAIM_SCHEMA_VERSION,
                "compatibility_policy_digest": compatibility_policy_digest,
                "confidence_policy_digest": confidence_policy_digest,
                "derivation_class": derivation.value,
                "evidence_record_digest": evidence_record_digest,
                "normalizer_fingerprint": normalizer_fingerprint,
                "proposal": proposal_identity,
            }
        )
        semantic_fingerprint = canonical_digest(
            {
                "activity": proposal.activity,
                "actor_role": None if actor_role is None else actor_role.value,
                "claim_kind": proposal.claim_kind.value,
                "phase": proposal.phase,
                "predicate": proposal.predicate,
                "semantic_family": proposal.semantic_family,
                "semantic_payload": proposal.semantic_payload,
                "subject_role": subject_role.value,
            }
        )
        body = {
            "activity": proposal.activity,
            "actor_role": None if actor_role is None else actor_role.value,
            "alternative_group_key": alternative_group_key,
            "binding_policy_digest": binding_policy_digest,
            "claim_id": claim_id,
            "claim_kind": proposal.claim_kind.value,
            "compatibility_policy_digest": compatibility_policy_digest,
            "confidence_policy_digest": confidence_policy_digest,
            "created_at": utc_text(created),
            "derivation_class": derivation.value,
            "effective_confidence": effective_confidence,
            "evidence_record_digest": evidence_record_digest,
            "evidence_record_id": evidence_record_id,
            "human_summary": proposal.human_summary,
            "local_alternative_group_id": proposal.local_alternative_group_id,
            "normalizer_confidence": proposal.normalizer_confidence,
            "normalizer_fingerprint": normalizer_fingerprint,
            "phase": proposal.phase,
            "predicate": proposal.predicate,
            "schema_version": CLAIM_SCHEMA_VERSION,
            "semantic_family": proposal.semantic_family,
            "semantic_fingerprint": semantic_fingerprint,
            "semantic_payload": proposal.semantic_payload,
            "source_confidence": source_confidence,
            "source_epistemic_class": source_epistemic.value,
            "subject_role": subject_role.value,
            "time_end": utc_text(time_end),
            "time_start": utc_text(time_start),
            "time_uncertainty_ms": time_uncertainty_ms,
        }
        return BehaviorClaim(
            evidence_record_id, evidence_record_digest, subject_role, actor_role,
            time_start, time_end, time_uncertainty_ms, proposal.claim_kind,
            proposal.semantic_family, proposal.predicate, proposal.activity, proposal.phase,
            proposal.semantic_payload, proposal.human_summary, source_epistemic, derivation,
            source_confidence, proposal.normalizer_confidence, effective_confidence,
            proposal.local_alternative_group_id, alternative_group_key, normalizer_fingerprint,
            compatibility_policy_digest, binding_policy_digest, confidence_policy_digest,
            semantic_fingerprint, claim_id, created, canonical_digest(body))
