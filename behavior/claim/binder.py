"""将 Proposal 的系统字段绑定到当前单条 Evidence。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from behavior.claim.compatibility import ClaimCompatibilityPolicy
from behavior.claim.confidence import ClaimConfidencePolicy
from behavior.claim.model import BehaviorClaim, DerivationClass, source_epistemic_class
from behavior.claim.normalizer import ClaimNormalizerKind
from behavior.claim.proposal import ClaimSemanticProposal
from behavior.config import ClaimNormalizationConfig
from behavior.errors import BehaviorClaimSchemaError, ClaimCompatibilityError
from behavior.evidence.record import BehaviorEvidenceRecord
from behavior.evidence.trust import BehaviorSourceTrust
from foundation.integrity import canonical_digest, canonical_json

CLAIM_BINDING_POLICY_VERSION = "claim_binding_v1"
ALTERNATIVE_GROUP_POLICY_VERSION = "claim_alternative_group_v1"

@dataclass(frozen=True)
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


class ClaimBinder:
    def __init__(
        self,
        *,
        config: ClaimNormalizationConfig,
        compatibility: ClaimCompatibilityPolicy | None = None,
        binding: ClaimBindingPolicy | None = None,
        confidence: ClaimConfidencePolicy | None = None,
    ) -> None:
        if not isinstance(config, ClaimNormalizationConfig):
            raise TypeError("config must be ClaimNormalizationConfig")
        self.config = config
        self.compatibility = compatibility or ClaimCompatibilityPolicy()
        self.binding = binding or ClaimBindingPolicy()
        self.confidence = confidence or ClaimConfidencePolicy()

    def bind(
        self,
        record: BehaviorEvidenceRecord,
        proposal: ClaimSemanticProposal,
        *,
        normalizer_fingerprint: str,
        normalizer_kind: ClaimNormalizerKind,
        derivation_class: DerivationClass,
        created_at: datetime,
    ) -> BehaviorClaim:
        if not isinstance(record, BehaviorEvidenceRecord):
            raise TypeError("record must be BehaviorEvidenceRecord")
        if not isinstance(proposal, ClaimSemanticProposal):
            raise TypeError("proposal must be ClaimSemanticProposal")
        kind = ClaimNormalizerKind(normalizer_kind)
        derivation = DerivationClass(derivation_class)
        expected_derivation = (
            DerivationClass.DETERMINISTIC
            if kind is ClaimNormalizerKind.DETERMINISTIC
            else DerivationClass.MODEL
        )
        if derivation is not expected_derivation:
            raise BehaviorClaimSchemaError(
                "Normalizer kind and Claim derivation class disagree"
            )
        compatibility = self.compatibility.evaluate(
            record_kind=record.semantic_content.record_kind,
            subject_role=record.semantic_content.subject_role,
            actor_role=record.semantic_content.actor_role,
            normalizer_kind=kind,
            claim_kind=proposal.claim_kind,
        )
        if not compatibility.allowed:
            raise ClaimCompatibilityError(
                f"Claim compatibility rejected proposal: {compatibility.reason_code}"
            )
        if len(canonical_json(proposal.semantic_payload)) > self.config.max_semantic_payload_chars:
            raise BehaviorClaimSchemaError("Claim semantic payload exceeds configured boundary")
        if (
            proposal.human_summary is not None
            and len(proposal.human_summary) > self.config.max_human_summary_chars
        ):
            raise BehaviorClaimSchemaError("Claim human summary exceeds configured boundary")
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
                    "normalizer_fingerprint": normalizer_fingerprint,
                    "policy_version": self.binding.alternative_group_version,
                }
            )
        )
        return BehaviorClaim(
            evidence_record_id=record.evidence_record_id,
            evidence_record_digest=record.content_digest,
            subject_role=record.semantic_content.subject_role,
            actor_role=record.semantic_content.actor_role,
            time_start=record.semantic_content.event_time_start,
            time_end=record.semantic_content.event_time_end,
            time_uncertainty_ms=record.semantic_content.event_time_uncertainty_ms,
            claim_kind=proposal.claim_kind,
            semantic_family=proposal.semantic_family,
            predicate=proposal.predicate,
            activity=proposal.activity,
            phase=proposal.phase,
            semantic_payload=proposal.semantic_payload,
            human_summary=proposal.human_summary,
            source_epistemic_class=source_epistemic_class(record.source_trust),
            derivation_class=derivation,
            source_confidence=record.semantic_content.source_confidence,
            normalizer_confidence=proposal.normalizer_confidence,
            effective_confidence=effective,
            local_alternative_group_id=proposal.local_alternative_group_id,
            alternative_group_key=alternative_key,
            normalizer_fingerprint=normalizer_fingerprint,
            compatibility_policy_digest=self.compatibility.digest,
            binding_policy_digest=self.binding.digest,
            confidence_policy_digest=self.confidence.digest,
            created_at=created_at,
        )


__all__ = ["ClaimBinder"]
