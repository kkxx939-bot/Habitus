"""Bind one proposal to one immutable semantic record and its Manifest."""

from __future__ import annotations

from behavior.claim.confidence import ClaimConfidencePolicy
from behavior.claim.model import Claim, EpistemicClass
from behavior.claim.normalizer import ClaimNormalizerKind, NormalizerFingerprint
from behavior.claim.policy import ClaimBindingPolicy, ClaimDerivationClass
from behavior.claim.proposal import ClaimSemanticProposal, ClaimSemanticProposalContract
from behavior.config import ClaimConfig
from behavior.errors import ClaimBindingError
from behavior.evidence.manifest import EvidenceManifest
from behavior.ingress.model import OwnerScopedSemanticRecord
from behavior.ingress.service import Clock, SystemClock
from behavior.ingress.trust import IngressTrustClass
from foundation.integrity import canonical_digest

_EPISTEMIC_MAP = {
    IngressTrustClass.DIRECT_SYSTEM_LOG: EpistemicClass.DIRECT_SOURCE,
    IngressTrustClass.DIRECT_DEVICE_FACT: EpistemicClass.DIRECT_SOURCE,
    IngressTrustClass.OWNER_EXPLICIT: EpistemicClass.USER_EXPLICIT,
    IngressTrustClass.SENSOR_INFERRED: EpistemicClass.SENSOR_INFERRED,
    IngressTrustClass.MODEL_INFERRED: EpistemicClass.MODEL_INFERRED,
    IngressTrustClass.MULTIMODAL_MODEL_INFERRED: EpistemicClass.MULTIMODAL_MODEL_INFERRED,
}


class ClaimBinder:
    def __init__(
        self,
        *,
        config: ClaimConfig,
        confidence_policy: ClaimConfidencePolicy | None = None,
        binding_policy: ClaimBindingPolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        self.config = config
        self.confidence_policy = confidence_policy or ClaimConfidencePolicy()
        self.binding_policy = binding_policy or ClaimBindingPolicy()
        self.clock = clock or SystemClock()
        if not isinstance(self.confidence_policy, ClaimConfidencePolicy):
            raise TypeError("confidence_policy must be ClaimConfidencePolicy")
        if not isinstance(self.binding_policy, ClaimBindingPolicy):
            raise TypeError("binding_policy must be ClaimBindingPolicy")
        if not isinstance(self.clock, Clock):
            raise TypeError("clock must implement Clock")

    def bind(
        self,
        manifest: EvidenceManifest,
        record: OwnerScopedSemanticRecord,
        proposal: ClaimSemanticProposal,
        normalizer_fingerprint: NormalizerFingerprint,
    ) -> Claim:
        if not isinstance(manifest, EvidenceManifest):
            raise TypeError("manifest must be EvidenceManifest")
        if not isinstance(record, OwnerScopedSemanticRecord):
            raise TypeError("record must be OwnerScopedSemanticRecord")
        if not isinstance(normalizer_fingerprint, NormalizerFingerprint):
            raise TypeError("normalizer_fingerprint must be NormalizerFingerprint")
        derivation = (
            ClaimDerivationClass.DETERMINISTIC
            if normalizer_fingerprint.normalizer_kind is ClaimNormalizerKind.DETERMINISTIC
            else ClaimDerivationClass.MODEL
        )
        allowed = (
            frozenset({proposal.claim_kind})
            if derivation is ClaimDerivationClass.DETERMINISTIC
            else self.binding_policy.compatibility.allowed_model_kinds(
                record.semantic_input.record_kind,
                record.semantic_input.subject_role,
                record.semantic_input.actor_role,
            )
        )
        proposal = ClaimSemanticProposalContract.model_validate(proposal.to_dict(), self.config, allowed)
        if manifest.owner_identity_digest != record.owner_identity_digest:
            raise ClaimBindingError("Manifest and semantic record Owner scope differ")
        snapshot = next(
            (item for item in manifest.ordered_record_snapshots if item.semantic_record_id == record.semantic_record_id),
            None,
        )
        if snapshot is None or snapshot.semantic_record_digest != record.semantic_digest:
            raise ClaimBindingError("semantic record is not bound to this Manifest")
        compatibility = self.binding_policy.compatibility.evaluate(
            record_kind=record.semantic_input.record_kind,
            subject_role=record.semantic_input.subject_role,
            actor_role=record.semantic_input.actor_role,
            derivation_class=derivation,
            claim_kind=proposal.claim_kind,
        )
        if not compatibility.allowed:
            raise ClaimBindingError(compatibility.reason_code)
        allowed_refs = set(record.semantic_input.object_refs) | set(record.semantic_input.entity_refs)
        if not set(proposal.object_refs).issubset(allowed_refs):
            raise ClaimBindingError("Proposal object_refs are outside the current semantic record")
        if proposal.location_ref is not None and proposal.location_ref != record.semantic_input.location_ref:
            raise ClaimBindingError("Proposal location_ref is outside the current semantic record")
        if derivation is ClaimDerivationClass.DETERMINISTIC and proposal.normalizer_confidence != 1.0:
            raise ClaimBindingError("deterministic Proposal confidence must be one")
        effective = self.confidence_policy.effective(
            source_confidence=record.semantic_input.source_confidence,
            normalizer_confidence=proposal.normalizer_confidence,
            derivation_class=derivation,
        )
        local_group = proposal.local_alternative_group_id
        alternative_key = (
            None
            if local_group is None
            else canonical_digest(
                {
                    "semantic_record_id": record.semantic_record_id,
                    "normalizer_fingerprint": normalizer_fingerprint.digest,
                    "local_alternative_group_id": local_group,
                    "policy_version": self.binding_policy.alternative_group_policy_version,
                }
            )
        )
        return Claim.create(
            owner_identity_digest=record.owner_identity_digest,
            semantic_record_id=record.semantic_record_id,
            semantic_record_digest=record.semantic_digest,
            manifest_id=manifest.manifest_id,
            manifest_digest=manifest.manifest_semantic_digest,
            subject_role=record.semantic_input.subject_role,
            actor_role=record.semantic_input.actor_role,
            time_start=record.semantic_input.event_time_start,
            time_end=record.semantic_input.event_time_end,
            time_uncertainty_ms=record.semantic_input.event_time_uncertainty_ms,
            source_epistemic_class=_EPISTEMIC_MAP[record.ingress_trust_class],
            derivation_class=derivation,
            source_confidence=record.semantic_input.source_confidence,
            normalizer_confidence=proposal.normalizer_confidence,
            effective_confidence=effective,
            confidence_policy_digest=self.confidence_policy.digest,
            binding_policy_digest=self.binding_policy.digest,
            normalizer_fingerprint=normalizer_fingerprint.digest,
            proposal=proposal,
            local_alternative_group_id=local_group,
            alternative_group_key=alternative_key,
            created_at=self.clock.now(),
        )


__all__ = ["ClaimBinder"]
