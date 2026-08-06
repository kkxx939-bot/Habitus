"""把单条语义记录的系统字段绑定到 Claim。"""

from __future__ import annotations

from behavior.claim.confidence import ClaimConfidencePolicy
from behavior.claim.model import Claim, ClaimBatch, EpistemicClass
from behavior.claim.normalizer import ClaimNormalizerKind, NormalizerFingerprint
from behavior.claim.proposal import ClaimKind, ClaimSemanticProposal
from behavior.config import ClaimConfig
from behavior.errors import ClaimBindingError
from behavior.evidence.manifest import EvidenceManifest
from behavior.ingress.model import OwnerScopedSemanticRecord, SemanticRecordKind
from behavior.ingress.service import Clock, SystemClock
from behavior.ingress.trust import IngressTrustClass
from foundation.integrity import canonical_json

_EPISTEMIC_MAP = {
    IngressTrustClass.DIRECT_SYSTEM_LOG: EpistemicClass.DIRECT_SOURCE,
    IngressTrustClass.DIRECT_DEVICE_FACT: EpistemicClass.DIRECT_SOURCE,
    IngressTrustClass.OWNER_EXPLICIT: EpistemicClass.USER_EXPLICIT,
    IngressTrustClass.SENSOR_INFERRED: EpistemicClass.SENSOR_INFERRED,
    IngressTrustClass.MODEL_INFERRED: EpistemicClass.MODEL_INFERRED,
    IngressTrustClass.MULTIMODAL_MODEL_INFERRED: EpistemicClass.MULTIMODAL_MODEL_INFERRED,
}

_DETERMINISTIC_KIND_MAP = {
    SemanticRecordKind.OWNER_ACTIVITY_SEGMENT: ClaimKind.ACTIVITY_PHASE,
    SemanticRecordKind.OWNER_UTTERANCE_SEGMENT: ClaimKind.UTTERANCE,
    SemanticRecordKind.OWNER_STATE_ASSERTION: ClaimKind.STATE_ASSERTION,
    SemanticRecordKind.OWNER_STATE_TRANSITION: ClaimKind.STATE_TRANSITION,
    SemanticRecordKind.OWNER_INTERACTION_SEGMENT: ClaimKind.INTERACTION,
    SemanticRecordKind.ROBOT_ACTION_EVENT: ClaimKind.ROBOT_ACTION,
    SemanticRecordKind.AGENT_ACTION_EVENT: ClaimKind.AGENT_ACTION,
    SemanticRecordKind.TOOL_RESULT_EVENT: ClaimKind.TOOL_RESULT,
    SemanticRecordKind.OWNER_SENSOR_FACT: ClaimKind.STATE_ASSERTION,
    SemanticRecordKind.ENVIRONMENT_SENSOR_FACT: ClaimKind.STATE_ASSERTION,
    SemanticRecordKind.DEVICE_STATE: ClaimKind.STATE_ASSERTION,
    SemanticRecordKind.ENVIRONMENT_CHANGE: ClaimKind.ENVIRONMENT_CHANGE,
    SemanticRecordKind.COVERAGE_INTERVAL: ClaimKind.COVERAGE,
}

_MODEL_ALLOWED_KINDS = frozenset(
    {
        ClaimKind.STATE_ASSERTION,
        ClaimKind.STATE_TRANSITION,
        ClaimKind.ACTIVITY_PHASE,
        ClaimKind.INTERACTION,
        ClaimKind.UTTERANCE,
        ClaimKind.ENVIRONMENT_CHANGE,
        ClaimKind.FREE_TEXT_SEMANTIC,
    }
)


class ClaimBinder:
    def __init__(
        self,
        *,
        config: ClaimConfig,
        confidence_policy: ClaimConfidencePolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        policy = confidence_policy or ClaimConfidencePolicy()
        if not isinstance(policy, ClaimConfidencePolicy):
            raise TypeError("confidence_policy must be ClaimConfidencePolicy")
        resolved_clock = clock or SystemClock()
        if not isinstance(resolved_clock, Clock):
            raise TypeError("clock must implement Clock")
        self.config = config
        self.confidence_policy = policy
        self.clock = resolved_clock

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
        if not isinstance(proposal, ClaimSemanticProposal):
            raise TypeError("proposal must be ClaimSemanticProposal")
        if not isinstance(normalizer_fingerprint, NormalizerFingerprint):
            raise TypeError("normalizer_fingerprint must be NormalizerFingerprint")
        if manifest.owner_identity_digest != record.owner_identity_digest:
            raise ClaimBindingError("Manifest and semantic record Owner scope differ")
        snapshot = next(
            (
                item
                for item in manifest.ordered_record_snapshots
                if item.semantic_record_id == record.semantic_record_id
            ),
            None,
        )
        if snapshot is None or snapshot.semantic_record_digest != record.canonical_digest:
            raise ClaimBindingError("semantic record is not bound to this Manifest")
        allowed_refs = set(record.semantic_input.object_refs) | set(record.semantic_input.entity_refs)
        if not set(proposal.object_refs).issubset(allowed_refs):
            raise ClaimBindingError("Proposal object_refs are outside the current semantic record")
        if proposal.location_ref is not None and proposal.location_ref != record.semantic_input.location_ref:
            raise ClaimBindingError("Proposal location_ref is outside the current semantic record")
        if len(canonical_json(proposal.semantic_payload)) > self.config.max_semantic_payload_chars:
            raise ClaimBindingError("Proposal semantic_payload exceeds its configured boundary")
        if len(proposal.human_summary) > self.config.max_human_summary_chars:
            raise ClaimBindingError("Proposal human_summary exceeds its configured boundary")
        if normalizer_fingerprint.normalizer_kind is ClaimNormalizerKind.DETERMINISTIC:
            expected_kind = _DETERMINISTIC_KIND_MAP.get(record.semantic_input.record_kind)
            if expected_kind is None or proposal.claim_kind is not expected_kind:
                raise ClaimBindingError("deterministic Proposal kind does not match its semantic record")
            if proposal.normalizer_confidence != 1.0:
                raise ClaimBindingError("deterministic Proposal confidence must be one")
        elif proposal.claim_kind not in _MODEL_ALLOWED_KINDS:
            raise ClaimBindingError("model Proposal attempted to create a system-log Claim kind")
        effective = self.confidence_policy.effective(
            source_confidence=record.semantic_input.source_confidence,
            normalizer_confidence=proposal.normalizer_confidence,
            normalizer_kind=normalizer_fingerprint.normalizer_kind,
        )
        batch_id = ClaimBatch.identity_for(
            manifest_digest=manifest.content_digest,
            semantic_record_id=record.semantic_record_id,
            normalizer_fingerprint=normalizer_fingerprint.digest,
        )
        semantic_fingerprint = Claim.semantic_identity(
            owner_identity_digest=record.owner_identity_digest,
            subject_role=record.semantic_input.subject_role,
            actor_role=record.semantic_input.actor_role,
            proposal=proposal,
        )
        claim_id = Claim.identity_for(
            owner_identity_digest=record.owner_identity_digest,
            semantic_record_digest=record.canonical_digest,
            manifest_digest=manifest.content_digest,
            normalizer_fingerprint=normalizer_fingerprint.digest,
            proposal=proposal,
        )
        return Claim(
            claim_id=claim_id,
            claim_batch_id=batch_id,
            owner_identity_digest=record.owner_identity_digest,
            semantic_record_id=record.semantic_record_id,
            semantic_record_digest=record.canonical_digest,
            manifest_id=manifest.manifest_id,
            manifest_digest=manifest.content_digest,
            subject_role=record.semantic_input.subject_role,
            actor_role=record.semantic_input.actor_role,
            time_start=record.semantic_input.event_time_start,
            time_end=record.semantic_input.event_time_end,
            time_uncertainty_ms=record.semantic_input.event_time_uncertainty_ms,
            epistemic_class=_EPISTEMIC_MAP[record.ingress_trust_class],
            source_confidence=record.semantic_input.source_confidence,
            normalizer_confidence=proposal.normalizer_confidence,
            effective_confidence=effective,
            normalizer_fingerprint=normalizer_fingerprint.digest,
            proposal=proposal,
            semantic_fingerprint=semantic_fingerprint,
            created_at=self.clock.now(),
        )


__all__ = ["ClaimBinder"]
