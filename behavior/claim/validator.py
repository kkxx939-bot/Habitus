"""ClaimProposal 对 EvidenceManifest 的确定性忠实绑定。"""

from __future__ import annotations

from behavior.claim.model import CLAIM_SCHEMA_VERSION, Claim
from behavior.claim.producer import ClaimProducerKind, ProducerFingerprint
from behavior.claim.proposal import ClaimKind, ClaimProposal, EpistemicClass, SubjectRole
from behavior.config import ClaimConfig
from behavior.errors import ClaimValidationError
from behavior.evidence.manifest import EvidenceManifest
from behavior.persistence.contracts import BehaviorEvidenceClaimStore
from behavior.source.model import SourceType
from foundation.integrity import canonical_json


class ClaimValidator:
    def __init__(self, store: BehaviorEvidenceClaimStore, *, config: ClaimConfig) -> None:
        if not isinstance(store, BehaviorEvidenceClaimStore):
            raise TypeError("store must implement BehaviorEvidenceClaimStore")
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        self.store = store
        self.config = config

    def validate_and_bind(
        self,
        *,
        manifest: EvidenceManifest,
        proposal: ClaimProposal,
        producer: ProducerFingerprint,
        producer_kind: str,
        claim_batch_id: str,
    ) -> Claim:
        stored = self.store.read_manifest(manifest.manifest_id)
        if stored is None:
            raise ClaimValidationError("EvidenceManifest does not exist")
        if stored.manifest_digest != manifest.manifest_digest:
            raise ClaimValidationError("EvidenceManifest digest mismatch")
        manifest_sources = {record.source_record_id: record for record in stored.ordered_source_records}
        if not proposal.source_record_ids or len(set(proposal.source_record_ids)) != len(proposal.source_record_ids):
            raise ClaimValidationError("source_record_ids must be non-empty and unique")
        try:
            referenced = tuple(manifest_sources[source_id] for source_id in proposal.source_record_ids)
        except KeyError as exc:
            raise ClaimValidationError("ClaimProposal references a SourceRecord outside the Manifest") from exc
        if not stored.started_at <= proposal.time_start <= proposal.time_end <= stored.ended_at:
            raise ClaimValidationError("ClaimProposal time range is outside the Manifest")
        if proposal.subject_role is SubjectRole.OWNER:
            owner_digest = self.store.owner_binding_digest()
            if owner_digest is None or owner_digest != stored.owner_binding_digest:
                raise ClaimValidationError("OWNER ClaimProposal does not match the Store owner binding")
        if proposal.scene_ref is not None:
            source_scenes = {record.scene_ref for record in referenced if record.scene_ref is not None}
            if proposal.scene_ref != stored.scene_ref and proposal.scene_ref not in source_scenes:
                raise ClaimValidationError("ClaimProposal scene_ref is outside the Manifest scope")
        allowed_objects = {
            reference
            for record in referenced
            for reference in (*record.track_refs, *record.entity_refs)
        }
        if not set(proposal.object_refs) <= allowed_objects:
            raise ClaimValidationError("ClaimProposal object_refs are outside declared source references")
        if (
            proposal.location_ref is not None
            and proposal.location_ref not in allowed_objects
            and proposal.location_ref != stored.scene_ref
        ):
            raise ClaimValidationError("ClaimProposal location_ref is outside declared source references")
        if len(canonical_json(proposal.semantic_payload)) > self.config.max_semantic_payload_chars:
            raise ClaimValidationError("ClaimProposal semantic_payload exceeds its configured boundary")
        if len(proposal.human_summary) > self.config.max_human_summary_chars:
            raise ClaimValidationError("ClaimProposal human_summary exceeds its configured boundary")
        if proposal.claim_kind is ClaimKind.ACTIVITY_PHASE:
            if proposal.activity is None or proposal.phase is None:
                raise ClaimValidationError("ACTIVITY_PHASE requires activity and phase")
        elif proposal.activity is not None or proposal.phase is not None:
            raise ClaimValidationError("activity and phase are reserved for ACTIVITY_PHASE")
        if proposal.epistemic_class is EpistemicClass.DIRECT_SOURCE and producer_kind != ClaimProducerKind.DIRECT:
            raise ClaimValidationError("model Producer cannot create DIRECT_SOURCE claims")
        if proposal.epistemic_class is EpistemicClass.USER_EXPLICIT and any(
            record.source_type not in {SourceType.CONVERSATION_REFERENCE, SourceType.ASR_SEGMENT}
            for record in referenced
        ):
            raise ClaimValidationError("USER_EXPLICIT requires Conversation or ASR source records")
        if producer_kind == ClaimProducerKind.MODEL and proposal.epistemic_class is EpistemicClass.DIRECT_SOURCE:
            raise ClaimValidationError("model inference cannot masquerade as a direct source")
        claim_id = Claim.identity_for(
            manifest_digest=stored.manifest_digest,
            producer_fingerprint=producer.digest,
            proposal=proposal,
        )
        return Claim(
            claim_id=claim_id,
            claim_batch_id=claim_batch_id,
            owner_binding_digest=stored.owner_binding_digest,
            evidence_manifest_id=stored.manifest_id,
            evidence_manifest_digest=stored.manifest_digest,
            producer_fingerprint=producer.digest,
            proposal=proposal,
            semantic_fingerprint=Claim.semantic_identity(proposal),
            created_at=stored.sealed_at,
            schema_version=CLAIM_SCHEMA_VERSION,
        )


__all__ = ["ClaimValidator"]
