"""Domain-side durable Store protocol for Behavior Schema V3."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from behavior.claim.admission import ClaimAdmissionDecision, ClaimAdmissionPolicy, StaticAdmissionResult
    from behavior.claim.model import Claim, ClaimBatch, ClaimNormalizerAttempt, ClaimProcessingReceipt
    from behavior.claim.policy import ClaimProcessingLane
    from behavior.config import BehaviorConfig
    from behavior.evidence.bundle import (
        EvidenceSealReason,
        SemanticEvidenceBundle,
        SemanticEvidenceBundleAssembler,
        SemanticIngestResult,
    )
    from behavior.evidence.manifest import EvidenceManifest
    from behavior.ingress.model import IngressDecision, OwnerScopedSemanticRecord
    from behavior.ingress.service import AcceptedSemanticIngress


@runtime_checkable
class BehaviorEvidenceClaimStore(Protocol):
    root: Path
    path: Path
    initialized: bool
    config: BehaviorConfig

    def initialize(self) -> None: ...
    def readiness(self) -> tuple[bool, str]: ...
    def health_snapshot(self, *, admission_policy_digest: str | None = None) -> dict[str, int | str | bool]: ...
    def owner_identity_digest(self) -> str | None: ...

    def record_ingress_decision(
        self, decision: IngressDecision, *, record: OwnerScopedSemanticRecord
    ) -> IngressDecision: ...
    def read_ingress_decision(self, decision_id: str) -> IngressDecision | None: ...

    def ingest_semantic_record(
        self,
        accepted: AcceptedSemanticIngress,
        assembler: SemanticEvidenceBundleAssembler,
        *,
        sealed_at: datetime,
    ) -> SemanticIngestResult: ...

    def read_semantic_record(self, semantic_record_id: str) -> OwnerScopedSemanticRecord | None: ...
    def read_active_bundle(self, bundle_id: str) -> SemanticEvidenceBundle | None: ...
    def seal_bundle(
        self,
        bundle_id: str,
        *,
        reason: EvidenceSealReason,
        assembler: SemanticEvidenceBundleAssembler,
        sealed_at: datetime,
    ) -> EvidenceManifest | None: ...
    def read_manifest(self, manifest_id: str) -> EvidenceManifest | None: ...
    def read_manifest_for_bundle(self, bundle_id: str) -> EvidenceManifest | None: ...
    def list_manifests(
        self, *, start: datetime, end: datetime, limit: int, cursor: str | None = None
    ) -> tuple[EvidenceManifest, ...]: ...

    def read_claim(self, claim_id: str) -> Claim | None: ...
    def list_claims(
        self, *, start: datetime, end: datetime, limit: int, cursor: str | None = None
    ) -> tuple[Claim, ...]: ...
    def list_accepted_claims(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        admission_policy_digest: str,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]: ...
    def list_claims_by_processing(
        self, processing_identity: str, *, limit: int, cursor: str | None = None
    ) -> tuple[Claim, ...]: ...
    def read_claim_decision(
        self,
        claim_id: str,
        *,
        processing_identity: str | None = None,
        admission_policy_digest: str | None = None,
    ) -> ClaimAdmissionDecision | None: ...
    def read_receipt(self, processing_identity: str) -> ClaimProcessingReceipt | None: ...
    def read_claims_by_ids(self, claim_ids: tuple[str, ...]) -> tuple[Claim, ...]: ...
    def read_decisions_by_ids(self, decision_ids: tuple[str, ...]) -> tuple[ClaimAdmissionDecision, ...]: ...
    def read_attempts_by_ids(self, attempt_ids: tuple[str, ...]) -> tuple[ClaimNormalizerAttempt, ...]: ...
    def read_batches_by_ids(self, batch_ids: tuple[str, ...]) -> tuple[ClaimBatch, ...]: ...
    def read_latest_attempt(
        self, processing_identity: str, semantic_record_id: str, normalizer_fingerprint: str
    ) -> ClaimNormalizerAttempt | None: ...
    def next_attempt_number(
        self, processing_identity: str, semantic_record_id: str, normalizer_fingerprint: str
    ) -> int: ...
    def record_failed_attempt(self, attempt: ClaimNormalizerAttempt) -> ClaimNormalizerAttempt: ...
    def publish_lane(
        self,
        *,
        processing_identity: str,
        processing_lane: ClaimProcessingLane,
        scope_semantic_record_id: str | None,
        manifest: EvidenceManifest,
        routing_policy_digest: str,
        binding_policy_digest: str,
        confidence_policy_digest: str,
        attempts: tuple[ClaimNormalizerAttempt, ...],
        batches: tuple[ClaimBatch, ...],
        batch_claim_ids: tuple[tuple[str, ...], ...],
        claims: tuple[Claim, ...],
        static_results: tuple[StaticAdmissionResult, ...],
        admission_policy: ClaimAdmissionPolicy,
        processing_completed_at: datetime,
    ) -> tuple[ClaimProcessingReceipt, bool]: ...


__all__ = ["BehaviorEvidenceClaimStore"]
