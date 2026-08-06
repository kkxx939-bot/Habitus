"""Behavior Evidence & Claim Store 的领域侧耐久契约。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from behavior.claim.admission import (
        ClaimAdmissionDecision,
        ClaimAdmissionPolicy,
        StaticAdmissionResult,
    )
    from behavior.claim.model import (
        Claim,
        ClaimBatch,
        ClaimNormalizerRun,
        ClaimProcessingReceipt,
    )
    from behavior.evidence.bundle import (
        EvidenceSealReason,
        SemanticEvidenceBundle,
        SemanticEvidenceBundleAssembler,
        SemanticIngestResult,
    )
    from behavior.evidence.manifest import EvidenceManifest
    from behavior.ingress.model import IngressDecision, OwnerScopedSemanticRecord


@runtime_checkable
class BehaviorEvidenceClaimStore(Protocol):
    root: Path
    path: Path
    initialized: bool

    def initialize(self) -> None: ...

    def readiness(self) -> tuple[bool, str]: ...

    def health_snapshot(self) -> dict[str, int | str | bool]: ...

    def owner_identity_digest(self) -> str | None: ...

    def record_ingress_decision(
        self,
        decision: IngressDecision,
        *,
        record: OwnerScopedSemanticRecord,
    ) -> IngressDecision: ...

    def ingest_semantic_record(
        self,
        record: OwnerScopedSemanticRecord,
        decision: IngressDecision,
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
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[EvidenceManifest, ...]: ...

    def read_claim(self, claim_id: str) -> Claim | None: ...

    def list_claims(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]: ...

    def list_accepted_claims(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]: ...

    def list_claims_by_processing(
        self,
        processing_identity: str,
        *,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]: ...

    def read_claim_decision(
        self,
        claim_id: str,
        *,
        processing_identity: str | None = None,
    ) -> ClaimAdmissionDecision | None: ...

    def read_receipt(self, processing_identity: str) -> ClaimProcessingReceipt | None: ...

    def read_normalizer_runs(self, processing_identity: str) -> tuple[ClaimNormalizerRun, ...]: ...

    def read_decisions(self, processing_identity: str) -> tuple[ClaimAdmissionDecision, ...]: ...

    def publish_processing(
        self,
        *,
        processing_identity: str,
        manifest: EvidenceManifest,
        normalizer_fingerprints: tuple[str, ...],
        normalizer_runs: tuple[ClaimNormalizerRun, ...],
        batches: tuple[ClaimBatch, ...],
        claims: tuple[Claim, ...],
        static_results: tuple[StaticAdmissionResult, ...],
        admission_policy: ClaimAdmissionPolicy,
        decided_at: datetime,
        published_at: datetime,
        completed_at: datetime,
    ) -> tuple[ClaimProcessingReceipt, bool]: ...


__all__ = ["BehaviorEvidenceClaimStore"]
