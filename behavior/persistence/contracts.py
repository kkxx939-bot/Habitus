"""Evidence & Claim Store 的领域侧耐久契约。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from behavior.claim.admission import ClaimAdmissionDecision
    from behavior.claim.model import Claim, ClaimBatch, ClaimProcessingReceipt, ClaimProducerRun
    from behavior.evidence.manifest import EvidenceManifest
    from behavior.evidence.model import EvidenceSealReason, EvidenceWindow, SourceIngestResult
    from behavior.evidence.window import EvidenceWindowAssembler
    from behavior.source.model import SourceRecord


@runtime_checkable
class BehaviorEvidenceClaimStore(Protocol):
    root: Path
    path: Path
    initialized: bool
    max_claim_capacity: int

    def initialize(self) -> None: ...

    def readiness(self) -> tuple[bool, str]: ...

    def owner_binding_digest(self) -> str | None: ...

    def ingest_source(
        self,
        record: SourceRecord,
        assembler: EvidenceWindowAssembler,
    ) -> SourceIngestResult: ...

    def read_source(self, source_record_id: str) -> SourceRecord | None: ...

    def read_active_window(self, window_id: str) -> EvidenceWindow | None: ...

    def seal_window(
        self,
        window_id: str,
        *,
        reason: EvidenceSealReason,
        assembler: EvidenceWindowAssembler,
    ) -> EvidenceManifest | None: ...

    def read_manifest(self, manifest_id: str) -> EvidenceManifest | None: ...

    def read_manifest_for_window(self, window_id: str) -> EvidenceManifest | None: ...

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

    def find_recent_accepted_claim(
        self,
        *,
        semantic_fingerprint: str,
        since: datetime,
        until: datetime,
    ) -> Claim | None: ...

    def claim_count(self) -> int: ...

    def read_receipt(self, processing_identity: str) -> ClaimProcessingReceipt | None: ...

    def read_producer_runs(self, processing_identity: str) -> tuple[ClaimProducerRun, ...]: ...

    def read_decisions(self, processing_identity: str) -> tuple[ClaimAdmissionDecision, ...]: ...

    def publish_processing(
        self,
        *,
        receipt: ClaimProcessingReceipt,
        producer_runs: tuple[ClaimProducerRun, ...],
        batches: tuple[ClaimBatch, ...],
        accepted_claims: tuple[Claim, ...],
        decisions: tuple[ClaimAdmissionDecision, ...],
    ) -> tuple[ClaimProcessingReceipt, bool]: ...


__all__ = ["BehaviorEvidenceClaimStore"]
