"""Manifest 到可审计 Claim 的无 Worker 应用编排。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from behavior._validation import identifier, strict_utc
from behavior.claim.admission import (
    ClaimAdmissionDecision,
    ClaimAdmissionPolicy,
    ClaimAdmissionStatus,
)
from behavior.claim.binder import ClaimBinder
from behavior.claim.model import (
    Claim,
    ClaimBatch,
    ClaimNormalizerRun,
    ClaimNormalizerRunStatus,
    ClaimProcessingReceipt,
)
from behavior.claim.normalizer import ClaimNormalizer
from behavior.claim.registry import ClaimNormalizerRegistry
from behavior.claim.router import ClaimNormalizationRouter
from behavior.config import ClaimConfig
from behavior.errors import ClaimProcessingConflictError, ClaimStoreError
from behavior.evidence.bundle import EvidenceSealReason, SemanticIngestResult
from behavior.evidence.manifest import EvidenceManifest
from behavior.evidence.service import EvidenceService
from behavior.ingress.model import IngressDecision, OwnerScopedSemanticRecord
from behavior.ingress.service import Clock, SemanticRecordService, SystemClock
from behavior.owner import ConfirmedOwnerBinding
from behavior.persistence.contracts import BehaviorEvidenceClaimStore
from foundation.integrity import canonical_digest
from foundation.observability import ObservationEvent, ObservationStatus, Observer


@dataclass(frozen=True)
class SemanticPipelineIngestResult:
    decision: IngressDecision
    bundle_result: SemanticIngestResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, IngressDecision):
            raise TypeError("decision must be IngressDecision")
        if self.bundle_result is not None and not isinstance(self.bundle_result, SemanticIngestResult):
            raise TypeError("bundle_result must be SemanticIngestResult or None")


@dataclass(frozen=True)
class ClaimProcessingResult:
    manifest_id: str
    processing_identity: str
    normalizer_runs: tuple[ClaimNormalizerRun, ...]
    validated_claims: tuple[Claim, ...]
    accepted_claims: tuple[Claim, ...]
    rejected_decisions: tuple[ClaimAdmissionDecision, ...]
    reused: bool
    completed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_id", identifier(self.manifest_id, "manifest_id"))
        object.__setattr__(
            self,
            "processing_identity",
            identifier(self.processing_identity, "processing_identity"),
        )
        if any(not isinstance(item, ClaimNormalizerRun) for item in self.normalizer_runs):
            raise TypeError("normalizer_runs must contain ClaimNormalizerRun values")
        if any(not isinstance(item, Claim) for item in (*self.validated_claims, *self.accepted_claims)):
            raise TypeError("Claim result collections must contain Claim values")
        if not {item.claim_id for item in self.accepted_claims}.issubset(
            {item.claim_id for item in self.validated_claims}
        ):
            raise ValueError("accepted Claims must belong to the validated Claim collection")
        if any(
            not isinstance(item, ClaimAdmissionDecision) or item.status is ClaimAdmissionStatus.ACCEPTED
            for item in self.rejected_decisions
        ):
            raise TypeError("rejected_decisions must contain non-accepted decisions")
        if not isinstance(self.reused, bool):
            raise TypeError("reused must be boolean")
        object.__setattr__(self, "completed_at", strict_utc(self.completed_at, "completed_at"))


class ClaimPipelineService:
    """自动路由单条记录；模型调用发生在最终 SQLite 事务之外。"""

    def __init__(
        self,
        store: BehaviorEvidenceClaimStore,
        ingress_service: SemanticRecordService,
        evidence_service: EvidenceService,
        normalizers: ClaimNormalizerRegistry,
        router: ClaimNormalizationRouter,
        *,
        config: ClaimConfig,
        observer: Observer,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(store, BehaviorEvidenceClaimStore):
            raise TypeError("store must implement BehaviorEvidenceClaimStore")
        if not isinstance(ingress_service, SemanticRecordService) or ingress_service.store is not store:
            raise ValueError("ingress_service must use the shared Store")
        if not isinstance(evidence_service, EvidenceService) or evidence_service.store is not store:
            raise ValueError("evidence_service must use the shared Store")
        if not isinstance(normalizers, ClaimNormalizerRegistry):
            raise TypeError("normalizers must be ClaimNormalizerRegistry")
        if not isinstance(router, ClaimNormalizationRouter) or router.registry is not normalizers:
            raise ValueError("router must use the shared Normalizer Registry")
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        if not callable(getattr(observer, "record", None)):
            raise TypeError("observer must implement record")
        resolved_clock = clock or SystemClock()
        if not isinstance(resolved_clock, Clock):
            raise TypeError("clock must implement Clock")
        self.store = store
        self.ingress_service = ingress_service
        self.evidence_service = evidence_service
        self.normalizers = normalizers
        self.router = router
        self.config = config
        self.observer = observer
        self.clock = resolved_clock
        self.binder = ClaimBinder(config=config, clock=resolved_clock)
        self.admission = ClaimAdmissionPolicy(config=config)

    async def ingest_semantic(
        self,
        adapter_name: str,
        payload: object,
        *,
        owner_binding: ConfirmedOwnerBinding,
    ) -> tuple[SemanticPipelineIngestResult, ...]:
        prepared = await self.ingress_service.prepare(
            adapter_name,
            payload,
            owner_binding=owner_binding,
        )
        results: list[SemanticPipelineIngestResult] = []
        for item in prepared:
            bundle_result = None if item.record is None else self.evidence_service.ingest(item.record, item.decision)
            results.append(
                SemanticPipelineIngestResult(
                    item.decision if bundle_result is None else bundle_result.decision,
                    bundle_result,
                )
            )
        return tuple(results)

    def seal_bundle(
        self,
        bundle_id: str,
        *,
        reason: EvidenceSealReason = EvidenceSealReason.EXPLICIT,
    ) -> EvidenceManifest | None:
        return self.evidence_service.seal_bundle(bundle_id, reason=reason)

    def read_manifest(self, manifest_id: str) -> EvidenceManifest | None:
        return self.store.read_manifest(manifest_id)

    def read_claim(self, claim_id: str) -> Claim | None:
        return self.store.read_claim(claim_id)

    def list_claims(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]:
        return self.store.list_claims(start=start, end=end, limit=limit, cursor=cursor)

    def list_accepted_claims(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]:
        return self.store.list_accepted_claims(start=start, end=end, limit=limit, cursor=cursor)

    async def process_manifest(self, manifest_id: str) -> ClaimProcessingResult:
        manifest = self.store.read_manifest(manifest_id)
        if manifest is None:
            raise ClaimStoreError("EvidenceManifest does not exist")
        routed: list[tuple[OwnerScopedSemanticRecord, ClaimNormalizer]] = []
        for snapshot in manifest.ordered_record_snapshots:
            record = self.store.read_semantic_record(snapshot.semantic_record_id)
            if record is None or record.canonical_digest != snapshot.semantic_record_digest:
                raise ClaimProcessingConflictError("Manifest references a missing or conflicting semantic record")
            routed.extend((record, normalizer) for normalizer in self.router.route(record))
        if not 1 <= len(routed) <= self.config.max_normalizers_per_processing:
            raise ClaimProcessingConflictError("automatic Normalizer route exceeds its processing boundary")
        fingerprints = tuple(normalizer.fingerprint.digest for _, normalizer in routed)
        processing = ClaimProcessingReceipt.processing_identity_for(
            manifest_digest=manifest.content_digest,
            normalizer_fingerprints=fingerprints,
        )
        existing = self.store.read_receipt(processing)
        if existing is not None:
            result = self._result_from_receipt(existing, reused=True)
            self._observe(
                "claim_processing_reused",
                {"reused": True, "result_count": len(result.accepted_claims)},
            )
            return result

        runs: list[ClaimNormalizerRun] = []
        batches: list[ClaimBatch] = []
        claims: list[Claim] = []
        for record, normalizer in routed:
            started_at = self._now()
            started = time.monotonic()
            proposal_batch = await normalizer.normalize(record)
            completed_at = self._now()
            if len(proposal_batch.claims) > self.config.max_claims_per_record:
                raise ClaimProcessingConflictError("Normalizer output exceeds the per-record Claim boundary")
            groups: dict[str, int] = {}
            for proposal in proposal_batch.claims:
                if proposal.alternative_group_id is not None:
                    groups[proposal.alternative_group_id] = groups.get(proposal.alternative_group_id, 0) + 1
            if groups and max(groups.values()) > self.config.max_alternative_group_size:
                raise ClaimProcessingConflictError("Normalizer alternative group exceeds its configured boundary")
            proposal_digest = canonical_digest(proposal_batch.to_dict())
            local_claims = tuple(
                self.binder.bind(manifest, record, proposal, normalizer.fingerprint)
                for proposal in proposal_batch.claims
            )
            claims.extend(local_claims)
            batch_id = ClaimBatch.identity_for(
                manifest_digest=manifest.content_digest,
                semantic_record_id=record.semantic_record_id,
                normalizer_fingerprint=normalizer.fingerprint.digest,
            )
            batch = ClaimBatch(
                claim_batch_id=batch_id,
                processing_identity=processing,
                manifest_id=manifest.manifest_id,
                manifest_digest=manifest.content_digest,
                semantic_record_id=record.semantic_record_id,
                normalizer_name=normalizer.name,
                normalizer_fingerprint=normalizer.fingerprint.digest,
                abstained=proposal_batch.abstained,
                claim_ids=tuple(item.claim_id for item in local_claims),
                proposal_digest=proposal_digest,
                created_at=completed_at,
            )
            if any(item.claim_batch_id != batch.claim_batch_id for item in local_claims):
                raise ClaimProcessingConflictError("bound Claim and ClaimBatch identities diverged")
            batches.append(batch)
            run = ClaimNormalizerRun(
                run_id="run_"
                + canonical_digest(
                    {
                        "manifest_id": manifest.manifest_id,
                        "normalizer_fingerprint": normalizer.fingerprint.digest,
                        "processing_identity": processing,
                        "semantic_record_id": record.semantic_record_id,
                    }
                ),
                processing_identity=processing,
                manifest_id=manifest.manifest_id,
                semantic_record_id=record.semantic_record_id,
                normalizer_name=normalizer.name,
                normalizer_fingerprint=normalizer.fingerprint.digest,
                status=(
                    ClaimNormalizerRunStatus.ABSTAINED
                    if proposal_batch.abstained
                    else ClaimNormalizerRunStatus.COMPLETED
                ),
                proposal_digest=proposal_digest,
                claim_count=len(local_claims),
                normalization_started_at=started_at,
                normalization_completed_at=completed_at,
            )
            runs.append(run)
            self._observe(
                "normalizer_abstained" if proposal_batch.abstained else "normalizer_completed",
                {
                    "normalizer_name": normalizer.name,
                    "claim_count": len(local_claims),
                    "duration": max(0.0, time.monotonic() - started),
                },
            )
        if len({item.claim_id for item in claims}) != len(claims):
            raise ClaimProcessingConflictError("processing produced duplicate deterministic Claim identities")
        owner_digest = self.store.owner_identity_digest()
        static_results = tuple(
            self.admission.evaluate_static(claim, owner_identity_digest=owner_digest) for claim in claims
        )
        decided_at = self._now()
        published_at = self._now()
        completed_at = self._now()
        receipt, reused = self.store.publish_processing(
            processing_identity=processing,
            manifest=manifest,
            normalizer_fingerprints=fingerprints,
            normalizer_runs=tuple(runs),
            batches=tuple(batches),
            claims=tuple(claims),
            static_results=static_results,
            admission_policy=self.admission,
            decided_at=decided_at,
            published_at=published_at,
            completed_at=completed_at,
        )
        result = self._result_from_receipt(receipt, reused=reused)
        if not reused and result.validated_claims:
            self._observe("claim_validated", {"claim_count": len(result.validated_claims)})
        if not reused and result.accepted_claims:
            self._observe(
                "claim_admitted",
                {
                    "claim_count": len(result.accepted_claims),
                    "reason_code": "claim_passed_admission",
                },
            )
        if not reused:
            rejected_counts: dict[str, int] = {}
            for decision in result.rejected_decisions:
                rejected_counts[decision.reason_code] = rejected_counts.get(decision.reason_code, 0) + 1
            for reason_code, count in sorted(rejected_counts.items()):
                self._observe(
                    "claim_rejected",
                    {"claim_count": count, "reason_code": reason_code},
                )
        self._observe(
            "claim_batch_published" if not reused else "claim_processing_reused",
            {"claim_count": len(result.accepted_claims), "reused": reused},
        )
        return result

    def _result_from_receipt(
        self,
        receipt: ClaimProcessingReceipt,
        *,
        reused: bool,
    ) -> ClaimProcessingResult:
        validated = (
            self.store.list_claims_by_processing(
                receipt.processing_identity,
                limit=max(1, len(receipt.claim_ids)),
            )
            if receipt.claim_ids
            else ()
        )
        by_id = {item.claim_id: item for item in validated}
        if set(by_id) != set(receipt.claim_ids):
            raise ClaimProcessingConflictError("ProcessingReceipt references missing validated Claims")
        accepted = tuple(by_id[item] for item in receipt.accepted_claim_ids)
        decisions = self.store.read_decisions(receipt.processing_identity)
        if tuple(item.decision_id for item in decisions) != receipt.decision_ids:
            by_decision = {item.decision_id: item for item in decisions}
            try:
                decisions = tuple(by_decision[item] for item in receipt.decision_ids)
            except KeyError as exc:
                raise ClaimProcessingConflictError("ProcessingReceipt references a missing AdmissionDecision") from exc
        return ClaimProcessingResult(
            manifest_id=receipt.manifest_id,
            processing_identity=receipt.processing_identity,
            normalizer_runs=self._ordered_runs(receipt),
            validated_claims=tuple(by_id[item] for item in receipt.claim_ids),
            accepted_claims=accepted,
            rejected_decisions=tuple(item for item in decisions if item.status is not ClaimAdmissionStatus.ACCEPTED),
            reused=reused,
            completed_at=receipt.completed_at,
        )

    def _ordered_runs(self, receipt: ClaimProcessingReceipt) -> tuple[ClaimNormalizerRun, ...]:
        runs = self.store.read_normalizer_runs(receipt.processing_identity)
        by_id = {item.run_id: item for item in runs}
        try:
            return tuple(by_id[item] for item in receipt.normalizer_run_ids)
        except KeyError as exc:
            raise ClaimProcessingConflictError("ProcessingReceipt references a missing NormalizerRun") from exc

    def _now(self) -> datetime:
        return strict_utc(self.clock.now(), "clock.now")

    def _observe(self, operation: str, attributes: dict[str, str | int | float | bool]) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category="behavior",
                    operation=operation,
                    status=ObservationStatus.SUCCESS,
                    duration_seconds=0.0,
                    attributes=attributes,
                )
            )
        except Exception:
            return


__all__ = [
    "ClaimPipelineService",
    "ClaimProcessingResult",
    "SemanticPipelineIngestResult",
]
