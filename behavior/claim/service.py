"""Manifest 到幂等 Claim 发布的无 Worker 应用编排。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from behavior._validation import strict_utc, utc_text
from behavior.claim.admission import (
    ClaimAdmissionDecision,
    ClaimAdmissionGate,
    ClaimAdmissionStatus,
)
from behavior.claim.model import (
    CLAIM_SCHEMA_VERSION,
    PIPELINE_VERSION,
    Claim,
    ClaimBatch,
    ClaimProcessingReceipt,
    ClaimProducerRun,
    ClaimProducerRunStatus,
)
from behavior.claim.registry import ClaimProducerRegistry
from behavior.claim.validator import ClaimValidator
from behavior.config import ClaimConfig
from behavior.errors import ClaimProcessingConflictError, ClaimStoreError
from behavior.evidence.manifest import EvidenceManifest
from behavior.evidence.model import EvidenceSealReason, SourceIngestResult
from behavior.evidence.service import EvidenceService
from behavior.persistence.contracts import BehaviorEvidenceClaimStore
from behavior.source.model import SourceRecord, SourceRecordBatch
from behavior.source.service import SourceRecordService
from foundation.integrity import canonical_digest
from foundation.observability import ObservationEvent, ObservationStatus, Observer


@dataclass(frozen=True)
class ClaimProcessingResult:
    manifest_id: str
    processing_identity: str
    producer_runs: tuple[ClaimProducerRun, ...]
    accepted_claims: tuple[Claim, ...]
    rejected_decisions: tuple[ClaimAdmissionDecision, ...]
    reused: bool
    completed_at: datetime

    def __post_init__(self) -> None:
        if not self.manifest_id or not self.processing_identity:
            raise ValueError("processing result identities must be non-empty")
        if any(not isinstance(run, ClaimProducerRun) for run in self.producer_runs):
            raise TypeError("producer_runs must contain ClaimProducerRun values")
        if any(not isinstance(claim, Claim) for claim in self.accepted_claims):
            raise TypeError("accepted_claims must contain Claim values")
        if any(
            not isinstance(decision, ClaimAdmissionDecision)
            or decision.status is ClaimAdmissionStatus.ACCEPTED
            for decision in self.rejected_decisions
        ):
            raise TypeError("rejected_decisions must contain non-accepted decisions")
        if not isinstance(self.reused, bool):
            raise TypeError("reused must be boolean")
        object.__setattr__(self, "completed_at", strict_utc(self.completed_at, "completed_at"))


class ClaimPipelineService:
    """模型调用在事务外，最终结果由 Store 单事务发布。"""

    def __init__(
        self,
        store: BehaviorEvidenceClaimStore,
        source_service: SourceRecordService,
        evidence_service: EvidenceService,
        producers: ClaimProducerRegistry,
        *,
        config: ClaimConfig,
        observer: Observer,
    ) -> None:
        if not isinstance(store, BehaviorEvidenceClaimStore):
            raise TypeError("store must implement BehaviorEvidenceClaimStore")
        if not isinstance(source_service, SourceRecordService) or source_service.store is not store:
            raise ValueError("source_service must use the shared Store")
        if not isinstance(evidence_service, EvidenceService) or evidence_service.store is not store:
            raise ValueError("evidence_service must use the shared Store")
        if not isinstance(producers, ClaimProducerRegistry):
            raise TypeError("producers must be ClaimProducerRegistry")
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        if not callable(getattr(observer, "record", None)):
            raise TypeError("observer must implement record")
        self.store = store
        self.source_service = source_service
        self.evidence_service = evidence_service
        self.producers = producers
        self.config = config
        self.observer = observer
        self.validator = ClaimValidator(store, config=config)
        self.admission = ClaimAdmissionGate(store, config=config)

    def ingest_source(self, record: SourceRecord) -> SourceIngestResult:
        return self.evidence_service.ingest_source(record)

    def ingest_source_batch(self, batch: SourceRecordBatch) -> tuple[SourceIngestResult, ...]:
        if not isinstance(batch, SourceRecordBatch):
            raise TypeError("batch must be SourceRecordBatch")
        validated = SourceRecordBatch(batch.records, config=self.source_service.config)
        ordered = tuple(sorted(validated.records, key=lambda record: record.stable_sort_key))
        return tuple(self.ingest_source(record) for record in ordered)

    def seal_window(
        self,
        window_id: str,
        *,
        reason: EvidenceSealReason = EvidenceSealReason.EXPLICIT,
    ) -> EvidenceManifest | None:
        return self.evidence_service.seal_window(window_id, reason=reason)

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

    async def process_manifest(
        self,
        manifest_id: str,
        producer_names: tuple[str, ...],
    ) -> ClaimProcessingResult:
        manifest = self.store.read_manifest(manifest_id)
        if manifest is None:
            raise ClaimStoreError("EvidenceManifest does not exist")
        if not isinstance(producer_names, tuple):
            raise TypeError("producer_names must be a tuple")
        if not 1 <= len(producer_names) <= self.config.max_producers_per_processing:
            raise ValueError("producer_names size is outside its configured boundary")
        resolved = tuple(self.producers.get(name) for name in producer_names)
        if len({producer.name for producer in resolved}) != len(resolved):
            raise ValueError("producer_names cannot contain duplicates")
        fingerprints = tuple(producer.fingerprint.digest for producer in resolved)
        processing_identity = "processing_" + canonical_digest(
            {
                "claim_schema_version": CLAIM_SCHEMA_VERSION,
                "manifest_digest": manifest.manifest_digest,
                "pipeline_version": PIPELINE_VERSION,
                "producer_fingerprints": fingerprints,
            }
        )
        existing = self.store.read_receipt(processing_identity)
        if existing is not None:
            result = self._result_from_receipt(existing, reused=True)
            self._observe("claim_processing_reused", {"reused": True, "result_count": len(result.accepted_claims)})
            return result

        runs: list[ClaimProducerRun] = []
        batches: list[ClaimBatch] = []
        accepted: list[Claim] = []
        decisions: list[ClaimAdmissionDecision] = []
        for producer in resolved:
            started = time.monotonic()
            proposal_batch = await producer.produce(manifest)
            if len(proposal_batch.claims) > self.config.max_claims_per_batch:
                raise ClaimProcessingConflictError(
                    "Producer output exceeds the configured ClaimBatch boundary"
                )
            alternative_groups: dict[str, int] = {}
            for proposal in proposal_batch.claims:
                if proposal.alternative_group_id is not None:
                    alternative_groups[proposal.alternative_group_id] = (
                        alternative_groups.get(proposal.alternative_group_id, 0) + 1
                    )
            if (
                alternative_groups
                and max(alternative_groups.values()) > self.config.max_alternative_group_size
            ):
                raise ClaimProcessingConflictError(
                    "Producer alternative group exceeds its configured boundary"
                )
            proposal_digest = canonical_digest(proposal_batch.to_dict())
            batch_id = "batch_" + canonical_digest(
                {
                    "manifest_digest": manifest.manifest_digest,
                    "processing_identity": processing_identity,
                    "producer_fingerprint": producer.fingerprint.digest,
                    "proposal_batch": proposal_batch.to_dict(),
                    "schema_version": CLAIM_SCHEMA_VERSION,
                }
            )
            validated = tuple(
                self.validator.validate_and_bind(
                    manifest=manifest,
                    proposal=proposal,
                    producer=producer.fingerprint,
                    producer_kind=producer.kind,
                    claim_batch_id=batch_id,
                )
                for proposal in proposal_batch.claims
            )
            if validated:
                self._observe("claim_validated", {"claim_count": len(validated)})
            local_ids: list[str] = []
            for claim in validated:
                decision = self.admission.decide(
                    claim,
                    processing_identity=processing_identity,
                    pending_accepted=tuple(accepted),
                )
                decisions.append(decision)
                local_ids.append(claim.claim_id)
                if decision.status is ClaimAdmissionStatus.ACCEPTED:
                    accepted.append(claim)
                    self._observe("claim_admitted", {"reason_code": decision.reason_code, "claim_count": 1})
                else:
                    self._observe("claim_rejected", {"reason_code": decision.reason_code, "claim_count": 1})
            batches.append(
                ClaimBatch(
                    claim_batch_id=batch_id,
                    manifest_id=manifest.manifest_id,
                    manifest_digest=manifest.manifest_digest,
                    producer_name=producer.name,
                    producer_fingerprint=producer.fingerprint.digest,
                    abstained=proposal_batch.abstained,
                    claim_ids=tuple(local_ids),
                    proposal_digest=proposal_digest,
                    created_at=manifest.sealed_at,
                )
            )
            run_status = (
                ClaimProducerRunStatus.ABSTAINED
                if proposal_batch.abstained
                else ClaimProducerRunStatus.COMPLETED
            )
            runs.append(
                ClaimProducerRun(
                    run_id="run_" + canonical_digest(
                        {"processing_identity": processing_identity, "producer_fingerprint": producer.fingerprint.digest}
                    ),
                    processing_identity=processing_identity,
                    manifest_id=manifest.manifest_id,
                    producer_name=producer.name,
                    producer_fingerprint=producer.fingerprint.digest,
                    status=run_status,
                    proposal_digest=proposal_digest,
                    claim_count=len(validated),
                    completed_at=manifest.sealed_at,
                )
            )
            self._observe(
                "producer_abstained" if proposal_batch.abstained else "producer_completed",
                {
                    "producer_name": producer.name,
                    "claim_count": len(validated),
                    "duration": max(0.0, time.monotonic() - started),
                },
            )
        receipt_payload = {
            "processing_identity": processing_identity,
            "manifest_id": manifest.manifest_id,
            "manifest_digest": manifest.manifest_digest,
            "producer_fingerprints": fingerprints,
            "claim_batch_ids": tuple(batch.claim_batch_id for batch in batches),
            "accepted_claim_ids": tuple(claim.claim_id for claim in accepted),
            "decision_ids": tuple(decision.decision_id for decision in decisions),
            "completed_at": manifest.sealed_at,
            "schema_version": CLAIM_SCHEMA_VERSION,
        }
        receipt_digest_payload = {
            **receipt_payload,
            "completed_at": utc_text(manifest.sealed_at),
        }
        receipt = ClaimProcessingReceipt(
            processing_identity=processing_identity,
            manifest_id=manifest.manifest_id,
            manifest_digest=manifest.manifest_digest,
            producer_fingerprints=fingerprints,
            claim_batch_ids=tuple(batch.claim_batch_id for batch in batches),
            accepted_claim_ids=tuple(claim.claim_id for claim in accepted),
            decision_ids=tuple(decision.decision_id for decision in decisions),
            completed_at=manifest.sealed_at,
            receipt_digest=canonical_digest(receipt_digest_payload),
            schema_version=CLAIM_SCHEMA_VERSION,
        )
        published, reused = self.store.publish_processing(
            receipt=receipt,
            producer_runs=tuple(runs),
            batches=tuple(batches),
            accepted_claims=tuple(accepted),
            decisions=tuple(decisions),
        )
        if reused:
            return self._result_from_receipt(published, reused=True)
        self._observe("claim_batch_published", {"claim_count": len(accepted), "reused": False})
        return ClaimProcessingResult(
            manifest_id=manifest.manifest_id,
            processing_identity=processing_identity,
            producer_runs=tuple(runs),
            accepted_claims=tuple(accepted),
            rejected_decisions=tuple(
                decision for decision in decisions if decision.status is not ClaimAdmissionStatus.ACCEPTED
            ),
            reused=False,
            completed_at=receipt.completed_at,
        )

    def _result_from_receipt(
        self,
        receipt: ClaimProcessingReceipt,
        *,
        reused: bool,
    ) -> ClaimProcessingResult:
        claims: list[Claim] = []
        for claim_id in receipt.accepted_claim_ids:
            claim = self.store.read_claim(claim_id)
            if claim is None:
                raise ClaimProcessingConflictError("ProcessingReceipt references a missing Claim")
            claims.append(claim)
        decisions = self.store.read_decisions(receipt.processing_identity)
        return ClaimProcessingResult(
            manifest_id=receipt.manifest_id,
            processing_identity=receipt.processing_identity,
            producer_runs=self._ordered_runs(receipt),
            accepted_claims=tuple(claims),
            rejected_decisions=tuple(
                decision for decision in decisions if decision.status is not ClaimAdmissionStatus.ACCEPTED
            ),
            reused=reused,
            completed_at=receipt.completed_at,
        )

    def _ordered_runs(self, receipt: ClaimProcessingReceipt) -> tuple[ClaimProducerRun, ...]:
        runs = self.store.read_producer_runs(receipt.processing_identity)
        by_fingerprint = {run.producer_fingerprint: run for run in runs}
        try:
            return tuple(
                by_fingerprint[fingerprint] for fingerprint in receipt.producer_fingerprints
            )
        except KeyError as exc:
            raise ClaimProcessingConflictError(
                "ProcessingReceipt references a missing ProducerRun"
            ) from exc

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


__all__ = ["ClaimPipelineService", "ClaimProcessingResult"]
