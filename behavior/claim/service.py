"""Core-first, independently retryable Claim normalization and publication."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from behavior._validation import identifier, strict_utc
from behavior.claim.admission import ClaimAdmissionDecision, ClaimAdmissionPolicy, ClaimAdmissionStatus
from behavior.claim.binder import ClaimBinder
from behavior.claim.confidence import ClaimConfidencePolicy
from behavior.claim.model import (
    Claim,
    ClaimBatch,
    ClaimNormalizerAttempt,
    ClaimNormalizerAttemptStatus,
    ClaimProcessingReceipt,
)
from behavior.claim.normalizer import ClaimNormalizerKind
from behavior.claim.policy import ClaimBindingPolicy, ClaimProcessingLane
from behavior.claim.registry import ClaimNormalizerRegistry
from behavior.claim.router import ClaimNormalizationPlan, ClaimNormalizationRoute, ClaimNormalizationRouter
from behavior.config import ClaimConfig
from behavior.errors import (
    ClaimBindingError,
    ClaimModelAuthenticationError,
    ClaimModelConfigurationError,
    ClaimModelContentSafetyError,
    ClaimModelInputError,
    ClaimModelPermissionError,
    ClaimModelQuotaError,
    ClaimModelSchemaError,
    ClaimModelTransportError,
    ClaimProcessingConflictError,
    ClaimProductionError,
    ClaimSchemaError,
    ClaimStoreError,
)
from behavior.evidence.bundle import EvidenceSealReason, SemanticIngestResult
from behavior.evidence.manifest import EvidenceManifest
from behavior.evidence.service import EvidenceService
from behavior.ingress.model import IngressDecision, OwnerScopedSemanticRecord
from behavior.ingress.service import Clock, SemanticRecordService, SystemClock
from behavior.owner import ConfirmedOwnerBinding
from behavior.persistence.contracts import BehaviorEvidenceClaimStore
from foundation.integrity import canonical_digest
from foundation.observability import ObservationEvent, ObservationStatus, Observer


class ManifestClaimProcessingStatus(str, Enum):
    COMPLETE = "COMPLETE"
    CORE_COMMITTED_ENHANCEMENT_PENDING = "CORE_COMMITTED_ENHANCEMENT_PENDING"
    CORE_COMMITTED_WITH_DEGRADATION = "CORE_COMMITTED_WITH_DEGRADATION"


@dataclass(frozen=True)
class SemanticPipelineIngestResult:
    decision: IngressDecision
    bundle_result: SemanticIngestResult | None


@dataclass(frozen=True)
class ClaimLaneProcessingResult:
    processing_identity: str
    processing_lane: ClaimProcessingLane
    scope_semantic_record_id: str | None
    attempts: tuple[ClaimNormalizerAttempt, ...]
    validated_claims: tuple[Claim, ...]
    accepted_claims: tuple[Claim, ...]
    rejected_decisions: tuple[ClaimAdmissionDecision, ...]
    receipt: ClaimProcessingReceipt | None
    reused: bool


@dataclass(frozen=True)
class ClaimProcessingDegradation:
    semantic_record_id: str
    normalizer_name: str
    attempt_id: str
    status: ClaimNormalizerAttemptStatus
    error_code: str
    retryable: bool


@dataclass(frozen=True)
class ManifestClaimProcessingResult:
    manifest_id: str
    core_result: ClaimLaneProcessingResult
    enhancement_results: tuple[ClaimLaneProcessingResult, ...]
    degradations: tuple[ClaimProcessingDegradation, ...]
    status: ManifestClaimProcessingStatus


class ClaimPipelineService:
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
        binding_policy: ClaimBindingPolicy | None = None,
        confidence_policy: ClaimConfidencePolicy | None = None,
        admission_policy: ClaimAdmissionPolicy | None = None,
    ) -> None:
        if not isinstance(store, BehaviorEvidenceClaimStore):
            raise TypeError("store must implement BehaviorEvidenceClaimStore")
        if not isinstance(ingress_service, SemanticRecordService) or ingress_service.store is not store:
            raise ValueError("ingress_service must use the shared Store")
        if not isinstance(evidence_service, EvidenceService) or evidence_service.store is not store:
            raise ValueError("evidence_service must use the shared Store")
        if evidence_service.adapters is not ingress_service.adapters:
            raise ValueError("evidence_service must use the authoritative ingress Adapter Registry")
        if not isinstance(normalizers, ClaimNormalizerRegistry):
            raise TypeError("normalizers must be ClaimNormalizerRegistry")
        if not isinstance(router, ClaimNormalizationRouter) or router.registry is not normalizers:
            raise ValueError("router must use the shared Normalizer Registry")
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        if not callable(getattr(observer, "record", None)):
            raise TypeError("observer must implement record")
        self.store = store
        self.ingress_service = ingress_service
        self.evidence_service = evidence_service
        self.normalizers = normalizers
        self.router = router
        self.config = config
        self.observer = observer
        self.clock = clock or SystemClock()
        if not isinstance(self.clock, Clock):
            raise TypeError("clock must implement Clock")
        self.binding_policy = binding_policy or ClaimBindingPolicy()
        if not isinstance(self.binding_policy, ClaimBindingPolicy):
            raise TypeError("binding_policy must be ClaimBindingPolicy")
        resolved_confidence = confidence_policy or ClaimConfidencePolicy()
        if not isinstance(resolved_confidence, ClaimConfidencePolicy):
            raise TypeError("confidence_policy must be ClaimConfidencePolicy")
        self.binder = ClaimBinder(
            config=config,
            binding_policy=self.binding_policy,
            confidence_policy=resolved_confidence,
            clock=self.clock,
        )
        store_config = getattr(store, "config", None)
        if store_config is None:
            raise ValueError("Behavior Store must expose its bounded configuration")
        self.admission = admission_policy or ClaimAdmissionPolicy(
            config=config,
            max_accepted_claims=store_config.store.max_accepted_claims,
        )
        if not isinstance(self.admission, ClaimAdmissionPolicy):
            raise TypeError("admission_policy must be ClaimAdmissionPolicy")
        if self.admission.config != config:
            raise ValueError("admission_policy must use the Pipeline Claim configuration")
        if self.admission.max_accepted_claims != store_config.store.max_accepted_claims:
            raise ValueError("admission_policy must use the Store accepted Claim capacity")
        for name in self.normalizers.names():
            normalizer = self.normalizers.get(name)
            compatibility = normalizer.compatibility_policy
            if (
                normalizer.kind is ClaimNormalizerKind.MODEL
                and compatibility is not None
                and compatibility.digest != self.binding_policy.compatibility.digest
            ):
                raise ValueError("Normalizer and Binder compatibility policies must match")

    async def ingest_semantic(
        self,
        adapter_name: str,
        payload: object,
        *,
        owner_binding: ConfirmedOwnerBinding,
    ) -> tuple[SemanticPipelineIngestResult, ...]:
        prepared = await self.ingress_service.prepare(adapter_name, payload, owner_binding=owner_binding)
        results: list[SemanticPipelineIngestResult] = []
        for item in prepared:
            bundle_result = None if item.accepted is None else self.evidence_service.ingest(item.accepted)
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
        return self.store.list_accepted_claims(
            start=start,
            end=end,
            limit=limit,
            cursor=cursor,
            admission_policy_digest=self.admission.digest,
        )

    async def process_manifest(self, manifest_id: str) -> ManifestClaimProcessingResult:
        manifest, records, plan = self._preflight(manifest_id)
        by_id = {item.semantic_record_id: item for item in records}
        core_processing = self._processing_identity(
            manifest,
            ClaimProcessingLane.CORE,
            plan.core_routes,
            scope_record=None,
            routing_policy_digest=plan.routing_policy_digest,
        )
        existing = self.store.read_receipt(core_processing)
        if existing is None:
            core_result = await self._execute_core(
                manifest,
                plan,
                by_id,
                core_processing,
            )
        else:
            core_result = self._result_from_receipt(existing, reused=True)
            self._observe("claim_processing_reused", {"reused": True, "result_count": len(core_result.accepted_claims)})
        enhancements: list[ClaimLaneProcessingResult] = []
        degradations: list[ClaimProcessingDegradation] = []
        for route in plan.enhancement_routes:
            result, degradation = await self._execute_enhancement(
                manifest,
                plan,
                route,
                by_id[route.semantic_record_id],
                explicit_retry=False,
            )
            if result is not None:
                enhancements.append(result)
            if degradation is not None:
                degradations.append(degradation)
        if not degradations:
            status = ManifestClaimProcessingStatus.COMPLETE
        elif any(item.retryable for item in degradations):
            status = ManifestClaimProcessingStatus.CORE_COMMITTED_ENHANCEMENT_PENDING
        else:
            status = ManifestClaimProcessingStatus.CORE_COMMITTED_WITH_DEGRADATION
        return ManifestClaimProcessingResult(
            manifest.manifest_id,
            core_result,
            tuple(enhancements),
            tuple(degradations),
            status,
        )

    async def retry_enhancement(
        self,
        manifest_id: str,
        semantic_record_id: str,
        normalizer_name: str,
    ) -> ClaimLaneProcessingResult | ClaimProcessingDegradation:
        manifest, records, plan = self._preflight(manifest_id)
        record_id = identifier(semantic_record_id, "semantic_record_id")
        normalized_name = self.normalizers.normalize_name(normalizer_name)
        route = next(
            (
                item
                for item in plan.enhancement_routes
                if item.semantic_record_id == record_id
                and self.normalizers.normalize_name(item.normalizer_name) == normalized_name
            ),
            None,
        )
        if route is None:
            raise ClaimProcessingConflictError("requested Enhancement route is not in the current plan")
        result, degradation = await self._execute_enhancement(
            manifest,
            plan,
            route,
            {item.semantic_record_id: item for item in records}[record_id],
            explicit_retry=True,
        )
        if result is not None:
            return result
        if degradation is None:
            raise ClaimProcessingConflictError("Enhancement retry produced no durable result")
        return degradation

    def _preflight(
        self,
        manifest_id: str,
    ) -> tuple[EvidenceManifest, tuple[OwnerScopedSemanticRecord, ...], ClaimNormalizationPlan]:
        manifest = self.store.read_manifest(manifest_id)
        if manifest is None:
            raise ClaimStoreError("EvidenceManifest does not exist")
        records: list[OwnerScopedSemanticRecord] = []
        for snapshot in manifest.ordered_record_snapshots:
            record = self.store.read_semantic_record(snapshot.semantic_record_id)
            if record is None or record.semantic_digest != snapshot.semantic_record_digest:
                raise ClaimProcessingConflictError("Manifest references a missing or conflicting semantic record")
            if record.owner_identity_digest != manifest.owner_identity_digest:
                raise ClaimProcessingConflictError("Manifest references a semantic record from another Owner")
            records.append(record)
        plan = self.router.plan(manifest, tuple(records))
        return manifest, tuple(records), plan

    async def _execute_core(
        self,
        manifest: EvidenceManifest,
        plan: ClaimNormalizationPlan,
        by_id: dict[str, OwnerScopedSemanticRecord],
        processing: str,
    ) -> ClaimLaneProcessingResult:
        attempts: list[ClaimNormalizerAttempt] = []
        batches: list[ClaimBatch] = []
        memberships: list[tuple[str, ...]] = []
        claims: list[Claim] = []
        for route in plan.core_routes:
            attempt, batch, local_claims = await self._normalize_success(
                manifest,
                route,
                by_id[route.semantic_record_id],
                processing,
                ClaimProcessingLane.CORE,
                1,
            )
            attempts.append(attempt)
            batches.append(batch)
            memberships.append(tuple(item.claim_id for item in local_claims))
            claims.extend(local_claims)
        receipt, reused = self.store.publish_lane(
            processing_identity=processing,
            processing_lane=ClaimProcessingLane.CORE,
            scope_semantic_record_id=None,
            manifest=manifest,
            routing_policy_digest=plan.routing_policy_digest,
            binding_policy_digest=self.binding_policy.digest,
            confidence_policy_digest=self.binder.confidence_policy.digest,
            attempts=tuple(attempts),
            batches=tuple(batches),
            batch_claim_ids=tuple(memberships),
            claims=tuple(claims),
            static_results=tuple(
                self.admission.evaluate_static(item, owner_identity_digest=self.store.owner_identity_digest())
                for item in claims
            ),
            admission_policy=self.admission,
            processing_completed_at=self._now(),
        )
        return self._result_from_receipt(receipt, reused=reused)

    async def _execute_enhancement(
        self,
        manifest: EvidenceManifest,
        plan: ClaimNormalizationPlan,
        route: ClaimNormalizationRoute,
        record: OwnerScopedSemanticRecord,
        *,
        explicit_retry: bool,
    ) -> tuple[ClaimLaneProcessingResult | None, ClaimProcessingDegradation | None]:
        processing = self._processing_identity(
            manifest,
            ClaimProcessingLane.ENHANCEMENT,
            (route,),
            scope_record=record,
            routing_policy_digest=plan.routing_policy_digest,
        )
        existing = self.store.read_receipt(processing)
        if existing is not None:
            return self._result_from_receipt(existing, reused=True), None
        previous = self.store.read_latest_attempt(
            processing,
            record.semantic_record_id,
            route.normalizer_fingerprint,
        )
        if previous is not None:
            if previous.status in {
                ClaimNormalizerAttemptStatus.COMPLETED,
                ClaimNormalizerAttemptStatus.ABSTAINED,
            }:
                committed = self.store.read_receipt(processing)
                if committed is not None:
                    return self._result_from_receipt(committed, reused=True), None
                raise ClaimProcessingConflictError(
                    "successful Enhancement Attempt exists without its atomic Receipt"
                )
            if not explicit_retry:
                return None, self._degradation_from_attempt(previous)
            if previous.status is not ClaimNormalizerAttemptStatus.FAILED_RETRYABLE:
                raise ClaimProcessingConflictError("the latest Enhancement failure is not retryable")
        attempt_number = 1 if previous is None else previous.attempt_number + 1
        started_at = self._now()
        try:
            attempt, batch, claims = await self._normalize_success(
                manifest,
                route,
                record,
                processing,
                ClaimProcessingLane.ENHANCEMENT,
                attempt_number,
                started_at=started_at,
            )
            receipt, reused = self.store.publish_lane(
                processing_identity=processing,
                processing_lane=ClaimProcessingLane.ENHANCEMENT,
                scope_semantic_record_id=record.semantic_record_id,
                manifest=manifest,
                routing_policy_digest=plan.routing_policy_digest,
                binding_policy_digest=self.binding_policy.digest,
                confidence_policy_digest=self.binder.confidence_policy.digest,
                attempts=(attempt,),
                batches=(batch,),
                batch_claim_ids=(tuple(item.claim_id for item in claims),),
                claims=claims,
                static_results=tuple(
                    self.admission.evaluate_static(item, owner_identity_digest=self.store.owner_identity_digest())
                    for item in claims
                ),
                admission_policy=self.admission,
                processing_completed_at=self._now(),
            )
            return self._result_from_receipt(receipt, reused=reused), None
        except ClaimStoreError:
            raise
        except (
            ClaimModelTransportError,
            ClaimModelSchemaError,
            ClaimModelInputError,
            ClaimModelContentSafetyError,
            ClaimModelAuthenticationError,
            ClaimModelPermissionError,
            ClaimModelConfigurationError,
            ClaimModelQuotaError,
            ClaimBindingError,
            ClaimSchemaError,
            ClaimProductionError,
        ) as exc:
            status, code = self._failure_status(exc)
            attempt = ClaimNormalizerAttempt.create(
                processing_identity=processing,
                processing_lane=ClaimProcessingLane.ENHANCEMENT,
                manifest_id=manifest.manifest_id,
                semantic_record_id=record.semantic_record_id,
                normalizer_name=route.normalizer_name,
                normalizer_fingerprint=route.normalizer_fingerprint,
                attempt_number=attempt_number,
                status=status,
                proposal_digest=None,
                claim_count=0,
                error_code=code,
                retryable=status is ClaimNormalizerAttemptStatus.FAILED_RETRYABLE,
                normalization_started_at=started_at,
                normalization_completed_at=self._now(),
            )
            try:
                durable_attempt = self.store.record_failed_attempt(attempt)
            except ClaimProcessingConflictError:
                committed = self.store.read_receipt(processing)
                if committed is not None:
                    return self._result_from_receipt(committed, reused=True), None
                raise
            return None, self._degradation_from_attempt(durable_attempt)

    @staticmethod
    def _degradation_from_attempt(attempt: ClaimNormalizerAttempt) -> ClaimProcessingDegradation:
        if attempt.error_code is None:
            raise ClaimProcessingConflictError("failed Enhancement Attempt is missing its error code")
        return ClaimProcessingDegradation(
            attempt.semantic_record_id,
            attempt.normalizer_name,
            attempt.attempt_id,
            attempt.status,
            attempt.error_code,
            attempt.retryable,
        )

    async def _normalize_success(
        self,
        manifest: EvidenceManifest,
        route: ClaimNormalizationRoute,
        record: OwnerScopedSemanticRecord,
        processing: str,
        lane: ClaimProcessingLane,
        attempt_number: int,
        *,
        started_at: datetime | None = None,
    ) -> tuple[ClaimNormalizerAttempt, ClaimBatch, tuple[Claim, ...]]:
        started = self._now() if started_at is None else started_at
        monotonic = time.monotonic()
        proposal_batch = await route.normalizer.normalize(record)
        if lane is ClaimProcessingLane.CORE and proposal_batch.abstained:
            raise ClaimProductionError("required Core Normalizer cannot abstain")
        completed = self._now()
        claims = tuple(
            self.binder.bind(manifest, record, proposal, route.normalizer.fingerprint)
            for proposal in proposal_batch.claims
        )
        if len(claims) > self.config.max_claims_per_record:
            raise ClaimProcessingConflictError("Normalizer output exceeds the per-record Claim boundary")
        proposal_digest = canonical_digest(proposal_batch.to_dict())
        status = (
            ClaimNormalizerAttemptStatus.ABSTAINED
            if proposal_batch.abstained
            else ClaimNormalizerAttemptStatus.COMPLETED
        )
        attempt = ClaimNormalizerAttempt.create(
            processing_identity=processing,
            processing_lane=lane,
            manifest_id=manifest.manifest_id,
            semantic_record_id=record.semantic_record_id,
            normalizer_name=route.normalizer_name,
            normalizer_fingerprint=route.normalizer_fingerprint,
            attempt_number=attempt_number,
            status=status,
            proposal_digest=proposal_digest,
            claim_count=len(claims),
            error_code=None,
            retryable=False,
            normalization_started_at=started,
            normalization_completed_at=completed,
        )
        batch = ClaimBatch.create(
            processing_identity=processing,
            processing_lane=lane,
            manifest_id=manifest.manifest_id,
            manifest_digest=manifest.manifest_semantic_digest,
            semantic_record_id=record.semantic_record_id,
            normalizer_name=route.normalizer_name,
            normalizer_fingerprint=route.normalizer_fingerprint,
            abstained=proposal_batch.abstained,
            proposal_digest=proposal_digest,
            claim_count=len(claims),
            created_at=completed,
        )
        self._observe(
            "normalizer_abstained" if proposal_batch.abstained else "normalizer_completed",
            {
                "normalizer_name": route.normalizer_name,
                "claim_count": len(claims),
                "duration": max(0.0, time.monotonic() - monotonic),
            },
        )
        return attempt, batch, claims

    def _processing_identity(
        self,
        manifest: EvidenceManifest,
        lane: ClaimProcessingLane,
        routes: tuple[ClaimNormalizationRoute, ...],
        *,
        scope_record: OwnerScopedSemanticRecord | None,
        routing_policy_digest: str,
    ) -> str:
        return ClaimProcessingReceipt.processing_identity_for(
            processing_lane=lane,
            manifest_digest=manifest.manifest_semantic_digest,
            route_identities=tuple(item.route_identity for item in routes),
            scope_semantic_record_digest=None if scope_record is None else scope_record.semantic_digest,
            routing_policy_digest=routing_policy_digest,
            binding_policy_digest=self.binding_policy.digest,
            confidence_policy_digest=self.binder.confidence_policy.digest,
            admission_policy_digest=self.admission.digest,
        )

    def _result_from_receipt(
        self,
        receipt: ClaimProcessingReceipt,
        *,
        reused: bool,
    ) -> ClaimLaneProcessingResult:
        claims = self.store.read_claims_by_ids(receipt.claim_ids)
        decisions = self.store.read_decisions_by_ids(receipt.decision_ids)
        attempts = self.store.read_attempts_by_ids(receipt.normalizer_attempt_ids)
        by_id = {item.claim_id: item for item in claims}
        return ClaimLaneProcessingResult(
            receipt.processing_identity,
            receipt.processing_lane,
            receipt.scope_semantic_record_id,
            attempts,
            claims,
            tuple(by_id[item] for item in receipt.accepted_claim_ids),
            tuple(item for item in decisions if item.status is not ClaimAdmissionStatus.ACCEPTED),
            receipt,
            reused,
        )

    @staticmethod
    def _failure_status(
        exc: Exception,
    ) -> tuple[ClaimNormalizerAttemptStatus, str]:
        if isinstance(exc, ClaimModelTransportError):
            return ClaimNormalizerAttemptStatus.FAILED_RETRYABLE, "model_transport_retryable"
        if isinstance(exc, ClaimModelContentSafetyError):
            return ClaimNormalizerAttemptStatus.FAILED_POLICY, "content_safety_blocked"
        if isinstance(exc, ClaimModelInputError):
            return ClaimNormalizerAttemptStatus.FAILED_NON_RETRYABLE, "model_input_invalid"
        if isinstance(exc, ClaimModelSchemaError):
            return ClaimNormalizerAttemptStatus.FAILED_NON_RETRYABLE, "model_output_schema_invalid"
        if isinstance(exc, ClaimBindingError):
            return ClaimNormalizerAttemptStatus.FAILED_NON_RETRYABLE, "claim_binding_incompatible"
        if isinstance(exc, ClaimSchemaError):
            return ClaimNormalizerAttemptStatus.FAILED_NON_RETRYABLE, "claim_schema_incompatible"
        if isinstance(exc, ClaimModelAuthenticationError):
            return ClaimNormalizerAttemptStatus.FAILED_NON_RETRYABLE, "model_authentication_failed"
        if isinstance(exc, ClaimModelPermissionError):
            return ClaimNormalizerAttemptStatus.FAILED_NON_RETRYABLE, "model_permission_failed"
        if isinstance(exc, ClaimModelConfigurationError):
            return ClaimNormalizerAttemptStatus.FAILED_NON_RETRYABLE, "model_configuration_failed"
        if isinstance(exc, ClaimModelQuotaError):
            return ClaimNormalizerAttemptStatus.FAILED_NON_RETRYABLE, "model_quota_failed"
        return ClaimNormalizerAttemptStatus.FAILED_NON_RETRYABLE, "normalizer_contract_failed"

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
    "ClaimLaneProcessingResult",
    "ClaimPipelineService",
    "ClaimProcessingDegradation",
    "ManifestClaimProcessingResult",
    "ManifestClaimProcessingStatus",
    "SemanticPipelineIngestResult",
]
