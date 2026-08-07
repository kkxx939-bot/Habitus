"""单 Evidence Claim Normalization 的锁定、失败隔离与发布服务。"""

from __future__ import annotations

import time
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from behavior.claim.binder import ClaimBinder
from behavior.claim.ledger import BehaviorClaimLedger, ClaimPage
from behavior.claim.model import BehaviorClaim, BehaviorClaimLedgerEntry, DerivationClass
from behavior.claim.normalizer import ClaimNormalizerKind
from behavior.claim.planner import (
    ClaimNormalizationPlan,
    ClaimNormalizationPlanner,
    ClaimNormalizationRoute,
    NormalizationLane,
)
from behavior.claim.proposal import ClaimSemanticProposal, proposal_to_dict
from behavior.claim.receipt import (
    NO_CORE_NORMALIZER_FINGERPRINT,
    AttemptStatus,
    ClaimNormalizationAttempt,
    ClaimNormalizationReceipt,
    ReceiptStatus,
    attempt_identity,
    processing_identity,
)
from behavior.claim.registry import ClaimNormalizerRegistry
from behavior.errors import (
    BehaviorClaimSchemaError,
    ClaimCompatibilityError,
    ClaimModelAuthenticationError,
    ClaimModelConfigurationError,
    ClaimModelContentSafetyError,
    ClaimModelInputError,
    ClaimModelPermissionError,
    ClaimModelQuotaError,
    ClaimModelSchemaError,
    ClaimModelTransportError,
    ClaimNormalizationConflictError,
    ClaimNormalizationError,
)
from behavior.evidence.ingress import Clock, SystemClock
from behavior.evidence.ledger import BehaviorEvidenceLedger
from behavior.evidence.record import BehaviorEvidenceRecord
from foundation.integrity import canonical_digest
from foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer

_FAILURE_RULES: tuple[tuple[type[Exception], AttemptStatus, str, bool], ...] = (
    (ClaimModelTransportError, AttemptStatus.FAILED_RETRYABLE, "MODEL_TRANSPORT", True),
    (ClaimModelContentSafetyError, AttemptStatus.FAILED_POLICY, "MODEL_CONTENT_SAFETY", False),
    (ClaimModelSchemaError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_SCHEMA", False),
    (ClaimModelInputError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_INPUT", False),
    (ClaimModelAuthenticationError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_AUTHENTICATION", False),
    (ClaimModelPermissionError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_PERMISSION", False),
    (ClaimModelConfigurationError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_CONFIGURATION", False),
    (ClaimModelQuotaError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_QUOTA", False),
    (ClaimCompatibilityError, AttemptStatus.FAILED_POLICY, "COMPATIBILITY_POLICY", False),
    (BehaviorClaimSchemaError, AttemptStatus.FAILED_NON_RETRYABLE, "CLAIM_SCHEMA", False),
)


@runtime_checkable
class ProcessingLock(Protocol):
    def acquire(self, processing_identity: str) -> AbstractAsyncContextManager[object]: ...


class ClaimNormalizationResultStatus(str, Enum):
    COMPLETE = "COMPLETE"
    CORE_COMMITTED_ENHANCEMENT_PENDING = "CORE_COMMITTED_ENHANCEMENT_PENDING"
    CORE_COMMITTED_WITH_DEGRADATION = "CORE_COMMITTED_WITH_DEGRADATION"


@dataclass(frozen=True)
class ClaimProcessingDegradation:
    normalizer_name: str
    error_code: str
    retryable: bool


@dataclass(frozen=True)
class ClaimNormalizationResult:
    evidence_record_id: str
    status: ClaimNormalizationResultStatus
    core_receipt: ClaimNormalizationReceipt
    enhancement_receipts: tuple[ClaimNormalizationReceipt, ...]
    degradations: tuple[ClaimProcessingDegradation, ...]


class ClaimNormalizationService:
    def __init__(
        self,
        evidence_ledger: BehaviorEvidenceLedger,
        claim_ledger: BehaviorClaimLedger,
        normalizers: ClaimNormalizerRegistry,
        planner: ClaimNormalizationPlanner,
        binder: ClaimBinder,
        processing_lock: ProcessingLock,
        *,
        clock: Clock | None = None,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(evidence_ledger, BehaviorEvidenceLedger):
            raise TypeError("evidence_ledger must implement BehaviorEvidenceLedger")
        if not isinstance(claim_ledger, BehaviorClaimLedger):
            raise TypeError("claim_ledger must implement BehaviorClaimLedger")
        if not isinstance(normalizers, ClaimNormalizerRegistry):
            raise TypeError("normalizers must be ClaimNormalizerRegistry")
        if not isinstance(planner, ClaimNormalizationPlanner):
            raise TypeError("planner must be ClaimNormalizationPlanner")
        if not isinstance(binder, ClaimBinder):
            raise TypeError("binder must be ClaimBinder")
        if not isinstance(processing_lock, ProcessingLock):
            raise TypeError("processing_lock must implement ProcessingLock")
        self.evidence_ledger = evidence_ledger
        self.claim_ledger = claim_ledger
        self.normalizers = normalizers
        self.planner = planner
        self.binder = binder
        self.processing_lock = processing_lock
        self.clock = clock or SystemClock()
        self.observer = observer or NullObserver()

    async def normalize(self, evidence_record_id: str) -> ClaimNormalizationResult:
        record = self.evidence_ledger.read(evidence_record_id)
        if record is None:
            raise ClaimNormalizationError("Evidence record does not exist")
        plan = self.planner.plan(record)
        core_receipt = await self._process_core(record, plan)
        enhancement_receipts: list[ClaimNormalizationReceipt] = []
        degradations: list[ClaimProcessingDegradation] = []
        for route in plan.enhancement_routes:
            receipt, degradation = await self._process_enhancement(record, plan, route, retry=False)
            if receipt is not None:
                enhancement_receipts.append(receipt)
            if degradation is not None:
                degradations.append(degradation)
        if any(item.retryable for item in degradations) and all(item.retryable for item in degradations):
            status = ClaimNormalizationResultStatus.CORE_COMMITTED_ENHANCEMENT_PENDING
        elif degradations:
            status = ClaimNormalizationResultStatus.CORE_COMMITTED_WITH_DEGRADATION
        else:
            status = ClaimNormalizationResultStatus.COMPLETE
        return ClaimNormalizationResult(
            evidence_record_id=record.evidence_record_id,
            status=status,
            core_receipt=core_receipt,
            enhancement_receipts=tuple(enhancement_receipts),
            degradations=tuple(degradations),
        )

    async def retry_enhancement(
        self,
        evidence_record_id: str,
        normalizer_name: str,
    ) -> ClaimNormalizationResult:
        record = self.evidence_ledger.read(evidence_record_id)
        if record is None:
            raise ClaimNormalizationError("Evidence record does not exist")
        plan = self.planner.plan(record)
        route = next(
            (item for item in plan.enhancement_routes if item.normalizer_name == normalizer_name),
            None,
        )
        if route is None:
            raise ClaimNormalizationError("requested Enhancement route is not planned")
        core_identity = self._core_identity(record, plan)
        core_receipt = self.claim_ledger.read_receipt(core_identity)
        if core_receipt is None:
            raise ClaimNormalizationConflictError(
                "Core must be completed before retrying an Enhancement"
            )
        receipt, degradation = await self._process_enhancement(record, plan, route, retry=True)
        if degradation is not None:
            status = (
                ClaimNormalizationResultStatus.CORE_COMMITTED_ENHANCEMENT_PENDING
                if degradation.retryable
                else ClaimNormalizationResultStatus.CORE_COMMITTED_WITH_DEGRADATION
            )
        else:
            status = ClaimNormalizationResultStatus.COMPLETE
        return ClaimNormalizationResult(
            evidence_record_id=record.evidence_record_id,
            status=status,
            core_receipt=core_receipt,
            enhancement_receipts=() if receipt is None else (receipt,),
            degradations=() if degradation is None else (degradation,),
        )

    def read_claim(self, claim_id: str) -> BehaviorClaim | None:
        return self.claim_ledger.read_claim(claim_id)

    def list_claims_for_evidence(
        self,
        evidence_record_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> ClaimPage:
        return self.claim_ledger.list_for_evidence(evidence_record_id, limit, cursor)

    def list_claims_after_sequence(
        self,
        sequence: int,
        limit: int,
    ) -> tuple[BehaviorClaimLedgerEntry, ...]:
        return self.claim_ledger.list_after_sequence(sequence, limit)

    async def _process_core(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
    ) -> ClaimNormalizationReceipt:
        if not plan.core_routes:
            return await self._publish_no_core(record, plan)
        route = plan.core_routes[0]
        receipt, degradation = await self._process_route(record, plan, route, retry=False)
        if degradation is not None or receipt is None:
            raise ClaimNormalizationError(
                "Deterministic Core failed; Evidence remains committed and no Core Receipt was published"
            )
        return receipt

    async def _publish_no_core(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
    ) -> ClaimNormalizationReceipt:
        identity = self._core_identity(record, plan)
        prior = self.claim_ledger.read_receipt(identity)
        if prior is not None:
            self._observe("claim_core_reused", time.monotonic(), record, prior, None)
            return prior
        started = time.monotonic()
        async with self.processing_lock.acquire(identity):
            prior = self.claim_ledger.read_receipt(identity)
            if prior is not None:
                self._observe("claim_core_reused", started, record, prior, None)
                return prior
            now = self.clock.now()
            receipt = ClaimNormalizationReceipt(
                processing_identity=identity,
                evidence_record_id=record.evidence_record_id,
                lane=NormalizationLane.CORE,
                normalizer_fingerprint=NO_CORE_NORMALIZER_FINGERPRINT,
                planner_policy_digest=plan.planner_policy_digest,
                compatibility_policy_digest=self.binder.compatibility.digest,
                binding_policy_digest=self.binder.binding.digest,
                confidence_policy_digest=self.binder.confidence.digest,
                status=ReceiptStatus.NO_CORE_REQUIRED,
                attempt_ids=(),
                claim_ids=(),
                completed_at=now,
                publication_recorded_at=now,
            )
            stored, reused = self.claim_ledger.publish_route(None, (), receipt)
            self._observe(
                "claim_core_reused" if reused else "claim_core_noop",
                started,
                record,
                stored,
                None,
            )
            return stored

    async def _process_enhancement(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
        route: ClaimNormalizationRoute,
        *,
        retry: bool,
    ) -> tuple[ClaimNormalizationReceipt | None, ClaimProcessingDegradation | None]:
        receipt, degradation = await self._process_route(record, plan, route, retry=retry)
        return receipt, degradation

    async def _process_route(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
        route: ClaimNormalizationRoute,
        *,
        retry: bool,
    ) -> tuple[ClaimNormalizationReceipt | None, ClaimProcessingDegradation | None]:
        identity = self._route_identity(record, plan, route)
        prior = self.claim_ledger.read_receipt(identity)
        if prior is not None:
            self._observe(
                "claim_core_reused" if route.lane is NormalizationLane.CORE else "claim_enhancement_completed",
                time.monotonic(),
                record,
                prior,
                None,
            )
            return prior, None
        started_metric = time.monotonic()
        async with self.processing_lock.acquire(identity):
            prior = self.claim_ledger.read_receipt(identity)
            if prior is not None:
                return prior, None
            latest = self.claim_ledger.read_latest_attempt(identity)
            if latest is not None:
                if retry:
                    if latest.status is not AttemptStatus.FAILED_RETRYABLE:
                        raise ClaimNormalizationConflictError(
                            "only the latest retryable Enhancement failure can be retried"
                        )
                else:
                    degradation = ClaimProcessingDegradation(
                        route.normalizer_name,
                        latest.error_code or "NORMALIZATION_FAILED",
                        latest.retryable,
                    )
                    return None, degradation
            elif retry:
                raise ClaimNormalizationConflictError("Enhancement has no retryable failed Attempt")
            attempt_number = 1 if latest is None else latest.attempt_number + 1
            started_at = self.clock.now()
            try:
                proposals, claims = await self._normalize_route(record, route)
            except (
                BehaviorClaimSchemaError,
                ClaimCompatibilityError,
                ClaimModelAuthenticationError,
                ClaimModelConfigurationError,
                ClaimModelContentSafetyError,
                ClaimModelInputError,
                ClaimModelPermissionError,
                ClaimModelQuotaError,
                ClaimModelSchemaError,
                ClaimModelTransportError,
                TypeError,
                ValueError,
            ) as exc:
                degradation = self._publish_failure(
                    identity=identity,
                    record=record,
                    route=route,
                    attempt_number=attempt_number,
                    started_at=started_at,
                    started_metric=started_metric,
                    error=exc,
                )
                if route.lane is NormalizationLane.CORE:
                    raise ClaimNormalizationError("Deterministic Core normalization failed") from exc
                return None, degradation
            attempt, receipt = self._success_publication(
                identity=identity,
                record=record,
                plan=plan,
                route=route,
                attempt_number=attempt_number,
                started_at=started_at,
                proposals=proposals,
                claims=claims,
            )
            stored, reused = self.claim_ledger.publish_route(attempt, claims, receipt)
            abstained = not proposals
            if route.lane is NormalizationLane.CORE:
                operation = "claim_core_reused" if reused else "claim_core_completed"
            elif abstained:
                operation = "claim_enhancement_abstained"
            elif retry:
                operation = "claim_enhancement_retried"
            else:
                operation = "claim_enhancement_completed"
            self._observe(operation, started_metric, record, stored, None)
            return stored, None

    async def _normalize_route(
        self,
        record: BehaviorEvidenceRecord,
        route: ClaimNormalizationRoute,
    ) -> tuple[tuple[ClaimSemanticProposal, ...], tuple[BehaviorClaim, ...]]:
        proposals = await self.normalizers.get(route.normalizer_name).normalize(record)
        self._validate_proposals(proposals)
        derivation = (
            DerivationClass.DETERMINISTIC
            if route.normalizer_kind is ClaimNormalizerKind.DETERMINISTIC
            else DerivationClass.MODEL
        )
        claims = tuple(
            self.binder.bind(
                record,
                proposal,
                normalizer_fingerprint=route.normalizer_fingerprint,
                normalizer_kind=route.normalizer_kind,
                derivation_class=derivation,
                created_at=self.clock.now(),
            )
            for proposal in proposals
        )
        return proposals, claims

    def _validate_proposals(self, proposals: object) -> None:
        if not isinstance(proposals, tuple) or any(
            not isinstance(proposal, ClaimSemanticProposal) for proposal in proposals
        ):
            raise BehaviorClaimSchemaError("Normalizer returned an invalid Proposal sequence")
        if len(proposals) > self.planner.config.max_claims_per_record:
            raise BehaviorClaimSchemaError("Normalizer exceeded Claim count capacity")
        alternative_counts: dict[str, int] = {}
        for proposal in proposals:
            group = proposal.local_alternative_group_id
            if group is None:
                continue
            alternative_counts[group] = alternative_counts.get(group, 0) + 1
            if alternative_counts[group] > self.planner.config.max_alternative_group_size:
                raise BehaviorClaimSchemaError("Normalizer exceeded alternative group capacity")

    def _publish_failure(
        self,
        *,
        identity: str,
        record: BehaviorEvidenceRecord,
        route: ClaimNormalizationRoute,
        attempt_number: int,
        started_at: datetime,
        started_metric: float,
        error: Exception,
    ) -> ClaimProcessingDegradation:
        status, code, retryable = self._failure(error)
        attempt = ClaimNormalizationAttempt(
            processing_identity=identity,
            evidence_record_id=record.evidence_record_id,
            normalizer_name=route.normalizer_name,
            normalizer_fingerprint=route.normalizer_fingerprint,
            lane=route.lane,
            attempt_number=attempt_number,
            status=status,
            proposal_digest=None,
            claim_count=0,
            error_code=code,
            retryable=retryable,
            started_at=started_at,
            completed_at=self.clock.now(),
        )
        self.claim_ledger.publish_failed_attempt(attempt)
        operation = (
            "claim_core_completed"
            if route.lane is NormalizationLane.CORE
            else "claim_enhancement_failed"
        )
        self._observe(operation, started_metric, record, None, code, retryable)
        return ClaimProcessingDegradation(route.normalizer_name, code, retryable)

    def _success_publication(
        self,
        *,
        identity: str,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
        route: ClaimNormalizationRoute,
        attempt_number: int,
        started_at: datetime,
        proposals: tuple[ClaimSemanticProposal, ...],
        claims: tuple[BehaviorClaim, ...],
    ) -> tuple[ClaimNormalizationAttempt, ClaimNormalizationReceipt]:
        completed_at = self.clock.now()
        abstained = not proposals
        attempt = ClaimNormalizationAttempt(
            processing_identity=identity,
            evidence_record_id=record.evidence_record_id,
            normalizer_name=route.normalizer_name,
            normalizer_fingerprint=route.normalizer_fingerprint,
            lane=route.lane,
            attempt_number=attempt_number,
            status=AttemptStatus.ABSTAINED if abstained else AttemptStatus.COMPLETED,
            proposal_digest=canonical_digest([proposal_to_dict(item) for item in proposals]),
            claim_count=len(claims),
            error_code=None,
            retryable=False,
            started_at=started_at,
            completed_at=completed_at,
        )
        receipt = ClaimNormalizationReceipt(
            processing_identity=identity,
            evidence_record_id=record.evidence_record_id,
            lane=route.lane,
            normalizer_fingerprint=route.normalizer_fingerprint,
            planner_policy_digest=plan.planner_policy_digest,
            compatibility_policy_digest=self.binder.compatibility.digest,
            binding_policy_digest=self.binder.binding.digest,
            confidence_policy_digest=self.binder.confidence.digest,
            status=ReceiptStatus.ABSTAINED if abstained else ReceiptStatus.COMPLETED,
            attempt_ids=tuple(
                attempt_identity(identity, number)
                for number in range(1, attempt.attempt_number + 1)
            ),
            claim_ids=tuple(claim.claim_id for claim in claims),
            completed_at=completed_at,
            publication_recorded_at=self.clock.now(),
        )
        return attempt, receipt

    def _core_identity(self, record: BehaviorEvidenceRecord, plan: ClaimNormalizationPlan) -> str:
        fingerprint = (
            NO_CORE_NORMALIZER_FINGERPRINT
            if not plan.core_routes
            else plan.core_routes[0].normalizer_fingerprint
        )
        return processing_identity(
            evidence_record_digest=record.content_digest,
            lane=NormalizationLane.CORE,
            normalizer_fingerprint=fingerprint,
            planner_policy_digest=plan.planner_policy_digest,
            compatibility_policy_digest=self.binder.compatibility.digest,
            binding_policy_digest=self.binder.binding.digest,
            confidence_policy_digest=self.binder.confidence.digest,
        )

    def _route_identity(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
        route: ClaimNormalizationRoute,
    ) -> str:
        return processing_identity(
            evidence_record_digest=record.content_digest,
            lane=route.lane,
            normalizer_fingerprint=route.normalizer_fingerprint,
            planner_policy_digest=plan.planner_policy_digest,
            compatibility_policy_digest=self.binder.compatibility.digest,
            binding_policy_digest=self.binder.binding.digest,
            confidence_policy_digest=self.binder.confidence.digest,
        )

    @staticmethod
    def _failure(exc: Exception) -> tuple[AttemptStatus, str, bool]:
        for error_type, status, code, retryable in _FAILURE_RULES:
            if isinstance(exc, error_type):
                return status, code, retryable
        return AttemptStatus.FAILED_NON_RETRYABLE, "NORMALIZATION_ERROR", False

    def _observe(
        self,
        operation: str,
        started: float,
        record: BehaviorEvidenceRecord,
        receipt: ClaimNormalizationReceipt | None,
        error_code: str | None,
        retryable: bool = False,
    ) -> None:
        attributes: dict[str, str | int | float | bool] = {
            "record_kind": record.semantic_content.record_kind.value,
            "result_count": 0 if receipt is None else len(receipt.claim_ids),
            "retryable": retryable,
        }
        if receipt is not None:
            attributes["status"] = receipt.status.value
        if error_code is not None:
            attributes["error_code"] = error_code
        try:
            self.observer.record(
                ObservationEvent(
                    category="behavior",
                    operation=operation,
                    status=(ObservationStatus.FAILURE if error_code else ObservationStatus.SUCCESS),
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:
            return


__all__ = ["ClaimNormalizationService"]
