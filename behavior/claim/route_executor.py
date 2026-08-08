"""单条 NormalizationRoute 的锁、执行、分类与原子发布。"""

from __future__ import annotations

import time
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from behavior.claim.binder import ClaimFactory
from behavior.claim.ledger import BehaviorClaimLedger
from behavior.claim.planner import (
    ClaimNormalizationPlan,
    ClaimNormalizationRoute,
    NormalizationLane,
)
from behavior.claim.proposal import ClaimProposalParser
from behavior.claim.publication import ClaimPublicationFactory, ProcessingIdentity
from behavior.claim.receipt import (
    NO_CORE_NORMALIZER_FINGERPRINT,
    AttemptStatus,
    ClaimNormalizationAttempt,
    ClaimNormalizationReceipt,
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
    NormalizerOutputError,
)
from behavior.evidence.ingress import Clock
from behavior.evidence.record import BehaviorEvidenceRecord
from foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer


@runtime_checkable
class ProcessingLock(Protocol):
    def acquire(self, processing_identity: str) -> AbstractAsyncContextManager[object]: ...


@dataclass(frozen=True, slots=True)
class ClaimProcessingDegradation:
    normalizer_name: str
    error_code: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class RouteExecutionResult:
    receipt: ClaimNormalizationReceipt | None
    degradation: ClaimProcessingDegradation | None


class NormalizationFailureClassifier:
    _RULES: tuple[tuple[type[Exception], AttemptStatus, str, bool], ...] = (
        (ClaimModelTransportError, AttemptStatus.FAILED_RETRYABLE, "MODEL_TRANSPORT", True),
        (ClaimModelContentSafetyError, AttemptStatus.FAILED_POLICY, "MODEL_CONTENT_SAFETY", False),
        (ClaimModelSchemaError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_SCHEMA", False),
        (ClaimModelInputError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_INPUT", False),
        (ClaimModelAuthenticationError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_AUTHENTICATION", False),
        (ClaimModelPermissionError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_PERMISSION", False),
        (ClaimModelConfigurationError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_CONFIGURATION", False),
        (ClaimModelQuotaError, AttemptStatus.FAILED_NON_RETRYABLE, "MODEL_QUOTA", False),
        (ClaimCompatibilityError, AttemptStatus.FAILED_POLICY, "COMPATIBILITY_POLICY", False),
        (NormalizerOutputError, AttemptStatus.FAILED_NON_RETRYABLE, "NORMALIZER_OUTPUT", False),
        (BehaviorClaimSchemaError, AttemptStatus.FAILED_NON_RETRYABLE, "CLAIM_SCHEMA", False),
    )

    def classify(self, error: Exception) -> tuple[AttemptStatus, str, bool]:
        for error_type, status, code, retryable in self._RULES:
            if isinstance(error, error_type):
                return status, code, retryable
        raise TypeError("undeclared normalization failure")


_DECLARED_FAILURES = (
    ClaimModelAuthenticationError,
    ClaimModelConfigurationError,
    ClaimModelContentSafetyError,
    ClaimModelInputError,
    ClaimModelPermissionError,
    ClaimModelQuotaError,
    ClaimModelSchemaError,
    ClaimModelTransportError,
    ClaimCompatibilityError,
    NormalizerOutputError,
    BehaviorClaimSchemaError,
)


class NormalizationRouteExecutor:
    def __init__(
        self,
        *,
        ledger: BehaviorClaimLedger,
        normalizers: ClaimNormalizerRegistry,
        proposal_parser: ClaimProposalParser,
        claim_factory: ClaimFactory,
        publication_factory: ClaimPublicationFactory,
        processing_lock: ProcessingLock,
        clock: Clock,
        observer: Observer | None = None,
        failure_classifier: NormalizationFailureClassifier | None = None,
    ) -> None:
        self.ledger = ledger
        self.normalizers = normalizers
        self.proposal_parser = proposal_parser
        self.claim_factory = claim_factory
        self.publication_factory = publication_factory
        self.processing_lock = processing_lock
        self.clock = clock
        self.observer = observer or NullObserver()
        self.failure_classifier = failure_classifier or NormalizationFailureClassifier()

    async def publish_no_core(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
    ) -> ClaimNormalizationReceipt:
        identity = self.publication_factory.identity(
            record,
            plan,
            lane=NormalizationLane.CORE,
            normalizer_fingerprint=NO_CORE_NORMALIZER_FINGERPRINT,
        )
        prior = self.ledger.read_receipt(identity.value)
        if prior is not None:
            return prior
        async with self.processing_lock.acquire(identity.value):
            prior = self.ledger.read_receipt(identity.value)
            if prior is not None:
                return prior
            now = self.clock.now()
            publication = self.publication_factory.no_core(
                record,
                plan,
                completed_at=now,
                publication_recorded_at=now,
            )
            stored, _ = self.ledger.publish(publication)
            return stored

    async def execute(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
        route: ClaimNormalizationRoute,
        *,
        retry: bool,
    ) -> RouteExecutionResult:
        identity = self.identity(record, plan, route)
        prior = self.ledger.read_receipt(identity.value)
        if prior is not None:
            return RouteExecutionResult(prior, None)
        started_metric = time.monotonic()
        async with self.processing_lock.acquire(identity.value):
            prior = self.ledger.read_receipt(identity.value)
            if prior is not None:
                return RouteExecutionResult(prior, None)
            latest = self.ledger.read_latest_attempt(identity.value)
            blocked = self._retry_state(route, latest, retry)
            if blocked is not None:
                return RouteExecutionResult(None, blocked)
            attempt_number = 1 if latest is None else latest.attempt_number + 1
            started_at = self.clock.now()
            try:
                raw = await self.normalizers.get(route.normalizer_name).normalize(record)
                proposals = (
                    self.proposal_parser.parse_batch(raw).proposals
                    if isinstance(raw, dict)
                    else self.proposal_parser.validate_batch(raw).proposals
                )
                claims = tuple(
                    self.claim_factory.create(
                        record,
                        proposal,
                        route,
                        created_at=self.clock.now(),
                    )
                    for proposal in proposals
                )
                now = self.clock.now()
                publication = self.publication_factory.success(
                    record,
                    plan,
                    route,
                    attempt_number=attempt_number,
                    started_at=started_at,
                    completed_at=now,
                    publication_recorded_at=self.clock.now(),
                    proposals=proposals,
                    claims=claims,
                )
            except _DECLARED_FAILURES as exc:
                degradation = self._publish_failure(
                    identity,
                    record,
                    route,
                    attempt_number,
                    started_at,
                    exc,
                )
                if route.lane is NormalizationLane.CORE:
                    raise ClaimNormalizationError("Deterministic Core normalization failed") from exc
                self._observe("claim_enhancement_failed", started_metric, record, None, degradation.error_code)
                return RouteExecutionResult(None, degradation)
            stored, reused = self.ledger.publish(publication)
            self._observe(
                self._operation(route, retry, bool(proposals), reused),
                started_metric,
                record,
                stored,
                None,
            )
            return RouteExecutionResult(stored, None)

    def identity(
        self,
        record: BehaviorEvidenceRecord,
        plan: ClaimNormalizationPlan,
        route: ClaimNormalizationRoute,
    ) -> ProcessingIdentity:
        return self.publication_factory.identity(
            record,
            plan,
            lane=route.lane,
            normalizer_fingerprint=route.normalizer_fingerprint,
        )

    @staticmethod
    def _retry_state(
        route: ClaimNormalizationRoute,
        latest: ClaimNormalizationAttempt | None,
        retry: bool,
    ) -> ClaimProcessingDegradation | None:
        if latest is None:
            if retry:
                raise ClaimNormalizationConflictError("Enhancement has no failed Attempt")
            return None
        status = latest.status
        if retry:
            if status is not AttemptStatus.FAILED_RETRYABLE:
                raise ClaimNormalizationConflictError(
                    "only the latest retryable Enhancement failure can be retried"
                )
            return None
        return ClaimProcessingDegradation(
            route.normalizer_name,
            latest.error_code or "NORMALIZATION_FAILED",
            bool(latest.retryable),
        )

    def _publish_failure(
        self,
        identity: ProcessingIdentity,
        record: BehaviorEvidenceRecord,
        route: ClaimNormalizationRoute,
        attempt_number: int,
        started_at: datetime,
        error: Exception,
    ) -> ClaimProcessingDegradation:
        status, code, retryable = self.failure_classifier.classify(error)
        attempt = self.publication_factory.failed_attempt(
            identity,
            record,
            route,
            attempt_number=attempt_number,
            status=status,
            error_code=code,
            retryable=retryable,
            started_at=started_at,
            completed_at=self.clock.now(),
        )
        self.ledger.publish_failed_attempt(attempt)
        return ClaimProcessingDegradation(route.normalizer_name, code, retryable)

    @staticmethod
    def _operation(
        route: ClaimNormalizationRoute,
        retry: bool,
        has_proposals: bool,
        reused: bool,
    ) -> str:
        if route.lane is NormalizationLane.CORE:
            return "claim_core_reused" if reused else "claim_core_completed"
        if not has_proposals:
            return "claim_enhancement_abstained"
        return "claim_enhancement_retried" if retry else "claim_enhancement_completed"

    def _observe(
        self,
        operation: str,
        started: float,
        record: BehaviorEvidenceRecord,
        receipt: ClaimNormalizationReceipt | None,
        error_code: str | None,
    ) -> None:
        attributes: dict[str, str | int | float | bool] = {
            "record_kind": record.semantic_content.record_kind.value,
            "result_count": 0 if receipt is None else len(receipt.claim_ids),
        }
        if error_code is not None:
            attributes["error_code"] = error_code
        try:
            self.observer.record(
                ObservationEvent(
                    category="behavior",
                    operation=operation,
                    status=ObservationStatus.FAILURE if error_code else ObservationStatus.SUCCESS,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:
            return
