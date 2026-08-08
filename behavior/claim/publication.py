"""Claim 路由的 Processing Identity 与原子提交聚合。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from behavior.claim.model import CLAIM_PIPELINE_VERSION, CLAIM_SCHEMA_VERSION, BehaviorClaim, DerivationClass
from behavior.claim.planner import ClaimNormalizationPlan, ClaimNormalizationRoute, NormalizationLane
from behavior.claim.proposal import ClaimSemanticProposal, proposal_to_dict
from behavior.claim.receipt import (
    NO_CORE_NORMALIZER_FINGERPRINT,
    AttemptStatus,
    ClaimNormalizationAttempt,
    ClaimNormalizationReceipt,
    ReceiptStatus,
    attempt_identity,
)
from behavior.errors import ClaimNormalizationConflictError, NormalizerOutputError
from behavior.evidence.record import BehaviorEvidenceRecord
from foundation.integrity import canonical_digest


@dataclass(frozen=True, slots=True, init=False)
class ProcessingIdentity:
    value: str

    @classmethod
    def create(cls, *, evidence_record_digest: str, lane: NormalizationLane,
               normalizer_fingerprint: str, planner_policy_digest: str,
               compatibility_policy_digest: str, binding_policy_digest: str,
               confidence_policy_digest: str) -> ProcessingIdentity:
        instance = object.__new__(cls)
        value = canonical_digest({
            "binding_policy_digest": binding_policy_digest,
            "claim_schema_version": CLAIM_SCHEMA_VERSION,
            "compatibility_policy_digest": compatibility_policy_digest,
            "confidence_policy_digest": confidence_policy_digest,
            "evidence_record_digest": evidence_record_digest,
            "lane": lane.value,
            "normalizer_fingerprint": normalizer_fingerprint,
            "pipeline_version": CLAIM_PIPELINE_VERSION,
            "planner_policy_digest": planner_policy_digest,
        })
        object.__setattr__(instance, "value", value)
        return instance


@dataclass(frozen=True, slots=True)
class ClaimPublication:
    identity: ProcessingIdentity
    attempt: ClaimNormalizationAttempt | None
    claims: tuple[BehaviorClaim, ...]
    receipt: ClaimNormalizationReceipt

    def __post_init__(self) -> None:
        if self.receipt.processing_identity != self.identity.value:
            raise ClaimNormalizationConflictError("Receipt processing identity is not factory-owned")
        if self.attempt is None:
            if self.receipt.status is not ReceiptStatus.NO_CORE_REQUIRED:
                raise ClaimNormalizationConflictError("only NO_CORE_REQUIRED omits an Attempt")
            if self.claims:
                raise ClaimNormalizationConflictError("NO_CORE_REQUIRED cannot publish Claims")
        elif (
            self.attempt.processing_identity != self.identity.value
            or self.attempt.evidence_record_id != self.receipt.evidence_record_id
            or self.attempt.lane is not self.receipt.lane
            or self.attempt.normalizer_fingerprint != self.receipt.normalizer_fingerprint
            or self.receipt.attempt_ids[-1] != self.attempt.attempt_id
        ):
            raise ClaimNormalizationConflictError("Attempt and Receipt aggregate ownership disagree")
        elif (
            self.attempt.status not in {AttemptStatus.COMPLETED, AttemptStatus.ABSTAINED}
            or self.attempt.claim_count != len(self.claims)
            or (self.attempt.status is AttemptStatus.ABSTAINED)
            != (self.receipt.status is ReceiptStatus.ABSTAINED)
        ):
            raise ClaimNormalizationConflictError("Attempt result disagrees with publication")
        if self.receipt.claim_ids != tuple(claim.claim_id for claim in self.claims):
            raise ClaimNormalizationConflictError("Receipt Claim members disagree with publication")
        expected_derivation = (
            DerivationClass.DETERMINISTIC
            if self.receipt.lane is NormalizationLane.CORE
            else DerivationClass.MODEL
        )
        if any(
            claim.evidence_record_id != self.receipt.evidence_record_id
            or claim.normalizer_fingerprint != self.receipt.normalizer_fingerprint
            or claim.compatibility_policy_digest != self.receipt.compatibility_policy_digest
            or claim.binding_policy_digest != self.receipt.binding_policy_digest
            or claim.confidence_policy_digest != self.receipt.confidence_policy_digest
            or claim.derivation_class is not expected_derivation
            for claim in self.claims
        ):
            raise ClaimNormalizationConflictError("Claim ownership disagrees with Receipt")


class ClaimPublicationFactory:
    def __init__(self, *, compatibility_policy_digest: str, binding_policy_digest: str,
                 confidence_policy_digest: str) -> None:
        self.compatibility_policy_digest = compatibility_policy_digest
        self.binding_policy_digest = binding_policy_digest
        self.confidence_policy_digest = confidence_policy_digest

    def identity(self, record: BehaviorEvidenceRecord, plan: ClaimNormalizationPlan, *,
                 lane: NormalizationLane, normalizer_fingerprint: str) -> ProcessingIdentity:
        return ProcessingIdentity.create(
            evidence_record_digest=record.content_digest,
            lane=lane,
            normalizer_fingerprint=normalizer_fingerprint,
            planner_policy_digest=plan.planner_policy_digest,
            compatibility_policy_digest=self.compatibility_policy_digest,
            binding_policy_digest=self.binding_policy_digest,
            confidence_policy_digest=self.confidence_policy_digest,
        )

    def no_core(self, record: BehaviorEvidenceRecord, plan: ClaimNormalizationPlan, *,
                completed_at: datetime, publication_recorded_at: datetime) -> ClaimPublication:
        identity = self.identity(record, plan, lane=NormalizationLane.CORE,
                                 normalizer_fingerprint=NO_CORE_NORMALIZER_FINGERPRINT)
        receipt = self._receipt(
            identity,
            record,
            plan,
            lane=NormalizationLane.CORE,
            normalizer_fingerprint=NO_CORE_NORMALIZER_FINGERPRINT,
            status=ReceiptStatus.NO_CORE_REQUIRED,
            attempt_ids=(),
            claim_ids=(),
            completed_at=completed_at,
            publication_recorded_at=publication_recorded_at,
        )
        return ClaimPublication(identity, None, (), receipt)

    def success(self, record: BehaviorEvidenceRecord, plan: ClaimNormalizationPlan,
                route: ClaimNormalizationRoute, *, attempt_number: int, started_at: datetime,
                completed_at: datetime, publication_recorded_at: datetime,
                proposals: tuple[ClaimSemanticProposal, ...],
                claims: tuple[BehaviorClaim, ...]) -> ClaimPublication:
        if route.lane is NormalizationLane.CORE and not proposals:
            raise NormalizerOutputError("Deterministic Core cannot abstain")
        identity = self.identity(record, plan, lane=route.lane,
                                 normalizer_fingerprint=route.normalizer_fingerprint)
        abstained = not proposals
        attempt = ClaimNormalizationAttempt(
            processing_identity=identity.value,
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
        receipt = self._receipt(
            identity,
            record,
            plan,
            lane=route.lane,
            normalizer_fingerprint=route.normalizer_fingerprint,
            status=ReceiptStatus.ABSTAINED if abstained else ReceiptStatus.COMPLETED,
            attempt_ids=tuple(attempt_identity(identity.value, number)
                              for number in range(1, attempt_number + 1)),
            claim_ids=tuple(claim.claim_id for claim in claims),
            completed_at=completed_at,
            publication_recorded_at=publication_recorded_at,
        )
        return ClaimPublication(identity, attempt, claims, receipt)

    def failed_attempt(self, identity: ProcessingIdentity, record: BehaviorEvidenceRecord,
                       route: ClaimNormalizationRoute, *, attempt_number: int,
                       status: AttemptStatus, error_code: str, retryable: bool,
                       started_at: datetime, completed_at: datetime) -> ClaimNormalizationAttempt:
        return ClaimNormalizationAttempt(
            processing_identity=identity.value,
            evidence_record_id=record.evidence_record_id,
            normalizer_name=route.normalizer_name,
            normalizer_fingerprint=route.normalizer_fingerprint,
            lane=route.lane,
            attempt_number=attempt_number,
            status=status,
            proposal_digest=None,
            claim_count=0,
            error_code=error_code,
            retryable=retryable,
            started_at=started_at,
            completed_at=completed_at,
        )

    def _receipt(self, identity: ProcessingIdentity, record: BehaviorEvidenceRecord,
                 plan: ClaimNormalizationPlan, *, lane: NormalizationLane,
                 normalizer_fingerprint: str, status: ReceiptStatus,
                 attempt_ids: tuple[str, ...], claim_ids: tuple[str, ...],
                 completed_at: datetime,
                 publication_recorded_at: datetime) -> ClaimNormalizationReceipt:
        return ClaimNormalizationReceipt(
            processing_identity=identity.value,
            evidence_record_id=record.evidence_record_id,
            lane=lane,
            normalizer_fingerprint=normalizer_fingerprint,
            planner_policy_digest=plan.planner_policy_digest,
            compatibility_policy_digest=self.compatibility_policy_digest,
            binding_policy_digest=self.binding_policy_digest,
            confidence_policy_digest=self.confidence_policy_digest,
            status=status,
            attempt_ids=attempt_ids,
            claim_ids=claim_ids,
            completed_at=completed_at,
            publication_recorded_at=publication_recorded_at,
        )
