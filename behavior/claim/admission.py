"""Claim 静态准入策略与事务内动态决策规则。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from behavior._validation import (
    identifier,
    optional_identifier,
    parse_utc,
    require_fields,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.claim.model import Claim, EpistemicClass
from behavior.claim.proposal import ClaimKind
from behavior.config import ClaimConfig
from behavior.errors import ClaimAdmissionError
from foundation.integrity import canonical_digest


class ClaimAdmissionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    REPEATED_STATE_SUPPRESSED = "REPEATED_STATE_SUPPRESSED"
    BELOW_SCORE_THRESHOLD = "BELOW_SCORE_THRESHOLD"
    NO_INFORMATION_GAIN = "NO_INFORMATION_GAIN"
    OWNER_SCOPE_REJECTED = "OWNER_SCOPE_REJECTED"
    CAPACITY_REJECTED = "CAPACITY_REJECTED"


@dataclass(frozen=True)
class StaticAdmissionResult:
    claim_id: str
    rejection_status: ClaimAdmissionStatus | None
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", identifier(self.claim_id, "claim_id"))
        if self.rejection_status is not None:
            status = ClaimAdmissionStatus(self.rejection_status)
            if status in {
                ClaimAdmissionStatus.ACCEPTED,
                ClaimAdmissionStatus.EXACT_DUPLICATE,
                ClaimAdmissionStatus.REPEATED_STATE_SUPPRESSED,
                ClaimAdmissionStatus.NO_INFORMATION_GAIN,
                ClaimAdmissionStatus.CAPACITY_REJECTED,
            }:
                raise ClaimAdmissionError("static Admission cannot decide a dynamic status")
            object.__setattr__(self, "rejection_status", status)
        object.__setattr__(self, "reason_code", identifier(self.reason_code, "reason_code"))


@dataclass(frozen=True)
class ClaimAdmissionDecision:
    decision_id: str
    processing_identity: str
    claim_id: str
    status: ClaimAdmissionStatus
    reason_code: str
    decided_at: datetime
    existing_claim_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("decision_id", "processing_identity", "claim_id", "reason_code"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        object.__setattr__(self, "status", ClaimAdmissionStatus(self.status))
        object.__setattr__(self, "decided_at", strict_utc(self.decided_at, "decided_at"))
        object.__setattr__(
            self,
            "existing_claim_id",
            optional_identifier(self.existing_claim_id, "existing_claim_id"),
        )
        expected = self.identity_for(
            processing_identity=self.processing_identity,
            claim_id=self.claim_id,
            status=self.status,
            reason_code=self.reason_code,
            existing_claim_id=self.existing_claim_id,
        )
        if self.decision_id != expected:
            raise ClaimAdmissionError("AdmissionDecision identity mismatch")

    @staticmethod
    def identity_for(
        *,
        processing_identity: str,
        claim_id: str,
        status: ClaimAdmissionStatus,
        reason_code: str,
        existing_claim_id: str | None,
    ) -> str:
        return "decision_" + canonical_digest(
            {
                "processing_identity": processing_identity,
                "claim_id": claim_id,
                "status": ClaimAdmissionStatus(status).value,
                "reason_code": reason_code,
                "existing_claim_id": existing_claim_id,
            }
        )

    @classmethod
    def create(
        cls,
        claim: Claim,
        status: ClaimAdmissionStatus,
        reason_code: str,
        *,
        processing_identity: str,
        decided_at: datetime,
        existing_claim_id: str | None = None,
    ) -> ClaimAdmissionDecision:
        decision_id = cls.identity_for(
            processing_identity=processing_identity,
            claim_id=claim.claim_id,
            status=status,
            reason_code=reason_code,
            existing_claim_id=existing_claim_id,
        )
        return cls(
            decision_id=decision_id,
            processing_identity=processing_identity,
            claim_id=claim.claim_id,
            status=status,
            reason_code=reason_code,
            decided_at=decided_at,
            existing_claim_id=existing_claim_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "processing_identity": self.processing_identity,
            "claim_id": self.claim_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "decided_at": utc_text(self.decided_at),
            "existing_claim_id": self.existing_claim_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> ClaimAdmissionDecision:
        fields = frozenset(
            {
                "decision_id",
                "processing_identity",
                "claim_id",
                "status",
                "reason_code",
                "decided_at",
                "existing_claim_id",
            }
        )
        data = strict_fields(value, "claim_admission_decision", fields)
        require_fields(data, "claim_admission_decision", fields)
        return cls(
            decision_id=data["decision_id"],
            processing_identity=data["processing_identity"],
            claim_id=data["claim_id"],
            status=ClaimAdmissionStatus(data["status"]),
            reason_code=data["reason_code"],
            decided_at=parse_utc(data["decided_at"], "decided_at"),
            existing_claim_id=data["existing_claim_id"],
        )


class ClaimAdmissionPolicy:
    def __init__(self, *, config: ClaimConfig) -> None:
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        self.config = config

    def evaluate_static(self, claim: Claim, *, owner_identity_digest: str | None) -> StaticAdmissionResult:
        if not isinstance(claim, Claim):
            raise TypeError("claim must be Claim")
        if owner_identity_digest is None or claim.owner_identity_digest != owner_identity_digest:
            return StaticAdmissionResult(
                claim.claim_id,
                ClaimAdmissionStatus.OWNER_SCOPE_REJECTED,
                "owner_scope_mismatch",
            )
        if claim.epistemic_class is EpistemicClass.SENSOR_INFERRED:
            threshold = self.config.min_sensor_confidence
        elif claim.epistemic_class in {
            EpistemicClass.MODEL_INFERRED,
            EpistemicClass.MULTIMODAL_MODEL_INFERRED,
        }:
            threshold = self.config.min_model_confidence
        else:
            threshold = self.config.min_direct_confidence
        if claim.effective_confidence < threshold:
            return StaticAdmissionResult(
                claim.claim_id,
                ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD,
                "confidence_below_configured_threshold",
            )
        return StaticAdmissionResult(claim.claim_id, None, "static_admission_passed")

    def evaluate_dynamic(
        self,
        claim: Claim,
        *,
        exact_claim_id: str | None,
        same_batch_claim_id: str | None,
        recent_state_claim_id: str | None,
        capacity_reached: bool,
    ) -> tuple[ClaimAdmissionStatus, str, str | None]:
        if exact_claim_id is not None:
            return (
                ClaimAdmissionStatus.EXACT_DUPLICATE,
                "claim_identity_already_published",
                exact_claim_id,
            )
        if same_batch_claim_id is not None:
            return (
                ClaimAdmissionStatus.NO_INFORMATION_GAIN,
                "same_processing_semantic_duplicate",
                same_batch_claim_id,
            )
        if claim.proposal.claim_kind is ClaimKind.STATE_ASSERTION and recent_state_claim_id is not None:
            return (
                ClaimAdmissionStatus.REPEATED_STATE_SUPPRESSED,
                "state_repeated_within_configured_window",
                recent_state_claim_id,
            )
        if not isinstance(capacity_reached, bool):
            raise TypeError("capacity_reached must be boolean")
        if capacity_reached:
            return (
                ClaimAdmissionStatus.CAPACITY_REJECTED,
                "accepted_claim_capacity_reached",
                None,
            )
        return ClaimAdmissionStatus.ACCEPTED, "claim_passed_admission", None


__all__ = [
    "ClaimAdmissionDecision",
    "ClaimAdmissionPolicy",
    "ClaimAdmissionStatus",
    "StaticAdmissionResult",
]
