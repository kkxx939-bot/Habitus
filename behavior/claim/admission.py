"""Versioned static Admission and immutable decision audit values."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from behavior._validation import (
    finite_score,
    identifier,
    optional_identifier,
    parse_utc,
    require_fields,
    sha256_digest,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.claim.model import Claim, EpistemicClass
from behavior.claim.policy import ClaimAdmissionPolicyIdentity, ClaimDerivationClass
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
    required_threshold: float
    evaluated_confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", identifier(self.claim_id, "claim_id"))
        if self.rejection_status is not None:
            status = ClaimAdmissionStatus(self.rejection_status)
            if status not in {
                ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD,
                ClaimAdmissionStatus.OWNER_SCOPE_REJECTED,
            }:
                raise ClaimAdmissionError("static Admission cannot decide a dynamic status")
            object.__setattr__(self, "rejection_status", status)
        object.__setattr__(self, "reason_code", identifier(self.reason_code, "reason_code"))
        object.__setattr__(self, "required_threshold", finite_score(self.required_threshold, "required_threshold"))
        object.__setattr__(
            self,
            "evaluated_confidence",
            finite_score(self.evaluated_confidence, "evaluated_confidence"),
        )


@dataclass(frozen=True)
class ClaimAdmissionDecision:
    decision_id: str
    processing_identity: str
    claim_id: str
    admission_policy_digest: str
    status: ClaimAdmissionStatus
    reason_code: str
    required_threshold: float
    evaluated_confidence: float
    admission_decided_at: datetime
    existing_claim_id: str | None
    content_digest: str

    def __post_init__(self) -> None:
        for name in ("decision_id", "processing_identity", "claim_id", "reason_code"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "admission_policy_digest",
            sha256_digest(self.admission_policy_digest, "admission_policy_digest"),
        )
        object.__setattr__(self, "status", ClaimAdmissionStatus(self.status))
        object.__setattr__(self, "required_threshold", finite_score(self.required_threshold, "required_threshold"))
        object.__setattr__(
            self,
            "evaluated_confidence",
            finite_score(self.evaluated_confidence, "evaluated_confidence"),
        )
        object.__setattr__(
            self,
            "admission_decided_at",
            strict_utc(self.admission_decided_at, "admission_decided_at"),
        )
        object.__setattr__(
            self,
            "existing_claim_id",
            optional_identifier(self.existing_claim_id, "existing_claim_id"),
        )
        if self.decision_id != self.identity_for(
            processing_identity=self.processing_identity,
            claim_id=self.claim_id,
            admission_policy_digest=self.admission_policy_digest,
            status=self.status,
            reason_code=self.reason_code,
            existing_claim_id=self.existing_claim_id,
        ):
            raise ClaimAdmissionError("AdmissionDecision identity mismatch")
        if self.content_digest != canonical_digest(self._content_payload()):
            raise ClaimAdmissionError("AdmissionDecision content digest mismatch")

    @staticmethod
    def identity_for(
        *,
        processing_identity: str,
        claim_id: str,
        admission_policy_digest: str,
        status: ClaimAdmissionStatus,
        reason_code: str,
        existing_claim_id: str | None,
    ) -> str:
        return "decision_" + canonical_digest(
            {
                "processing_identity": processing_identity,
                "claim_id": claim_id,
                "admission_policy_digest": admission_policy_digest,
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
        admission_policy_digest: str,
        required_threshold: float,
        evaluated_confidence: float,
        admission_decided_at: datetime,
        existing_claim_id: str | None = None,
    ) -> ClaimAdmissionDecision:
        decision_id = cls.identity_for(
            processing_identity=processing_identity,
            claim_id=claim.claim_id,
            admission_policy_digest=admission_policy_digest,
            status=status,
            reason_code=reason_code,
            existing_claim_id=existing_claim_id,
        )
        content = {
            "decision_id": decision_id,
            "processing_identity": processing_identity,
            "claim_id": claim.claim_id,
            "admission_policy_digest": admission_policy_digest,
            "status": ClaimAdmissionStatus(status).value,
            "reason_code": reason_code,
            "required_threshold": required_threshold,
            "evaluated_confidence": evaluated_confidence,
            "admission_decided_at": utc_text(admission_decided_at),
            "existing_claim_id": existing_claim_id,
        }
        return cls(
            decision_id=decision_id,
            processing_identity=processing_identity,
            claim_id=claim.claim_id,
            admission_policy_digest=admission_policy_digest,
            status=ClaimAdmissionStatus(status),
            reason_code=reason_code,
            required_threshold=required_threshold,
            evaluated_confidence=evaluated_confidence,
            admission_decided_at=admission_decided_at,
            existing_claim_id=existing_claim_id,
            content_digest=canonical_digest(content),
        )

    def _content_payload(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "processing_identity": self.processing_identity,
            "claim_id": self.claim_id,
            "admission_policy_digest": self.admission_policy_digest,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "required_threshold": self.required_threshold,
            "evaluated_confidence": self.evaluated_confidence,
            "admission_decided_at": utc_text(self.admission_decided_at),
            "existing_claim_id": self.existing_claim_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._content_payload(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, value: object) -> ClaimAdmissionDecision:
        fields = frozenset({*cls.__dataclass_fields__})
        data = strict_fields(value, "claim_admission_decision", fields)
        require_fields(data, "claim_admission_decision", fields)
        return cls(
            decision_id=data["decision_id"],
            processing_identity=data["processing_identity"],
            claim_id=data["claim_id"],
            admission_policy_digest=data["admission_policy_digest"],
            status=ClaimAdmissionStatus(data["status"]),
            reason_code=data["reason_code"],
            required_threshold=data["required_threshold"],
            evaluated_confidence=data["evaluated_confidence"],
            admission_decided_at=parse_utc(data["admission_decided_at"], "admission_decided_at"),
            existing_claim_id=data["existing_claim_id"],
            content_digest=data["content_digest"],
        )


class ClaimAdmissionPolicy:
    def __init__(
        self,
        *,
        config: ClaimConfig,
        max_accepted_claims: int,
        version: str = "3",
    ) -> None:
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        self.config = config
        if isinstance(max_accepted_claims, bool) or not isinstance(max_accepted_claims, int) or max_accepted_claims < 1:
            raise ValueError("max_accepted_claims must be a positive integer")
        self.max_accepted_claims = max_accepted_claims
        identity = ClaimAdmissionPolicyIdentity.from_config(
            config,
            max_accepted_claims=max_accepted_claims,
            version=identifier(version, "version", maximum=32),
        )
        self.version = identity.version
        self.digest = identity.digest

    def evaluate_static(self, claim: Claim, *, owner_identity_digest: str | None) -> StaticAdmissionResult:
        if claim.source_epistemic_class in {EpistemicClass.DIRECT_SOURCE, EpistemicClass.USER_EXPLICIT}:
            source_threshold = self.config.min_direct_confidence
        elif claim.source_epistemic_class is EpistemicClass.SENSOR_INFERRED:
            source_threshold = self.config.min_sensor_confidence
        else:
            source_threshold = self.config.min_model_confidence
        derivation_threshold = (
            0.0
            if claim.derivation_class is ClaimDerivationClass.DETERMINISTIC
            else self.config.min_model_confidence
        )
        required = max(source_threshold, derivation_threshold)
        if owner_identity_digest is None or claim.owner_identity_digest != owner_identity_digest:
            return StaticAdmissionResult(
                claim.claim_id,
                ClaimAdmissionStatus.OWNER_SCOPE_REJECTED,
                "owner_scope_mismatch",
                required,
                claim.effective_confidence,
            )
        if claim.effective_confidence < required:
            return StaticAdmissionResult(
                claim.claim_id,
                ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD,
                "confidence_below_required_threshold",
                required,
                claim.effective_confidence,
            )
        return StaticAdmissionResult(
            claim.claim_id,
            None,
            "static_admission_passed",
            required,
            claim.effective_confidence,
        )


__all__ = [
    "ClaimAdmissionDecision",
    "ClaimAdmissionPolicy",
    "ClaimAdmissionStatus",
    "StaticAdmissionResult",
]
