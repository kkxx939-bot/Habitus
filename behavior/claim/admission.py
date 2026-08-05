"""合法 Claim 的有界重复、增量与质量准入。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
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
from behavior.claim.model import Claim
from behavior.claim.proposal import ClaimKind, EpistemicClass
from behavior.config import ClaimConfig
from behavior.errors import ClaimAdmissionError
from behavior.persistence.contracts import BehaviorEvidenceClaimStore
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
class ClaimAdmissionDecision:
    decision_id: str
    processing_identity: str
    claim_id: str
    status: ClaimAdmissionStatus
    reason_code: str
    decided_at: datetime
    existing_claim_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", identifier(self.decision_id, "decision_id"))
        object.__setattr__(
            self,
            "processing_identity",
            identifier(self.processing_identity, "processing_identity"),
        )
        object.__setattr__(self, "claim_id", identifier(self.claim_id, "claim_id"))
        object.__setattr__(self, "status", ClaimAdmissionStatus(self.status))
        object.__setattr__(self, "reason_code", identifier(self.reason_code, "reason_code"))
        object.__setattr__(self, "decided_at", strict_utc(self.decided_at, "decided_at"))
        object.__setattr__(
            self,
            "existing_claim_id",
            optional_identifier(self.existing_claim_id, "existing_claim_id"),
        )
        expected = self.identity_for(
            claim_id=self.claim_id,
            processing_identity=self.processing_identity,
            status=self.status,
            reason_code=self.reason_code,
            existing_claim_id=self.existing_claim_id,
        )
        if self.decision_id != expected:
            raise ClaimAdmissionError("decision_id does not match deterministic identity")

    @staticmethod
    def identity_for(
        *,
        claim_id: str,
        processing_identity: str,
        status: ClaimAdmissionStatus,
        reason_code: str,
        existing_claim_id: str | None,
    ) -> str:
        return "decision_" + canonical_digest(
            {
                "claim_id": claim_id,
                "processing_identity": processing_identity,
                "existing_claim_id": existing_claim_id,
                "reason_code": reason_code,
                "status": ClaimAdmissionStatus(status).value,
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
        existing_claim_id: str | None = None,
    ) -> ClaimAdmissionDecision:
        return cls(
            decision_id=cls.identity_for(
                claim_id=claim.claim_id,
                processing_identity=processing_identity,
                status=status,
                reason_code=reason_code,
                existing_claim_id=existing_claim_id,
            ),
            processing_identity=processing_identity,
            claim_id=claim.claim_id,
            status=status,
            reason_code=reason_code,
            decided_at=claim.created_at,
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


class ClaimAdmissionGate:
    def __init__(self, store: BehaviorEvidenceClaimStore, *, config: ClaimConfig) -> None:
        if not isinstance(store, BehaviorEvidenceClaimStore):
            raise TypeError("store must implement BehaviorEvidenceClaimStore")
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        self.store = store
        self.config = config

    def decide(
        self,
        claim: Claim,
        *,
        processing_identity: str,
        pending_accepted: tuple[Claim, ...] = (),
    ) -> ClaimAdmissionDecision:
        if not isinstance(claim, Claim):
            raise TypeError("claim must be Claim")
        owner_digest = self.store.owner_binding_digest()
        if owner_digest is None or owner_digest != claim.owner_binding_digest:
            return ClaimAdmissionDecision.create(
                claim,
                ClaimAdmissionStatus.OWNER_SCOPE_REJECTED,
                "owner_scope_mismatch",
                processing_identity=processing_identity,
            )
        existing = self.store.read_claim(claim.claim_id)
        if existing is not None:
            return ClaimAdmissionDecision.create(
                claim,
                ClaimAdmissionStatus.EXACT_DUPLICATE,
                "claim_identity_already_published",
                processing_identity=processing_identity,
                existing_claim_id=existing.claim_id,
            )
        threshold = (
            self.config.min_direct_score
            if claim.proposal.epistemic_class is EpistemicClass.DIRECT_SOURCE
            else self.config.min_model_score
        )
        if claim.proposal.raw_score < threshold:
            return ClaimAdmissionDecision.create(
                claim,
                ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD,
                "score_below_configured_threshold",
                processing_identity=processing_identity,
            )
        exact_pending = next(
            (item for item in pending_accepted if item.claim_id == claim.claim_id),
            None,
        )
        semantic_pending = next(
            (
                item
                for item in pending_accepted
                if claim.proposal.claim_kind is ClaimKind.STATE_ASSERTION
                and item.proposal.claim_kind is ClaimKind.STATE_ASSERTION
                and item.semantic_fingerprint == claim.semantic_fingerprint
            ),
            None,
        )
        no_gain = exact_pending or semantic_pending
        if no_gain is not None:
            return ClaimAdmissionDecision.create(
                claim,
                ClaimAdmissionStatus.NO_INFORMATION_GAIN,
                "same_batch_semantic_duplicate",
                processing_identity=processing_identity,
                existing_claim_id=no_gain.claim_id,
            )
        if claim.proposal.claim_kind is ClaimKind.STATE_ASSERTION:
            since = claim.proposal.time_start - timedelta(seconds=self.config.repeat_state_suppression_seconds)
            previous = self.store.find_recent_accepted_claim(
                semantic_fingerprint=claim.semantic_fingerprint,
                since=since,
                until=claim.proposal.time_end,
            )
            if previous is not None:
                return ClaimAdmissionDecision.create(
                    claim,
                    ClaimAdmissionStatus.REPEATED_STATE_SUPPRESSED,
                    "state_repeated_within_configured_window",
                    processing_identity=processing_identity,
                    existing_claim_id=previous.claim_id,
                )
        if self.store.claim_count() >= self.store.max_claim_capacity:
            return ClaimAdmissionDecision.create(
                claim,
                ClaimAdmissionStatus.CAPACITY_REJECTED,
                "claim_store_capacity_reached",
                processing_identity=processing_identity,
            )
        return ClaimAdmissionDecision.create(
            claim,
            ClaimAdmissionStatus.ACCEPTED,
            "claim_passed_admission",
            processing_identity=processing_identity,
        )


__all__ = ["ClaimAdmissionDecision", "ClaimAdmissionGate", "ClaimAdmissionStatus"]
