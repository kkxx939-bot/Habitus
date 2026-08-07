"""Normalizer Attempt、成功 Receipt 与处理身份。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from behavior._validation import (
    identifier,
    non_negative_int,
    optional_identifier,
    positive_int,
    sha256_digest,
    strict_utc,
    utc_text,
)
from behavior.claim.model import CLAIM_PIPELINE_VERSION, CLAIM_SCHEMA_VERSION
from behavior.claim.planner import NormalizationLane
from foundation.integrity import canonical_digest

NO_CORE_NORMALIZER_FINGERPRINT = canonical_digest({"sentinel": "NO_CORE_REQUIRED", "version": "1"})


def attempt_identity(processing_identity_value: str, attempt_number_value: int) -> str:
    identity = sha256_digest(processing_identity_value, "attempt.processing_identity")
    number = positive_int(attempt_number_value, "attempt.attempt_number")
    return "attempt_" + canonical_digest(
        {"attempt_number": number, "processing_identity": identity}
    )


class AttemptStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ABSTAINED = "ABSTAINED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_NON_RETRYABLE = "FAILED_NON_RETRYABLE"
    FAILED_POLICY = "FAILED_POLICY"


class ReceiptStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ABSTAINED = "ABSTAINED"
    NO_CORE_REQUIRED = "NO_CORE_REQUIRED"


def processing_identity(
    *,
    evidence_record_digest: str,
    lane: NormalizationLane,
    normalizer_fingerprint: str,
    planner_policy_digest: str,
    compatibility_policy_digest: str,
    binding_policy_digest: str,
    confidence_policy_digest: str,
) -> str:
    return canonical_digest(
        {
            "binding_policy_digest": sha256_digest(binding_policy_digest, "binding_policy_digest"),
            "claim_schema_version": CLAIM_SCHEMA_VERSION,
            "compatibility_policy_digest": sha256_digest(
                compatibility_policy_digest,
                "compatibility_policy_digest",
            ),
            "confidence_policy_digest": sha256_digest(
                confidence_policy_digest,
                "confidence_policy_digest",
            ),
            "evidence_record_digest": sha256_digest(
                evidence_record_digest,
                "evidence_record_digest",
            ),
            "lane": NormalizationLane(lane).value,
            "normalizer_fingerprint": sha256_digest(normalizer_fingerprint, "normalizer_fingerprint"),
            "pipeline_version": CLAIM_PIPELINE_VERSION,
            "planner_policy_digest": sha256_digest(planner_policy_digest, "planner_policy_digest"),
        }
    )


@dataclass(frozen=True)
class ClaimNormalizationAttempt:
    processing_identity: str
    evidence_record_id: str
    normalizer_name: str
    normalizer_fingerprint: str
    lane: NormalizationLane
    attempt_number: int
    status: AttemptStatus
    proposal_digest: str | None
    claim_count: int
    error_code: str | None
    retryable: bool
    started_at: datetime
    completed_at: datetime
    attempt_id: str = field(init=False)
    content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        identity = sha256_digest(self.processing_identity, "attempt.processing_identity")
        evidence_id = identifier(self.evidence_record_id, "attempt.evidence_record_id")
        normalizer_name = identifier(self.normalizer_name, "attempt.normalizer_name")
        fingerprint = sha256_digest(self.normalizer_fingerprint, "attempt.normalizer_fingerprint")
        lane = NormalizationLane(self.lane)
        number = positive_int(self.attempt_number, "attempt.attempt_number")
        status = AttemptStatus(self.status)
        proposal_digest = (
            None
            if self.proposal_digest is None
            else sha256_digest(self.proposal_digest, "attempt.proposal_digest")
        )
        claim_count = non_negative_int(self.claim_count, "attempt.claim_count")
        error_code = optional_identifier(self.error_code, "attempt.error_code")
        if not isinstance(self.retryable, bool):
            raise TypeError("attempt.retryable must be boolean")
        started_at = strict_utc(self.started_at, "attempt.started_at")
        completed_at = strict_utc(self.completed_at, "attempt.completed_at")
        if completed_at < started_at:
            raise ValueError("attempt completed_at cannot precede started_at")
        failed = status in {
            AttemptStatus.FAILED_RETRYABLE,
            AttemptStatus.FAILED_NON_RETRYABLE,
            AttemptStatus.FAILED_POLICY,
        }
        if failed and (claim_count != 0 or error_code is None):
            raise ValueError("failed Attempt requires an error and no Claims")
        if not failed and error_code is not None:
            raise ValueError("successful Attempt cannot contain an error")
        if failed and proposal_digest is not None:
            raise ValueError("failed Attempt cannot publish a Proposal digest")
        if not failed and proposal_digest is None:
            raise ValueError("successful Attempt requires a Proposal digest")
        if self.retryable is not (status is AttemptStatus.FAILED_RETRYABLE):
            raise ValueError("Attempt retryable flag disagrees with status")
        if status is AttemptStatus.ABSTAINED and claim_count != 0:
            raise ValueError("ABSTAINED Attempt cannot contain Claims")
        if status is AttemptStatus.COMPLETED and claim_count == 0:
            raise ValueError("COMPLETED Attempt requires Claims")
        attempt_id = attempt_identity(identity, number)
        body = {
            "attempt_id": attempt_id,
            "attempt_number": number,
            "claim_count": claim_count,
            "completed_at": utc_text(completed_at),
            "error_code": error_code,
            "evidence_record_id": evidence_id,
            "lane": lane.value,
            "normalizer_fingerprint": fingerprint,
            "normalizer_name": normalizer_name,
            "processing_identity": identity,
            "proposal_digest": proposal_digest,
            "retryable": self.retryable,
            "started_at": utc_text(started_at),
            "status": status.value,
        }
        object.__setattr__(self, "processing_identity", identity)
        object.__setattr__(self, "evidence_record_id", evidence_id)
        object.__setattr__(self, "normalizer_name", normalizer_name)
        object.__setattr__(self, "normalizer_fingerprint", fingerprint)
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "attempt_number", number)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "proposal_digest", proposal_digest)
        object.__setattr__(self, "claim_count", claim_count)
        object.__setattr__(self, "error_code", error_code)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "completed_at", completed_at)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "content_digest", canonical_digest(body))


@dataclass(frozen=True)
class ClaimNormalizationReceipt:
    processing_identity: str
    evidence_record_id: str
    lane: NormalizationLane
    normalizer_fingerprint: str
    planner_policy_digest: str
    compatibility_policy_digest: str
    binding_policy_digest: str
    confidence_policy_digest: str
    status: ReceiptStatus
    attempt_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    completed_at: datetime
    publication_recorded_at: datetime
    content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        digests = {
            "processing_identity": sha256_digest(self.processing_identity, "receipt.processing_identity"),
            "normalizer_fingerprint": sha256_digest(
                self.normalizer_fingerprint,
                "receipt.normalizer_fingerprint",
            ),
            "planner_policy_digest": sha256_digest(
                self.planner_policy_digest,
                "receipt.planner_policy_digest",
            ),
            "compatibility_policy_digest": sha256_digest(
                self.compatibility_policy_digest,
                "receipt.compatibility_policy_digest",
            ),
            "binding_policy_digest": sha256_digest(
                self.binding_policy_digest,
                "receipt.binding_policy_digest",
            ),
            "confidence_policy_digest": sha256_digest(
                self.confidence_policy_digest,
                "receipt.confidence_policy_digest",
            ),
        }
        evidence_id = identifier(self.evidence_record_id, "receipt.evidence_record_id")
        lane = NormalizationLane(self.lane)
        status = ReceiptStatus(self.status)
        if not isinstance(self.attempt_ids, tuple) or not isinstance(self.claim_ids, tuple):
            raise TypeError("Receipt members must be tuples")
        attempt_ids = tuple(identifier(item, "receipt.attempt_id") for item in self.attempt_ids)
        claim_ids = tuple(identifier(item, "receipt.claim_id") for item in self.claim_ids)
        if len(attempt_ids) != len(set(attempt_ids)) or len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Receipt members must be unique")
        if status is ReceiptStatus.NO_CORE_REQUIRED:
            if lane is not NormalizationLane.CORE or attempt_ids or claim_ids:
                raise ValueError("NO_CORE_REQUIRED must be an empty Core Receipt")
        elif not attempt_ids:
            raise ValueError("completed or abstained Receipt requires an Attempt")
        if status is ReceiptStatus.ABSTAINED and claim_ids:
            raise ValueError("ABSTAINED Receipt cannot contain Claims")
        if status is ReceiptStatus.COMPLETED and not claim_ids:
            raise ValueError("COMPLETED Receipt requires Claims")
        completed_at = strict_utc(self.completed_at, "receipt.completed_at")
        publication_at = strict_utc(
            self.publication_recorded_at,
            "receipt.publication_recorded_at",
        )
        if publication_at < completed_at:
            raise ValueError("publication time cannot precede completion time")
        body: dict[str, Any] = {
            **digests,
            "attempt_ids": attempt_ids,
            "claim_ids": claim_ids,
            "completed_at": utc_text(completed_at),
            "evidence_record_id": evidence_id,
            "lane": lane.value,
            "publication_recorded_at": utc_text(publication_at),
            "status": status.value,
        }
        for name, value in digests.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "evidence_record_id", evidence_id)
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "attempt_ids", attempt_ids)
        object.__setattr__(self, "claim_ids", claim_ids)
        object.__setattr__(self, "completed_at", completed_at)
        object.__setattr__(self, "publication_recorded_at", publication_at)
        object.__setattr__(self, "content_digest", canonical_digest(body))


def attempt_to_dict(value: ClaimNormalizationAttempt) -> dict[str, Any]:
    return {
        "attempt_id": value.attempt_id,
        "processing_identity": value.processing_identity,
        "evidence_record_id": value.evidence_record_id,
        "normalizer_name": value.normalizer_name,
        "normalizer_fingerprint": value.normalizer_fingerprint,
        "lane": value.lane.value,
        "attempt_number": value.attempt_number,
        "status": value.status.value,
        "proposal_digest": value.proposal_digest,
        "claim_count": value.claim_count,
        "error_code": value.error_code,
        "retryable": value.retryable,
        "started_at": utc_text(value.started_at),
        "completed_at": utc_text(value.completed_at),
        "content_digest": value.content_digest,
    }


def normalization_receipt_to_dict(value: ClaimNormalizationReceipt) -> dict[str, Any]:
    return {
        "processing_identity": value.processing_identity,
        "evidence_record_id": value.evidence_record_id,
        "lane": value.lane.value,
        "normalizer_fingerprint": value.normalizer_fingerprint,
        "planner_policy_digest": value.planner_policy_digest,
        "compatibility_policy_digest": value.compatibility_policy_digest,
        "binding_policy_digest": value.binding_policy_digest,
        "confidence_policy_digest": value.confidence_policy_digest,
        "status": value.status.value,
        "attempt_ids": value.attempt_ids,
        "claim_ids": value.claim_ids,
        "completed_at": utc_text(value.completed_at),
        "publication_recorded_at": utc_text(value.publication_recorded_at),
        "content_digest": value.content_digest,
    }


__all__ = [
    "AttemptStatus",
    "ClaimNormalizationAttempt",
    "ClaimNormalizationReceipt",
    "ReceiptStatus",
]
