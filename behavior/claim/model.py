"""系统绑定后的耐久 Claim、批次与处理收据。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from behavior._validation import (
    identifier,
    identifier_tuple,
    parse_utc,
    require_fields,
    sha256_digest,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.claim.proposal import ClaimProposal
from behavior.errors import ClaimSchemaError
from foundation.integrity import canonical_digest

CLAIM_SCHEMA_VERSION = "1"
PIPELINE_VERSION = "1"


class ClaimProducerRunStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ABSTAINED = "ABSTAINED"


@dataclass(frozen=True)
class ClaimProducerRun:
    run_id: str
    processing_identity: str
    manifest_id: str
    producer_name: str
    producer_fingerprint: str
    status: ClaimProducerRunStatus
    proposal_digest: str
    claim_count: int
    completed_at: datetime

    def __post_init__(self) -> None:
        from behavior._validation import non_negative_int

        for name in ("run_id", "processing_identity", "manifest_id", "producer_name"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        for name in ("producer_fingerprint", "proposal_digest"):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name))
        object.__setattr__(self, "status", ClaimProducerRunStatus(self.status))
        object.__setattr__(self, "claim_count", non_negative_int(self.claim_count, "claim_count"))
        object.__setattr__(self, "completed_at", strict_utc(self.completed_at, "completed_at"))

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "processing_identity": self.processing_identity,
            "manifest_id": self.manifest_id,
            "producer_name": self.producer_name,
            "producer_fingerprint": self.producer_fingerprint,
            "status": self.status.value,
            "proposal_digest": self.proposal_digest,
            "claim_count": self.claim_count,
            "completed_at": utc_text(self.completed_at),
        }

    @classmethod
    def from_dict(cls, value: object) -> ClaimProducerRun:
        fields = frozenset(
            {"run_id", "processing_identity", "manifest_id", "producer_name", "producer_fingerprint", "status", "proposal_digest", "claim_count", "completed_at"}
        )
        data = strict_fields(value, "claim_producer_run", fields)
        require_fields(data, "claim_producer_run", fields)
        return cls(
            run_id=data["run_id"],
            processing_identity=data["processing_identity"],
            manifest_id=data["manifest_id"],
            producer_name=data["producer_name"],
            producer_fingerprint=data["producer_fingerprint"],
            status=ClaimProducerRunStatus(data["status"]),
            proposal_digest=data["proposal_digest"],
            claim_count=data["claim_count"],
            completed_at=parse_utc(data["completed_at"], "completed_at"),
        )


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim_batch_id: str
    owner_binding_digest: str
    evidence_manifest_id: str
    evidence_manifest_digest: str
    producer_fingerprint: str
    proposal: ClaimProposal
    semantic_fingerprint: str
    created_at: datetime
    schema_version: str = CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("claim_id", "claim_batch_id", "evidence_manifest_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        for name in (
            "owner_binding_digest",
            "evidence_manifest_digest",
            "producer_fingerprint",
            "semantic_fingerprint",
        ):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name))
        if not isinstance(self.proposal, ClaimProposal):
            raise TypeError("proposal must be ClaimProposal")
        object.__setattr__(self, "created_at", strict_utc(self.created_at, "created_at"))
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))
        expected = self.identity_for(
            manifest_digest=self.evidence_manifest_digest,
            producer_fingerprint=self.producer_fingerprint,
            proposal=self.proposal,
            schema_version=self.schema_version,
        )
        if self.claim_id != expected:
            raise ClaimSchemaError("claim_id does not match deterministic identity")

    @staticmethod
    def identity_for(
        *,
        manifest_digest: str,
        producer_fingerprint: str,
        proposal: ClaimProposal,
        schema_version: str = CLAIM_SCHEMA_VERSION,
    ) -> str:
        return "claim_" + canonical_digest(
            {
                "manifest_digest": manifest_digest,
                "producer_fingerprint": producer_fingerprint,
                "proposal": proposal.to_dict(),
                "schema_version": schema_version,
            }
        )

    @staticmethod
    def semantic_identity(proposal: ClaimProposal) -> str:
        return canonical_digest(
            {
                "activity": proposal.activity,
                "actor_role": proposal.actor_role.value,
                "claim_kind": proposal.claim_kind.value,
                "location_ref": proposal.location_ref,
                "object_refs": proposal.object_refs,
                "phase": proposal.phase,
                "predicate": proposal.predicate,
                "semantic_family": proposal.semantic_family,
                "semantic_payload": proposal.semantic_payload,
                "subject_role": proposal.subject_role.value,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "claim_batch_id": self.claim_batch_id,
            "owner_binding_digest": self.owner_binding_digest,
            "evidence_manifest_id": self.evidence_manifest_id,
            "evidence_manifest_digest": self.evidence_manifest_digest,
            "producer_fingerprint": self.producer_fingerprint,
            "proposal": self.proposal.to_dict(),
            "semantic_fingerprint": self.semantic_fingerprint,
            "created_at": utc_text(self.created_at),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> Claim:
        fields = frozenset(
            {
                "claim_id",
                "claim_batch_id",
                "owner_binding_digest",
                "evidence_manifest_id",
                "evidence_manifest_digest",
                "producer_fingerprint",
                "proposal",
                "semantic_fingerprint",
                "created_at",
                "schema_version",
            }
        )
        data = strict_fields(value, "claim", fields)
        require_fields(data, "claim", fields)
        return cls(
            claim_id=data["claim_id"],
            claim_batch_id=data["claim_batch_id"],
            owner_binding_digest=data["owner_binding_digest"],
            evidence_manifest_id=data["evidence_manifest_id"],
            evidence_manifest_digest=data["evidence_manifest_digest"],
            producer_fingerprint=data["producer_fingerprint"],
            proposal=ClaimProposal.model_validate(data["proposal"]),
            semantic_fingerprint=data["semantic_fingerprint"],
            created_at=parse_utc(data["created_at"], "created_at"),
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class ClaimBatch:
    claim_batch_id: str
    manifest_id: str
    manifest_digest: str
    producer_name: str
    producer_fingerprint: str
    abstained: bool
    claim_ids: tuple[str, ...]
    proposal_digest: str
    created_at: datetime
    schema_version: str = CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("claim_batch_id", "manifest_id", "producer_name"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        for name in ("manifest_digest", "producer_fingerprint", "proposal_digest"):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name))
        if not isinstance(self.abstained, bool):
            raise TypeError("abstained must be boolean")
        claim_ids = identifier_tuple(self.claim_ids, "claim_ids", maximum_items=1_000)
        if self.abstained == bool(claim_ids):
            raise ClaimSchemaError("abstained ClaimBatch cannot contain claims")
        object.__setattr__(self, "claim_ids", claim_ids)
        object.__setattr__(self, "created_at", strict_utc(self.created_at, "created_at"))
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_batch_id": self.claim_batch_id,
            "manifest_id": self.manifest_id,
            "manifest_digest": self.manifest_digest,
            "producer_name": self.producer_name,
            "producer_fingerprint": self.producer_fingerprint,
            "abstained": self.abstained,
            "claim_ids": self.claim_ids,
            "proposal_digest": self.proposal_digest,
            "created_at": utc_text(self.created_at),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> ClaimBatch:
        fields = frozenset(
            {"claim_batch_id", "manifest_id", "manifest_digest", "producer_name", "producer_fingerprint", "abstained", "claim_ids", "proposal_digest", "created_at", "schema_version"}
        )
        data = strict_fields(value, "claim_batch", fields)
        require_fields(data, "claim_batch", fields)
        return cls(
            claim_batch_id=data["claim_batch_id"],
            manifest_id=data["manifest_id"],
            manifest_digest=data["manifest_digest"],
            producer_name=data["producer_name"],
            producer_fingerprint=data["producer_fingerprint"],
            abstained=data["abstained"],
            claim_ids=tuple(data["claim_ids"]),
            proposal_digest=data["proposal_digest"],
            created_at=parse_utc(data["created_at"], "created_at"),
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class ClaimProcessingReceipt:
    processing_identity: str
    manifest_id: str
    manifest_digest: str
    producer_fingerprints: tuple[str, ...]
    claim_batch_ids: tuple[str, ...]
    accepted_claim_ids: tuple[str, ...]
    decision_ids: tuple[str, ...]
    completed_at: datetime
    receipt_digest: str
    schema_version: str = CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("processing_identity", "manifest_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        object.__setattr__(self, "manifest_digest", sha256_digest(self.manifest_digest, "manifest_digest"))
        fingerprints = tuple(sha256_digest(value, "producer_fingerprint") for value in self.producer_fingerprints)
        if not fingerprints or len(set(fingerprints)) != len(fingerprints):
            raise ClaimSchemaError("producer_fingerprints must be non-empty and unique")
        object.__setattr__(self, "producer_fingerprints", fingerprints)
        for name in ("claim_batch_ids", "accepted_claim_ids", "decision_ids"):
            object.__setattr__(self, name, identifier_tuple(getattr(self, name), name, maximum_items=10_000))
        if len(self.claim_batch_ids) != len(fingerprints):
            raise ClaimSchemaError("one ClaimBatch identity is required for each producer fingerprint")
        object.__setattr__(self, "completed_at", strict_utc(self.completed_at, "completed_at"))
        object.__setattr__(self, "receipt_digest", sha256_digest(self.receipt_digest, "receipt_digest"))
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))
        expected_processing_identity = "processing_" + canonical_digest(
            {
                "claim_schema_version": self.schema_version,
                "manifest_digest": self.manifest_digest,
                "pipeline_version": PIPELINE_VERSION,
                "producer_fingerprints": fingerprints,
            }
        )
        if self.processing_identity != expected_processing_identity:
            raise ClaimSchemaError("processing_identity does not match the deterministic request identity")
        if self.receipt_digest != canonical_digest(self._digest_payload()):
            raise ClaimSchemaError("processing receipt digest mismatch")

    def _digest_payload(self) -> dict[str, object]:
        return {
            "processing_identity": self.processing_identity,
            "manifest_id": self.manifest_id,
            "manifest_digest": self.manifest_digest,
            "producer_fingerprints": self.producer_fingerprints,
            "claim_batch_ids": self.claim_batch_ids,
            "accepted_claim_ids": self.accepted_claim_ids,
            "decision_ids": self.decision_ids,
            "completed_at": utc_text(self.completed_at),
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._digest_payload(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: object) -> ClaimProcessingReceipt:
        fields = frozenset(
            {"processing_identity", "manifest_id", "manifest_digest", "producer_fingerprints", "claim_batch_ids", "accepted_claim_ids", "decision_ids", "completed_at", "receipt_digest", "schema_version"}
        )
        data = strict_fields(value, "claim_processing_receipt", fields)
        require_fields(data, "claim_processing_receipt", fields)
        return cls(
            processing_identity=data["processing_identity"],
            manifest_id=data["manifest_id"],
            manifest_digest=data["manifest_digest"],
            producer_fingerprints=tuple(data["producer_fingerprints"]),
            claim_batch_ids=tuple(data["claim_batch_ids"]),
            accepted_claim_ids=tuple(data["accepted_claim_ids"]),
            decision_ids=tuple(data["decision_ids"]),
            completed_at=parse_utc(data["completed_at"], "completed_at"),
            receipt_digest=data["receipt_digest"],
            schema_version=data["schema_version"],
        )


__all__ = [
    "CLAIM_SCHEMA_VERSION",
    "PIPELINE_VERSION",
    "Claim",
    "ClaimBatch",
    "ClaimProcessingReceipt",
    "ClaimProducerRun",
    "ClaimProducerRunStatus",
]
