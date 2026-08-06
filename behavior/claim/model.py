"""系统绑定后的耐久 Claim、Normalizer 运行、批次与处理收据。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from behavior._validation import (
    finite_score,
    identifier,
    identifier_tuple,
    non_negative_int,
    parse_utc,
    require_fields,
    sha256_digest,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.claim.proposal import ClaimSemanticProposal
from behavior.errors import ClaimSchemaError
from behavior.ingress.model import SemanticActorRole, SemanticSubjectRole
from foundation.integrity import canonical_digest

CLAIM_SCHEMA_VERSION = "2"
PIPELINE_VERSION = "2"


class EpistemicClass(str, Enum):
    DIRECT_SOURCE = "DIRECT_SOURCE"
    USER_EXPLICIT = "USER_EXPLICIT"
    SENSOR_INFERRED = "SENSOR_INFERRED"
    MODEL_INFERRED = "MODEL_INFERRED"
    MULTIMODAL_MODEL_INFERRED = "MULTIMODAL_MODEL_INFERRED"


class ClaimNormalizerRunStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ABSTAINED = "ABSTAINED"


@dataclass(frozen=True)
class ClaimNormalizerRun:
    run_id: str
    processing_identity: str
    manifest_id: str
    semantic_record_id: str
    normalizer_name: str
    normalizer_fingerprint: str
    status: ClaimNormalizerRunStatus
    proposal_digest: str
    claim_count: int
    normalization_started_at: datetime
    normalization_completed_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "processing_identity",
            "manifest_id",
            "semantic_record_id",
            "normalizer_name",
        ):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        for name in ("normalizer_fingerprint", "proposal_digest"):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name))
        object.__setattr__(self, "status", ClaimNormalizerRunStatus(self.status))
        object.__setattr__(self, "claim_count", non_negative_int(self.claim_count, "claim_count"))
        object.__setattr__(
            self,
            "normalization_started_at",
            strict_utc(self.normalization_started_at, "normalization_started_at"),
        )
        object.__setattr__(
            self,
            "normalization_completed_at",
            strict_utc(self.normalization_completed_at, "normalization_completed_at"),
        )
        if self.normalization_completed_at < self.normalization_started_at:
            raise ClaimSchemaError("Normalizer completion cannot precede its start")
        expected = "run_" + canonical_digest(
            {
                "manifest_id": self.manifest_id,
                "normalizer_fingerprint": self.normalizer_fingerprint,
                "processing_identity": self.processing_identity,
                "semantic_record_id": self.semantic_record_id,
            }
        )
        if self.run_id != expected:
            raise ClaimSchemaError("Normalizer run identity mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "processing_identity": self.processing_identity,
            "manifest_id": self.manifest_id,
            "semantic_record_id": self.semantic_record_id,
            "normalizer_name": self.normalizer_name,
            "normalizer_fingerprint": self.normalizer_fingerprint,
            "status": self.status.value,
            "proposal_digest": self.proposal_digest,
            "claim_count": self.claim_count,
            "normalization_started_at": utc_text(self.normalization_started_at),
            "normalization_completed_at": utc_text(self.normalization_completed_at),
        }

    @classmethod
    def from_dict(cls, value: object) -> ClaimNormalizerRun:
        fields = frozenset(
            {
                "run_id",
                "processing_identity",
                "manifest_id",
                "semantic_record_id",
                "normalizer_name",
                "normalizer_fingerprint",
                "status",
                "proposal_digest",
                "claim_count",
                "normalization_started_at",
                "normalization_completed_at",
            }
        )
        data = strict_fields(value, "claim_normalizer_run", fields)
        require_fields(data, "claim_normalizer_run", fields)
        return cls(
            run_id=data["run_id"],
            processing_identity=data["processing_identity"],
            manifest_id=data["manifest_id"],
            semantic_record_id=data["semantic_record_id"],
            normalizer_name=data["normalizer_name"],
            normalizer_fingerprint=data["normalizer_fingerprint"],
            status=ClaimNormalizerRunStatus(data["status"]),
            proposal_digest=data["proposal_digest"],
            claim_count=data["claim_count"],
            normalization_started_at=parse_utc(data["normalization_started_at"], "normalization_started_at"),
            normalization_completed_at=parse_utc(data["normalization_completed_at"], "normalization_completed_at"),
        )


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim_batch_id: str
    owner_identity_digest: str
    semantic_record_id: str
    semantic_record_digest: str
    manifest_id: str
    manifest_digest: str
    subject_role: SemanticSubjectRole
    actor_role: SemanticActorRole
    time_start: datetime
    time_end: datetime
    time_uncertainty_ms: int
    epistemic_class: EpistemicClass
    source_confidence: float
    normalizer_confidence: float
    effective_confidence: float
    normalizer_fingerprint: str
    proposal: ClaimSemanticProposal
    semantic_fingerprint: str
    created_at: datetime
    schema_version: str = CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("claim_id", "claim_batch_id", "semantic_record_id", "manifest_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        for name in (
            "owner_identity_digest",
            "semantic_record_digest",
            "manifest_digest",
            "normalizer_fingerprint",
            "semantic_fingerprint",
        ):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name))
        object.__setattr__(self, "subject_role", SemanticSubjectRole(self.subject_role))
        object.__setattr__(self, "actor_role", SemanticActorRole(self.actor_role))
        object.__setattr__(self, "time_start", strict_utc(self.time_start, "time_start"))
        object.__setattr__(self, "time_end", strict_utc(self.time_end, "time_end"))
        if self.time_end < self.time_start:
            raise ClaimSchemaError("Claim time end cannot precede start")
        object.__setattr__(
            self,
            "time_uncertainty_ms",
            non_negative_int(self.time_uncertainty_ms, "time_uncertainty_ms"),
        )
        object.__setattr__(self, "epistemic_class", EpistemicClass(self.epistemic_class))
        for name in ("source_confidence", "normalizer_confidence", "effective_confidence"):
            object.__setattr__(self, name, finite_score(getattr(self, name), name))
        if not isinstance(self.proposal, ClaimSemanticProposal):
            raise TypeError("proposal must be ClaimSemanticProposal")
        object.__setattr__(self, "created_at", strict_utc(self.created_at, "created_at"))
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))
        expected = self.identity_for(
            owner_identity_digest=self.owner_identity_digest,
            semantic_record_digest=self.semantic_record_digest,
            manifest_digest=self.manifest_digest,
            normalizer_fingerprint=self.normalizer_fingerprint,
            proposal=self.proposal,
            schema_version=self.schema_version,
        )
        if self.claim_id != expected:
            raise ClaimSchemaError("claim_id does not match deterministic content")
        expected_semantic = self.semantic_identity(
            owner_identity_digest=self.owner_identity_digest,
            subject_role=self.subject_role,
            actor_role=self.actor_role,
            proposal=self.proposal,
        )
        if self.semantic_fingerprint != expected_semantic:
            raise ClaimSchemaError("Claim semantic fingerprint mismatch")

    @staticmethod
    def identity_for(
        *,
        owner_identity_digest: str,
        semantic_record_digest: str,
        manifest_digest: str,
        normalizer_fingerprint: str,
        proposal: ClaimSemanticProposal,
        schema_version: str = CLAIM_SCHEMA_VERSION,
    ) -> str:
        semantic = proposal.to_dict()
        semantic.pop("human_summary")
        return "claim_" + canonical_digest(
            {
                "owner_identity_digest": owner_identity_digest,
                "semantic_record_digest": semantic_record_digest,
                "manifest_digest": manifest_digest,
                "normalizer_fingerprint": normalizer_fingerprint,
                "claim_semantic": semantic,
                "schema_version": schema_version,
            }
        )

    @staticmethod
    def semantic_identity(
        *,
        owner_identity_digest: str,
        subject_role: SemanticSubjectRole,
        actor_role: SemanticActorRole,
        proposal: ClaimSemanticProposal,
    ) -> str:
        return canonical_digest(
            {
                "owner_identity_digest": owner_identity_digest,
                "subject_role": subject_role.value,
                "actor_role": actor_role.value,
                "claim_kind": proposal.claim_kind.value,
                "predicate": proposal.predicate,
                "semantic_family": proposal.semantic_family,
                "activity": proposal.activity,
                "phase": proposal.phase,
                "object_refs": proposal.object_refs,
                "location_ref": proposal.location_ref,
                "semantic_payload": proposal.semantic_payload,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "claim_batch_id": self.claim_batch_id,
            "owner_identity_digest": self.owner_identity_digest,
            "semantic_record_id": self.semantic_record_id,
            "semantic_record_digest": self.semantic_record_digest,
            "manifest_id": self.manifest_id,
            "manifest_digest": self.manifest_digest,
            "subject_role": self.subject_role.value,
            "actor_role": self.actor_role.value,
            "time_start": utc_text(self.time_start),
            "time_end": utc_text(self.time_end),
            "time_uncertainty_ms": self.time_uncertainty_ms,
            "epistemic_class": self.epistemic_class.value,
            "source_confidence": self.source_confidence,
            "normalizer_confidence": self.normalizer_confidence,
            "effective_confidence": self.effective_confidence,
            "normalizer_fingerprint": self.normalizer_fingerprint,
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
                "owner_identity_digest",
                "semantic_record_id",
                "semantic_record_digest",
                "manifest_id",
                "manifest_digest",
                "subject_role",
                "actor_role",
                "time_start",
                "time_end",
                "time_uncertainty_ms",
                "epistemic_class",
                "source_confidence",
                "normalizer_confidence",
                "effective_confidence",
                "normalizer_fingerprint",
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
            owner_identity_digest=data["owner_identity_digest"],
            semantic_record_id=data["semantic_record_id"],
            semantic_record_digest=data["semantic_record_digest"],
            manifest_id=data["manifest_id"],
            manifest_digest=data["manifest_digest"],
            subject_role=SemanticSubjectRole(data["subject_role"]),
            actor_role=SemanticActorRole(data["actor_role"]),
            time_start=parse_utc(data["time_start"], "time_start"),
            time_end=parse_utc(data["time_end"], "time_end"),
            time_uncertainty_ms=data["time_uncertainty_ms"],
            epistemic_class=EpistemicClass(data["epistemic_class"]),
            source_confidence=data["source_confidence"],
            normalizer_confidence=data["normalizer_confidence"],
            effective_confidence=data["effective_confidence"],
            normalizer_fingerprint=data["normalizer_fingerprint"],
            proposal=ClaimSemanticProposal.model_validate(data["proposal"]),
            semantic_fingerprint=data["semantic_fingerprint"],
            created_at=parse_utc(data["created_at"], "created_at"),
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class ClaimBatch:
    claim_batch_id: str
    processing_identity: str
    manifest_id: str
    manifest_digest: str
    semantic_record_id: str
    normalizer_name: str
    normalizer_fingerprint: str
    abstained: bool
    claim_ids: tuple[str, ...]
    proposal_digest: str
    created_at: datetime
    schema_version: str = CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "claim_batch_id",
            "processing_identity",
            "manifest_id",
            "semantic_record_id",
            "normalizer_name",
        ):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        for name in ("manifest_digest", "normalizer_fingerprint", "proposal_digest"):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name))
        if not isinstance(self.abstained, bool):
            raise TypeError("abstained must be boolean")
        ids = identifier_tuple(self.claim_ids, "claim_ids", maximum_items=10_000)
        if self.abstained == bool(ids):
            raise ClaimSchemaError("abstained ClaimBatch cannot contain claims")
        object.__setattr__(self, "claim_ids", ids)
        object.__setattr__(self, "created_at", strict_utc(self.created_at, "created_at"))
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))
        expected = self.identity_for(
            manifest_digest=self.manifest_digest,
            semantic_record_id=self.semantic_record_id,
            normalizer_fingerprint=self.normalizer_fingerprint,
        )
        if self.claim_batch_id != expected:
            raise ClaimSchemaError("ClaimBatch identity mismatch")

    @staticmethod
    def identity_for(
        *,
        manifest_digest: str,
        semantic_record_id: str,
        normalizer_fingerprint: str,
    ) -> str:
        return "batch_" + canonical_digest(
            {
                "manifest_digest": manifest_digest,
                "semantic_record_id": semantic_record_id,
                "normalizer_fingerprint": normalizer_fingerprint,
                "schema_version": CLAIM_SCHEMA_VERSION,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_batch_id": self.claim_batch_id,
            "processing_identity": self.processing_identity,
            "manifest_id": self.manifest_id,
            "manifest_digest": self.manifest_digest,
            "semantic_record_id": self.semantic_record_id,
            "normalizer_name": self.normalizer_name,
            "normalizer_fingerprint": self.normalizer_fingerprint,
            "abstained": self.abstained,
            "claim_ids": self.claim_ids,
            "proposal_digest": self.proposal_digest,
            "created_at": utc_text(self.created_at),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> ClaimBatch:
        fields = frozenset(
            {
                "claim_batch_id",
                "processing_identity",
                "manifest_id",
                "manifest_digest",
                "semantic_record_id",
                "normalizer_name",
                "normalizer_fingerprint",
                "abstained",
                "claim_ids",
                "proposal_digest",
                "created_at",
                "schema_version",
            }
        )
        data = strict_fields(value, "claim_batch", fields)
        require_fields(data, "claim_batch", fields)
        return cls(
            claim_batch_id=data["claim_batch_id"],
            processing_identity=data["processing_identity"],
            manifest_id=data["manifest_id"],
            manifest_digest=data["manifest_digest"],
            semantic_record_id=data["semantic_record_id"],
            normalizer_name=data["normalizer_name"],
            normalizer_fingerprint=data["normalizer_fingerprint"],
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
    normalizer_fingerprints: tuple[str, ...]
    normalizer_run_ids: tuple[str, ...]
    claim_batch_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    accepted_claim_ids: tuple[str, ...]
    decision_ids: tuple[str, ...]
    completed_at: datetime
    published_at: datetime
    receipt_digest: str
    schema_version: str = CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("processing_identity", "manifest_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        object.__setattr__(self, "manifest_digest", sha256_digest(self.manifest_digest, "manifest_digest"))
        fingerprints = tuple(sha256_digest(item, "normalizer_fingerprint") for item in self.normalizer_fingerprints)
        if not fingerprints:
            raise ClaimSchemaError("normalizer_fingerprints cannot be empty")
        object.__setattr__(self, "normalizer_fingerprints", fingerprints)
        for name in (
            "normalizer_run_ids",
            "claim_batch_ids",
            "claim_ids",
            "accepted_claim_ids",
            "decision_ids",
        ):
            object.__setattr__(
                self,
                name,
                identifier_tuple(getattr(self, name), name, maximum_items=100_000),
            )
        if len(self.normalizer_run_ids) != len(fingerprints) or len(self.claim_batch_ids) != len(fingerprints):
            raise ClaimSchemaError("each routed Normalizer requires one run and one ClaimBatch")
        if not set(self.accepted_claim_ids).issubset(self.claim_ids):
            raise ClaimSchemaError("accepted Claim identities must belong to the complete Claim set")
        if len(self.decision_ids) != len(self.claim_ids):
            raise ClaimSchemaError("each validated Claim requires one AdmissionDecision")
        object.__setattr__(self, "completed_at", strict_utc(self.completed_at, "completed_at"))
        object.__setattr__(self, "published_at", strict_utc(self.published_at, "published_at"))
        if self.completed_at < self.published_at:
            raise ClaimSchemaError("processing completion cannot precede publication")
        object.__setattr__(self, "receipt_digest", sha256_digest(self.receipt_digest, "receipt_digest"))
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))
        expected_processing = self.processing_identity_for(
            manifest_digest=self.manifest_digest,
            normalizer_fingerprints=fingerprints,
            schema_version=self.schema_version,
        )
        if self.processing_identity != expected_processing:
            raise ClaimSchemaError("processing_identity does not match its deterministic request")
        if self.receipt_digest != canonical_digest(self._digest_payload()):
            raise ClaimSchemaError("processing receipt digest mismatch")

    @staticmethod
    def processing_identity_for(
        *,
        manifest_digest: str,
        normalizer_fingerprints: tuple[str, ...],
        schema_version: str = CLAIM_SCHEMA_VERSION,
    ) -> str:
        return "processing_" + canonical_digest(
            {
                "manifest_digest": manifest_digest,
                "ordered_normalizer_fingerprints": normalizer_fingerprints,
                "claim_schema_version": schema_version,
                "pipeline_version": PIPELINE_VERSION,
            }
        )

    def _digest_payload(self) -> dict[str, object]:
        return {
            "processing_identity": self.processing_identity,
            "manifest_id": self.manifest_id,
            "manifest_digest": self.manifest_digest,
            "normalizer_fingerprints": self.normalizer_fingerprints,
            "normalizer_run_ids": self.normalizer_run_ids,
            "claim_batch_ids": self.claim_batch_ids,
            "claim_ids": self.claim_ids,
            "accepted_claim_ids": self.accepted_claim_ids,
            "decision_ids": self.decision_ids,
            "completed_at": utc_text(self.completed_at),
            "published_at": utc_text(self.published_at),
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._digest_payload(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: object) -> ClaimProcessingReceipt:
        fields = frozenset(
            {
                "processing_identity",
                "manifest_id",
                "manifest_digest",
                "normalizer_fingerprints",
                "normalizer_run_ids",
                "claim_batch_ids",
                "claim_ids",
                "accepted_claim_ids",
                "decision_ids",
                "completed_at",
                "published_at",
                "receipt_digest",
                "schema_version",
            }
        )
        data = strict_fields(value, "claim_processing_receipt", fields)
        require_fields(data, "claim_processing_receipt", fields)
        return cls(
            processing_identity=data["processing_identity"],
            manifest_id=data["manifest_id"],
            manifest_digest=data["manifest_digest"],
            normalizer_fingerprints=tuple(data["normalizer_fingerprints"]),
            normalizer_run_ids=tuple(data["normalizer_run_ids"]),
            claim_batch_ids=tuple(data["claim_batch_ids"]),
            claim_ids=tuple(data["claim_ids"]),
            accepted_claim_ids=tuple(data["accepted_claim_ids"]),
            decision_ids=tuple(data["decision_ids"]),
            completed_at=parse_utc(data["completed_at"], "completed_at"),
            published_at=parse_utc(data["published_at"], "published_at"),
            receipt_digest=data["receipt_digest"],
            schema_version=data["schema_version"],
        )


__all__ = [
    "CLAIM_SCHEMA_VERSION",
    "PIPELINE_VERSION",
    "Claim",
    "ClaimBatch",
    "ClaimNormalizerRun",
    "ClaimNormalizerRunStatus",
    "ClaimProcessingReceipt",
    "EpistemicClass",
]
