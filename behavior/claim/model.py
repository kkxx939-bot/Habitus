"""System-bound Claim values, route attempts, batches, and lane receipts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, cast

from behavior._validation import (
    finite_score,
    identifier,
    identifier_tuple,
    non_negative_int,
    optional_identifier,
    parse_utc,
    require_fields,
    sha256_digest,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.claim.policy import ClaimDerivationClass, ClaimProcessingLane
from behavior.claim.proposal import ClaimKind, ClaimSemanticProposal, ClaimSemanticProposalContract
from behavior.config import ClaimConfig
from behavior.errors import ClaimSchemaError
from behavior.ingress.model import SemanticActorRole, SemanticSubjectRole
from foundation.integrity import canonical_digest

CLAIM_SCHEMA_VERSION = "3"
PIPELINE_VERSION = "3"


class EpistemicClass(str, Enum):
    DIRECT_SOURCE = "DIRECT_SOURCE"
    USER_EXPLICIT = "USER_EXPLICIT"
    SENSOR_INFERRED = "SENSOR_INFERRED"
    MODEL_INFERRED = "MODEL_INFERRED"
    MULTIMODAL_MODEL_INFERRED = "MULTIMODAL_MODEL_INFERRED"


class ClaimNormalizerAttemptStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ABSTAINED = "ABSTAINED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_NON_RETRYABLE = "FAILED_NON_RETRYABLE"
    FAILED_POLICY = "FAILED_POLICY"


@dataclass(frozen=True)
class ClaimNormalizerAttempt:
    attempt_id: str
    processing_identity: str
    processing_lane: ClaimProcessingLane
    manifest_id: str
    semantic_record_id: str
    normalizer_name: str
    normalizer_fingerprint: str
    attempt_number: int
    status: ClaimNormalizerAttemptStatus
    proposal_digest: str | None
    claim_count: int
    error_code: str | None
    retryable: bool
    normalization_started_at: datetime
    normalization_completed_at: datetime
    content_digest: str

    def __post_init__(self) -> None:
        for name in (
            "attempt_id",
            "processing_identity",
            "manifest_id",
            "semantic_record_id",
            "normalizer_name",
        ):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        object.__setattr__(self, "processing_lane", ClaimProcessingLane(self.processing_lane))
        object.__setattr__(
            self,
            "normalizer_fingerprint",
            sha256_digest(self.normalizer_fingerprint, "normalizer_fingerprint"),
        )
        object.__setattr__(self, "attempt_number", non_negative_int(self.attempt_number, "attempt_number"))
        if self.attempt_number < 1:
            raise ClaimSchemaError("attempt_number must be positive")
        object.__setattr__(self, "status", ClaimNormalizerAttemptStatus(self.status))
        if self.proposal_digest is not None:
            object.__setattr__(self, "proposal_digest", sha256_digest(self.proposal_digest, "proposal_digest"))
        object.__setattr__(self, "claim_count", non_negative_int(self.claim_count, "claim_count"))
        object.__setattr__(self, "error_code", optional_identifier(self.error_code, "error_code"))
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be boolean")
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
        successful = self.status in {
            ClaimNormalizerAttemptStatus.COMPLETED,
            ClaimNormalizerAttemptStatus.ABSTAINED,
        }
        if successful != (self.proposal_digest is not None):
            raise ClaimSchemaError("only successful attempts have a proposal digest")
        if self.status is ClaimNormalizerAttemptStatus.ABSTAINED and self.claim_count != 0:
            raise ClaimSchemaError("abstained attempts cannot contain Claims")
        if self.status is ClaimNormalizerAttemptStatus.COMPLETED and self.claim_count == 0:
            raise ClaimSchemaError("completed attempts must contain at least one Claim")
        if not successful and (self.claim_count != 0 or self.error_code is None):
            raise ClaimSchemaError("failed attempts require an error code and cannot contain Claims")
        if successful and self.error_code is not None:
            raise ClaimSchemaError("successful attempts cannot contain an error code")
        if self.retryable != (self.status is ClaimNormalizerAttemptStatus.FAILED_RETRYABLE):
            raise ClaimSchemaError("retryable must match FAILED_RETRYABLE")
        expected_id = self.identity_for(
            processing_identity=self.processing_identity,
            semantic_record_id=self.semantic_record_id,
            normalizer_fingerprint=self.normalizer_fingerprint,
            attempt_number=self.attempt_number,
        )
        if self.attempt_id != expected_id:
            raise ClaimSchemaError("Normalizer attempt identity mismatch")
        if self.content_digest != canonical_digest(self._content_payload()):
            raise ClaimSchemaError("Normalizer attempt content digest mismatch")

    @staticmethod
    def identity_for(
        *,
        processing_identity: str,
        semantic_record_id: str,
        normalizer_fingerprint: str,
        attempt_number: int,
    ) -> str:
        return "attempt_" + canonical_digest(
            {
                "processing_identity": processing_identity,
                "semantic_record_id": semantic_record_id,
                "normalizer_fingerprint": normalizer_fingerprint,
                "attempt_number": attempt_number,
            }
        )

    @classmethod
    def create(cls, **values: object) -> ClaimNormalizerAttempt:
        payload = dict(values)
        payload["attempt_id"] = cls.identity_for(
            processing_identity=str(payload["processing_identity"]),
            semantic_record_id=str(payload["semantic_record_id"]),
            normalizer_fingerprint=str(payload["normalizer_fingerprint"]),
            attempt_number=int(cast(Any, payload["attempt_number"])),
        )
        content = {
            "attempt_id": payload["attempt_id"],
            "processing_identity": payload["processing_identity"],
            "processing_lane": ClaimProcessingLane(cast(Any, payload["processing_lane"])).value,
            "manifest_id": payload["manifest_id"],
            "semantic_record_id": payload["semantic_record_id"],
            "normalizer_name": payload["normalizer_name"],
            "normalizer_fingerprint": payload["normalizer_fingerprint"],
            "attempt_number": payload["attempt_number"],
            "status": ClaimNormalizerAttemptStatus(cast(Any, payload["status"])).value,
            "proposal_digest": payload["proposal_digest"],
            "claim_count": payload["claim_count"],
            "error_code": payload["error_code"],
            "retryable": payload["retryable"],
            "normalization_started_at": utc_text(strict_utc(payload["normalization_started_at"], "normalization_started_at")),
            "normalization_completed_at": utc_text(strict_utc(payload["normalization_completed_at"], "normalization_completed_at")),
        }
        return cast(Any, cls)(**payload, content_digest=canonical_digest(content))

    def _content_payload(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "processing_identity": self.processing_identity,
            "processing_lane": self.processing_lane.value,
            "manifest_id": self.manifest_id,
            "semantic_record_id": self.semantic_record_id,
            "normalizer_name": self.normalizer_name,
            "normalizer_fingerprint": self.normalizer_fingerprint,
            "attempt_number": self.attempt_number,
            "status": self.status.value,
            "proposal_digest": self.proposal_digest,
            "claim_count": self.claim_count,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "normalization_started_at": utc_text(self.normalization_started_at),
            "normalization_completed_at": utc_text(self.normalization_completed_at),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._content_payload(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, value: object) -> ClaimNormalizerAttempt:
        fields = frozenset(
            {
                "attempt_id",
                "processing_identity",
                "processing_lane",
                "manifest_id",
                "semantic_record_id",
                "normalizer_name",
                "normalizer_fingerprint",
                "attempt_number",
                "status",
                "proposal_digest",
                "claim_count",
                "error_code",
                "retryable",
                "normalization_started_at",
                "normalization_completed_at",
                "content_digest",
            }
        )
        data = strict_fields(value, "claim_normalizer_attempt", fields)
        require_fields(data, "claim_normalizer_attempt", fields)
        return cls(
            attempt_id=data["attempt_id"],
            processing_identity=data["processing_identity"],
            processing_lane=ClaimProcessingLane(data["processing_lane"]),
            manifest_id=data["manifest_id"],
            semantic_record_id=data["semantic_record_id"],
            normalizer_name=data["normalizer_name"],
            normalizer_fingerprint=data["normalizer_fingerprint"],
            attempt_number=data["attempt_number"],
            status=ClaimNormalizerAttemptStatus(data["status"]),
            proposal_digest=data["proposal_digest"],
            claim_count=data["claim_count"],
            error_code=data["error_code"],
            retryable=data["retryable"],
            normalization_started_at=parse_utc(data["normalization_started_at"], "normalization_started_at"),
            normalization_completed_at=parse_utc(data["normalization_completed_at"], "normalization_completed_at"),
            content_digest=data["content_digest"],
        )


@dataclass(frozen=True)
class Claim:
    claim_id: str
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
    source_epistemic_class: EpistemicClass
    derivation_class: ClaimDerivationClass
    source_confidence: float
    normalizer_confidence: float
    effective_confidence: float
    confidence_policy_digest: str
    binding_policy_digest: str
    normalizer_fingerprint: str
    proposal: ClaimSemanticProposal
    local_alternative_group_id: str | None
    alternative_group_key: str | None
    semantic_fingerprint: str
    created_at: datetime
    content_digest: str
    schema_version: str = CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("claim_id", "semantic_record_id", "manifest_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        for name in (
            "owner_identity_digest",
            "semantic_record_digest",
            "manifest_digest",
            "confidence_policy_digest",
            "binding_policy_digest",
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
        object.__setattr__(self, "time_uncertainty_ms", non_negative_int(self.time_uncertainty_ms, "time_uncertainty_ms"))
        object.__setattr__(self, "source_epistemic_class", EpistemicClass(self.source_epistemic_class))
        object.__setattr__(self, "derivation_class", ClaimDerivationClass(self.derivation_class))
        for name in ("source_confidence", "normalizer_confidence", "effective_confidence"):
            object.__setattr__(self, name, finite_score(getattr(self, name), name))
        if not isinstance(self.proposal, ClaimSemanticProposal):
            raise TypeError("proposal must be ClaimSemanticProposal")
        object.__setattr__(
            self,
            "local_alternative_group_id",
            optional_identifier(self.local_alternative_group_id, "local_alternative_group_id"),
        )
        if self.alternative_group_key is not None:
            object.__setattr__(
                self,
                "alternative_group_key",
                sha256_digest(self.alternative_group_key, "alternative_group_key"),
            )
        if (self.local_alternative_group_id is None) != (self.alternative_group_key is None):
            raise ClaimSchemaError("local and system alternative group identities must appear together")
        object.__setattr__(self, "created_at", strict_utc(self.created_at, "created_at"))
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))
        if self.claim_id != self.identity_for(
            owner_identity_digest=self.owner_identity_digest,
            semantic_record_digest=self.semantic_record_digest,
            manifest_digest=self.manifest_digest,
            normalizer_fingerprint=self.normalizer_fingerprint,
            derivation_class=self.derivation_class,
            binding_policy_digest=self.binding_policy_digest,
            confidence_policy_digest=self.confidence_policy_digest,
            proposal=self.proposal,
            alternative_group_key=self.alternative_group_key,
            schema_version=self.schema_version,
        ):
            raise ClaimSchemaError("claim_id does not match deterministic content")
        if self.semantic_fingerprint != self.semantic_identity(
            owner_identity_digest=self.owner_identity_digest,
            subject_role=self.subject_role,
            actor_role=self.actor_role,
            proposal=self.proposal,
        ):
            raise ClaimSchemaError("Claim semantic fingerprint mismatch")
        if self.content_digest != canonical_digest(self._content_payload()):
            raise ClaimSchemaError("Claim content digest mismatch")

    @staticmethod
    def _semantic_proposal(proposal: ClaimSemanticProposal) -> dict[str, object]:
        value = proposal.to_dict()
        value.pop("human_summary")
        return value

    @staticmethod
    def identity_for(
        *,
        owner_identity_digest: str,
        semantic_record_digest: str,
        manifest_digest: str,
        normalizer_fingerprint: str,
        derivation_class: ClaimDerivationClass,
        binding_policy_digest: str,
        confidence_policy_digest: str,
        proposal: ClaimSemanticProposal,
        alternative_group_key: str | None,
        schema_version: str = CLAIM_SCHEMA_VERSION,
    ) -> str:
        return "claim_" + canonical_digest(
            {
                "owner_identity_digest": owner_identity_digest,
                "semantic_record_digest": semantic_record_digest,
                "manifest_digest": manifest_digest,
                "normalizer_fingerprint": normalizer_fingerprint,
                "derivation_class": ClaimDerivationClass(derivation_class).value,
                "binding_policy_digest": binding_policy_digest,
                "confidence_policy_digest": confidence_policy_digest,
                "claim_semantic": Claim._semantic_proposal(proposal),
                "alternative_group_key": alternative_group_key,
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
        semantic = Claim._semantic_proposal(proposal)
        semantic.pop("local_alternative_group_id")
        semantic.pop("normalizer_confidence")
        return canonical_digest(
            {
                "owner_identity_digest": owner_identity_digest,
                "subject_role": SemanticSubjectRole(subject_role).value,
                "actor_role": SemanticActorRole(actor_role).value,
                "claim_semantic": semantic,
            }
        )

    def _content_payload(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
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
            "source_epistemic_class": self.source_epistemic_class.value,
            "derivation_class": self.derivation_class.value,
            "source_confidence": self.source_confidence,
            "normalizer_confidence": self.normalizer_confidence,
            "effective_confidence": self.effective_confidence,
            "confidence_policy_digest": self.confidence_policy_digest,
            "binding_policy_digest": self.binding_policy_digest,
            "normalizer_fingerprint": self.normalizer_fingerprint,
            "proposal": self.proposal.to_dict(),
            "local_alternative_group_id": self.local_alternative_group_id,
            "alternative_group_key": self.alternative_group_key,
            "semantic_fingerprint": self.semantic_fingerprint,
            "created_at": utc_text(self.created_at),
            "schema_version": self.schema_version,
        }

    @classmethod
    def create(cls, **values: object) -> Claim:
        proposal = cast(ClaimSemanticProposal, values["proposal"])
        payload = dict(values)
        payload["claim_id"] = cls.identity_for(
            owner_identity_digest=str(payload["owner_identity_digest"]),
            semantic_record_digest=str(payload["semantic_record_digest"]),
            manifest_digest=str(payload["manifest_digest"]),
            normalizer_fingerprint=str(payload["normalizer_fingerprint"]),
            derivation_class=ClaimDerivationClass(cast(Any, payload["derivation_class"])),
            binding_policy_digest=str(payload["binding_policy_digest"]),
            confidence_policy_digest=str(payload["confidence_policy_digest"]),
            proposal=proposal,
            alternative_group_key=cast(str | None, payload["alternative_group_key"]),
        )
        payload["semantic_fingerprint"] = cls.semantic_identity(
            owner_identity_digest=str(payload["owner_identity_digest"]),
            subject_role=SemanticSubjectRole(cast(Any, payload["subject_role"])),
            actor_role=SemanticActorRole(cast(Any, payload["actor_role"])),
            proposal=proposal,
        )
        content = {
            "claim_id": payload["claim_id"],
            "owner_identity_digest": payload["owner_identity_digest"],
            "semantic_record_id": payload["semantic_record_id"],
            "semantic_record_digest": payload["semantic_record_digest"],
            "manifest_id": payload["manifest_id"],
            "manifest_digest": payload["manifest_digest"],
            "subject_role": SemanticSubjectRole(cast(Any, payload["subject_role"])).value,
            "actor_role": SemanticActorRole(cast(Any, payload["actor_role"])).value,
            "time_start": utc_text(strict_utc(cast(Any, payload["time_start"]), "time_start")),
            "time_end": utc_text(strict_utc(cast(Any, payload["time_end"]), "time_end")),
            "time_uncertainty_ms": payload["time_uncertainty_ms"],
            "source_epistemic_class": EpistemicClass(cast(Any, payload["source_epistemic_class"])).value,
            "derivation_class": ClaimDerivationClass(cast(Any, payload["derivation_class"])).value,
            "source_confidence": payload["source_confidence"],
            "normalizer_confidence": payload["normalizer_confidence"],
            "effective_confidence": payload["effective_confidence"],
            "confidence_policy_digest": payload["confidence_policy_digest"],
            "binding_policy_digest": payload["binding_policy_digest"],
            "normalizer_fingerprint": payload["normalizer_fingerprint"],
            "proposal": proposal.to_dict(),
            "local_alternative_group_id": payload["local_alternative_group_id"],
            "alternative_group_key": payload["alternative_group_key"],
            "semantic_fingerprint": payload["semantic_fingerprint"],
            "created_at": utc_text(strict_utc(cast(Any, payload["created_at"]), "created_at")),
            "schema_version": CLAIM_SCHEMA_VERSION,
        }
        return cast(Any, cls)(**payload, content_digest=canonical_digest(content))

    def to_dict(self) -> dict[str, object]:
        return {**self._content_payload(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, value: object, *, config: ClaimConfig) -> Claim:
        fields = frozenset({*cls.__dataclass_fields__})
        data = strict_fields(value, "claim", fields)
        require_fields(data, "claim", fields)
        proposal = ClaimSemanticProposalContract.model_validate(
            data["proposal"], config, frozenset(ClaimKind)
        )
        return cls(
            claim_id=data["claim_id"],
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
            source_epistemic_class=EpistemicClass(data["source_epistemic_class"]),
            derivation_class=ClaimDerivationClass(data["derivation_class"]),
            source_confidence=data["source_confidence"],
            normalizer_confidence=data["normalizer_confidence"],
            effective_confidence=data["effective_confidence"],
            confidence_policy_digest=data["confidence_policy_digest"],
            binding_policy_digest=data["binding_policy_digest"],
            normalizer_fingerprint=data["normalizer_fingerprint"],
            proposal=proposal,
            local_alternative_group_id=data["local_alternative_group_id"],
            alternative_group_key=data["alternative_group_key"],
            semantic_fingerprint=data["semantic_fingerprint"],
            created_at=parse_utc(data["created_at"], "created_at"),
            content_digest=data["content_digest"],
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class ClaimBatch:
    claim_batch_id: str
    processing_identity: str
    processing_lane: ClaimProcessingLane
    manifest_id: str
    manifest_digest: str
    semantic_record_id: str
    normalizer_name: str
    normalizer_fingerprint: str
    abstained: bool
    proposal_digest: str
    claim_count: int
    created_at: datetime
    content_digest: str
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
        object.__setattr__(self, "processing_lane", ClaimProcessingLane(self.processing_lane))
        for name in ("manifest_digest", "normalizer_fingerprint", "proposal_digest"):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name))
        if not isinstance(self.abstained, bool):
            raise TypeError("abstained must be boolean")
        object.__setattr__(self, "claim_count", non_negative_int(self.claim_count, "claim_count"))
        if self.abstained != (self.claim_count == 0):
            raise ClaimSchemaError("abstained Batch must have zero Claims")
        object.__setattr__(self, "created_at", strict_utc(self.created_at, "created_at"))
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))
        if self.claim_batch_id != self.identity_for(
            processing_identity=self.processing_identity,
            semantic_record_id=self.semantic_record_id,
            normalizer_fingerprint=self.normalizer_fingerprint,
            schema_version=self.schema_version,
        ):
            raise ClaimSchemaError("ClaimBatch identity mismatch")
        if self.content_digest != canonical_digest(self._content_payload()):
            raise ClaimSchemaError("ClaimBatch content digest mismatch")

    @staticmethod
    def identity_for(
        *,
        processing_identity: str,
        semantic_record_id: str,
        normalizer_fingerprint: str,
        schema_version: str = CLAIM_SCHEMA_VERSION,
    ) -> str:
        return "batch_" + canonical_digest(
            {
                "processing_identity": processing_identity,
                "semantic_record_id": semantic_record_id,
                "normalizer_fingerprint": normalizer_fingerprint,
                "schema_version": schema_version,
            }
        )

    @classmethod
    def create(cls, **values: object) -> ClaimBatch:
        payload = dict(values)
        payload["claim_batch_id"] = cls.identity_for(
            processing_identity=str(payload["processing_identity"]),
            semantic_record_id=str(payload["semantic_record_id"]),
            normalizer_fingerprint=str(payload["normalizer_fingerprint"]),
        )
        content = {
            "claim_batch_id": payload["claim_batch_id"],
            "processing_identity": payload["processing_identity"],
            "processing_lane": ClaimProcessingLane(cast(Any, payload["processing_lane"])).value,
            "manifest_id": payload["manifest_id"],
            "manifest_digest": payload["manifest_digest"],
            "semantic_record_id": payload["semantic_record_id"],
            "normalizer_name": payload["normalizer_name"],
            "normalizer_fingerprint": payload["normalizer_fingerprint"],
            "abstained": payload["abstained"],
            "proposal_digest": payload["proposal_digest"],
            "claim_count": payload["claim_count"],
            "created_at": utc_text(strict_utc(payload["created_at"], "created_at")),
            "schema_version": CLAIM_SCHEMA_VERSION,
        }
        return cast(Any, cls)(**payload, content_digest=canonical_digest(content))

    def _content_payload(self) -> dict[str, object]:
        return {
            "claim_batch_id": self.claim_batch_id,
            "processing_identity": self.processing_identity,
            "processing_lane": self.processing_lane.value,
            "manifest_id": self.manifest_id,
            "manifest_digest": self.manifest_digest,
            "semantic_record_id": self.semantic_record_id,
            "normalizer_name": self.normalizer_name,
            "normalizer_fingerprint": self.normalizer_fingerprint,
            "abstained": self.abstained,
            "proposal_digest": self.proposal_digest,
            "claim_count": self.claim_count,
            "created_at": utc_text(self.created_at),
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._content_payload(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, value: object) -> ClaimBatch:
        fields = frozenset({*cls.__dataclass_fields__})
        data = strict_fields(value, "claim_batch", fields)
        require_fields(data, "claim_batch", fields)
        return cls(
            claim_batch_id=data["claim_batch_id"],
            processing_identity=data["processing_identity"],
            processing_lane=ClaimProcessingLane(data["processing_lane"]),
            manifest_id=data["manifest_id"],
            manifest_digest=data["manifest_digest"],
            semantic_record_id=data["semantic_record_id"],
            normalizer_name=data["normalizer_name"],
            normalizer_fingerprint=data["normalizer_fingerprint"],
            abstained=data["abstained"],
            proposal_digest=data["proposal_digest"],
            claim_count=data["claim_count"],
            created_at=parse_utc(data["created_at"], "created_at"),
            content_digest=data["content_digest"],
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class ClaimProcessingReceipt:
    processing_identity: str
    processing_lane: ClaimProcessingLane
    scope_semantic_record_id: str | None
    manifest_id: str
    manifest_digest: str
    routing_policy_digest: str
    binding_policy_digest: str
    confidence_policy_digest: str
    admission_policy_digest: str
    normalizer_attempt_ids: tuple[str, ...]
    claim_batch_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    accepted_claim_ids: tuple[str, ...]
    decision_ids: tuple[str, ...]
    publication_recorded_at: datetime
    processing_completed_at: datetime
    content_digest: str
    schema_version: str = CLAIM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "processing_identity", identifier(self.processing_identity, "processing_identity"))
        object.__setattr__(self, "processing_lane", ClaimProcessingLane(self.processing_lane))
        object.__setattr__(
            self,
            "scope_semantic_record_id",
            optional_identifier(self.scope_semantic_record_id, "scope_semantic_record_id"),
        )
        object.__setattr__(self, "manifest_id", identifier(self.manifest_id, "manifest_id"))
        for name in (
            "manifest_digest",
            "routing_policy_digest",
            "binding_policy_digest",
            "confidence_policy_digest",
            "admission_policy_digest",
        ):
            object.__setattr__(self, name, sha256_digest(getattr(self, name), name))
        for name in (
            "normalizer_attempt_ids",
            "claim_batch_ids",
            "claim_ids",
            "accepted_claim_ids",
            "decision_ids",
        ):
            object.__setattr__(self, name, identifier_tuple(getattr(self, name), name, maximum_items=100_000))
        if self.processing_lane is ClaimProcessingLane.CORE and self.scope_semantic_record_id is not None:
            raise ClaimSchemaError("CORE Receipt cannot have a record scope")
        if self.processing_lane is ClaimProcessingLane.ENHANCEMENT and self.scope_semantic_record_id is None:
            raise ClaimSchemaError("ENHANCEMENT Receipt requires a record scope")
        if len(self.claim_ids) != len(self.decision_ids):
            raise ClaimSchemaError("each validated Claim requires one AdmissionDecision")
        if not set(self.accepted_claim_ids).issubset(self.claim_ids):
            raise ClaimSchemaError("accepted Claims must belong to the Receipt Claim set")
        object.__setattr__(
            self,
            "publication_recorded_at",
            strict_utc(self.publication_recorded_at, "publication_recorded_at"),
        )
        object.__setattr__(
            self,
            "processing_completed_at",
            strict_utc(self.processing_completed_at, "processing_completed_at"),
        )
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))
        if self.content_digest != canonical_digest(self._content_payload()):
            raise ClaimSchemaError("ProcessingReceipt content digest mismatch")

    @staticmethod
    def processing_identity_for(
        *,
        processing_lane: ClaimProcessingLane,
        manifest_digest: str,
        route_identities: tuple[str, ...],
        scope_semantic_record_digest: str | None,
        routing_policy_digest: str,
        binding_policy_digest: str,
        confidence_policy_digest: str,
        admission_policy_digest: str,
        schema_version: str = CLAIM_SCHEMA_VERSION,
        pipeline_version: str = PIPELINE_VERSION,
    ) -> str:
        return "processing_" + canonical_digest(
            {
                "processing_lane": ClaimProcessingLane(processing_lane).value,
                "manifest_digest": manifest_digest,
                "route_identities": route_identities,
                "scope_semantic_record_digest": scope_semantic_record_digest,
                "routing_policy_digest": routing_policy_digest,
                "binding_policy_digest": binding_policy_digest,
                "confidence_policy_digest": confidence_policy_digest,
                "admission_policy_digest": admission_policy_digest,
                "claim_schema_version": schema_version,
                "pipeline_version": pipeline_version,
            }
        )

    @classmethod
    def create(cls, **values: object) -> ClaimProcessingReceipt:
        content = {
            "processing_identity": values["processing_identity"],
            "processing_lane": ClaimProcessingLane(cast(Any, values["processing_lane"])).value,
            "scope_semantic_record_id": values["scope_semantic_record_id"],
            "manifest_id": values["manifest_id"],
            "manifest_digest": values["manifest_digest"],
            "routing_policy_digest": values["routing_policy_digest"],
            "binding_policy_digest": values["binding_policy_digest"],
            "confidence_policy_digest": values["confidence_policy_digest"],
            "admission_policy_digest": values["admission_policy_digest"],
            "normalizer_attempt_ids": values["normalizer_attempt_ids"],
            "claim_batch_ids": values["claim_batch_ids"],
            "claim_ids": values["claim_ids"],
            "accepted_claim_ids": values["accepted_claim_ids"],
            "decision_ids": values["decision_ids"],
            "publication_recorded_at": utc_text(strict_utc(values["publication_recorded_at"], "publication_recorded_at")),
            "processing_completed_at": utc_text(strict_utc(values["processing_completed_at"], "processing_completed_at")),
            "schema_version": CLAIM_SCHEMA_VERSION,
        }
        return cast(Any, cls)(**values, content_digest=canonical_digest(content))

    def _content_payload(self) -> dict[str, object]:
        return {
            "processing_identity": self.processing_identity,
            "processing_lane": self.processing_lane.value,
            "scope_semantic_record_id": self.scope_semantic_record_id,
            "manifest_id": self.manifest_id,
            "manifest_digest": self.manifest_digest,
            "routing_policy_digest": self.routing_policy_digest,
            "binding_policy_digest": self.binding_policy_digest,
            "confidence_policy_digest": self.confidence_policy_digest,
            "admission_policy_digest": self.admission_policy_digest,
            "normalizer_attempt_ids": self.normalizer_attempt_ids,
            "claim_batch_ids": self.claim_batch_ids,
            "claim_ids": self.claim_ids,
            "accepted_claim_ids": self.accepted_claim_ids,
            "decision_ids": self.decision_ids,
            "publication_recorded_at": utc_text(self.publication_recorded_at),
            "processing_completed_at": utc_text(self.processing_completed_at),
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._content_payload(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, value: object) -> ClaimProcessingReceipt:
        fields = frozenset({*cls.__dataclass_fields__})
        data = strict_fields(value, "claim_processing_receipt", fields)
        require_fields(data, "claim_processing_receipt", fields)
        return cls(
            processing_identity=data["processing_identity"],
            processing_lane=ClaimProcessingLane(data["processing_lane"]),
            scope_semantic_record_id=data["scope_semantic_record_id"],
            manifest_id=data["manifest_id"],
            manifest_digest=data["manifest_digest"],
            routing_policy_digest=data["routing_policy_digest"],
            binding_policy_digest=data["binding_policy_digest"],
            confidence_policy_digest=data["confidence_policy_digest"],
            admission_policy_digest=data["admission_policy_digest"],
            normalizer_attempt_ids=tuple(data["normalizer_attempt_ids"]),
            claim_batch_ids=tuple(data["claim_batch_ids"]),
            claim_ids=tuple(data["claim_ids"]),
            accepted_claim_ids=tuple(data["accepted_claim_ids"]),
            decision_ids=tuple(data["decision_ids"]),
            publication_recorded_at=parse_utc(data["publication_recorded_at"], "publication_recorded_at"),
            processing_completed_at=parse_utc(data["processing_completed_at"], "processing_completed_at"),
            content_digest=data["content_digest"],
            schema_version=data["schema_version"],
        )


__all__ = [
    "CLAIM_SCHEMA_VERSION",
    "PIPELINE_VERSION",
    "Claim",
    "ClaimBatch",
    "ClaimNormalizerAttempt",
    "ClaimNormalizerAttemptStatus",
    "ClaimProcessingReceipt",
    "EpistemicClass",
]
