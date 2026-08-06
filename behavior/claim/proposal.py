"""Configuration-aware, system-field-free Claim semantic proposal contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from behavior._validation import (
    bounded_text,
    finite_score,
    identifier,
    identifier_tuple,
    json_snapshot,
    optional_identifier,
    require_fields,
    strict_fields,
)
from behavior.config import ClaimConfig
from behavior.errors import ClaimSchemaError


class ClaimKind(str, Enum):
    STATE_ASSERTION = "STATE_ASSERTION"
    STATE_TRANSITION = "STATE_TRANSITION"
    ACTIVITY_PHASE = "ACTIVITY_PHASE"
    INTERACTION = "INTERACTION"
    UTTERANCE = "UTTERANCE"
    ROBOT_ACTION = "ROBOT_ACTION"
    AGENT_ACTION = "AGENT_ACTION"
    TOOL_RESULT = "TOOL_RESULT"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    COVERAGE = "COVERAGE"


_PROPOSAL_FIELDS = frozenset(
    {
        "claim_kind",
        "predicate",
        "semantic_family",
        "activity",
        "phase",
        "object_refs",
        "location_ref",
        "semantic_payload",
        "human_summary",
        "local_alternative_group_id",
        "normalizer_confidence",
    }
)
_BATCH_FIELDS = frozenset({"abstained", "claims"})
_PROTOCOL_MAX_PAYLOAD_CHARS = 1_000_000
_PROTOCOL_MAX_SUMMARY_CHARS = 1_000_000
_PROTOCOL_MAX_OBJECT_REFS = 100_000
_PROTOCOL_MAX_CLAIMS = 100_000


@dataclass(frozen=True)
class ClaimSemanticProposal:
    claim_kind: ClaimKind
    predicate: str
    semantic_family: str
    activity: str | None
    phase: str | None
    object_refs: tuple[str, ...]
    location_ref: str | None
    semantic_payload: Mapping[str, Any]
    human_summary: str
    local_alternative_group_id: str | None
    normalizer_confidence: float

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "claim_kind", ClaimKind(self.claim_kind))
            object.__setattr__(self, "predicate", identifier(self.predicate, "predicate"))
            object.__setattr__(self, "semantic_family", identifier(self.semantic_family, "semantic_family"))
            object.__setattr__(self, "activity", optional_identifier(self.activity, "activity"))
            object.__setattr__(self, "phase", optional_identifier(self.phase, "phase"))
            object.__setattr__(
                self,
                "object_refs",
                identifier_tuple(self.object_refs, "object_refs", maximum_items=_PROTOCOL_MAX_OBJECT_REFS),
            )
            object.__setattr__(self, "location_ref", optional_identifier(self.location_ref, "location_ref"))
            object.__setattr__(
                self,
                "semantic_payload",
                json_snapshot(
                    self.semantic_payload,
                    "semantic_payload",
                    maximum_chars=_PROTOCOL_MAX_PAYLOAD_CHARS,
                ),
            )
            object.__setattr__(
                self,
                "human_summary",
                bounded_text(self.human_summary, "human_summary", maximum=_PROTOCOL_MAX_SUMMARY_CHARS),
            )
            object.__setattr__(
                self,
                "local_alternative_group_id",
                optional_identifier(self.local_alternative_group_id, "local_alternative_group_id"),
            )
            object.__setattr__(
                self,
                "normalizer_confidence",
                finite_score(self.normalizer_confidence, "normalizer_confidence"),
            )
            if self.claim_kind is ClaimKind.ACTIVITY_PHASE:
                if self.activity is None or self.phase is None:
                    raise ClaimSchemaError("ACTIVITY_PHASE requires activity and phase")
            elif self.activity is not None or self.phase is not None:
                raise ClaimSchemaError("activity and phase are only valid for ACTIVITY_PHASE")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ClaimSchemaError):
                raise
            raise ClaimSchemaError(str(exc)) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_kind": self.claim_kind.value,
            "predicate": self.predicate,
            "semantic_family": self.semantic_family,
            "activity": self.activity,
            "phase": self.phase,
            "object_refs": self.object_refs,
            "location_ref": self.location_ref,
            "semantic_payload": self.semantic_payload,
            "human_summary": self.human_summary,
            "local_alternative_group_id": self.local_alternative_group_id,
            "normalizer_confidence": self.normalizer_confidence,
        }


@dataclass(frozen=True)
class ClaimSemanticProposalBatch:
    abstained: bool
    claims: tuple[ClaimSemanticProposal, ...]

    def __init__(self, abstained: bool, claims: Sequence[ClaimSemanticProposal]) -> None:
        if not isinstance(abstained, bool):
            raise TypeError("abstained must be boolean")
        if isinstance(claims, str | bytes) or not isinstance(claims, Sequence):
            raise TypeError("claims must be an array")
        resolved = tuple(claims)
        if len(resolved) > _PROTOCOL_MAX_CLAIMS or any(
            not isinstance(item, ClaimSemanticProposal) for item in resolved
        ):
            raise ClaimSchemaError("claims exceed the protocol boundary")
        if abstained == bool(resolved):
            raise ClaimSchemaError("abstained is true exactly when claims is empty")
        object.__setattr__(self, "abstained", abstained)
        object.__setattr__(self, "claims", resolved)

    def to_dict(self) -> dict[str, object]:
        return {"abstained": self.abstained, "claims": tuple(item.to_dict() for item in self.claims)}


class ClaimSemanticProposalContract:
    @staticmethod
    def model_validate(
        value: object,
        config: ClaimConfig,
        allowed_claim_kinds: frozenset[ClaimKind],
    ) -> ClaimSemanticProposal:
        if not isinstance(config, ClaimConfig):
            raise TypeError("config must be ClaimConfig")
        try:
            data = strict_fields(value, "claim_semantic_proposal", _PROPOSAL_FIELDS)
            require_fields(data, "claim_semantic_proposal", _PROPOSAL_FIELDS)
            if not isinstance(data["object_refs"], list | tuple):
                raise TypeError("object_refs must be an array")
            if not isinstance(data["semantic_payload"], Mapping):
                raise TypeError("semantic_payload must be an object")
            proposal = ClaimSemanticProposal(
                claim_kind=ClaimKind(data["claim_kind"]),
                predicate=data["predicate"],
                semantic_family=data["semantic_family"],
                activity=data["activity"],
                phase=data["phase"],
                object_refs=tuple(data["object_refs"]),
                location_ref=data["location_ref"],
                semantic_payload=data["semantic_payload"],
                human_summary=data["human_summary"],
                local_alternative_group_id=data["local_alternative_group_id"],
                normalizer_confidence=data["normalizer_confidence"],
            )
            if proposal.claim_kind not in allowed_claim_kinds:
                raise ClaimSchemaError("claim_kind is not allowed for this normalization route")
            if len(proposal.object_refs) > config.max_alternative_group_size * 8:
                raise ClaimSchemaError("object_refs exceed the Runtime Claim boundary")
            json_snapshot(
                proposal.semantic_payload,
                "semantic_payload",
                maximum_chars=config.max_semantic_payload_chars,
            )
            bounded_text(
                proposal.human_summary,
                "human_summary",
                maximum=config.max_human_summary_chars,
            )
            return proposal
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ClaimSchemaError):
                raise
            raise ClaimSchemaError(str(exc)) from exc

    @staticmethod
    def model_json_schema(
        config: ClaimConfig,
        allowed_claim_kinds: frozenset[ClaimKind],
    ) -> dict[str, object]:
        if not allowed_claim_kinds:
            raise ClaimSchemaError("allowed Claim kinds cannot be empty")
        identifier_schema: dict[str, object] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 256,
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}$",
        }
        nullable_identifier = {"anyOf": [identifier_schema, {"type": "null"}]}
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": sorted(_PROPOSAL_FIELDS),
            "properties": {
                "claim_kind": {
                    "type": "string",
                    "enum": sorted(item.value for item in allowed_claim_kinds),
                },
                "predicate": identifier_schema,
                "semantic_family": identifier_schema,
                "activity": nullable_identifier,
                "phase": nullable_identifier,
                "object_refs": {
                    "type": "array",
                    "maxItems": config.max_alternative_group_size * 8,
                    "uniqueItems": True,
                    "items": identifier_schema,
                },
                "location_ref": nullable_identifier,
                "semantic_payload": {"type": "object", "maxProperties": 128},
                "human_summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": config.max_human_summary_chars,
                },
                "local_alternative_group_id": nullable_identifier,
                "normalizer_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }


class ClaimSemanticProposalBatchContract:
    @staticmethod
    def model_validate(
        value: object,
        config: ClaimConfig,
        allowed_claim_kinds: frozenset[ClaimKind],
    ) -> ClaimSemanticProposalBatch:
        try:
            data = strict_fields(value, "claim_semantic_proposal_batch", _BATCH_FIELDS)
            require_fields(data, "claim_semantic_proposal_batch", _BATCH_FIELDS)
            if not isinstance(data["abstained"], bool):
                raise TypeError("abstained must be boolean")
            if not isinstance(data["claims"], list | tuple):
                raise TypeError("claims must be an array")
            if len(data["claims"]) > min(config.max_claims_per_record, config.max_claims_per_batch):
                raise ClaimSchemaError("claims exceed the Runtime Claim boundary")
            claims = tuple(
                ClaimSemanticProposalContract.model_validate(item, config, allowed_claim_kinds)
                for item in data["claims"]
            )
            groups: dict[str, int] = {}
            for proposal in claims:
                if proposal.local_alternative_group_id is not None:
                    groups[proposal.local_alternative_group_id] = (
                        groups.get(proposal.local_alternative_group_id, 0) + 1
                    )
            if groups and max(groups.values()) > config.max_alternative_group_size:
                raise ClaimSchemaError("alternative group exceeds the Runtime boundary")
            return ClaimSemanticProposalBatch(data["abstained"], claims)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ClaimSchemaError):
                raise
            raise ClaimSchemaError(str(exc)) from exc

    @staticmethod
    def model_json_schema(
        config: ClaimConfig,
        allowed_claim_kinds: frozenset[ClaimKind],
    ) -> dict[str, object]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": sorted(_BATCH_FIELDS),
            "properties": {
                "abstained": {"type": "boolean"},
                "claims": {
                    "type": "array",
                    "maxItems": min(config.max_claims_per_record, config.max_claims_per_batch),
                    "items": ClaimSemanticProposalContract.model_json_schema(config, allowed_claim_kinds),
                },
            },
        }


__all__ = [
    "ClaimKind",
    "ClaimSemanticProposal",
    "ClaimSemanticProposalBatch",
    "ClaimSemanticProposalBatchContract",
    "ClaimSemanticProposalContract",
]
