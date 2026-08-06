"""Normalizer 唯一允许输出的无系统字段 Claim 语义提案。"""

from __future__ import annotations

from collections.abc import Mapping
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
    FREE_TEXT_SEMANTIC = "FREE_TEXT_SEMANTIC"


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
        "alternative_group_id",
        "normalizer_confidence",
    }
)


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
    alternative_group_id: str | None
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
                identifier_tuple(self.object_refs, "object_refs", maximum_items=128),
            )
            object.__setattr__(self, "location_ref", optional_identifier(self.location_ref, "location_ref"))
            object.__setattr__(
                self,
                "semantic_payload",
                json_snapshot(
                    self.semantic_payload,
                    "semantic_payload",
                    maximum_chars=ClaimConfig().max_semantic_payload_chars,
                ),
            )
            object.__setattr__(
                self,
                "human_summary",
                bounded_text(
                    self.human_summary,
                    "human_summary",
                    maximum=ClaimConfig().max_human_summary_chars,
                ),
            )
            object.__setattr__(
                self,
                "alternative_group_id",
                optional_identifier(self.alternative_group_id, "alternative_group_id"),
            )
            object.__setattr__(
                self,
                "normalizer_confidence",
                finite_score(self.normalizer_confidence, "normalizer_confidence"),
            )
            if self.claim_kind is ClaimKind.ACTIVITY_PHASE:
                if self.activity is None or self.phase is None:
                    raise ValueError("ACTIVITY_PHASE requires activity and phase")
            elif self.activity is not None or self.phase is not None:
                raise ValueError("activity and phase are only valid for ACTIVITY_PHASE")
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
            "alternative_group_id": self.alternative_group_id,
            "normalizer_confidence": self.normalizer_confidence,
        }

    @classmethod
    def model_validate(cls, value: object) -> ClaimSemanticProposal:
        try:
            data = strict_fields(value, "claim_semantic_proposal", _PROPOSAL_FIELDS)
            require_fields(data, "claim_semantic_proposal", _PROPOSAL_FIELDS)
            if not isinstance(data["object_refs"], list | tuple):
                raise TypeError("claim_semantic_proposal.object_refs must be an array")
            if not isinstance(data["semantic_payload"], Mapping):
                raise TypeError("claim_semantic_proposal.semantic_payload must be an object")
            return cls(
                claim_kind=ClaimKind(data["claim_kind"]),
                predicate=data["predicate"],
                semantic_family=data["semantic_family"],
                activity=data["activity"],
                phase=data["phase"],
                object_refs=tuple(data["object_refs"]),
                location_ref=data["location_ref"],
                semantic_payload=data["semantic_payload"],
                human_summary=data["human_summary"],
                alternative_group_id=data["alternative_group_id"],
                normalizer_confidence=data["normalizer_confidence"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ClaimSchemaError):
                raise
            raise ClaimSchemaError(str(exc)) from exc

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
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
                "claim_kind": {"type": "string", "enum": [item.value for item in ClaimKind]},
                "predicate": identifier_schema,
                "semantic_family": identifier_schema,
                "activity": nullable_identifier,
                "phase": nullable_identifier,
                "object_refs": {
                    "type": "array",
                    "maxItems": 128,
                    "uniqueItems": True,
                    "items": identifier_schema,
                },
                "location_ref": nullable_identifier,
                "semantic_payload": {"type": "object", "maxProperties": 128},
                "human_summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": ClaimConfig().max_human_summary_chars,
                },
                "alternative_group_id": nullable_identifier,
                "normalizer_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "oneOf": [
                {
                    "properties": {
                        "claim_kind": {"const": ClaimKind.ACTIVITY_PHASE.value},
                        "activity": identifier_schema,
                        "phase": identifier_schema,
                    }
                },
                {
                    "properties": {
                        "claim_kind": {
                            "enum": [item.value for item in ClaimKind if item is not ClaimKind.ACTIVITY_PHASE]
                        },
                        "activity": {"type": "null"},
                        "phase": {"type": "null"},
                    }
                },
            ],
        }


@dataclass(frozen=True)
class ClaimSemanticProposalBatch:
    abstained: bool
    claims: tuple[ClaimSemanticProposal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.abstained, bool):
            raise ClaimSchemaError("abstained must be boolean")
        claims = tuple(self.claims)
        if len(claims) > ClaimConfig().max_claims_per_batch or any(
            not isinstance(item, ClaimSemanticProposal) for item in claims
        ):
            raise ClaimSchemaError("semantic proposal batch exceeds its boundary")
        if self.abstained == bool(claims):
            raise ClaimSchemaError("abstained must be true exactly when claims is empty")
        groups: dict[str, int] = {}
        for claim in claims:
            if claim.alternative_group_id is not None:
                groups[claim.alternative_group_id] = groups.get(claim.alternative_group_id, 0) + 1
        if groups and max(groups.values()) > ClaimConfig().max_alternative_group_size:
            raise ClaimSchemaError("alternative group exceeds its boundary")
        object.__setattr__(self, "claims", claims)

    def to_dict(self) -> dict[str, object]:
        return {"abstained": self.abstained, "claims": tuple(item.to_dict() for item in self.claims)}

    @classmethod
    def model_validate(cls, value: object) -> ClaimSemanticProposalBatch:
        fields = frozenset({"abstained", "claims"})
        try:
            data = strict_fields(value, "claim_semantic_proposal_batch", fields)
            require_fields(data, "claim_semantic_proposal_batch", fields)
            if not isinstance(data["abstained"], bool):
                raise TypeError("abstained must be boolean")
            if not isinstance(data["claims"], list | tuple):
                raise TypeError("claims must be an array")
            return cls(
                abstained=data["abstained"],
                claims=tuple(ClaimSemanticProposal.model_validate(item) for item in data["claims"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ClaimSchemaError):
                raise
            raise ClaimSchemaError(str(exc)) from exc

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["abstained", "claims"],
            "properties": {
                "abstained": {"type": "boolean"},
                "claims": {
                    "type": "array",
                    "maxItems": ClaimConfig().max_claims_per_batch,
                    "items": ClaimSemanticProposal.model_json_schema(),
                },
            },
            "oneOf": [
                {"properties": {"abstained": {"const": True}, "claims": {"maxItems": 0}}},
                {"properties": {"abstained": {"const": False}, "claims": {"minItems": 1}}},
            ],
        }


__all__ = [
    "ClaimKind",
    "ClaimSemanticProposal",
    "ClaimSemanticProposalBatch",
]
