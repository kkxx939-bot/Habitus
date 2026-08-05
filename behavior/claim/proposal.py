"""模型和确定性 Producer 唯一允许输出的严格 ClaimProposal。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from behavior._validation import (
    bounded_text,
    finite_score,
    identifier,
    identifier_tuple,
    json_snapshot,
    non_negative_int,
    optional_identifier,
    parse_utc,
    require_fields,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.config import ClaimConfig
from behavior.errors import ClaimSchemaError


class ClaimKind(str, Enum):
    STATE_ASSERTION = "STATE_ASSERTION"
    STATE_TRANSITION = "STATE_TRANSITION"
    ACTIVITY_PHASE = "ACTIVITY_PHASE"
    INTERACTION = "INTERACTION"
    FEEDBACK = "FEEDBACK"
    ROBOT_ACTION = "ROBOT_ACTION"
    AGENT_ACTION = "AGENT_ACTION"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    COVERAGE = "COVERAGE"


class SubjectRole(str, Enum):
    OWNER = "OWNER"
    ROBOT = "ROBOT"
    ENVIRONMENT = "ENVIRONMENT"
    OTHER_ANONYMOUS = "OTHER_ANONYMOUS"
    SYSTEM = "SYSTEM"
    TOOL = "TOOL"
    AGENT = "AGENT"


ActorRole = SubjectRole


class EpistemicClass(str, Enum):
    DIRECT_SOURCE = "DIRECT_SOURCE"
    USER_EXPLICIT = "USER_EXPLICIT"
    SENSOR_INFERRED = "SENSOR_INFERRED"
    MODEL_INFERRED = "MODEL_INFERRED"
    MULTIMODAL_MODEL_INFERRED = "MULTIMODAL_MODEL_INFERRED"


_PROPOSAL_FIELDS = frozenset(
    {
        "claim_kind",
        "subject_role",
        "actor_role",
        "predicate",
        "semantic_family",
        "activity",
        "phase",
        "object_refs",
        "location_ref",
        "scene_ref",
        "time_start",
        "time_end",
        "time_uncertainty_ms",
        "epistemic_class",
        "raw_score",
        "alternative_group_id",
        "semantic_payload",
        "human_summary",
        "source_record_ids",
    }
)


@dataclass(frozen=True)
class ClaimProposal:
    claim_kind: ClaimKind
    subject_role: SubjectRole
    actor_role: ActorRole
    predicate: str
    semantic_family: str
    activity: str | None
    phase: str | None
    object_refs: tuple[str, ...]
    location_ref: str | None
    scene_ref: str | None
    time_start: datetime
    time_end: datetime
    time_uncertainty_ms: int
    epistemic_class: EpistemicClass
    raw_score: float
    alternative_group_id: str | None
    semantic_payload: Mapping[str, Any]
    human_summary: str
    source_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "claim_kind", ClaimKind(self.claim_kind))
            object.__setattr__(self, "subject_role", SubjectRole(self.subject_role))
            object.__setattr__(self, "actor_role", SubjectRole(self.actor_role))
            object.__setattr__(self, "predicate", identifier(self.predicate, "predicate"))
            object.__setattr__(self, "semantic_family", identifier(self.semantic_family, "semantic_family"))
            object.__setattr__(self, "activity", optional_identifier(self.activity, "activity"))
            object.__setattr__(self, "phase", optional_identifier(self.phase, "phase"))
            object.__setattr__(self, "object_refs", identifier_tuple(self.object_refs, "object_refs", maximum_items=128))
            object.__setattr__(self, "location_ref", optional_identifier(self.location_ref, "location_ref"))
            object.__setattr__(self, "scene_ref", optional_identifier(self.scene_ref, "scene_ref"))
            object.__setattr__(self, "time_start", strict_utc(self.time_start, "time_start"))
            object.__setattr__(self, "time_end", strict_utc(self.time_end, "time_end"))
            if self.time_end < self.time_start:
                raise ValueError("time_end cannot be earlier than time_start")
            object.__setattr__(
                self,
                "time_uncertainty_ms",
                non_negative_int(self.time_uncertainty_ms, "time_uncertainty_ms"),
            )
            object.__setattr__(self, "epistemic_class", EpistemicClass(self.epistemic_class))
            object.__setattr__(self, "raw_score", finite_score(self.raw_score, "raw_score"))
            object.__setattr__(
                self,
                "alternative_group_id",
                optional_identifier(self.alternative_group_id, "alternative_group_id"),
            )
            object.__setattr__(
                self,
                "semantic_payload",
                json_snapshot(self.semantic_payload, "semantic_payload", maximum_chars=ClaimConfig().max_semantic_payload_chars),
            )
            object.__setattr__(
                self,
                "human_summary",
                bounded_text(self.human_summary, "human_summary", maximum=ClaimConfig().max_human_summary_chars),
            )
            source_ids = identifier_tuple(self.source_record_ids, "source_record_ids", maximum_items=1_000)
            if not source_ids:
                raise ValueError("source_record_ids cannot be empty")
            object.__setattr__(self, "source_record_ids", source_ids)
            if self.claim_kind is ClaimKind.ACTIVITY_PHASE:
                if self.activity is None or self.phase is None:
                    raise ValueError("ACTIVITY_PHASE requires activity and phase")
            elif self.activity is not None or self.phase is not None:
                raise ValueError("activity and phase are only valid for ACTIVITY_PHASE")
            if self.claim_kind is ClaimKind.INTERACTION and not self.object_refs:
                raise ValueError("INTERACTION requires at least one object_ref")
            if self.claim_kind is ClaimKind.ROBOT_ACTION and self.actor_role is not SubjectRole.ROBOT:
                raise ValueError("ROBOT_ACTION requires actor_role ROBOT")
            if self.claim_kind is ClaimKind.AGENT_ACTION and self.actor_role is not SubjectRole.AGENT:
                raise ValueError("AGENT_ACTION requires actor_role AGENT")
            if (
                self.claim_kind is ClaimKind.ENVIRONMENT_CHANGE
                and self.subject_role is not SubjectRole.ENVIRONMENT
            ):
                raise ValueError("ENVIRONMENT_CHANGE requires subject_role ENVIRONMENT")
            if self.claim_kind is ClaimKind.COVERAGE and self.subject_role not in {
                SubjectRole.SYSTEM,
                SubjectRole.ENVIRONMENT,
            }:
                raise ValueError("COVERAGE requires SYSTEM or ENVIRONMENT subject_role")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ClaimSchemaError):
                raise
            raise ClaimSchemaError(str(exc)) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_kind": self.claim_kind.value,
            "subject_role": self.subject_role.value,
            "actor_role": self.actor_role.value,
            "predicate": self.predicate,
            "semantic_family": self.semantic_family,
            "activity": self.activity,
            "phase": self.phase,
            "object_refs": self.object_refs,
            "location_ref": self.location_ref,
            "scene_ref": self.scene_ref,
            "time_start": utc_text(self.time_start),
            "time_end": utc_text(self.time_end),
            "time_uncertainty_ms": self.time_uncertainty_ms,
            "epistemic_class": self.epistemic_class.value,
            "raw_score": self.raw_score,
            "alternative_group_id": self.alternative_group_id,
            "semantic_payload": self.semantic_payload,
            "human_summary": self.human_summary,
            "source_record_ids": self.source_record_ids,
        }

    @classmethod
    def model_validate(cls, value: object) -> ClaimProposal:
        try:
            data = strict_fields(value, "claim_proposal", _PROPOSAL_FIELDS)
            require_fields(data, "claim_proposal", _PROPOSAL_FIELDS)
            for name in ("object_refs", "source_record_ids"):
                if not isinstance(data[name], list):
                    raise TypeError(f"claim_proposal.{name} must be a JSON array")
            if not isinstance(data["semantic_payload"], Mapping):
                raise TypeError("claim_proposal.semantic_payload must be a JSON object")
            return cls(
                claim_kind=ClaimKind(data["claim_kind"]),
                subject_role=SubjectRole(data["subject_role"]),
                actor_role=SubjectRole(data["actor_role"]),
                predicate=data["predicate"],
                semantic_family=data["semantic_family"],
                activity=data["activity"],
                phase=data["phase"],
                object_refs=tuple(data["object_refs"]),
                location_ref=data["location_ref"],
                scene_ref=data["scene_ref"],
                time_start=parse_utc(data["time_start"], "claim_proposal.time_start"),
                time_end=parse_utc(data["time_end"], "claim_proposal.time_end"),
                time_uncertainty_ms=data["time_uncertainty_ms"],
                epistemic_class=EpistemicClass(data["epistemic_class"]),
                raw_score=data["raw_score"],
                alternative_group_id=data["alternative_group_id"],
                semantic_payload=data["semantic_payload"],
                human_summary=data["human_summary"],
                source_record_ids=tuple(data["source_record_ids"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, ClaimSchemaError):
                raise
            raise ClaimSchemaError(str(exc)) from exc

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        role_values = [item.value for item in SubjectRole]
        identifier_pattern = r"^[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}$"
        identifier_schema = {
            "type": "string",
            "minLength": 1,
            "maxLength": 256,
            "pattern": identifier_pattern,
        }
        nullable_identifier = {
            "anyOf": [identifier_schema, {"type": "null"}],
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": sorted(_PROPOSAL_FIELDS),
            "properties": {
                "claim_kind": {"type": "string", "enum": [item.value for item in ClaimKind]},
                "subject_role": {"type": "string", "enum": role_values},
                "actor_role": {"type": "string", "enum": role_values},
                "predicate": identifier_schema,
                "semantic_family": identifier_schema,
                "activity": nullable_identifier,
                "phase": nullable_identifier,
                "object_refs": {"type": "array", "maxItems": 128, "uniqueItems": True, "items": identifier_schema},
                "location_ref": nullable_identifier,
                "scene_ref": nullable_identifier,
                "time_start": {"type": "string", "format": "date-time"},
                "time_end": {"type": "string", "format": "date-time"},
                "time_uncertainty_ms": {"type": "integer", "minimum": 0},
                "epistemic_class": {"type": "string", "enum": [item.value for item in EpistemicClass]},
                "raw_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "alternative_group_id": nullable_identifier,
                "semantic_payload": {"type": "object", "maxProperties": 128},
                "human_summary": {"type": "string", "minLength": 1, "maxLength": ClaimConfig().max_human_summary_chars},
                "source_record_ids": {"type": "array", "minItems": 1, "maxItems": 1_000, "uniqueItems": True, "items": identifier_schema},
            },
            "allOf": [
                {
                    "if": {"properties": {"claim_kind": {"const": ClaimKind.ACTIVITY_PHASE.value}}},
                    "then": {"properties": {"activity": {"type": "string"}, "phase": {"type": "string"}}},
                    "else": {"properties": {"activity": {"type": "null"}, "phase": {"type": "null"}}},
                },
                {
                    "if": {"properties": {"claim_kind": {"const": ClaimKind.INTERACTION.value}}},
                    "then": {"properties": {"object_refs": {"minItems": 1}}},
                },
                {
                    "if": {"properties": {"claim_kind": {"const": ClaimKind.ROBOT_ACTION.value}}},
                    "then": {"properties": {"actor_role": {"const": SubjectRole.ROBOT.value}}},
                },
                {
                    "if": {"properties": {"claim_kind": {"const": ClaimKind.AGENT_ACTION.value}}},
                    "then": {"properties": {"actor_role": {"const": SubjectRole.AGENT.value}}},
                },
                {
                    "if": {"properties": {"claim_kind": {"const": ClaimKind.ENVIRONMENT_CHANGE.value}}},
                    "then": {"properties": {"subject_role": {"const": SubjectRole.ENVIRONMENT.value}}},
                },
                {
                    "if": {"properties": {"claim_kind": {"const": ClaimKind.COVERAGE.value}}},
                    "then": {
                        "properties": {
                            "subject_role": {
                                "enum": [SubjectRole.SYSTEM.value, SubjectRole.ENVIRONMENT.value]
                            }
                        }
                    },
                },
            ],
        }


@dataclass(frozen=True)
class ClaimProposalBatch:
    abstained: bool
    claims: tuple[ClaimProposal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.abstained, bool):
            raise ClaimSchemaError("abstained must be boolean")
        claims = tuple(self.claims)
        if len(claims) > ClaimConfig().max_claims_per_batch or any(
            not isinstance(claim, ClaimProposal) for claim in claims
        ):
            raise ClaimSchemaError("claims exceed their boundary or contain invalid values")
        if self.abstained == bool(claims):
            raise ClaimSchemaError("abstained must be true exactly when claims is empty")
        groups: dict[str, int] = {}
        for claim in claims:
            if claim.alternative_group_id is not None:
                groups[claim.alternative_group_id] = groups.get(claim.alternative_group_id, 0) + 1
        if groups and max(groups.values()) > ClaimConfig().max_alternative_group_size:
            raise ClaimSchemaError("alternative group exceeds its configured boundary")
        object.__setattr__(self, "claims", claims)

    def to_dict(self) -> dict[str, object]:
        return {"abstained": self.abstained, "claims": tuple(claim.to_dict() for claim in self.claims)}

    @classmethod
    def model_validate(cls, value: object) -> ClaimProposalBatch:
        try:
            fields = frozenset({"abstained", "claims"})
            data = strict_fields(value, "claim_proposal_batch", fields)
            require_fields(data, "claim_proposal_batch", fields)
            if not isinstance(data["abstained"], bool):
                raise TypeError("abstained must be boolean")
            if not isinstance(data["claims"], list):
                raise TypeError("claims must be a JSON array")
            return cls(
                abstained=data["abstained"],
                claims=tuple(ClaimProposal.model_validate(item) for item in data["claims"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ClaimSchemaError):
                raise
            raise ClaimSchemaError(str(exc)) from exc

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        proposal_schema = ClaimProposal.model_json_schema()
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["abstained", "claims"],
            "properties": {
                "abstained": {"type": "boolean"},
                "claims": {"type": "array", "maxItems": ClaimConfig().max_claims_per_batch, "items": proposal_schema},
            },
            "oneOf": [
                {"properties": {"abstained": {"const": True}, "claims": {"maxItems": 0}}},
                {"properties": {"abstained": {"const": False}, "claims": {"minItems": 1}}},
            ],
        }


__all__ = [
    "ActorRole",
    "ClaimKind",
    "ClaimProposal",
    "ClaimProposalBatch",
    "EpistemicClass",
    "SubjectRole",
]
