"""Normalizer 唯一允许输出的无系统字段语义 Proposal。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from behavior._validation import (
    bounded_text,
    claim_semantic_json_snapshot,
    finite_score,
    identifier,
    optional_bounded_text,
    optional_identifier,
    require_fields,
    strict_fields,
)
from behavior.config import ClaimNormalizationConfig
from behavior.errors import BehaviorClaimSchemaError
from foundation.integrity import canonical_json


class ClaimKind(str, Enum):
    ACTIVITY = "ACTIVITY"
    UTTERANCE = "UTTERANCE"
    STATE_ASSERTION = "STATE_ASSERTION"
    STATE_TRANSITION = "STATE_TRANSITION"
    INTERACTION = "INTERACTION"
    ACTION = "ACTION"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    COVERAGE = "COVERAGE"
    FEEDBACK = "FEEDBACK"
    FREE_TEXT = "FREE_TEXT"


@dataclass(frozen=True)
class ClaimSemanticProposal:
    claim_kind: ClaimKind
    semantic_family: str | None
    predicate: str
    activity: str | None
    phase: str | None
    semantic_payload: Mapping[str, Any]
    human_summary: str | None
    local_alternative_group_id: str | None
    normalizer_confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_kind", ClaimKind(self.claim_kind))
        object.__setattr__(
            self,
            "semantic_family",
            optional_identifier(self.semantic_family, "proposal.semantic_family"),
        )
        object.__setattr__(self, "predicate", identifier(self.predicate, "proposal.predicate"))
        object.__setattr__(self, "activity", optional_identifier(self.activity, "proposal.activity"))
        object.__setattr__(self, "phase", optional_identifier(self.phase, "proposal.phase"))
        object.__setattr__(
            self,
            "semantic_payload",
            claim_semantic_json_snapshot(
                self.semantic_payload,
                "proposal.semantic_payload",
                maximum_chars=1_000_000,
                maximum_items=10_000,
                maximum_depth=32,
            ),
        )
        object.__setattr__(
            self,
            "human_summary",
            optional_bounded_text(self.human_summary, "proposal.human_summary", maximum=1_000_000),
        )
        object.__setattr__(
            self,
            "local_alternative_group_id",
            optional_identifier(
                self.local_alternative_group_id,
                "proposal.local_alternative_group_id",
            ),
        )
        object.__setattr__(
            self,
            "normalizer_confidence",
            finite_score(self.normalizer_confidence, "proposal.normalizer_confidence"),
        )


def proposal_to_dict(
    value: ClaimSemanticProposal,
    *,
    include_human_summary: bool = True,
) -> dict[str, Any]:
    result = {
        "claim_kind": value.claim_kind.value,
        "semantic_family": value.semantic_family,
        "predicate": value.predicate,
        "activity": value.activity,
        "phase": value.phase,
        "semantic_payload": value.semantic_payload,
        "local_alternative_group_id": value.local_alternative_group_id,
        "normalizer_confidence": value.normalizer_confidence,
    }
    if include_human_summary:
        result["human_summary"] = value.human_summary
    return result


def proposal_from_dict(value: object, config: ClaimNormalizationConfig) -> ClaimSemanticProposal:
    if not isinstance(config, ClaimNormalizationConfig):
        raise TypeError("config must be ClaimNormalizationConfig")
    fields = frozenset(
        {
            "claim_kind",
            "semantic_family",
            "predicate",
            "activity",
            "phase",
            "semantic_payload",
            "human_summary",
            "local_alternative_group_id",
            "normalizer_confidence",
        }
    )
    try:
        data = strict_fields(value, "claim_proposal", fields)
        require_fields(data, "claim_proposal", fields)
        proposal = ClaimSemanticProposal(
            claim_kind=ClaimKind(data["claim_kind"]),
            semantic_family=data["semantic_family"],
            predicate=data["predicate"],
            activity=data["activity"],
            phase=data["phase"],
            semantic_payload=data["semantic_payload"],
            human_summary=data["human_summary"],
            local_alternative_group_id=data["local_alternative_group_id"],
            normalizer_confidence=data["normalizer_confidence"],
        )
        if len(bounded_text(proposal.predicate, "proposal.predicate", maximum=256)) > 256:
            raise ValueError("proposal predicate boundary exceeded")
        if len(canonical_json(proposal.semantic_payload)) > config.max_semantic_payload_chars:
            raise ValueError("proposal semantic payload boundary exceeded")
        if proposal.human_summary is not None and len(proposal.human_summary) > config.max_human_summary_chars:
            raise ValueError("proposal human summary boundary exceeded")
        return proposal
    except (TypeError, ValueError) as exc:
        raise BehaviorClaimSchemaError("Claim proposal failed strict validation") from exc


def proposal_batch_from_dict(
    value: object,
    config: ClaimNormalizationConfig,
) -> tuple[ClaimSemanticProposal, ...]:
    data = strict_fields(value, "claim_proposals", frozenset({"proposals"}))
    require_fields(data, "claim_proposals", frozenset({"proposals"}))
    raw = data["proposals"]
    if not isinstance(raw, list | tuple):
        raise BehaviorClaimSchemaError("proposals must be an array")
    if len(raw) > config.max_claims_per_record:
        raise BehaviorClaimSchemaError("proposal count exceeds configured boundary")
    proposals = tuple(proposal_from_dict(item, config) for item in raw)
    alternative_counts: dict[str, int] = {}
    for proposal in proposals:
        if proposal.local_alternative_group_id is not None:
            group = proposal.local_alternative_group_id
            alternative_counts[group] = alternative_counts.get(group, 0) + 1
            if alternative_counts[group] > config.max_alternative_group_size:
                raise BehaviorClaimSchemaError("alternative group exceeds configured boundary")
    return proposals


def proposal_batch_json_schema(config: ClaimNormalizationConfig) -> dict[str, object]:
    if not isinstance(config, ClaimNormalizationConfig):
        raise TypeError("config must be ClaimNormalizationConfig")
    nullable_identifier: dict[str, object] = {
        "anyOf": [
            {"type": "string", "minLength": 1, "maxLength": 256},
            {"type": "null"},
        ]
    }
    item: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "claim_kind",
            "semantic_family",
            "predicate",
            "activity",
            "phase",
            "semantic_payload",
            "human_summary",
            "local_alternative_group_id",
            "normalizer_confidence",
        ],
        "properties": {
            "claim_kind": {"type": "string", "enum": [item.value for item in ClaimKind]},
            "semantic_family": nullable_identifier,
            "predicate": {"type": "string", "minLength": 1, "maxLength": 256},
            "activity": nullable_identifier,
            "phase": nullable_identifier,
            "semantic_payload": {"type": "object"},
            "human_summary": {
                "anyOf": [
                    {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": config.max_human_summary_chars,
                    },
                    {"type": "null"},
                ]
            },
            "local_alternative_group_id": nullable_identifier,
            "normalizer_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["proposals"],
        "properties": {
            "proposals": {
                "type": "array",
                "maxItems": config.max_claims_per_record,
                "items": item,
            }
        },
    }


__all__ = [
    "ClaimKind",
    "ClaimSemanticProposal",
    "proposal_batch_from_dict",
    "proposal_batch_json_schema",
    "proposal_from_dict",
    "proposal_to_dict",
]
