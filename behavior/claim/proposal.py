"""Normalizer 输出的唯一结构与容量边界。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from behavior._validation import (
    finite_score,
    identifier,
    json_snapshot,
    optional_bounded_text,
    optional_identifier,
    strict_object,
)
from behavior.config import ClaimNormalizationConfig
from behavior.errors import NormalizerOutputError

_PROPOSAL_FIELDS = frozenset(
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
_CLAIM_SYSTEM_FIELDS = frozenset(
    {
        "actor_role",
        "alternative_group_key",
        "attempt_id",
        "binding_policy_digest",
        "capability_digest",
        "claim_id",
        "claim_sequence",
        "compatibility_policy_digest",
        "confidence_policy_digest",
        "content_digest",
        "created_at",
        "derivation_class",
        "effective_confidence",
        "encoded_digest",
        "evidence_record_digest",
        "evidence_record_id",
        "evidence_sequence",
        "ingested_at",
        "normalizer_fingerprint",
        "processing_identity",
        "producer_fingerprint",
        "semantic_digest",
        "semantic_fingerprint",
        "source_confidence",
        "source_epistemic_class",
        "source_trust",
        "subject_role",
        "time_end",
        "time_start",
        "time_uncertainty_ms",
    }
)


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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class ValidatedClaimProposalBatch:
    proposals: tuple[ClaimSemanticProposal, ...]


class ClaimProposalParser:
    def __init__(self, config: ClaimNormalizationConfig) -> None:
        self.config = config

    def parse_batch(self, value: object) -> ValidatedClaimProposalBatch:
        try:
            data = strict_object(value, "claim_proposals", frozenset({"proposals"}))
            raw = data["proposals"]
            if not isinstance(raw, list | tuple):
                raise TypeError("proposals must be an array")
            return self._validated(tuple(self._parse(item) for item in raw))
        except NormalizerOutputError:
            raise
        except (TypeError, ValueError) as exc:
            raise NormalizerOutputError("Claim proposal batch failed strict validation") from exc

    def validate_batch(self, value: object) -> ValidatedClaimProposalBatch:
        if not isinstance(value, tuple | list):
            raise NormalizerOutputError("Normalizer output must be a Proposal sequence")
        if any(not isinstance(item, ClaimSemanticProposal) for item in value):
            raise NormalizerOutputError(
                "Normalizer output must contain ClaimSemanticProposal values"
            )
        return self.parse_batch(
            {"proposals": [proposal_to_dict(item) for item in value]}
        )

    def _parse(self, value: object) -> ClaimSemanticProposal:
        data = strict_object(value, "claim_proposal", _PROPOSAL_FIELDS)
        return ClaimSemanticProposal(
            claim_kind=ClaimKind(data["claim_kind"]),
            semantic_family=optional_identifier(
                data["semantic_family"],
                "proposal.semantic_family",
            ),
            predicate=identifier(data["predicate"], "proposal.predicate"),
            activity=optional_identifier(data["activity"], "proposal.activity"),
            phase=optional_identifier(data["phase"], "proposal.phase"),
            semantic_payload=json_snapshot(
                data["semantic_payload"],
                "proposal.semantic_payload",
                maximum_chars=self.config.max_semantic_payload_chars,
                maximum_items=128,
                maximum_depth=12,
                forbidden_keys=_CLAIM_SYSTEM_FIELDS,
            ),
            human_summary=optional_bounded_text(
                data["human_summary"],
                "proposal.human_summary",
                maximum=self.config.max_human_summary_chars,
            ),
            local_alternative_group_id=optional_identifier(
                data["local_alternative_group_id"],
                "proposal.local_alternative_group_id",
            ),
            normalizer_confidence=finite_score(
                data["normalizer_confidence"],
                "proposal.normalizer_confidence",
            ),
        )

    def _validated(
        self,
        proposals: tuple[ClaimSemanticProposal, ...],
    ) -> ValidatedClaimProposalBatch:
        if len(proposals) > self.config.max_claims_per_record:
            raise NormalizerOutputError("proposal count exceeds configured boundary")
        groups: dict[str, int] = {}
        for proposal in proposals:
            group = proposal.local_alternative_group_id
            if group is None:
                continue
            groups[group] = groups.get(group, 0) + 1
            if groups[group] > self.config.max_alternative_group_size:
                raise NormalizerOutputError("alternative group exceeds configured boundary")
        return ValidatedClaimProposalBatch(proposals)

    def json_schema(self) -> dict[str, object]:
        nullable_identifier: dict[str, object] = {
            "anyOf": [
                {"type": "string", "minLength": 1, "maxLength": 256},
                {"type": "null"},
            ]
        }
        item: dict[str, object] = {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(_PROPOSAL_FIELDS),
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
                            "maxLength": self.config.max_human_summary_chars,
                        },
                        {"type": "null"},
                    ]
                },
                "local_alternative_group_id": nullable_identifier,
                "normalizer_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["proposals"],
            "properties": {
                "proposals": {
                    "type": "array",
                    "maxItems": self.config.max_claims_per_record,
                    "items": item,
                }
            },
        }


def proposal_to_dict(
    value: ClaimSemanticProposal,
    *,
    include_human_summary: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, ClaimSemanticProposal):
        raise TypeError("value must be ClaimSemanticProposal")
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
