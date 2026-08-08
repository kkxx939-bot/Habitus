"""RecordKind 的唯一 Payload、角色与确定性 Claim 规格表。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from behavior.claim.proposal import ClaimKind, ClaimSemanticProposal
from behavior.evidence.content import (
    BehaviorModality,
    BehaviorRecordKind,
    BehaviorRole,
    BehaviorSemanticContent,
)
from behavior.evidence.payloads import (
    ActionEventPayload,
    ActivitySegmentPayload,
    CommunicationChannel,
    CoverageIntervalPayload,
    CoverageStatus,
    EnvironmentChangePayload,
    FeedbackPayload,
    FeedbackPolarity,
    FreeTextSemanticPayload,
    InteractionMode,
    InteractionSegmentPayload,
    PayloadCodec,
    PhaseHint,
    StateAssertionPayload,
    StateTransitionPayload,
    ToolCallPayload,
    ToolResultPayload,
    ToolResultStatus,
    UtteranceSegmentPayload,
)

RolePair = tuple[BehaviorRole, BehaviorRole | None]


@dataclass(frozen=True, slots=True)
class RolePolicy:
    allowed_pairs: frozenset[RolePair]
    distinct_counterparty: bool = False
    match_coverage_modality: bool = False

    def validate(self, content: BehaviorSemanticContent) -> None:
        pair = (content.subject_role, content.actor_role)
        if pair not in self.allowed_pairs:
            raise ValueError(f"{content.record_kind.value} role pair is not supported")
        if self.distinct_counterparty:
            if not isinstance(content.payload, InteractionSegmentPayload):
                raise TypeError("distinct counterparty policy requires Interaction payload")
            counterparty = content.payload.counterparty_role
            if counterparty is content.subject_role:
                raise ValueError("INTERACTION_SEGMENT counterparty must be distinct")
        if self.match_coverage_modality:
            if not isinstance(content.payload, CoverageIntervalPayload):
                raise TypeError("coverage modality policy requires Coverage payload")
            payload_modality = content.payload.modality
            if payload_modality is not content.modality:
                raise ValueError("coverage payload modality must match semantic modality")


@dataclass(frozen=True, slots=True)
class DeterministicClaimMapper:
    claim_kind: ClaimKind
    predicate_field: str | None = None
    predicate_value: str | None = None
    activity_field: str | None = None
    phase_field: str | None = None

    def map(self, content: BehaviorSemanticContent, codec: PayloadCodec) -> ClaimSemanticProposal:
        payload = content.payload
        predicate = (
            self.predicate_value
            if self.predicate_value is not None
            else getattr(payload, self.predicate_field or "")
        )
        if not isinstance(predicate, str):
            raise TypeError("deterministic mapper requires a text predicate")
        activity = None if self.activity_field is None else getattr(payload, self.activity_field)
        phase = None if self.phase_field is None else getattr(payload, self.phase_field)
        if isinstance(phase, Enum):
            phase = phase.value
        return ClaimSemanticProposal(
            claim_kind=self.claim_kind,
            semantic_family="behavior." + content.record_kind.value.casefold(),
            predicate=predicate,
            activity=activity,
            phase=phase,
            semantic_payload=codec.encode(payload),
            human_summary=None,
            local_alternative_group_id=None,
            normalizer_confidence=1.0,
        )


@dataclass(frozen=True, slots=True)
class RecordSpec:
    payload_codec: PayloadCodec
    role_policy: RolePolicy
    deterministic_mapper: DeterministicClaimMapper | None


def _pairs(subjects: set[BehaviorRole], actors: set[BehaviorRole | None]) -> frozenset[RolePair]:
    return frozenset((subject, actor) for subject in subjects for actor in actors)


def _labels(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or any(not isinstance(item, str) for item in value):
        raise TypeError("labels must be a string array")
    return tuple(value)


_SELF = frozenset(
    {
        (BehaviorRole.USER, BehaviorRole.USER),
        (BehaviorRole.AGENT, BehaviorRole.AGENT),
        (BehaviorRole.ROBOT, BehaviorRole.ROBOT),
    }
)
_STATE_SUBJECTS = {
    BehaviorRole.USER,
    BehaviorRole.AGENT,
    BehaviorRole.ROBOT,
    BehaviorRole.TOOL,
    BehaviorRole.ENVIRONMENT,
    BehaviorRole.SYSTEM,
}
_ALL_ROLE_PAIRS = _pairs(set(BehaviorRole), set(BehaviorRole) | {None})


def _spec(payload_type: type[Any], roles: RolePolicy, claim: ClaimKind | None, *,
          decoders: tuple[tuple[str, Callable[[object], object]], ...] = (),
          identifiers: tuple[str, ...] = (), references: tuple[str, ...] = (),
          texts: tuple[str, ...] = (), predicate_field: str | None = None,
          predicate_value: str | None = None, activity_field: str | None = None,
          phase_field: str | None = None) -> RecordSpec:
    codec = PayloadCodec(payload_type, decoders, frozenset(identifiers),
                         frozenset(references), frozenset(texts))
    mapper = None if claim is None else DeterministicClaimMapper(
        claim, predicate_field, predicate_value, activity_field, phase_field)
    return RecordSpec(codec, roles, mapper)


RECORD_SPECS: Mapping[BehaviorRecordKind, RecordSpec] = MappingProxyType(
    {
        BehaviorRecordKind.ACTIVITY_SEGMENT: _spec(
            ActivitySegmentPayload, RolePolicy(_SELF), ClaimKind.ACTIVITY,
            decoders=(("phase_hint", PhaseHint),), identifiers=("activity",),
            predicate_value="activity", activity_field="activity", phase_field="phase_hint"),
        BehaviorRecordKind.UTTERANCE_SEGMENT: _spec(
            UtteranceSegmentPayload, RolePolicy(_SELF), ClaimKind.UTTERANCE,
            decoders=(("interaction_mode", InteractionMode), ("communication_channel", CommunicationChannel)),
            identifiers=("language",), texts=("text",), predicate_value="utterance"),
        BehaviorRecordKind.STATE_ASSERTION: _spec(
            StateAssertionPayload, RolePolicy(_pairs(_STATE_SUBJECTS, {None})),
            ClaimKind.STATE_ASSERTION, identifiers=("state_name",), predicate_field="state_name"),
        BehaviorRecordKind.STATE_TRANSITION: _spec(
            StateTransitionPayload, RolePolicy(_pairs(_STATE_SUBJECTS, _STATE_SUBJECTS | {None})),
            ClaimKind.STATE_TRANSITION, identifiers=("state_name",), predicate_field="state_name"),
        BehaviorRecordKind.INTERACTION_SEGMENT: _spec(
            InteractionSegmentPayload, RolePolicy(_SELF, distinct_counterparty=True), ClaimKind.INTERACTION,
            decoders=(("counterparty_role", BehaviorRole), ("phase_hint", PhaseHint)),
            identifiers=("interaction_type",), predicate_field="interaction_type", phase_field="phase_hint"),
        BehaviorRecordKind.ACTION_EVENT: _spec(
            ActionEventPayload, RolePolicy(_SELF | {(BehaviorRole.SYSTEM, BehaviorRole.SYSTEM)}), ClaimKind.ACTION,
            identifiers=("action_name", "phase", "capability_ref", "target_ref"),
            predicate_field="action_name", activity_field="action_name", phase_field="phase"),
        BehaviorRecordKind.TOOL_CALL_EVENT: _spec(
            ToolCallPayload, RolePolicy(frozenset((BehaviorRole.TOOL, actor) for actor in
                (BehaviorRole.AGENT, BehaviorRole.ROBOT, BehaviorRole.SYSTEM))), ClaimKind.TOOL_CALL,
            identifiers=("tool_name", "tool_call_id", "capability_ref"),
            texts=("arguments_summary",), predicate_field="tool_name"),
        BehaviorRecordKind.TOOL_RESULT_EVENT: _spec(
            ToolResultPayload, RolePolicy(frozenset({(BehaviorRole.TOOL, BehaviorRole.TOOL)})),
            ClaimKind.TOOL_RESULT, decoders=(("status", ToolResultStatus),),
            identifiers=("tool_name", "tool_call_id"), references=("result_ref",),
            texts=("result_summary",), predicate_field="tool_name", phase_field="status"),
        BehaviorRecordKind.ENVIRONMENT_CHANGE: _spec(
            EnvironmentChangePayload, RolePolicy(frozenset((BehaviorRole.ENVIRONMENT, actor) for actor in
                (None, BehaviorRole.ENVIRONMENT, BehaviorRole.SYSTEM, BehaviorRole.TOOL))),
            ClaimKind.ENVIRONMENT_CHANGE, identifiers=("predicate",), predicate_field="predicate"),
        BehaviorRecordKind.COVERAGE_INTERVAL: _spec(
            CoverageIntervalPayload, RolePolicy(frozenset({(BehaviorRole.SYSTEM, BehaviorRole.SYSTEM)}),
                match_coverage_modality=True), ClaimKind.COVERAGE,
            decoders=(("modality", BehaviorModality), ("coverage_status", CoverageStatus)),
            identifiers=("coverage_scope_ref",), texts=("reason",),
            predicate_value="coverage", phase_field="coverage_status"),
        BehaviorRecordKind.FEEDBACK_EVENT: _spec(
            FeedbackPayload, RolePolicy(frozenset({(BehaviorRole.USER, BehaviorRole.USER),
                (BehaviorRole.SYSTEM, BehaviorRole.SYSTEM)})), ClaimKind.FEEDBACK,
            decoders=(("polarity", FeedbackPolarity),), identifiers=("feedback_kind", "target_ref"),
            references=("explicit_text_ref",), predicate_field="feedback_kind", phase_field="polarity"),
        BehaviorRecordKind.FREE_TEXT_SEMANTIC: _spec(
            FreeTextSemanticPayload, RolePolicy(_ALL_ROLE_PAIRS), None,
            decoders=(("labels", _labels),), identifiers=("language", "labels"), texts=("text",)),
    }
)


def record_spec(record_kind: BehaviorRecordKind) -> RecordSpec:
    return RECORD_SPECS[BehaviorRecordKind(record_kind)]


def payload_codec(payload: object) -> PayloadCodec:
    matches = tuple(
        spec.payload_codec
        for spec in RECORD_SPECS.values()
        if isinstance(payload, spec.payload_codec.payload_type)
    )
    if len(matches) != 1:
        raise TypeError("payload has no unique registered RecordSpec")
    return matches[0]
