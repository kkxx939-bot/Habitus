from __future__ import annotations

from dataclasses import fields, replace
from datetime import timedelta

import pytest

from behavior.evidence import (
    ActionEventPayload,
    ActivitySegmentPayload,
    BehaviorAdapterCapability,
    BehaviorModality,
    BehaviorOriginKind,
    BehaviorRecordKind,
    BehaviorRole,
    BehaviorSemanticContent,
    BehaviorSemanticInput,
    BehaviorSourceDescriptor,
    BehaviorSourceTrust,
    BehaviorTimeMode,
    CausalRef,
    CausalRefKind,
    ClockSyncStatus,
    CommunicationChannel,
    CorrelationRef,
    CoverageIntervalPayload,
    CoverageStatus,
    EnvironmentChangePayload,
    EvidenceIntegrity,
    EvidenceKind,
    EvidenceReference,
    FeedbackPayload,
    FeedbackPolarity,
    FreeTextSemanticPayload,
    InteractionMode,
    InteractionSegmentPayload,
    PhaseHint,
    ProjectionRef,
    SourceEventRef,
    StateAssertionPayload,
    StateTransitionPayload,
    StreamRef,
    ToolCallPayload,
    ToolResultPayload,
    ToolResultStatus,
    UtteranceSegmentPayload,
)
from behavior.evidence.content import content_from_dict, content_to_dict, validate_record_roles
from behavior.evidence.payloads import payload_from_dict
from behavior.evidence.provenance import descriptor_from_dict, descriptor_to_dict
from tests.unit.behavior.conftest import BASE_TIME, digest, source_descriptor


def content(
    kind: BehaviorRecordKind,
    payload,
    *,
    subject: BehaviorRole,
    actor: BehaviorRole | None,
    modality: BehaviorModality = BehaviorModality.SYSTEM,
    evidence_refs=(),
    uncertainty: int = 0,
) -> BehaviorSemanticContent:
    return BehaviorSemanticContent(
        record_kind=kind,
        subject_role=subject,
        actor_role=actor,
        modality=modality,
        event_time_start=BASE_TIME,
        event_time_end=BASE_TIME + timedelta(seconds=1),
        event_time_uncertainty_ms=uncertainty,
        clock_domain="utc",
        clock_sync_status=ClockSyncStatus.SYNCHRONIZED,
        scene_ref=None,
        location_ref=None,
        object_refs=(),
        entity_refs=(),
        payload=payload,
        evidence_refs=tuple(evidence_refs),
        source_confidence=0.75,
        integrity=EvidenceIntegrity.COMPLETE,
    )


@pytest.mark.parametrize(
    ("kind", "payload", "subject", "actor", "modality"),
    [
        (BehaviorRecordKind.ACTIVITY_SEGMENT, ActivitySegmentPayload("walking", PhaseHint.IN_PROGRESS, {}), BehaviorRole.USER, BehaviorRole.USER, BehaviorModality.VISION),
        (BehaviorRecordKind.UTTERANCE_SEGMENT, UtteranceSegmentPayload("hello", "en", InteractionMode.DIALOGUE, CommunicationChannel.VOICE), BehaviorRole.AGENT, BehaviorRole.AGENT, BehaviorModality.AUDIO),
        (BehaviorRecordKind.STATE_ASSERTION, StateAssertionPayload("door", "open", {}), BehaviorRole.ENVIRONMENT, None, BehaviorModality.SENSOR),
        (BehaviorRecordKind.STATE_TRANSITION, StateTransitionPayload("door", "closed", "open", {}), BehaviorRole.ENVIRONMENT, BehaviorRole.SYSTEM, BehaviorModality.SENSOR),
        (BehaviorRecordKind.INTERACTION_SEGMENT, InteractionSegmentPayload("handoff", BehaviorRole.ROBOT, PhaseHint.STARTED, {}), BehaviorRole.USER, BehaviorRole.USER, BehaviorModality.VISION),
        (BehaviorRecordKind.ACTION_EVENT, ActionEventPayload("move", "STARTED", None, None, None, {}), BehaviorRole.ROBOT, BehaviorRole.ROBOT, BehaviorModality.ROBOT),
        (BehaviorRecordKind.TOOL_CALL_EVENT, ToolCallPayload("search", "call-1", digest("args"), "bounded", None), BehaviorRole.TOOL, BehaviorRole.AGENT, BehaviorModality.TOOL),
        (BehaviorRecordKind.TOOL_RESULT_EVENT, ToolResultPayload("search", "call-1", ToolResultStatus.SUCCESS, "tool://result/1", digest("result"), "ok"), BehaviorRole.TOOL, BehaviorRole.TOOL, BehaviorModality.TOOL),
        (BehaviorRecordKind.ENVIRONMENT_CHANGE, EnvironmentChangePayload("light", "off", "on", {}), BehaviorRole.ENVIRONMENT, None, BehaviorModality.SENSOR),
        (BehaviorRecordKind.COVERAGE_INTERVAL, CoverageIntervalPayload(BehaviorModality.VISION, CoverageStatus.COVERED, None, None), BehaviorRole.SYSTEM, BehaviorRole.SYSTEM, BehaviorModality.VISION),
        (BehaviorRecordKind.FEEDBACK_EVENT, FeedbackPayload("explicit", None, FeedbackPolarity.POSITIVE, None, {}), BehaviorRole.USER, BehaviorRole.USER, BehaviorModality.TEXT),
        (BehaviorRecordKind.FREE_TEXT_SEMANTIC, FreeTextSemanticPayload("unstructured", "en", ("label",)), BehaviorRole.OTHER_ANONYMOUS, None, BehaviorModality.TEXT),
    ],
)
def test_every_record_kind_has_one_strict_payload_and_canonical_round_trip(
    kind, payload, subject, actor, modality
) -> None:
    value = content(kind, payload, subject=subject, actor=actor, modality=modality)
    assert content_from_dict(content_to_dict(value)) == value
    wrong_payload = (
        FreeTextSemanticPayload("wrong", None, ())
        if kind is not BehaviorRecordKind.FREE_TEXT_SEMANTIC
        else StateAssertionPayload("x", 1, {})
    )
    with pytest.raises(TypeError):
        content(kind, wrong_payload, subject=subject, actor=actor, modality=modality)


def test_source_identity_stream_generation_and_content_digest_are_distinct() -> None:
    source = SourceEventRef("camera", "event-7")
    assert source.identity_digest == SourceEventRef("camera", "event-7").identity_digest
    assert source.identity_digest != digest("actual-media")
    assert StreamRef("camera", "stream", 1) != StreamRef("camera", "stream", 2)
    descriptor = source_descriptor(content_digest=digest("actual-media"))
    assert descriptor.source_content_digest != descriptor.source_event_ref.identity_digest
    assert descriptor_from_dict(descriptor_to_dict(descriptor)) == descriptor
    correlation = CorrelationRef("scene", "tea", "root")
    causal = CausalRef(CausalRefKind.TOOL_RESULT, "tool-result-1", digest("tool-result"))
    projection = ProjectionRef("conversation", "message-1", digest("message"))
    assert correlation.root_value == "root"
    assert causal.kind is CausalRefKind.TOOL_RESULT
    projected = replace(
        descriptor,
        origin_kind=BehaviorOriginKind.CONVERSATION_PROJECTION,
        projection_ref=projection,
    )
    assert descriptor_from_dict(descriptor_to_dict(projected)).projection_ref == projection
    with pytest.raises(ValueError):
        replace(descriptor, projection_ref=projection)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SourceEventRef("test", "person@example.com"),
        lambda: SourceEventRef("test", "bad\x00value"),
        lambda: StreamRef("test", "stream", -1),
        lambda: CorrelationRef("test", "", None),
        lambda: CausalRef(CausalRefKind.OTHER, "x", "ABC"),
        lambda: ProjectionRef("test", "projection", "bad"),
    ],
)
def test_reference_objects_reject_pii_controls_and_invalid_bounds(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_evidence_reference_is_external_media_free_utc_and_overlap_checked() -> None:
    reference = EvidenceReference(
        "s3://bucket/frame",
        EvidenceKind.IMAGE_FRAME,
        digest("frame"),
        BASE_TIME,
        BASE_TIME + timedelta(seconds=1),
        "image/jpeg",
        100,
        "device://camera/1",
    )
    assert content(
        BehaviorRecordKind.ACTIVITY_SEGMENT,
        ActivitySegmentPayload("walking", PhaseHint.UNKNOWN, {}),
        subject=BehaviorRole.USER,
        actor=BehaviorRole.USER,
        modality=BehaviorModality.VISION,
        evidence_refs=(reference,),
    ).evidence_refs == (reference,)
    with pytest.raises(ValueError, match="overlap"):
        content(
            BehaviorRecordKind.ACTIVITY_SEGMENT,
            ActivitySegmentPayload("walking", PhaseHint.UNKNOWN, {}),
            subject=BehaviorRole.USER,
            actor=BehaviorRole.USER,
            modality=BehaviorModality.VISION,
            evidence_refs=(
                EvidenceReference(
                    "s3://bucket/old",
                    EvidenceKind.IMAGE_FRAME,
                    digest("old"),
                    BASE_TIME - timedelta(hours=1),
                    BASE_TIME - timedelta(minutes=59),
                ),
            ),
        )
    with pytest.raises(ValueError):
        EvidenceReference("data:image/png;base64,AAAA", EvidenceKind.IMAGE_FRAME, digest("x"), BASE_TIME, BASE_TIME)
    with pytest.raises(ValueError):
        EvidenceReference(
            "s3://bucket/frame",
            EvidenceKind.IMAGE_FRAME,
            digest("x"),
            BASE_TIME,
            BASE_TIME,
            size_bytes=2**63,
        )


@pytest.mark.parametrize(
    ("kind", "payload", "subject", "actor"),
    [
        (BehaviorRecordKind.STATE_ASSERTION, StateAssertionPayload("x", 1, {}), BehaviorRole.USER, BehaviorRole.SYSTEM),
        (BehaviorRecordKind.TOOL_CALL_EVENT, ToolCallPayload("x", "id", digest("a"), None, None), BehaviorRole.TOOL, BehaviorRole.USER),
        (BehaviorRecordKind.TOOL_RESULT_EVENT, ToolResultPayload("x", "id", ToolResultStatus.SUCCESS, None, digest("r"), None), BehaviorRole.TOOL, BehaviorRole.AGENT),
        (BehaviorRecordKind.ENVIRONMENT_CHANGE, EnvironmentChangePayload("x", 1, 2, {}), BehaviorRole.ENVIRONMENT, BehaviorRole.USER),
        (BehaviorRecordKind.COVERAGE_INTERVAL, CoverageIntervalPayload(BehaviorModality.SYSTEM, CoverageStatus.BLIND, None, None), BehaviorRole.USER, BehaviorRole.USER),
    ],
)
def test_role_compatibility_is_mechanical_and_actor_none_is_preserved(kind, payload, subject, actor) -> None:
    with pytest.raises(ValueError):
        content(kind, payload, subject=subject, actor=actor)
    state = content(
        BehaviorRecordKind.STATE_ASSERTION,
        StateAssertionPayload("ready", True, {}),
        subject=BehaviorRole.SYSTEM,
        actor=None,
    )
    assert state.actor_role is None


def test_record_kind_role_matrix_is_exhaustive() -> None:
    actors = (None, *tuple(BehaviorRole))
    payloads = {
        BehaviorRecordKind.ACTIVITY_SEGMENT: ActivitySegmentPayload("x", PhaseHint.UNKNOWN, {}),
        BehaviorRecordKind.UTTERANCE_SEGMENT: UtteranceSegmentPayload(
            "x", None, InteractionMode.UNKNOWN, CommunicationChannel.OTHER
        ),
        BehaviorRecordKind.STATE_ASSERTION: StateAssertionPayload("x", 1, {}),
        BehaviorRecordKind.STATE_TRANSITION: StateTransitionPayload("x", 1, 2, {}),
        BehaviorRecordKind.INTERACTION_SEGMENT: InteractionSegmentPayload(
            "x", BehaviorRole.ROBOT, PhaseHint.UNKNOWN, {}
        ),
        BehaviorRecordKind.ACTION_EVENT: ActionEventPayload("x", "UNKNOWN", None, None, None, {}),
        BehaviorRecordKind.TOOL_CALL_EVENT: ToolCallPayload("x", "id", digest("x"), None, None),
        BehaviorRecordKind.TOOL_RESULT_EVENT: ToolResultPayload(
            "x", "id", ToolResultStatus.UNKNOWN, None, digest("x"), None
        ),
        BehaviorRecordKind.ENVIRONMENT_CHANGE: EnvironmentChangePayload("x", 1, 2, {}),
        BehaviorRecordKind.COVERAGE_INTERVAL: CoverageIntervalPayload(
            BehaviorModality.SYSTEM, CoverageStatus.UNKNOWN, None, None
        ),
        BehaviorRecordKind.FEEDBACK_EVENT: FeedbackPayload(
            "x", None, FeedbackPolarity.NEUTRAL, None, {}
        ),
        BehaviorRecordKind.FREE_TEXT_SEMANTIC: FreeTextSemanticPayload("x", None, ()),
    }
    self_roles = {BehaviorRole.USER, BehaviorRole.AGENT, BehaviorRole.ROBOT}
    state_roles = set(BehaviorRole) - {BehaviorRole.OTHER_ANONYMOUS}
    action_roles = self_roles | {BehaviorRole.SYSTEM}

    def expected(kind, subject, actor):
        if kind in {BehaviorRecordKind.ACTIVITY_SEGMENT, BehaviorRecordKind.UTTERANCE_SEGMENT}:
            return subject in self_roles and actor is subject
        if kind is BehaviorRecordKind.STATE_ASSERTION:
            return subject in state_roles and actor is None
        if kind is BehaviorRecordKind.STATE_TRANSITION:
            return subject in state_roles and (actor is None or actor in state_roles)
        if kind is BehaviorRecordKind.INTERACTION_SEGMENT:
            return actor is subject and subject is not BehaviorRole.ROBOT
        if kind is BehaviorRecordKind.ACTION_EVENT:
            return subject in action_roles and actor is subject
        if kind is BehaviorRecordKind.TOOL_CALL_EVENT:
            return subject is BehaviorRole.TOOL and actor in {
                BehaviorRole.AGENT,
                BehaviorRole.ROBOT,
                BehaviorRole.SYSTEM,
            }
        if kind is BehaviorRecordKind.TOOL_RESULT_EVENT:
            return subject is actor is BehaviorRole.TOOL
        if kind is BehaviorRecordKind.ENVIRONMENT_CHANGE:
            return subject is BehaviorRole.ENVIRONMENT and actor in {
                None,
                BehaviorRole.ENVIRONMENT,
                BehaviorRole.SYSTEM,
                BehaviorRole.TOOL,
            }
        if kind is BehaviorRecordKind.COVERAGE_INTERVAL:
            return subject is actor is BehaviorRole.SYSTEM
        if kind is BehaviorRecordKind.FEEDBACK_EVENT:
            return (subject, actor) in {
                (BehaviorRole.USER, BehaviorRole.USER),
                (BehaviorRole.SYSTEM, BehaviorRole.SYSTEM),
            }
        return True

    for kind, payload in payloads.items():
        for subject in BehaviorRole:
            for actor in actors:
                if expected(kind, subject, actor):
                    validate_record_roles(kind, subject, actor, payload)
                else:
                    with pytest.raises(ValueError):
                        validate_record_roles(kind, subject, actor, payload)


def test_payload_unknown_fields_strict_types_nonfinite_binary_and_recursion_fail() -> None:
    with pytest.raises(ValueError, match="unknown"):
        payload_from_dict(
            BehaviorRecordKind.ACTIVITY_SEGMENT,
            {"activity": "walk", "phase_hint": "STARTED", "attributes": {}, "extra": 1},
        )
    with pytest.raises((TypeError, ValueError)):
        ActivitySegmentPayload("walk", PhaseHint.STARTED, {"score": float("nan")})
    with pytest.raises((TypeError, ValueError)):
        ActivitySegmentPayload("walk", PhaseHint.STARTED, {"score": float("inf")})
    with pytest.raises(TypeError):
        StateAssertionPayload("state", b"bytes", {})
    with pytest.raises(TypeError):
        StateAssertionPayload("state", "base64:AAAA", {})
    recursive = {}
    recursive["self"] = recursive
    with pytest.raises(ValueError, match="recursive"):
        StateAssertionPayload("state", recursive, {})
    with pytest.raises(TypeError):
        ToolCallPayload("tool", "id", digest("x"), 123, None)
    for reserved in (
        "claim_id",
        "evidence_record_id",
        "source_trust",
        "content_digest",
        "claim_sequence",
        "owner_identity",
    ):
        with pytest.raises(ValueError, match="reserved"):
            StateAssertionPayload("state", {"nested": {reserved: "forbidden"}}, {})

    allowed_semantic_names = {
        "user_action": "walk",
        "account_status": "active",
        "tenant_mode": "shared",
    }
    assert StateAssertionPayload("state", allowed_semantic_names, {}).value == allowed_semantic_names

    valid = content(
        BehaviorRecordKind.STATE_ASSERTION,
        StateAssertionPayload("ready", True, {}),
        subject=BehaviorRole.SYSTEM,
        actor=None,
    )
    with pytest.raises(ValueError):
        replace(valid, event_time_end=BASE_TIME - timedelta(microseconds=1))
    with pytest.raises(ValueError):
        replace(valid, event_time_start=BASE_TIME.replace(tzinfo=None))
    with pytest.raises(ValueError):
        replace(valid, event_time_uncertainty_ms=-1)


def test_semantic_input_has_only_content_and_source_and_capability_binds_trust() -> None:
    assert {item.name for item in fields(BehaviorSemanticInput)} == {"content", "source"}
    assert {item.name for item in fields(BehaviorSourceDescriptor)} == {
        "source_event_ref",
        "stream_ref",
        "source_sequence",
        "source_item_index",
        "origin_kind",
        "source_ref",
        "source_content_digest",
        "parent_source_event_refs",
        "correlation_refs",
        "causal_refs",
        "projection_ref",
    }
    capability = BehaviorAdapterCapability(
        BehaviorSourceTrust.USER_EXPLICIT,
        BehaviorTimeMode.LIVE,
        (BehaviorOriginKind.DIRECT_AMBIENT_ASR,),
        (BehaviorRecordKind.UTTERANCE_SEGMENT,),
        (BehaviorModality.AUDIO,),
        ((BehaviorRole.USER, BehaviorRole.USER),),
        10,
    )
    assert capability.source_trust is BehaviorSourceTrust.USER_EXPLICIT
    with pytest.raises(ValueError):
        BehaviorAdapterCapability(
            BehaviorSourceTrust.MODEL_INFERRED,
            BehaviorTimeMode.LIVE,
            (BehaviorOriginKind.DIRECT_AMBIENT_ASR,),
            (BehaviorRecordKind.UTTERANCE_SEGMENT,),
            (BehaviorModality.AUDIO,),
            ((BehaviorRole.USER, BehaviorRole.USER),),
            10,
        )
