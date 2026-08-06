from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta

import pytest

from behavior.errors import (
    BehaviorOwnerConflictError,
    SemanticIngressError,
    SemanticRecordConflictError,
    SemanticRecordError,
)
from behavior.evidence.service import EvidenceService
from behavior.ingress import (
    ActionEventPayload,
    ActivitySegmentPayload,
    ClockSyncStatus,
    CoverageIntervalPayload,
    CoverageStatus,
    EnvironmentChangePayload,
    EvidenceKind,
    EvidenceReference,
    FreeTextSemanticPayload,
    IngressDecisionStatus,
    IngressTrustClass,
    InteractionCounterpartyRole,
    InteractionSegmentPayload,
    PhaseHint,
    SemanticActorRole,
    SemanticIngressAdapterRegistry,
    SemanticModality,
    SemanticRecordInput,
    SemanticRecordKind,
    SemanticRecordService,
    SemanticSubjectRole,
    SensorFactPayload,
    StateAssertionPayload,
    StateTransitionPayload,
    ToolResultPayload,
    UtteranceChannel,
    UtteranceSegmentPayload,
)
from behavior.owner import ConfirmedOwnerBinding
from foundation.integrity import canonical_json
from foundation.observability import NullObserver
from ModelClient.schema_validation import JSONSchemaValidationError, validate_json_schema
from tests.unit.behavior.conftest import (
    BASE_TIME,
    FakeAdapter,
    FakeClock,
    accepted_ingress,
    bind_record,
    digest,
    make_input,
)


def test_owner_identity_excludes_resolver_audit_fields() -> None:
    first = ConfirmedOwnerBinding("owner-a", "resolver-v1", BASE_TIME, digest("one"))
    second = ConfirmedOwnerBinding(
        "owner-a",
        "resolver-v2",
        BASE_TIME + timedelta(days=1),
        digest("two"),
    )
    assert first.owner_identity_digest == second.owner_identity_digest
    assert first.to_dict()["resolver_fingerprint"] != second.to_dict()["resolver_fingerprint"]


def test_store_binds_only_one_owner_identity(store, owner) -> None:
    record = bind_record(owner)
    EvidenceService(store, config=store.config.evidence, observer=NullObserver()).ingest(
        accepted_ingress(record),
    )
    other = ConfirmedOwnerBinding("owner-b", "resolver-v2", BASE_TIME, digest("other"))
    conflict = bind_record(other, make_input(sequence=1))
    with pytest.raises(BehaviorOwnerConflictError):
        EvidenceService(store, config=store.config.evidence, observer=NullObserver()).ingest(
            accepted_ingress(conflict),
        )


def test_evidence_reference_is_external_bounded_and_media_free() -> None:
    reference = EvidenceReference(
        reference="blob://perception/frame/42",
        evidence_kind=EvidenceKind.IMAGE_FRAME,
        digest=digest("frame"),
        event_time_start=BASE_TIME,
        event_time_end=BASE_TIME,
        media_type="image/jpeg",
        size_bytes=1234,
        source_system_ref="perception-runtime",
    )
    assert reference.evidence_kind is EvidenceKind.IMAGE_FRAME
    for invalid in ("data:image/jpeg;base64,AAAA", "AAAA==", b"bytes"):
        with pytest.raises(SemanticRecordError):
            EvidenceReference(
                reference=invalid,
                evidence_kind=EvidenceKind.IMAGE_FRAME,
                digest=digest("frame"),
                event_time_start=BASE_TIME,
                event_time_end=BASE_TIME,
                media_type="image/jpeg",
                size_bytes=1,
                source_system_ref="source",
            )
    with pytest.raises(SemanticRecordError):
        EvidenceReference(
            reference="blob://camera/frame-1",
            evidence_kind=EvidenceKind.IMAGE_FRAME,
            digest=digest("frame"),
            event_time_start=BASE_TIME,
            event_time_end=BASE_TIME,
            media_type="image/jpeg",
            size_bytes=9_223_372_036_854_775_808,
            source_system_ref="camera-edge",
        )


def test_record_kind_has_no_raw_media_ingress_values() -> None:
    values = {item.value for item in SemanticRecordKind}
    assert values.isdisjoint({"CAMERA_FRAME", "VIDEO_CLIP", "AUDIO_CLIP"})
    assert {item.value for item in EvidenceKind}.issuperset({"IMAGE_FRAME", "VIDEO_CLIP", "AUDIO_SEGMENT"})
    assert {item.value for item in SemanticModality} == {
        "VISION",
        "AUDIO",
        "TEXT",
        "SENSOR",
        "IMU",
        "LOCATION",
        "DEVICE",
        "ROBOT",
        "AGENT",
        "TOOL",
        "MULTIMODAL",
    }


def test_record_kind_payload_role_and_trust_matrix(owner) -> None:
    cases = (
        (
            SemanticRecordKind.OWNER_ACTIVITY_SEGMENT,
            ActivitySegmentPayload("walking", PhaseHint.IN_PROGRESS, {}),
            "OWNER",
            "OWNER",
            IngressTrustClass.MODEL_INFERRED,
            SemanticModality.VISION,
        ),
        (
            SemanticRecordKind.OWNER_UTTERANCE_SEGMENT,
            UtteranceSegmentPayload("hello", "en", UtteranceChannel.VOICE),
            "OWNER",
            "OWNER",
            IngressTrustClass.OWNER_EXPLICIT,
            SemanticModality.AUDIO,
        ),
        (
            SemanticRecordKind.OWNER_STATE_ASSERTION,
            StateAssertionPayload("awake", True),
            "OWNER",
            "SYSTEM",
            IngressTrustClass.SENSOR_INFERRED,
            SemanticModality.SENSOR,
        ),
        (
            SemanticRecordKind.OWNER_STATE_TRANSITION,
            StateTransitionPayload("presence", False, True),
            "OWNER",
            "SYSTEM",
            IngressTrustClass.SENSOR_INFERRED,
            SemanticModality.SENSOR,
        ),
        (
            SemanticRecordKind.OWNER_INTERACTION_SEGMENT,
            InteractionSegmentPayload(
                "handover",
                InteractionCounterpartyRole.ROBOT,
                PhaseHint.IN_PROGRESS,
                {},
            ),
            "OWNER",
            "OWNER",
            IngressTrustClass.MODEL_INFERRED,
            SemanticModality.VISION,
        ),
        (
            SemanticRecordKind.ROBOT_ACTION_EVENT,
            ActionEventPayload("wave", "completed", None, {}),
            "ROBOT",
            "ROBOT",
            IngressTrustClass.DIRECT_SYSTEM_LOG,
            SemanticModality.ROBOT,
        ),
        (
            SemanticRecordKind.AGENT_ACTION_EVENT,
            ActionEventPayload("notify", "completed", "ok", {}),
            "AGENT",
            "AGENT",
            IngressTrustClass.DIRECT_SYSTEM_LOG,
            SemanticModality.AGENT,
        ),
        (
            SemanticRecordKind.TOOL_RESULT_EVENT,
            ToolResultPayload("weather", "call-1", "ok", "blob://result/1", "sunny"),
            "TOOL",
            "TOOL",
            IngressTrustClass.DIRECT_SYSTEM_LOG,
            SemanticModality.TOOL,
        ),
        (
            SemanticRecordKind.OWNER_SENSOR_FACT,
            SensorFactPayload("heart_rate", 70, "bpm", "sample", {}),
            "OWNER",
            "SYSTEM",
            IngressTrustClass.DIRECT_DEVICE_FACT,
            SemanticModality.SENSOR,
        ),
        (
            SemanticRecordKind.ENVIRONMENT_SENSOR_FACT,
            SensorFactPayload("temperature", 22, "celsius", "sample", {}),
            "ENVIRONMENT",
            "SYSTEM",
            IngressTrustClass.DIRECT_DEVICE_FACT,
            SemanticModality.SENSOR,
        ),
        (
            SemanticRecordKind.ENVIRONMENT_CHANGE,
            EnvironmentChangePayload("light_changed", "off", "on", {}),
            "ENVIRONMENT",
            "SYSTEM",
            IngressTrustClass.DIRECT_DEVICE_FACT,
            SemanticModality.DEVICE,
        ),
        (
            SemanticRecordKind.COVERAGE_INTERVAL,
            CoverageIntervalPayload("VISION", CoverageStatus.UNKNOWN, "camera-repositioned"),
            "ENVIRONMENT",
            "SYSTEM",
            IngressTrustClass.DIRECT_DEVICE_FACT,
            SemanticModality.VISION,
        ),
        (
            SemanticRecordKind.FREE_TEXT_SEMANTIC,
            FreeTextSemanticPayload("bounded semantic", "en", ()),
            "OWNER",
            "OWNER",
            IngressTrustClass.MODEL_INFERRED,
            SemanticModality.TEXT,
        ),
    )
    for sequence, (kind, payload, subject, actor, trust, modality) in enumerate(cases):
        semantic_input = make_input(
            sequence=sequence,
            kind=kind,
            payload=payload,
            subject_role=subject,
            actor_role=actor,
            modality=modality,
        )
        assert bind_record(owner, semantic_input, trust=trust).semantic_input.payload == payload


def test_record_kind_rejects_incompatible_roles() -> None:
    with pytest.raises(SemanticRecordError):
        make_input(
            kind=SemanticRecordKind.ROBOT_ACTION_EVENT,
            payload=ActionEventPayload("wave", "completed", None, {}),
        )


def test_payload_discriminant_and_unknown_fields_are_strict() -> None:
    value = make_input().to_dict()
    value["payload"] = {"device_ref": "d", "state_name": "power", "value": "on", "unknown": 1}
    with pytest.raises(SemanticRecordError):
        SemanticRecordInput.model_validate(value)
    with pytest.raises((SemanticRecordError, ValueError)):
        make_input(payload=replace(make_input().payload, value={"prediction": "future"}))
    value = make_input().to_dict()
    value["record_kind"] = SemanticRecordKind.FREE_TEXT_SEMANTIC.value
    with pytest.raises(SemanticRecordError):
        SemanticRecordInput.model_validate(value)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object(), {"nested": object()}])
def test_payload_rejects_noncanonical_values(invalid: object) -> None:
    with pytest.raises((SemanticRecordError, TypeError, ValueError)):
        make_input(payload=replace(make_input().payload, value=invalid))


def test_external_input_schema_excludes_all_system_owned_fields() -> None:
    properties = SemanticRecordInput.model_json_schema()["properties"]
    assert set(properties).isdisjoint(
        {
            "semantic_record_id",
            "owner_identity_digest",
            "trust_class",
            "epistemic_class",
            "claim_id",
            "manifest_id",
            "ingested_at",
        }
    )


def test_external_input_json_schema_matches_payload_and_evidence_structure() -> None:
    evidence = EvidenceReference(
        reference="blob://camera/frame-1",
        evidence_kind=EvidenceKind.IMAGE_FRAME,
        digest=digest("frame"),
        event_time_start=BASE_TIME,
        event_time_end=BASE_TIME,
        media_type="image/jpeg",
        size_bytes=123,
        source_system_ref="camera-edge",
    )
    value = json.loads(canonical_json(make_input(evidence_refs=(evidence,)).to_dict()))
    validate_json_schema(value, SemanticRecordInput.model_json_schema())
    evidence_values = value["evidence_refs"]
    assert isinstance(evidence_values, list)
    assert isinstance(evidence_values[0], dict)
    evidence_values[0]["unknown"] = True
    with pytest.raises(JSONSchemaValidationError):
        validate_json_schema(value, SemanticRecordInput.model_json_schema())


def test_semantic_record_identity_is_deterministic_and_ignores_audit_track(owner) -> None:
    first = bind_record(owner, make_input(upstream_subject_ref="track-a"))
    replay = bind_record(
        owner, make_input(upstream_subject_ref="track-a"), ingested_at=BASE_TIME + timedelta(seconds=9)
    )
    changed_track = bind_record(owner, make_input(upstream_subject_ref="track-b"))
    assert first.semantic_record_id == replay.semantic_record_id
    assert first.semantic_digest == replay.semantic_digest
    assert first.content_digest != replay.content_digest
    assert changed_track.semantic_record_id != first.semantic_record_id
    assert changed_track.owner_identity_digest == first.owner_identity_digest


def test_same_record_identity_with_changed_system_trust_is_a_conflict(store, owner) -> None:
    semantic_input = make_input(
        kind=SemanticRecordKind.OWNER_SENSOR_FACT,
        payload=SensorFactPayload("heart_rate", 70, "bpm", None, {}),
        subject_role=SemanticSubjectRole.OWNER,
        actor_role=SemanticActorRole.SYSTEM,
        modality=SemanticModality.SENSOR,
    )
    direct = bind_record(owner, semantic_input, trust=IngressTrustClass.DIRECT_DEVICE_FACT)
    inferred = bind_record(owner, semantic_input, trust=IngressTrustClass.SENSOR_INFERRED)
    assert direct.semantic_record_id == inferred.semantic_record_id
    assert direct.semantic_digest != inferred.semantic_digest
    evidence = EvidenceService(store, config=store.config.evidence, observer=NullObserver())
    evidence.ingest(accepted_ingress(direct))
    with pytest.raises(SemanticRecordConflictError, match="identity conflicts"):
        evidence.ingest(accepted_ingress(inferred))


def test_registry_rejects_duplicates_and_unknown_adapter() -> None:
    registry = SemanticIngressAdapterRegistry()
    adapter = FakeAdapter(make_input())
    registry.register(adapter)
    with pytest.raises(SemanticIngressError):
        registry.register(adapter)
    with pytest.raises(SemanticIngressError):
        registry.get("missing")


def test_trust_capability_cannot_be_supplied_by_payload(store, owner) -> None:
    free_text = make_input(
        kind=SemanticRecordKind.FREE_TEXT_SEMANTIC,
        payload=FreeTextSemanticPayload("semantic description", "en", ()),
        modality=SemanticModality.TEXT,
        subject_role=SemanticSubjectRole.OWNER,
    )
    with pytest.raises(SemanticIngressError):
        FakeAdapter(
            free_text,
            trust=IngressTrustClass.DIRECT_DEVICE_FACT,
            allowed=(SemanticRecordKind.FREE_TEXT_SEMANTIC,),
        )
    assert "trust_class" not in free_text.to_dict()
    assert "epistemic_class" not in free_text.to_dict()


def test_owner_explicit_requires_speaker_binding() -> None:
    utterance = make_input(
        kind=SemanticRecordKind.OWNER_UTTERANCE_SEGMENT,
        payload=UtteranceSegmentPayload("hello", "en", UtteranceChannel.VOICE),
        modality=SemanticModality.AUDIO,
        subject_role=SemanticSubjectRole.OWNER,
        actor_role=SemanticActorRole.OWNER,
    )
    with pytest.raises(SemanticIngressError):
        FakeAdapter(
            utterance,
            trust=IngressTrustClass.OWNER_EXPLICIT,
            allowed=(SemanticRecordKind.OWNER_UTTERANCE_SEGMENT,),
        )


def test_clock_rejections_are_audited_without_record_or_watermark(store, owner) -> None:
    future = make_input(offset_seconds=1_000)
    registry = SemanticIngressAdapterRegistry()
    registry.register(FakeAdapter(future))
    service = SemanticRecordService(
        store,
        registry,
        config=store.config.ingress,
        clock=FakeClock(),
    )
    result = asyncio.run(service.prepare("fake_semantic", {}, owner_binding=owner))[0]
    assert result.accepted is None
    assert result.decision.status is IngressDecisionStatus.CLOCK_SKEW_REJECTED
    assert store.read_semantic_record(result.decision.semantic_record_id) is None
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM semantic_ingress_decisions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM evidence_watermarks").fetchone()[0] == 0


def test_past_and_uncertainty_bounds_are_enforced(store, owner) -> None:
    old = make_input(offset_seconds=-(store.config.ingress.max_past_event_age_seconds + 1))
    registry = SemanticIngressAdapterRegistry()
    registry.register(FakeAdapter(old))
    service = SemanticRecordService(store, registry, clock=FakeClock(), config=store.config.ingress)
    result = asyncio.run(service.prepare("fake_semantic", None, owner_binding=owner))[0]
    assert result.decision.status is IngressDecisionStatus.EVENT_TOO_OLD_REJECTED
    with pytest.raises(SemanticRecordError):
        make_input(uncertainty_ms=store.config.ingress.max_event_time_uncertainty_ms + 1)


def test_unsynchronized_clock_is_accepted_but_does_not_advance_watermark(store, owner) -> None:
    value = make_input(clock_sync_status=ClockSyncStatus.UNKNOWN)
    record = bind_record(owner, value)
    result = EvidenceService(store, config=store.config.evidence, observer=NullObserver()).ingest(
        accepted_ingress(record),
    )
    assert result.active_bundle is not None
    assert result.active_bundle.watermark is None


def test_unsynchronized_clock_cannot_bypass_committed_lateness(store, owner) -> None:
    service = EvidenceService(store, config=store.config.evidence, observer=NullObserver())
    trusted = bind_record(
        owner,
        make_input(sequence=0, offset_seconds=100, boundary_signal="END"),
    )
    assert service.ingest(accepted_ingress(trusted)).manifest_ids
    untrusted = bind_record(
        owner,
        make_input(
            sequence=1,
            offset_seconds=0,
            clock_sync_status=ClockSyncStatus.UNKNOWN,
        ),
    )
    result = service.ingest(accepted_ingress(untrusted))
    assert result.status.name == "LATE_REJECTED"
    assert store.read_semantic_record(untrusted.semantic_record_id) is None
