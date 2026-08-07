"""上游已经完成基础解析后的纯语义内容。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from behavior._validation import (
    finite_score,
    identifier,
    identifier_tuple,
    non_negative_int,
    optional_identifier,
    parse_utc,
    strict_object,
    strict_utc,
    typed_tuple,
    utc_text,
)
from behavior.evidence.payloads import (
    BehaviorPayload,
    CoverageIntervalPayload,
    InteractionSegmentPayload,
    payload_from_dict,
    payload_to_dict,
    payload_type_for,
)
from behavior.evidence.refs import EvidenceKind, EvidenceReference

SEMANTIC_CONTENT_SCHEMA_VERSION = "behavior_semantic_content_v1"


class BehaviorRole(str, Enum):
    USER = "USER"
    AGENT = "AGENT"
    ROBOT = "ROBOT"
    TOOL = "TOOL"
    ENVIRONMENT = "ENVIRONMENT"
    SYSTEM = "SYSTEM"
    OTHER_ANONYMOUS = "OTHER_ANONYMOUS"


class BehaviorModality(str, Enum):
    VISION = "VISION"
    AUDIO = "AUDIO"
    TEXT = "TEXT"
    SENSOR = "SENSOR"
    LOCATION = "LOCATION"
    DEVICE = "DEVICE"
    AGENT = "AGENT"
    ROBOT = "ROBOT"
    TOOL = "TOOL"
    MULTIMODAL = "MULTIMODAL"
    SYSTEM = "SYSTEM"


class ClockSyncStatus(str, Enum):
    SYNCHRONIZED = "SYNCHRONIZED"
    OFFSET_ESTIMATED = "OFFSET_ESTIMATED"
    UNSYNCHRONIZED = "UNSYNCHRONIZED"
    UNKNOWN = "UNKNOWN"


class EvidenceIntegrity(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


class BehaviorRecordKind(str, Enum):
    ACTIVITY_SEGMENT = "ACTIVITY_SEGMENT"
    UTTERANCE_SEGMENT = "UTTERANCE_SEGMENT"
    STATE_ASSERTION = "STATE_ASSERTION"
    STATE_TRANSITION = "STATE_TRANSITION"
    INTERACTION_SEGMENT = "INTERACTION_SEGMENT"
    ACTION_EVENT = "ACTION_EVENT"
    TOOL_CALL_EVENT = "TOOL_CALL_EVENT"
    TOOL_RESULT_EVENT = "TOOL_RESULT_EVENT"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    COVERAGE_INTERVAL = "COVERAGE_INTERVAL"
    FEEDBACK_EVENT = "FEEDBACK_EVENT"
    FREE_TEXT_SEMANTIC = "FREE_TEXT_SEMANTIC"


_SELF_ROLES = frozenset({BehaviorRole.USER, BehaviorRole.AGENT, BehaviorRole.ROBOT})
_STATE_SUBJECTS = frozenset(
    {
        BehaviorRole.USER,
        BehaviorRole.AGENT,
        BehaviorRole.ROBOT,
        BehaviorRole.TOOL,
        BehaviorRole.ENVIRONMENT,
        BehaviorRole.SYSTEM,
    }
)


def validate_record_roles(
    record_kind: BehaviorRecordKind,
    subject_role: BehaviorRole,
    actor_role: BehaviorRole | None,
    payload: BehaviorPayload,
) -> None:
    kind = BehaviorRecordKind(record_kind)
    subject = BehaviorRole(subject_role)
    actor = None if actor_role is None else BehaviorRole(actor_role)
    if kind in {BehaviorRecordKind.ACTIVITY_SEGMENT, BehaviorRecordKind.UTTERANCE_SEGMENT}:
        if subject not in _SELF_ROLES or actor is not subject:
            raise ValueError(f"{kind.value} requires matching USER, AGENT, or ROBOT roles")
    elif kind is BehaviorRecordKind.STATE_ASSERTION:
        if subject not in _STATE_SUBJECTS or actor is not None:
            raise ValueError("STATE_ASSERTION requires a supported subject and actor=None")
    elif kind is BehaviorRecordKind.STATE_TRANSITION:
        if subject not in _STATE_SUBJECTS or actor not in _STATE_SUBJECTS | {None}:
            raise ValueError("STATE_TRANSITION role pair is not supported")
    elif kind is BehaviorRecordKind.INTERACTION_SEGMENT:
        if not isinstance(payload, InteractionSegmentPayload):
            raise TypeError("INTERACTION_SEGMENT requires InteractionSegmentPayload")
        if actor is None or actor is not subject or subject is payload.counterparty_role:
            raise ValueError("INTERACTION_SEGMENT requires actor=subject and a distinct counterparty")
    elif kind is BehaviorRecordKind.ACTION_EVENT:
        allowed = {BehaviorRole.USER, BehaviorRole.AGENT, BehaviorRole.ROBOT, BehaviorRole.SYSTEM}
        if subject not in allowed or actor is not subject:
            raise ValueError("ACTION_EVENT requires a supported matching role pair")
    elif kind is BehaviorRecordKind.TOOL_CALL_EVENT:
        if subject is not BehaviorRole.TOOL or actor not in {
            BehaviorRole.AGENT,
            BehaviorRole.ROBOT,
            BehaviorRole.SYSTEM,
        }:
            raise ValueError("TOOL_CALL_EVENT requires subject=TOOL and a runtime actor")
    elif kind is BehaviorRecordKind.TOOL_RESULT_EVENT:
        if subject is not BehaviorRole.TOOL or actor is not BehaviorRole.TOOL:
            raise ValueError("TOOL_RESULT_EVENT requires TOOL/TOOL")
    elif kind is BehaviorRecordKind.ENVIRONMENT_CHANGE:
        if subject is not BehaviorRole.ENVIRONMENT or actor not in {
            None,
            BehaviorRole.ENVIRONMENT,
            BehaviorRole.SYSTEM,
            BehaviorRole.TOOL,
        }:
            raise ValueError("ENVIRONMENT_CHANGE role pair is not supported")
    elif kind is BehaviorRecordKind.COVERAGE_INTERVAL:
        if subject is not BehaviorRole.SYSTEM or actor is not BehaviorRole.SYSTEM:
            raise ValueError("COVERAGE_INTERVAL requires SYSTEM/SYSTEM")
    elif kind is BehaviorRecordKind.FEEDBACK_EVENT:
        if (subject, actor) not in {
            (BehaviorRole.USER, BehaviorRole.USER),
            (BehaviorRole.SYSTEM, BehaviorRole.SYSTEM),
        }:
            raise ValueError("FEEDBACK_EVENT role pair is not supported")


@dataclass(frozen=True)
class BehaviorSemanticContent:
    record_kind: BehaviorRecordKind
    subject_role: BehaviorRole
    actor_role: BehaviorRole | None
    modality: BehaviorModality
    event_time_start: datetime
    event_time_end: datetime
    event_time_uncertainty_ms: int
    clock_domain: str
    clock_sync_status: ClockSyncStatus
    scene_ref: str | None
    location_ref: str | None
    object_refs: tuple[str, ...]
    entity_refs: tuple[str, ...]
    payload: BehaviorPayload
    evidence_refs: tuple[EvidenceReference, ...]
    source_confidence: float
    integrity: EvidenceIntegrity
    schema_version: str = SEMANTIC_CONTENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        kind = BehaviorRecordKind(self.record_kind)
        subject = BehaviorRole(self.subject_role)
        actor = None if self.actor_role is None else BehaviorRole(self.actor_role)
        modality = BehaviorModality(self.modality)
        start = strict_utc(self.event_time_start, "semantic_content.event_time_start")
        end = strict_utc(self.event_time_end, "semantic_content.event_time_end")
        if end < start:
            raise ValueError("semantic content end cannot precede start")
        uncertainty = non_negative_int(
            self.event_time_uncertainty_ms,
            "semantic_content.event_time_uncertainty_ms",
        )
        if not isinstance(self.payload, payload_type_for(kind)):
            raise TypeError(f"{kind.value} has the wrong payload type")
        validate_record_roles(kind, subject, actor, self.payload)
        evidence_refs = typed_tuple(
            self.evidence_refs,
            "semantic_content.evidence_refs",
            EvidenceReference,
            maximum_items=10_000,
        )
        expanded_start = start - timedelta(milliseconds=uncertainty)
        expanded_end = end + timedelta(milliseconds=uncertainty)
        for reference in evidence_refs:
            if reference.event_time_end < expanded_start or reference.event_time_start > expanded_end:
                raise ValueError("EvidenceReference time does not overlap semantic time uncertainty")
        if isinstance(self.payload, CoverageIntervalPayload) and self.payload.modality is not modality:
            raise ValueError("coverage payload modality must match semantic modality")
        if self.schema_version != SEMANTIC_CONTENT_SCHEMA_VERSION:
            raise ValueError("unsupported semantic content schema version")
        object.__setattr__(self, "record_kind", kind)
        object.__setattr__(self, "subject_role", subject)
        object.__setattr__(self, "actor_role", actor)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "event_time_start", start)
        object.__setattr__(self, "event_time_end", end)
        object.__setattr__(self, "event_time_uncertainty_ms", uncertainty)
        object.__setattr__(self, "clock_domain", identifier(self.clock_domain, "semantic_content.clock_domain"))
        object.__setattr__(self, "clock_sync_status", ClockSyncStatus(self.clock_sync_status))
        object.__setattr__(self, "scene_ref", optional_identifier(self.scene_ref, "semantic_content.scene_ref"))
        object.__setattr__(
            self,
            "location_ref",
            optional_identifier(self.location_ref, "semantic_content.location_ref"),
        )
        object.__setattr__(
            self,
            "object_refs",
            identifier_tuple(self.object_refs, "semantic_content.object_refs", maximum_items=10_000),
        )
        object.__setattr__(
            self,
            "entity_refs",
            identifier_tuple(self.entity_refs, "semantic_content.entity_refs", maximum_items=10_000),
        )
        object.__setattr__(self, "source_confidence", finite_score(self.source_confidence, "source_confidence"))
        object.__setattr__(self, "integrity", EvidenceIntegrity(self.integrity))


def content_to_dict(content: BehaviorSemanticContent) -> dict[str, Any]:
    return {
        "record_kind": content.record_kind.value,
        "subject_role": content.subject_role.value,
        "actor_role": None if content.actor_role is None else content.actor_role.value,
        "modality": content.modality.value,
        "event_time_start": utc_text(content.event_time_start),
        "event_time_end": utc_text(content.event_time_end),
        "event_time_uncertainty_ms": content.event_time_uncertainty_ms,
        "clock_domain": content.clock_domain,
        "clock_sync_status": content.clock_sync_status.value,
        "scene_ref": content.scene_ref,
        "location_ref": content.location_ref,
        "object_refs": content.object_refs,
        "entity_refs": content.entity_refs,
        "payload": payload_to_dict(content.payload),
        "evidence_refs": tuple(_evidence_reference_to_dict(reference) for reference in content.evidence_refs),
        "source_confidence": content.source_confidence,
        "integrity": content.integrity.value,
        "schema_version": content.schema_version,
    }


def content_from_dict(value: object) -> BehaviorSemanticContent:
    fields = frozenset(
        {
            "record_kind",
            "subject_role",
            "actor_role",
            "modality",
            "event_time_start",
            "event_time_end",
            "event_time_uncertainty_ms",
            "clock_domain",
            "clock_sync_status",
            "scene_ref",
            "location_ref",
            "object_refs",
            "entity_refs",
            "payload",
            "evidence_refs",
            "source_confidence",
            "integrity",
            "schema_version",
        }
    )
    data = strict_object(value, "semantic_content", fields)
    kind = BehaviorRecordKind(data["record_kind"])
    object_refs = _tuple_value(data["object_refs"], "semantic_content.object_refs")
    entity_refs = _tuple_value(data["entity_refs"], "semantic_content.entity_refs")
    evidence_values = _tuple_value(data["evidence_refs"], "semantic_content.evidence_refs")
    return BehaviorSemanticContent(
        record_kind=kind,
        subject_role=BehaviorRole(data["subject_role"]),
        actor_role=None if data["actor_role"] is None else BehaviorRole(data["actor_role"]),
        modality=BehaviorModality(data["modality"]),
        event_time_start=parse_utc(data["event_time_start"], "semantic_content.event_time_start"),
        event_time_end=parse_utc(data["event_time_end"], "semantic_content.event_time_end"),
        event_time_uncertainty_ms=data["event_time_uncertainty_ms"],
        clock_domain=data["clock_domain"],
        clock_sync_status=ClockSyncStatus(data["clock_sync_status"]),
        scene_ref=data["scene_ref"],
        location_ref=data["location_ref"],
        object_refs=tuple(object_refs),
        entity_refs=tuple(entity_refs),
        payload=payload_from_dict(kind, data["payload"]),
        evidence_refs=tuple(_evidence_reference_from_dict(item) for item in evidence_values),
        source_confidence=data["source_confidence"],
        integrity=EvidenceIntegrity(data["integrity"]),
        schema_version=data["schema_version"],
    )


def _tuple_value(value: object, field_name: str) -> tuple[Any, ...] | list[Any]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"{field_name} must be an array")
    return value


def _evidence_reference_to_dict(reference: EvidenceReference) -> dict[str, Any]:
    return {
        "reference": reference.reference,
        "evidence_kind": reference.evidence_kind.value,
        "digest": reference.digest,
        "event_time_start": utc_text(reference.event_time_start),
        "event_time_end": utc_text(reference.event_time_end),
        "media_type": reference.media_type,
        "size_bytes": reference.size_bytes,
        "source_system_ref": reference.source_system_ref,
    }


def _evidence_reference_from_dict(value: object) -> EvidenceReference:
    fields = frozenset(
        {
            "reference",
            "evidence_kind",
            "digest",
            "event_time_start",
            "event_time_end",
            "media_type",
            "size_bytes",
            "source_system_ref",
        }
    )
    data = strict_object(value, "evidence_reference", fields)
    return EvidenceReference(
        reference=data["reference"],
        evidence_kind=EvidenceKind(data["evidence_kind"]),
        digest=data["digest"],
        event_time_start=parse_utc(data["event_time_start"], "evidence_reference.event_time_start"),
        event_time_end=parse_utc(data["event_time_end"], "evidence_reference.event_time_end"),
        media_type=data["media_type"],
        size_bytes=data["size_bytes"],
        source_system_ref=data["source_system_ref"],
    )


__all__ = [
    "BehaviorModality",
    "BehaviorRecordKind",
    "BehaviorRole",
    "BehaviorSemanticContent",
    "ClockSyncStatus",
    "EvidenceIntegrity",
    "content_from_dict",
    "content_to_dict",
    "validate_record_roles",
]
