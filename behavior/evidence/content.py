"""上游已经完成基础解析后的纯语义内容。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
from behavior.evidence.payloads import BehaviorPayload
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


@dataclass(frozen=True, slots=True)
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
        uncertainty = non_negative_int(
            self.event_time_uncertainty_ms,
            "semantic_content.event_time_uncertainty_ms",
        )
        evidence_refs = typed_tuple(
            self.evidence_refs,
            "semantic_content.evidence_refs",
            EvidenceReference,
            maximum_items=10_000,
        )
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
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "source_confidence", finite_score(self.source_confidence, "source_confidence"))
        object.__setattr__(self, "integrity", EvidenceIntegrity(self.integrity))


def content_to_dict(content: BehaviorSemanticContent) -> dict[str, Any]:
    from behavior.evidence.specs import payload_codec

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
        "payload": payload_codec(content.payload).encode(content.payload),
        "evidence_refs": tuple(_evidence_reference_to_dict(reference) for reference in content.evidence_refs),
        "source_confidence": content.source_confidence,
        "integrity": content.integrity.value,
        "schema_version": content.schema_version,
    }


def content_from_dict(value: object) -> BehaviorSemanticContent:
    from behavior.evidence.specs import record_spec

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
        payload=record_spec(kind).payload_codec.decode(data["payload"]),
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
