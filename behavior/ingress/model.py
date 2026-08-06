"""外部语义输入与系统绑定后的 Owner-scoped 记录。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from behavior._validation import (
    finite_score,
    identifier,
    identifier_tuple,
    non_negative_int,
    optional_identifier,
    parse_utc,
    require_fields,
    sha256_digest,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.config import IngressConfig
from behavior.errors import SemanticRecordError
from behavior.ingress.evidence_ref import EvidenceReference
from behavior.ingress.payloads import SemanticPayload, payload_from_dict, payload_json_schema, validate_payload
from behavior.ingress.trust import (
    IngressTrustClass,
    ProducerFingerprint,
    require_record_trust_compatibility,
)
from behavior.owner import ConfirmedOwnerBinding
from foundation.integrity import canonical_digest, canonical_json

SEMANTIC_RECORD_SCHEMA_VERSION = "2"


class SemanticRecordKind(str, Enum):
    OWNER_ACTIVITY_SEGMENT = "OWNER_ACTIVITY_SEGMENT"
    OWNER_UTTERANCE_SEGMENT = "OWNER_UTTERANCE_SEGMENT"
    OWNER_STATE_ASSERTION = "OWNER_STATE_ASSERTION"
    OWNER_STATE_TRANSITION = "OWNER_STATE_TRANSITION"
    OWNER_INTERACTION_SEGMENT = "OWNER_INTERACTION_SEGMENT"
    ROBOT_ACTION_EVENT = "ROBOT_ACTION_EVENT"
    AGENT_ACTION_EVENT = "AGENT_ACTION_EVENT"
    TOOL_RESULT_EVENT = "TOOL_RESULT_EVENT"
    OWNER_SENSOR_FACT = "OWNER_SENSOR_FACT"
    ENVIRONMENT_SENSOR_FACT = "ENVIRONMENT_SENSOR_FACT"
    DEVICE_STATE = "DEVICE_STATE"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    COVERAGE_INTERVAL = "COVERAGE_INTERVAL"
    FREE_TEXT_SEMANTIC = "FREE_TEXT_SEMANTIC"


class SemanticModality(str, Enum):
    VISION = "VISION"
    AUDIO = "AUDIO"
    TEXT = "TEXT"
    SENSOR = "SENSOR"
    IMU = "IMU"
    LOCATION = "LOCATION"
    DEVICE = "DEVICE"
    ROBOT = "ROBOT"
    AGENT = "AGENT"
    TOOL = "TOOL"
    MULTIMODAL = "MULTIMODAL"


class SemanticSubjectRole(str, Enum):
    OWNER = "OWNER"
    ROBOT = "ROBOT"
    AGENT = "AGENT"
    TOOL = "TOOL"
    ENVIRONMENT = "ENVIRONMENT"
    SYSTEM = "SYSTEM"
    OTHER_ANONYMOUS = "OTHER_ANONYMOUS"


class SemanticActorRole(str, Enum):
    OWNER = "OWNER"
    ROBOT = "ROBOT"
    AGENT = "AGENT"
    TOOL = "TOOL"
    ENVIRONMENT = "ENVIRONMENT"
    SYSTEM = "SYSTEM"
    OTHER_ANONYMOUS = "OTHER_ANONYMOUS"


class RecordIntegrity(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


class ClockSyncStatus(str, Enum):
    SYNCHRONIZED = "SYNCHRONIZED"
    OFFSET_ESTIMATED = "OFFSET_ESTIMATED"
    UNSYNCHRONIZED = "UNSYNCHRONIZED"
    UNKNOWN = "UNKNOWN"


class BoundarySignal(str, Enum):
    CONTINUE = "CONTINUE"
    END = "END"


class IngressDecisionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REPLAYED = "REPLAYED"
    LATE_REJECTED = "LATE_REJECTED"
    CLOCK_SKEW_REJECTED = "CLOCK_SKEW_REJECTED"
    EVENT_TOO_OLD_REJECTED = "EVENT_TOO_OLD_REJECTED"
    CAPACITY_REJECTED = "CAPACITY_REJECTED"


_RECORD_ROLES: dict[
    SemanticRecordKind,
    tuple[SemanticSubjectRole, frozenset[SemanticActorRole]],
] = {
    SemanticRecordKind.OWNER_ACTIVITY_SEGMENT: (
        SemanticSubjectRole.OWNER,
        frozenset({SemanticActorRole.OWNER}),
    ),
    SemanticRecordKind.OWNER_UTTERANCE_SEGMENT: (
        SemanticSubjectRole.OWNER,
        frozenset({SemanticActorRole.OWNER}),
    ),
    SemanticRecordKind.OWNER_STATE_ASSERTION: (
        SemanticSubjectRole.OWNER,
        frozenset({SemanticActorRole.OWNER, SemanticActorRole.SYSTEM}),
    ),
    SemanticRecordKind.OWNER_STATE_TRANSITION: (
        SemanticSubjectRole.OWNER,
        frozenset({SemanticActorRole.OWNER, SemanticActorRole.SYSTEM}),
    ),
    SemanticRecordKind.OWNER_INTERACTION_SEGMENT: (
        SemanticSubjectRole.OWNER,
        frozenset({SemanticActorRole.OWNER}),
    ),
    SemanticRecordKind.ROBOT_ACTION_EVENT: (
        SemanticSubjectRole.ROBOT,
        frozenset({SemanticActorRole.ROBOT}),
    ),
    SemanticRecordKind.AGENT_ACTION_EVENT: (
        SemanticSubjectRole.AGENT,
        frozenset({SemanticActorRole.AGENT}),
    ),
    SemanticRecordKind.TOOL_RESULT_EVENT: (
        SemanticSubjectRole.TOOL,
        frozenset({SemanticActorRole.TOOL}),
    ),
    SemanticRecordKind.OWNER_SENSOR_FACT: (
        SemanticSubjectRole.OWNER,
        frozenset({SemanticActorRole.SYSTEM, SemanticActorRole.ENVIRONMENT}),
    ),
    SemanticRecordKind.ENVIRONMENT_SENSOR_FACT: (
        SemanticSubjectRole.ENVIRONMENT,
        frozenset({SemanticActorRole.SYSTEM, SemanticActorRole.ENVIRONMENT}),
    ),
    SemanticRecordKind.DEVICE_STATE: (
        SemanticSubjectRole.ENVIRONMENT,
        frozenset({SemanticActorRole.SYSTEM, SemanticActorRole.ENVIRONMENT}),
    ),
    SemanticRecordKind.ENVIRONMENT_CHANGE: (
        SemanticSubjectRole.ENVIRONMENT,
        frozenset({SemanticActorRole.SYSTEM, SemanticActorRole.ENVIRONMENT}),
    ),
    SemanticRecordKind.COVERAGE_INTERVAL: (
        SemanticSubjectRole.ENVIRONMENT,
        frozenset({SemanticActorRole.SYSTEM}),
    ),
}


def _require_record_roles(
    kind: SemanticRecordKind,
    subject: SemanticSubjectRole,
    actor: SemanticActorRole,
) -> None:
    expected = _RECORD_ROLES.get(kind)
    if expected is None:
        return
    expected_subject, allowed_actors = expected
    if subject is not expected_subject or actor not in allowed_actors:
        raise ValueError("record kind is incompatible with its semantic subject or actor role")


@dataclass(frozen=True, init=False)
class SemanticRecordInput:
    stream_id: str
    source_sequence: int
    record_kind: SemanticRecordKind
    subject_role: SemanticSubjectRole
    actor_role: SemanticActorRole
    modality: SemanticModality
    event_time_start: datetime
    event_time_end: datetime
    event_time_uncertainty_ms: int
    clock_domain: str
    clock_sync_status: ClockSyncStatus
    correlation_id: str
    boundary_signal: BoundarySignal
    scene_ref: str | None
    upstream_subject_ref: str | None
    object_refs: tuple[str, ...]
    entity_refs: tuple[str, ...]
    location_ref: str | None
    payload: SemanticPayload
    evidence_refs: tuple[EvidenceReference, ...]
    source_confidence: float
    integrity: RecordIntegrity
    schema_version: str

    def __init__(
        self,
        *,
        stream_id: object,
        source_sequence: object,
        record_kind: SemanticRecordKind | str,
        subject_role: SemanticSubjectRole | str,
        actor_role: SemanticActorRole | str,
        modality: SemanticModality | str,
        event_time_start: object,
        event_time_end: object,
        event_time_uncertainty_ms: object,
        clock_domain: object,
        clock_sync_status: ClockSyncStatus | str,
        correlation_id: object,
        boundary_signal: BoundarySignal | str,
        scene_ref: object,
        upstream_subject_ref: object,
        object_refs: object,
        entity_refs: object,
        location_ref: object,
        payload: object,
        evidence_refs: object,
        source_confidence: object,
        integrity: RecordIntegrity | str,
        schema_version: object = SEMANTIC_RECORD_SCHEMA_VERSION,
        config: IngressConfig | None = None,
    ) -> None:
        limits = config or IngressConfig()
        try:
            resolved_kind = SemanticRecordKind(record_kind)
            start = strict_utc(event_time_start, "event_time_start")
            end = strict_utc(event_time_end, "event_time_end")
            if end < start:
                raise ValueError("event_time_end cannot precede event_time_start")
            uncertainty = non_negative_int(event_time_uncertainty_ms, "event_time_uncertainty_ms")
            if uncertainty > limits.max_event_time_uncertainty_ms:
                raise ValueError("event_time_uncertainty_ms exceeds its configured boundary")
            if isinstance(evidence_refs, str | bytes) or not isinstance(evidence_refs, Sequence):
                raise TypeError("evidence_refs must be a sequence")
            refs = tuple(
                EvidenceReference.from_dict(item.to_dict(), config=limits)
                if isinstance(item, EvidenceReference)
                else EvidenceReference.from_dict(item, config=limits)
                for item in evidence_refs
            )
            if len(refs) > limits.max_evidence_refs:
                raise ValueError("evidence_refs exceeds its configured boundary")
            ref_identities = tuple((item.reference, item.digest) for item in refs)
            if len(set(ref_identities)) != len(ref_identities):
                raise ValueError("evidence_refs must not contain duplicates")
            resolved_payload = validate_payload(resolved_kind, payload, config=limits)
            if len(canonical_json(resolved_payload.to_dict())) > limits.max_payload_chars:
                raise ValueError("payload exceeds its configured canonical boundary")
            version = identifier(schema_version, "schema_version", maximum=32)
            if version != SEMANTIC_RECORD_SCHEMA_VERSION:
                raise ValueError("semantic record input schema version is unsupported")
            resolved_subject = SemanticSubjectRole(subject_role)
            resolved_actor = SemanticActorRole(actor_role)
            _require_record_roles(resolved_kind, resolved_subject, resolved_actor)
            resolved_sequence = non_negative_int(source_sequence, "source_sequence")
            if resolved_sequence > 9_223_372_036_854_775_807:
                raise ValueError("source_sequence exceeds the durable integer boundary")
            values: tuple[tuple[str, object], ...] = (
                ("stream_id", identifier(stream_id, "stream_id")),
                ("source_sequence", resolved_sequence),
                ("record_kind", resolved_kind),
                ("subject_role", resolved_subject),
                ("actor_role", resolved_actor),
                ("modality", SemanticModality(modality)),
                ("event_time_start", start),
                ("event_time_end", end),
                ("event_time_uncertainty_ms", uncertainty),
                ("clock_domain", identifier(clock_domain, "clock_domain")),
                ("clock_sync_status", ClockSyncStatus(clock_sync_status)),
                ("correlation_id", identifier(correlation_id, "correlation_id")),
                ("boundary_signal", BoundarySignal(boundary_signal)),
                ("scene_ref", optional_identifier(scene_ref, "scene_ref")),
                (
                    "upstream_subject_ref",
                    optional_identifier(upstream_subject_ref, "upstream_subject_ref"),
                ),
                (
                    "object_refs",
                    identifier_tuple(
                        object_refs,
                        "object_refs",
                        maximum_items=limits.max_object_refs,
                    ),
                ),
                (
                    "entity_refs",
                    identifier_tuple(
                        entity_refs,
                        "entity_refs",
                        maximum_items=limits.max_entity_refs,
                    ),
                ),
                ("location_ref", optional_identifier(location_ref, "location_ref")),
                ("payload", resolved_payload),
                ("evidence_refs", refs),
                ("source_confidence", finite_score(source_confidence, "source_confidence")),
                ("integrity", RecordIntegrity(integrity)),
                ("schema_version", version),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SemanticRecordError):
                raise
            raise SemanticRecordError(str(exc)) from exc
        for name, value in values:
            object.__setattr__(self, name, value)

    @property
    def payload_digest(self) -> str:
        return canonical_digest(self.payload.to_dict())

    @property
    def projection_chars(self) -> int:
        return len(canonical_json(self.payload.to_dict()))

    @property
    def stable_sort_key(self) -> tuple[datetime, datetime, str, int, str]:
        return (
            self.event_time_start,
            self.event_time_end,
            self.stream_id,
            self.source_sequence,
            self.payload_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stream_id": self.stream_id,
            "source_sequence": self.source_sequence,
            "record_kind": self.record_kind.value,
            "subject_role": self.subject_role.value,
            "actor_role": self.actor_role.value,
            "modality": self.modality.value,
            "event_time_start": utc_text(self.event_time_start),
            "event_time_end": utc_text(self.event_time_end),
            "event_time_uncertainty_ms": self.event_time_uncertainty_ms,
            "clock_domain": self.clock_domain,
            "clock_sync_status": self.clock_sync_status.value,
            "correlation_id": self.correlation_id,
            "boundary_signal": self.boundary_signal.value,
            "scene_ref": self.scene_ref,
            "upstream_subject_ref": self.upstream_subject_ref,
            "object_refs": self.object_refs,
            "entity_refs": self.entity_refs,
            "location_ref": self.location_ref,
            "payload": self.payload.to_dict(),
            "evidence_refs": tuple(item.to_dict() for item in self.evidence_refs),
            "source_confidence": self.source_confidence,
            "integrity": self.integrity.value,
            "schema_version": self.schema_version,
        }

    @classmethod
    def model_validate(cls, value: object, *, config: IngressConfig | None = None) -> SemanticRecordInput:
        fields = frozenset(
            {
                "stream_id",
                "source_sequence",
                "record_kind",
                "subject_role",
                "actor_role",
                "modality",
                "event_time_start",
                "event_time_end",
                "event_time_uncertainty_ms",
                "clock_domain",
                "clock_sync_status",
                "correlation_id",
                "boundary_signal",
                "scene_ref",
                "upstream_subject_ref",
                "object_refs",
                "entity_refs",
                "location_ref",
                "payload",
                "evidence_refs",
                "source_confidence",
                "integrity",
                "schema_version",
            }
        )
        try:
            data = strict_fields(value, "semantic_record_input", fields)
            require_fields(data, "semantic_record_input", fields)
            for name in ("object_refs", "entity_refs", "evidence_refs"):
                if not isinstance(data[name], list | tuple):
                    raise TypeError(f"semantic_record_input.{name} must be an array")
            kind = SemanticRecordKind(data["record_kind"])
            values = {key: data[key] for key in fields if key != "payload"}
            values["event_time_start"] = parse_utc(data["event_time_start"], "event_time_start")
            values["event_time_end"] = parse_utc(data["event_time_end"], "event_time_end")
            return cls(
                **values,
                payload=payload_from_dict(kind, data["payload"], config=config),
                config=config,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SemanticRecordError):
                raise
            raise SemanticRecordError(str(exc)) from exc

    @classmethod
    def model_json_schema(cls, *, config: IngressConfig | None = None) -> dict[str, object]:
        limits = config or IngressConfig()
        identifier_schema: dict[str, object] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 256,
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}$",
        }
        nullable_identifier = {"anyOf": [identifier_schema, {"type": "null"}]}
        fields = {
            "stream_id": identifier_schema,
            "source_sequence": {"type": "integer", "minimum": 0},
            "record_kind": {"type": "string", "enum": [item.value for item in SemanticRecordKind]},
            "subject_role": {"type": "string", "enum": [item.value for item in SemanticSubjectRole]},
            "actor_role": {"type": "string", "enum": [item.value for item in SemanticActorRole]},
            "modality": {"type": "string", "enum": [item.value for item in SemanticModality]},
            "event_time_start": {"type": "string", "format": "date-time"},
            "event_time_end": {"type": "string", "format": "date-time"},
            "event_time_uncertainty_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": limits.max_event_time_uncertainty_ms,
            },
            "clock_domain": identifier_schema,
            "clock_sync_status": {"type": "string", "enum": [item.value for item in ClockSyncStatus]},
            "correlation_id": identifier_schema,
            "boundary_signal": {"type": "string", "enum": [item.value for item in BoundarySignal]},
            "scene_ref": nullable_identifier,
            "upstream_subject_ref": nullable_identifier,
            "object_refs": {
                "type": "array",
                "maxItems": limits.max_object_refs,
                "uniqueItems": True,
                "items": identifier_schema,
            },
            "entity_refs": {
                "type": "array",
                "maxItems": limits.max_entity_refs,
                "uniqueItems": True,
                "items": identifier_schema,
            },
            "location_ref": nullable_identifier,
            "payload": {"type": "object"},
            "evidence_refs": {
                "type": "array",
                "maxItems": limits.max_evidence_refs,
                "items": EvidenceReference.model_json_schema(config=limits),
            },
            "source_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "integrity": {"type": "string", "enum": [item.value for item in RecordIntegrity]},
            "schema_version": {"const": SEMANTIC_RECORD_SCHEMA_VERSION},
        }
        variants = [
            {
                "type": "object",
                "required": ["record_kind", "payload"],
                "properties": {
                    "record_kind": {"const": kind.value},
                    "payload": payload_json_schema(kind, config=limits),
                },
            }
            for kind in SemanticRecordKind
        ]
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": sorted(fields),
            "properties": fields,
            "allOf": [{"oneOf": variants}],
        }


@dataclass(frozen=True)
class SemanticRecordInputBatch:
    records: tuple[SemanticRecordInput, ...]

    def __init__(self, records: Sequence[SemanticRecordInput], *, config: IngressConfig | None = None) -> None:
        limits = config or IngressConfig()
        if isinstance(records, str | bytes) or not isinstance(records, Sequence):
            raise TypeError("records must be a sequence")
        resolved = tuple(records)
        if not resolved or len(resolved) > limits.max_batch_size:
            raise SemanticRecordError("semantic input batch size is outside its configured boundary")
        validated = tuple(SemanticRecordInput.model_validate(item.to_dict(), config=limits) for item in resolved)
        object.__setattr__(self, "records", validated)


@dataclass(frozen=True, init=False)
class OwnerScopedSemanticRecord:
    semantic_record_id: str
    owner_binding: ConfirmedOwnerBinding
    owner_identity_digest: str
    producer_fingerprint: ProducerFingerprint
    ingress_trust_class: IngressTrustClass
    ingested_at: datetime
    payload_digest: str
    canonical_digest: str
    semantic_input: SemanticRecordInput

    def __init__(
        self,
        *,
        semantic_input: SemanticRecordInput,
        owner_binding: ConfirmedOwnerBinding,
        producer_fingerprint: ProducerFingerprint,
        ingress_trust_class: IngressTrustClass,
        ingested_at: datetime,
    ) -> None:
        if not isinstance(semantic_input, SemanticRecordInput):
            raise TypeError("semantic_input must be SemanticRecordInput")
        if not isinstance(owner_binding, ConfirmedOwnerBinding):
            raise TypeError("owner_binding must be ConfirmedOwnerBinding")
        if not isinstance(producer_fingerprint, ProducerFingerprint):
            raise TypeError("producer_fingerprint must be ProducerFingerprint")
        trust = IngressTrustClass(ingress_trust_class)
        require_record_trust_compatibility(semantic_input.record_kind, trust)
        timestamp = strict_utc(ingested_at, "ingested_at")
        from behavior.ingress.identity import SemanticRecordIdentityFactory

        stable_payload = {
            "ingress_trust_class": trust.value,
            "owner_identity_digest": owner_binding.owner_identity_digest,
            "producer_fingerprint": producer_fingerprint.digest,
            "semantic_input": semantic_input.to_dict(),
        }
        record_id = SemanticRecordIdentityFactory.create(
            owner_identity_digest=owner_binding.owner_identity_digest,
            producer_fingerprint=producer_fingerprint.digest,
            semantic_input=semantic_input,
        )
        object.__setattr__(self, "semantic_record_id", record_id)
        object.__setattr__(self, "owner_binding", owner_binding)
        object.__setattr__(self, "owner_identity_digest", owner_binding.owner_identity_digest)
        object.__setattr__(self, "producer_fingerprint", producer_fingerprint)
        object.__setattr__(self, "ingress_trust_class", trust)
        object.__setattr__(self, "ingested_at", timestamp)
        object.__setattr__(self, "payload_digest", semantic_input.payload_digest)
        object.__setattr__(self, "canonical_digest", canonical_digest(stable_payload))
        object.__setattr__(self, "semantic_input", semantic_input)

    @property
    def stable_sort_key(self) -> tuple[datetime, datetime, str, int, str]:
        value = self.semantic_input
        return (
            value.event_time_start,
            value.event_time_end,
            value.stream_id,
            value.source_sequence,
            self.semantic_record_id,
        )

    @property
    def projection_chars(self) -> int:
        return self.semantic_input.projection_chars

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic_record_id": self.semantic_record_id,
            "owner_binding": self.owner_binding.to_dict(),
            "owner_identity_digest": self.owner_identity_digest,
            "producer_fingerprint": self.producer_fingerprint.to_dict(),
            "ingress_trust_class": self.ingress_trust_class.value,
            "ingested_at": utc_text(self.ingested_at),
            "payload_digest": self.payload_digest,
            "canonical_digest": self.canonical_digest,
            "semantic_input": self.semantic_input.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object, *, config: IngressConfig | None = None) -> OwnerScopedSemanticRecord:
        fields = frozenset(
            {
                "semantic_record_id",
                "owner_binding",
                "owner_identity_digest",
                "producer_fingerprint",
                "ingress_trust_class",
                "ingested_at",
                "payload_digest",
                "canonical_digest",
                "semantic_input",
            }
        )
        try:
            data = strict_fields(value, "owner_scoped_semantic_record", fields)
            require_fields(data, "owner_scoped_semantic_record", fields)
            result = cls(
                semantic_input=SemanticRecordInput.model_validate(data["semantic_input"], config=config),
                owner_binding=ConfirmedOwnerBinding.from_dict(data["owner_binding"]),
                producer_fingerprint=ProducerFingerprint.from_dict(data["producer_fingerprint"]),
                ingress_trust_class=IngressTrustClass(data["ingress_trust_class"]),
                ingested_at=parse_utc(data["ingested_at"], "ingested_at"),
            )
            for name in (
                "semantic_record_id",
                "owner_identity_digest",
                "payload_digest",
                "canonical_digest",
            ):
                if data[name] != getattr(result, name):
                    raise SemanticRecordError(f"{name} does not match deterministic content")
            return result
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SemanticRecordError):
                raise
            raise SemanticRecordError(str(exc)) from exc


@dataclass(frozen=True, init=False)
class IngressDecision:
    decision_id: str
    status: IngressDecisionStatus
    reason_code: str
    semantic_record_id: str
    owner_identity_digest: str
    producer_fingerprint: str
    stream_id: str
    source_sequence: int
    record_kind: SemanticRecordKind
    decided_at: datetime
    content_digest: str

    def __init__(
        self,
        *,
        status: IngressDecisionStatus | str,
        reason_code: object,
        record: OwnerScopedSemanticRecord,
        decided_at: object,
    ) -> None:
        if not isinstance(record, OwnerScopedSemanticRecord):
            raise TypeError("record must be OwnerScopedSemanticRecord")
        resolved_status = IngressDecisionStatus(status)
        resolved_reason = identifier(reason_code, "reason_code")
        timestamp = strict_utc(decided_at, "decided_at")
        stable = {
            "status": resolved_status.value,
            "reason_code": resolved_reason,
            "semantic_record_id": record.semantic_record_id,
            "owner_identity_digest": record.owner_identity_digest,
            "producer_fingerprint": record.producer_fingerprint.digest,
            "stream_id": record.semantic_input.stream_id,
            "source_sequence": record.semantic_input.source_sequence,
            "record_kind": record.semantic_input.record_kind.value,
            "schema_version": SEMANTIC_RECORD_SCHEMA_VERSION,
        }
        digest = canonical_digest(stable)
        object.__setattr__(self, "decision_id", "ingress_" + digest)
        object.__setattr__(self, "status", resolved_status)
        object.__setattr__(self, "reason_code", resolved_reason)
        object.__setattr__(self, "semantic_record_id", record.semantic_record_id)
        object.__setattr__(self, "owner_identity_digest", record.owner_identity_digest)
        object.__setattr__(self, "producer_fingerprint", record.producer_fingerprint.digest)
        object.__setattr__(self, "stream_id", record.semantic_input.stream_id)
        object.__setattr__(self, "source_sequence", record.semantic_input.source_sequence)
        object.__setattr__(self, "record_kind", record.semantic_input.record_kind)
        object.__setattr__(self, "decided_at", timestamp)
        object.__setattr__(self, "content_digest", digest)

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "semantic_record_id": self.semantic_record_id,
            "owner_identity_digest": self.owner_identity_digest,
            "producer_fingerprint": self.producer_fingerprint,
            "stream_id": self.stream_id,
            "source_sequence": self.source_sequence,
            "record_kind": self.record_kind.value,
            "decided_at": utc_text(self.decided_at),
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> IngressDecision:
        fields = frozenset(
            {
                "decision_id",
                "status",
                "reason_code",
                "semantic_record_id",
                "owner_identity_digest",
                "producer_fingerprint",
                "stream_id",
                "source_sequence",
                "record_kind",
                "decided_at",
                "content_digest",
            }
        )
        try:
            data = strict_fields(value, "ingress_decision", fields)
            require_fields(data, "ingress_decision", fields)
            status = IngressDecisionStatus(data["status"])
            reason = identifier(data["reason_code"], "reason_code")
            semantic_record_id = identifier(data["semantic_record_id"], "semantic_record_id")
            owner_digest = sha256_digest(data["owner_identity_digest"], "owner_identity_digest")
            producer_digest = sha256_digest(data["producer_fingerprint"], "producer_fingerprint")
            stream_id = identifier(data["stream_id"], "stream_id")
            source_sequence = non_negative_int(data["source_sequence"], "source_sequence")
            record_kind = SemanticRecordKind(data["record_kind"])
            decided_at = parse_utc(data["decided_at"], "decided_at")
            stable = {
                "status": status.value,
                "reason_code": reason,
                "semantic_record_id": semantic_record_id,
                "owner_identity_digest": owner_digest,
                "producer_fingerprint": producer_digest,
                "stream_id": stream_id,
                "source_sequence": source_sequence,
                "record_kind": record_kind.value,
                "schema_version": SEMANTIC_RECORD_SCHEMA_VERSION,
            }
            digest = canonical_digest(stable)
            result = object.__new__(cls)
            object.__setattr__(result, "decision_id", "ingress_" + digest)
            object.__setattr__(result, "status", status)
            object.__setattr__(result, "reason_code", reason)
            object.__setattr__(result, "semantic_record_id", semantic_record_id)
            object.__setattr__(result, "owner_identity_digest", owner_digest)
            object.__setattr__(result, "producer_fingerprint", producer_digest)
            object.__setattr__(result, "stream_id", stream_id)
            object.__setattr__(result, "source_sequence", source_sequence)
            object.__setattr__(result, "record_kind", record_kind)
            object.__setattr__(result, "decided_at", decided_at)
            object.__setattr__(result, "content_digest", digest)
            if data["decision_id"] != result.decision_id or data["content_digest"] != result.content_digest:
                raise SemanticRecordError("IngressDecision deterministic identity mismatch")
            return result
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SemanticRecordError):
                raise
            raise SemanticRecordError(str(exc)) from exc


__all__ = [
    "BoundarySignal",
    "ClockSyncStatus",
    "IngressDecision",
    "IngressDecisionStatus",
    "OwnerScopedSemanticRecord",
    "RecordIntegrity",
    "SEMANTIC_RECORD_SCHEMA_VERSION",
    "SemanticActorRole",
    "SemanticModality",
    "SemanticRecordInput",
    "SemanticRecordInputBatch",
    "SemanticRecordKind",
    "SemanticSubjectRole",
]
