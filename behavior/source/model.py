"""异构来源的不可变规范记录。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from behavior._validation import (
    bounded_text,
    external_reference,
    identifier,
    identifier_tuple,
    json_snapshot,
    non_negative_int,
    optional_bounded_text,
    optional_identifier,
    parse_utc,
    require_fields,
    sha256_digest,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.config import SourceConfig
from behavior.errors import SourceRecordError
from behavior.owner import ConfirmedOwnerBinding
from foundation.integrity import canonical_digest, canonical_json

SOURCE_RECORD_SCHEMA_VERSION = "1"


class SourceType(str, Enum):
    CAMERA_FRAME = "CAMERA_FRAME"
    VIDEO_CLIP = "VIDEO_CLIP"
    VLM_OUTPUT = "VLM_OUTPUT"
    AUDIO_CLIP = "AUDIO_CLIP"
    AUDIO_SEMANTIC = "AUDIO_SEMANTIC"
    ASR_SEGMENT = "ASR_SEGMENT"
    SENSOR_SAMPLE = "SENSOR_SAMPLE"
    SENSOR_WINDOW = "SENSOR_WINDOW"
    DEVICE_STATE = "DEVICE_STATE"
    ROBOT_ACTION_LOG = "ROBOT_ACTION_LOG"
    AGENT_EVENT = "AGENT_EVENT"
    CONVERSATION_REFERENCE = "CONVERSATION_REFERENCE"
    TOOL_RESULT_REFERENCE = "TOOL_RESULT_REFERENCE"
    COVERAGE_SIGNAL = "COVERAGE_SIGNAL"
    UPSTREAM_SEMANTIC = "UPSTREAM_SEMANTIC"


class Modality(str, Enum):
    VISION = "VISION"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    TEXT = "TEXT"
    IMU = "IMU"
    LOCATION = "LOCATION"
    DEVICE_STATE = "DEVICE_STATE"
    ROBOT_ACTION = "ROBOT_ACTION"
    AGENT = "AGENT"
    MULTIMODAL = "MULTIMODAL"


class CaptureState(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    BLIND = "BLIND"
    STREAM_END = "STREAM_END"


@dataclass(frozen=True, init=False)
class SourceRecord:
    source_record_id: str
    stream_id: str
    source_sequence: int
    source_type: SourceType
    modality: Modality
    producer_ref: str
    event_time_start: datetime
    event_time_end: datetime
    ingested_at: datetime
    payload_ref: str
    payload_digest: str
    payload_media_type: str
    payload_size_bytes: int
    semantic_text: str | None
    semantic_data: Mapping[str, Any]
    scene_ref: str | None
    track_refs: tuple[str, ...]
    entity_refs: tuple[str, ...]
    correlation_key: str
    capture_state: CaptureState
    owner_binding: ConfirmedOwnerBinding
    attributes: Mapping[str, Any]
    schema_version: str

    def __init__(
        self,
        *,
        stream_id: object,
        source_sequence: object,
        source_type: SourceType | str,
        modality: Modality | str,
        producer_ref: object,
        event_time_start: object,
        event_time_end: object,
        ingested_at: object,
        payload_ref: object,
        payload_digest: object,
        payload_media_type: object,
        payload_size_bytes: object,
        semantic_text: object = None,
        semantic_data: object = None,
        scene_ref: object = None,
        track_refs: object = (),
        entity_refs: object = (),
        correlation_key: object,
        capture_state: CaptureState | str = CaptureState.COMPLETE,
        owner_binding: ConfirmedOwnerBinding,
        attributes: object = None,
        schema_version: object = SOURCE_RECORD_SCHEMA_VERSION,
        config: SourceConfig | None = None,
    ) -> None:
        limits = config or SourceConfig()
        if not isinstance(limits, SourceConfig):
            raise TypeError("config must be SourceConfig")
        try:
            resolved_type = SourceType(source_type)
            resolved_modality = Modality(modality)
            resolved_capture = CaptureState(capture_state)
            resolved_stream = identifier(stream_id, "stream_id")
            resolved_sequence = non_negative_int(source_sequence, "source_sequence")
            resolved_producer = identifier(producer_ref, "producer_ref")
            start = strict_utc(event_time_start, "event_time_start")
            end = strict_utc(event_time_end, "event_time_end")
            if end < start:
                raise ValueError("event_time_end cannot be earlier than event_time_start")
            ingested = strict_utc(ingested_at, "ingested_at")
            reference = external_reference(
                payload_ref,
                "payload_ref",
                maximum=limits.max_payload_ref_chars,
            )
            digest = sha256_digest(payload_digest, "payload_digest")
            media_type = bounded_text(payload_media_type, "payload_media_type", maximum=256)
            size = non_negative_int(payload_size_bytes, "payload_size_bytes")
            semantic = optional_bounded_text(
                semantic_text,
                "semantic_text",
                maximum=limits.max_semantic_text_chars,
            )
            if (
                semantic is not None
                and semantic.casefold().startswith("data:")
                and ";base64," in semantic.casefold()
            ):
                raise ValueError("semantic_text cannot contain inline base64 media")
            semantic_object = json_snapshot(
                {} if semantic_data is None else semantic_data,
                "semantic_data",
                maximum_chars=limits.max_semantic_data_chars,
            )
            scene = optional_identifier(scene_ref, "scene_ref")
            tracks = identifier_tuple(
                track_refs,
                "track_refs",
                maximum_items=limits.max_track_refs,
            )
            entities = identifier_tuple(
                entity_refs,
                "entity_refs",
                maximum_items=limits.max_entity_refs,
            )
            correlation = identifier(correlation_key, "correlation_key")
            if not isinstance(owner_binding, ConfirmedOwnerBinding):
                raise TypeError("owner_binding must be ConfirmedOwnerBinding")
            attribute_object = json_snapshot(
                {} if attributes is None else attributes,
                "attributes",
                maximum_chars=limits.max_attributes_chars,
            )
            version = identifier(schema_version, "schema_version", maximum=32)
            from behavior.source.identity import SourceRecordIdentityFactory

            record_id = SourceRecordIdentityFactory.create(
                stream_id=resolved_stream,
                source_sequence=resolved_sequence,
                payload_digest=digest,
                source_type=resolved_type,
                schema_version=version,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SourceRecordError):
                raise
            raise SourceRecordError(str(exc)) from exc
        for name, value in (
            ("source_record_id", record_id),
            ("stream_id", resolved_stream),
            ("source_sequence", resolved_sequence),
            ("source_type", resolved_type),
            ("modality", resolved_modality),
            ("producer_ref", resolved_producer),
            ("event_time_start", start),
            ("event_time_end", end),
            ("ingested_at", ingested),
            ("payload_ref", reference),
            ("payload_digest", digest),
            ("payload_media_type", media_type),
            ("payload_size_bytes", size),
            ("semantic_text", semantic),
            ("semantic_data", semantic_object),
            ("scene_ref", scene),
            ("track_refs", tracks),
            ("entity_refs", entities),
            ("correlation_key", correlation),
            ("capture_state", resolved_capture),
            ("owner_binding", owner_binding),
            ("attributes", attribute_object),
            ("schema_version", version),
        ):
            object.__setattr__(self, name, value)

    @property
    def semantic_projection_digest(self) -> str:
        return canonical_digest(
            {"semantic_text": self.semantic_text, "semantic_data": self.semantic_data}
        )

    @property
    def semantic_projection_chars(self) -> int:
        return len(self.semantic_text or "") + len(canonical_json(self.semantic_data))

    @property
    def canonical_digest(self) -> str:
        return canonical_digest(self.to_dict())

    @property
    def stable_sort_key(self) -> tuple[datetime, datetime, str, int, str]:
        return (
            self.event_time_start,
            self.event_time_end,
            self.stream_id,
            self.source_sequence,
            self.source_record_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_record_id": self.source_record_id,
            "stream_id": self.stream_id,
            "source_sequence": self.source_sequence,
            "source_type": self.source_type.value,
            "modality": self.modality.value,
            "producer_ref": self.producer_ref,
            "event_time_start": utc_text(self.event_time_start),
            "event_time_end": utc_text(self.event_time_end),
            "ingested_at": utc_text(self.ingested_at),
            "payload_ref": self.payload_ref,
            "payload_digest": self.payload_digest,
            "payload_media_type": self.payload_media_type,
            "payload_size_bytes": self.payload_size_bytes,
            "semantic_text": self.semantic_text,
            "semantic_data": self.semantic_data,
            "scene_ref": self.scene_ref,
            "track_refs": self.track_refs,
            "entity_refs": self.entity_refs,
            "correlation_key": self.correlation_key,
            "capture_state": self.capture_state.value,
            "owner_binding": self.owner_binding.to_dict(),
            "attributes": self.attributes,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object, *, config: SourceConfig | None = None) -> SourceRecord:
        fields = frozenset(
            {
                "source_record_id",
                "stream_id",
                "source_sequence",
                "source_type",
                "modality",
                "producer_ref",
                "event_time_start",
                "event_time_end",
                "ingested_at",
                "payload_ref",
                "payload_digest",
                "payload_media_type",
                "payload_size_bytes",
                "semantic_text",
                "semantic_data",
                "scene_ref",
                "track_refs",
                "entity_refs",
                "correlation_key",
                "capture_state",
                "owner_binding",
                "attributes",
                "schema_version",
            }
        )
        try:
            data = strict_fields(value, "source_record", fields)
            require_fields(data, "source_record", fields)
            result = cls(
                stream_id=data["stream_id"],
                source_sequence=data["source_sequence"],
                source_type=SourceType(data["source_type"]),
                modality=Modality(data["modality"]),
                producer_ref=data["producer_ref"],
                event_time_start=parse_utc(data["event_time_start"], "event_time_start"),
                event_time_end=parse_utc(data["event_time_end"], "event_time_end"),
                ingested_at=parse_utc(data["ingested_at"], "ingested_at"),
                payload_ref=data["payload_ref"],
                payload_digest=data["payload_digest"],
                payload_media_type=data["payload_media_type"],
                payload_size_bytes=data["payload_size_bytes"],
                semantic_text=data["semantic_text"],
                semantic_data=data["semantic_data"],
                scene_ref=data["scene_ref"],
                track_refs=data["track_refs"],
                entity_refs=data["entity_refs"],
                correlation_key=data["correlation_key"],
                capture_state=CaptureState(data["capture_state"]),
                owner_binding=ConfirmedOwnerBinding.from_dict(data["owner_binding"]),
                attributes=data["attributes"],
                schema_version=data["schema_version"],
                config=config,
            )
            if data["source_record_id"] != result.source_record_id:
                raise SourceRecordError("source_record_id does not match deterministic identity")
            return result
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SourceRecordError):
                raise
            raise SourceRecordError(str(exc)) from exc


@dataclass(frozen=True)
class SourceRecordBatch:
    records: tuple[SourceRecord, ...]

    def __init__(self, records: Sequence[SourceRecord], *, config: SourceConfig | None = None) -> None:
        limits = config or SourceConfig()
        if isinstance(records, str | bytes) or not isinstance(records, Sequence):
            raise TypeError("records must be a sequence of SourceRecord values")
        resolved = tuple(records)
        if not resolved or len(resolved) > limits.max_batch_size:
            raise SourceRecordError("source record batch size is outside its configured boundary")
        if any(not isinstance(record, SourceRecord) for record in resolved):
            raise TypeError("records must contain SourceRecord values")
        identifiers = tuple(record.source_record_id for record in resolved)
        if len(set(identifiers)) != len(identifiers):
            raise SourceRecordError("source record batch cannot contain duplicate identities")
        object.__setattr__(self, "records", resolved)


__all__ = [
    "CaptureState",
    "Modality",
    "SOURCE_RECORD_SCHEMA_VERSION",
    "SourceRecord",
    "SourceRecordBatch",
    "SourceType",
]
