"""原始证据只读外部引用。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from behavior._validation import (
    bounded_text,
    external_reference,
    non_negative_int,
    parse_utc,
    require_fields,
    sha256_digest,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.config import IngressConfig
from behavior.errors import SemanticRecordError


class EvidenceKind(str, Enum):
    IMAGE_FRAME = "IMAGE_FRAME"
    VIDEO_CLIP = "VIDEO_CLIP"
    AUDIO_SEGMENT = "AUDIO_SEGMENT"
    TRANSCRIPT = "TRANSCRIPT"
    SENSOR_WINDOW = "SENSOR_WINDOW"
    DEVICE_LOG = "DEVICE_LOG"
    ROBOT_LOG = "ROBOT_LOG"
    AGENT_LOG = "AGENT_LOG"
    TOOL_RESULT = "TOOL_RESULT"
    OTHER = "OTHER"


@dataclass(frozen=True, init=False)
class EvidenceReference:
    """只定位上游证据，不赋予该证据任何现实语义。"""

    reference: str
    evidence_kind: EvidenceKind
    digest: str
    event_time_start: datetime
    event_time_end: datetime
    media_type: str
    size_bytes: int
    source_system_ref: str

    def __init__(
        self,
        *,
        reference: object,
        evidence_kind: EvidenceKind | str,
        digest: object,
        event_time_start: object,
        event_time_end: object,
        media_type: object,
        size_bytes: object,
        source_system_ref: object,
        config: IngressConfig | None = None,
    ) -> None:
        limits = config or IngressConfig()
        try:
            resolved_reference = external_reference(
                reference,
                "evidence_reference.reference",
                maximum=limits.max_reference_chars,
            )
            resolved_kind = EvidenceKind(evidence_kind)
            resolved_digest = sha256_digest(digest, "evidence_reference.digest")
            start = strict_utc(event_time_start, "evidence_reference.event_time_start")
            end = strict_utc(event_time_end, "evidence_reference.event_time_end")
            if end < start:
                raise ValueError("evidence reference end cannot precede start")
            resolved_media_type = bounded_text(media_type, "evidence_reference.media_type", maximum=256)
            resolved_size = non_negative_int(size_bytes, "evidence_reference.size_bytes")
            if resolved_size > 9_223_372_036_854_775_807:
                raise ValueError("evidence_reference.size_bytes exceeds its metadata boundary")
            resolved_source = bounded_text(
                source_system_ref,
                "evidence_reference.source_system_ref",
                maximum=limits.max_identifier_chars,
            )
        except (TypeError, ValueError) as exc:
            raise SemanticRecordError(str(exc)) from exc
        object.__setattr__(self, "reference", resolved_reference)
        object.__setattr__(self, "evidence_kind", resolved_kind)
        object.__setattr__(self, "digest", resolved_digest)
        object.__setattr__(self, "event_time_start", start)
        object.__setattr__(self, "event_time_end", end)
        object.__setattr__(self, "media_type", resolved_media_type)
        object.__setattr__(self, "size_bytes", resolved_size)
        object.__setattr__(self, "source_system_ref", resolved_source)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "evidence_kind": self.evidence_kind.value,
            "digest": self.digest,
            "event_time_start": utc_text(self.event_time_start),
            "event_time_end": utc_text(self.event_time_end),
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "source_system_ref": self.source_system_ref,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        config: IngressConfig | None = None,
    ) -> EvidenceReference:
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
        try:
            data = strict_fields(value, "evidence_reference", fields)
            require_fields(data, "evidence_reference", fields)
            return cls(
                reference=data["reference"],
                evidence_kind=EvidenceKind(data["evidence_kind"]),
                digest=data["digest"],
                event_time_start=parse_utc(data["event_time_start"], "evidence_reference.event_time_start"),
                event_time_end=parse_utc(data["event_time_end"], "evidence_reference.event_time_end"),
                media_type=data["media_type"],
                size_bytes=data["size_bytes"],
                source_system_ref=data["source_system_ref"],
                config=config,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SemanticRecordError):
                raise
            raise SemanticRecordError(str(exc)) from exc

    @classmethod
    def model_json_schema(cls, *, config: IngressConfig | None = None) -> dict[str, object]:
        limits = config or IngressConfig()
        fields: dict[str, object] = {
            "reference": {
                "type": "string",
                "minLength": 1,
                "maxLength": limits.max_reference_chars,
                "allOf": [
                    {"pattern": r"^(?![dD][aA][tT][aA]:)[A-Za-z][A-Za-z0-9+.-]*:"},
                    {"not": {"pattern": r";[bB][aA][sS][eE]64,"}},
                ],
            },
            "evidence_kind": {"type": "string", "enum": [item.value for item in EvidenceKind]},
            "digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
            "event_time_start": {"type": "string", "format": "date-time"},
            "event_time_end": {"type": "string", "format": "date-time"},
            "media_type": {"type": "string", "minLength": 1, "maxLength": 256},
            "size_bytes": {
                "type": "integer",
                "minimum": 0,
                "maximum": 9_223_372_036_854_775_807,
            },
            "source_system_ref": {
                "type": "string",
                "minLength": 1,
                "maxLength": limits.max_identifier_chars,
            },
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(fields),
            "properties": fields,
        }


__all__ = ["EvidenceKind", "EvidenceReference"]
