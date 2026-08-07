"""Evidence 第一层的稳定引用值对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from behavior._validation import (
    bounded_text,
    external_reference,
    non_negative_int,
    optional_bounded_text,
    pii_safe_identifier,
    sha256_digest,
    strict_utc,
)
from foundation.integrity import canonical_digest

SOURCE_EVENT_REF_SCHEMA_VERSION = "source_event_ref_v1"


@dataclass(frozen=True)
class SourceEventRef:
    namespace: str
    value: str
    identity_digest: str = field(init=False)

    def __post_init__(self) -> None:
        namespace = pii_safe_identifier(self.namespace, "source_event_ref.namespace")
        value = pii_safe_identifier(self.value, "source_event_ref.value")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "value", value)
        object.__setattr__(
            self,
            "identity_digest",
            canonical_digest(
                {
                    "namespace": namespace,
                    "schema_version": SOURCE_EVENT_REF_SCHEMA_VERSION,
                    "value": value,
                }
            ),
        )


@dataclass(frozen=True)
class StreamRef:
    namespace: str
    value: str
    generation: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", pii_safe_identifier(self.namespace, "stream_ref.namespace"))
        object.__setattr__(self, "value", pii_safe_identifier(self.value, "stream_ref.value"))
        object.__setattr__(self, "generation", non_negative_int(self.generation, "stream_ref.generation"))


@dataclass(frozen=True)
class CorrelationRef:
    namespace: str
    value: str
    root_value: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", pii_safe_identifier(self.namespace, "correlation_ref.namespace"))
        object.__setattr__(self, "value", pii_safe_identifier(self.value, "correlation_ref.value"))
        root = None if self.root_value is None else pii_safe_identifier(self.root_value, "correlation_ref.root_value")
        object.__setattr__(self, "root_value", root)


class CausalRefKind(str, Enum):
    SOURCE_EVENT = "SOURCE_EVENT"
    CONVERSATION_MESSAGE = "CONVERSATION_MESSAGE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    ACTION_DIRECTIVE = "ACTION_DIRECTIVE"
    INJECTION_RECEIPT = "INJECTION_RECEIPT"
    PARENT_RUNTIME_EVENT = "PARENT_RUNTIME_EVENT"
    OTHER = "OTHER"


@dataclass(frozen=True)
class CausalRef:
    kind: CausalRefKind
    reference: str
    reference_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CausalRefKind(self.kind))
        object.__setattr__(self, "reference", bounded_text(self.reference, "causal_ref.reference", maximum=2_048))
        object.__setattr__(
            self,
            "reference_digest",
            sha256_digest(self.reference_digest, "causal_ref.reference_digest"),
        )


@dataclass(frozen=True)
class ProjectionRef:
    namespace: str
    value: str
    source_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", pii_safe_identifier(self.namespace, "projection_ref.namespace"))
        object.__setattr__(self, "value", pii_safe_identifier(self.value, "projection_ref.value"))
        object.__setattr__(self, "source_digest", sha256_digest(self.source_digest, "projection_ref.source_digest"))


class EvidenceKind(str, Enum):
    IMAGE_FRAME = "IMAGE_FRAME"
    VIDEO_CLIP = "VIDEO_CLIP"
    AUDIO_SEGMENT = "AUDIO_SEGMENT"
    TRANSCRIPT = "TRANSCRIPT"
    SENSOR_WINDOW = "SENSOR_WINDOW"
    DEVICE_LOG = "DEVICE_LOG"
    AGENT_LOG = "AGENT_LOG"
    TOOL_RESULT = "TOOL_RESULT"
    OTHER = "OTHER"


@dataclass(frozen=True)
class EvidenceReference:
    reference: str
    evidence_kind: EvidenceKind
    digest: str
    event_time_start: datetime
    event_time_end: datetime
    media_type: str | None = None
    size_bytes: int | None = None
    source_system_ref: str | None = None

    def __post_init__(self) -> None:
        reference = external_reference(self.reference, "evidence_reference.reference", maximum=2_048)
        start = strict_utc(self.event_time_start, "evidence_reference.event_time_start")
        end = strict_utc(self.event_time_end, "evidence_reference.event_time_end")
        if end < start:
            raise ValueError("evidence_reference end cannot precede start")
        if self.size_bytes is not None:
            size = non_negative_int(self.size_bytes, "evidence_reference.size_bytes")
        else:
            size = None
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "evidence_kind", EvidenceKind(self.evidence_kind))
        object.__setattr__(self, "digest", sha256_digest(self.digest, "evidence_reference.digest"))
        object.__setattr__(self, "event_time_start", start)
        object.__setattr__(self, "event_time_end", end)
        object.__setattr__(
            self,
            "media_type",
            optional_bounded_text(self.media_type, "evidence_reference.media_type", maximum=256),
        )
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(
            self,
            "source_system_ref",
            None
            if self.source_system_ref is None
            else external_reference(
                self.source_system_ref,
                "evidence_reference.source_system_ref",
                maximum=2_048,
            ),
        )


__all__ = [
    "CausalRef",
    "CausalRefKind",
    "CorrelationRef",
    "EvidenceKind",
    "EvidenceReference",
    "ProjectionRef",
    "SourceEventRef",
    "StreamRef",
]
