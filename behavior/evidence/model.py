"""EvidenceWindow 的耐久状态与显式处理结果。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from behavior._validation import (
    identifier,
    non_negative_int,
    optional_identifier,
    parse_utc,
    require_fields,
    sha256_digest,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.errors import EvidenceWindowError
from foundation.integrity import canonical_digest

EVIDENCE_WINDOW_SCHEMA_VERSION = "1"


class EvidenceWindowState(str, Enum):
    OPEN = "OPEN"
    SEALED = "SEALED"
    EXPIRED = "EXPIRED"


class EvidenceSealReason(str, Enum):
    MAX_GAP = "MAX_GAP"
    MAX_DURATION = "MAX_DURATION"
    MAX_RECORDS = "MAX_RECORDS"
    MAX_PROJECTION_SIZE = "MAX_PROJECTION_SIZE"
    EXPLICIT = "EXPLICIT"
    STREAM_END = "STREAM_END"


class SourceIngestStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REPLAYED = "REPLAYED"
    LATE_REJECTED = "LATE_REJECTED"


@dataclass(frozen=True)
class EvidenceWindow:
    window_id: str
    grouping_key: str
    generation: int
    owner_binding_digest: str
    correlation_key: str
    scene_ref: str | None
    primary_track_ref: str | None
    state: EvidenceWindowState
    started_at: datetime
    ended_at: datetime
    max_event_time: datetime
    watermark: datetime
    ordered_source_record_ids: tuple[str, ...]
    total_projection_chars: int
    schema_version: str = EVIDENCE_WINDOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_id", identifier(self.window_id, "window_id"))
        object.__setattr__(self, "grouping_key", sha256_digest(self.grouping_key, "grouping_key"))
        object.__setattr__(self, "generation", non_negative_int(self.generation, "generation"))
        object.__setattr__(
            self,
            "owner_binding_digest",
            sha256_digest(self.owner_binding_digest, "owner_binding_digest"),
        )
        object.__setattr__(self, "correlation_key", identifier(self.correlation_key, "correlation_key"))
        object.__setattr__(self, "scene_ref", optional_identifier(self.scene_ref, "scene_ref"))
        object.__setattr__(
            self,
            "primary_track_ref",
            optional_identifier(self.primary_track_ref, "primary_track_ref"),
        )
        object.__setattr__(self, "state", EvidenceWindowState(self.state))
        for name in ("started_at", "ended_at", "max_event_time", "watermark"):
            object.__setattr__(self, name, strict_utc(getattr(self, name), name))
        if self.ended_at < self.started_at:
            raise EvidenceWindowError("window ended_at cannot precede started_at")
        if self.max_event_time < self.ended_at:
            raise EvidenceWindowError("window max_event_time cannot precede ended_at")
        if not isinstance(self.ordered_source_record_ids, tuple) or not self.ordered_source_record_ids:
            raise EvidenceWindowError("window must contain at least one SourceRecord")
        identifiers = tuple(identifier(value, "source_record_id") for value in self.ordered_source_record_ids)
        if len(set(identifiers)) != len(identifiers):
            raise EvidenceWindowError("window SourceRecord identities must be unique")
        object.__setattr__(self, "ordered_source_record_ids", identifiers)
        object.__setattr__(
            self,
            "total_projection_chars",
            non_negative_int(self.total_projection_chars, "total_projection_chars"),
        )
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))

    @classmethod
    def create(
        cls,
        *,
        grouping_key: str,
        generation: int,
        owner_binding_digest: str,
        correlation_key: str,
        scene_ref: str | None,
        primary_track_ref: str | None,
        state: EvidenceWindowState,
        started_at: datetime,
        ended_at: datetime,
        max_event_time: datetime,
        watermark: datetime,
        ordered_source_record_ids: tuple[str, ...],
        total_projection_chars: int,
    ) -> EvidenceWindow:
        window_id = "window_" + canonical_digest(
            {"generation": generation, "grouping_key": grouping_key, "schema_version": EVIDENCE_WINDOW_SCHEMA_VERSION}
        )
        return cls(
            window_id=window_id,
            grouping_key=grouping_key,
            generation=generation,
            owner_binding_digest=owner_binding_digest,
            correlation_key=correlation_key,
            scene_ref=scene_ref,
            primary_track_ref=primary_track_ref,
            state=state,
            started_at=started_at,
            ended_at=ended_at,
            max_event_time=max_event_time,
            watermark=watermark,
            ordered_source_record_ids=ordered_source_record_ids,
            total_projection_chars=total_projection_chars,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "grouping_key": self.grouping_key,
            "generation": self.generation,
            "owner_binding_digest": self.owner_binding_digest,
            "correlation_key": self.correlation_key,
            "scene_ref": self.scene_ref,
            "primary_track_ref": self.primary_track_ref,
            "state": self.state.value,
            "started_at": utc_text(self.started_at),
            "ended_at": utc_text(self.ended_at),
            "max_event_time": utc_text(self.max_event_time),
            "watermark": utc_text(self.watermark),
            "ordered_source_record_ids": self.ordered_source_record_ids,
            "total_projection_chars": self.total_projection_chars,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> EvidenceWindow:
        fields = frozenset(
            {
                "window_id",
                "grouping_key",
                "generation",
                "owner_binding_digest",
                "correlation_key",
                "scene_ref",
                "primary_track_ref",
                "state",
                "started_at",
                "ended_at",
                "max_event_time",
                "watermark",
                "ordered_source_record_ids",
                "total_projection_chars",
                "schema_version",
            }
        )
        data = strict_fields(value, "evidence_window", fields)
        require_fields(data, "evidence_window", fields)
        identifiers = data["ordered_source_record_ids"]
        if not isinstance(identifiers, list | tuple):
            raise EvidenceWindowError("ordered_source_record_ids must be a sequence")
        return cls(
            window_id=data["window_id"],
            grouping_key=data["grouping_key"],
            generation=data["generation"],
            owner_binding_digest=data["owner_binding_digest"],
            correlation_key=data["correlation_key"],
            scene_ref=data["scene_ref"],
            primary_track_ref=data["primary_track_ref"],
            state=EvidenceWindowState(data["state"]),
            started_at=parse_utc(data["started_at"], "started_at"),
            ended_at=parse_utc(data["ended_at"], "ended_at"),
            max_event_time=parse_utc(data["max_event_time"], "max_event_time"),
            watermark=parse_utc(data["watermark"], "watermark"),
            ordered_source_record_ids=tuple(identifiers),
            total_projection_chars=data["total_projection_chars"],
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class SourceIngestResult:
    status: SourceIngestStatus
    source_record_id: str
    active_window: EvidenceWindow | None
    manifest_ids: tuple[str, ...] = ()
    reason_code: str | None = None
    window_opened: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SourceIngestStatus(self.status))
        object.__setattr__(self, "source_record_id", identifier(self.source_record_id, "source_record_id"))
        if self.active_window is not None and not isinstance(self.active_window, EvidenceWindow):
            raise TypeError("active_window must be EvidenceWindow or None")
        manifests = tuple(identifier(value, "manifest_id") for value in self.manifest_ids)
        object.__setattr__(self, "manifest_ids", manifests)
        if self.status is SourceIngestStatus.LATE_REJECTED and not self.reason_code:
            raise EvidenceWindowError("late rejection requires a stable reason_code")
        if not isinstance(self.window_opened, bool):
            raise TypeError("window_opened must be boolean")


__all__ = [
    "EVIDENCE_WINDOW_SCHEMA_VERSION",
    "EvidenceSealReason",
    "EvidenceWindow",
    "EvidenceWindowState",
    "SourceIngestResult",
    "SourceIngestStatus",
]
