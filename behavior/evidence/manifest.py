"""EvidenceWindow 的确定性不可变封存快照。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from behavior._validation import (
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
from behavior.errors import EvidenceManifestError
from behavior.evidence.model import EvidenceSealReason, EvidenceWindow, EvidenceWindowState
from behavior.source.model import CaptureState, Modality, SourceRecord, SourceType
from foundation.integrity import canonical_digest

EVIDENCE_MANIFEST_SCHEMA_VERSION = "1"


class EvidenceCoverageState(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    BLIND = "BLIND"


@dataclass(frozen=True)
class BlindInterval:
    started_at: datetime
    ended_at: datetime
    source_record_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "started_at", strict_utc(self.started_at, "started_at"))
        object.__setattr__(self, "ended_at", strict_utc(self.ended_at, "ended_at"))
        if self.ended_at < self.started_at:
            raise EvidenceManifestError("blind interval end cannot precede start")
        object.__setattr__(self, "source_record_id", identifier(self.source_record_id, "source_record_id"))

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at": utc_text(self.started_at),
            "ended_at": utc_text(self.ended_at),
            "source_record_id": self.source_record_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> BlindInterval:
        fields = frozenset({"started_at", "ended_at", "source_record_id"})
        data = strict_fields(value, "blind_interval", fields)
        require_fields(data, "blind_interval", fields)
        return cls(
            started_at=parse_utc(data["started_at"], "started_at"),
            ended_at=parse_utc(data["ended_at"], "ended_at"),
            source_record_id=data["source_record_id"],
        )


@dataclass(frozen=True)
class ManifestSourceRecord:
    source_record_id: str
    stream_id: str
    source_sequence: int
    source_type: SourceType
    modality: Modality
    event_time_start: datetime
    event_time_end: datetime
    payload_ref: str
    payload_digest: str
    semantic_projection_digest: str
    semantic_text: str | None
    semantic_data: Mapping[str, Any]
    scene_ref: str | None
    track_refs: tuple[str, ...]
    entity_refs: tuple[str, ...]
    capture_state: CaptureState

    def __post_init__(self) -> None:
        for name in ("source_record_id", "stream_id"):
            object.__setattr__(self, name, identifier(getattr(self, name), name))
        object.__setattr__(self, "source_sequence", non_negative_int(self.source_sequence, "source_sequence"))
        object.__setattr__(self, "source_type", SourceType(self.source_type))
        object.__setattr__(self, "modality", Modality(self.modality))
        object.__setattr__(self, "event_time_start", strict_utc(self.event_time_start, "event_time_start"))
        object.__setattr__(self, "event_time_end", strict_utc(self.event_time_end, "event_time_end"))
        if self.event_time_end < self.event_time_start:
            raise EvidenceManifestError("manifest source end cannot precede start")
        try:
            resolved_payload_ref = external_reference(
                self.payload_ref,
                "payload_ref",
                maximum=1_000_000,
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceManifestError("manifest payload_ref must be bounded reference text") from exc
        object.__setattr__(self, "payload_ref", resolved_payload_ref)
        object.__setattr__(self, "payload_digest", sha256_digest(self.payload_digest, "payload_digest"))
        object.__setattr__(
            self,
            "semantic_projection_digest",
            sha256_digest(self.semantic_projection_digest, "semantic_projection_digest"),
        )
        object.__setattr__(self, "semantic_text", optional_bounded_text(self.semantic_text, "semantic_text", maximum=1_000_000))
        object.__setattr__(self, "semantic_data", json_snapshot(self.semantic_data, "semantic_data", maximum_chars=1_000_000))
        object.__setattr__(self, "scene_ref", optional_identifier(self.scene_ref, "scene_ref"))
        object.__setattr__(self, "track_refs", identifier_tuple(self.track_refs, "track_refs", maximum_items=10_000))
        object.__setattr__(self, "entity_refs", identifier_tuple(self.entity_refs, "entity_refs", maximum_items=10_000))
        object.__setattr__(self, "capture_state", CaptureState(self.capture_state))
        expected_projection = canonical_digest(
            {"semantic_text": self.semantic_text, "semantic_data": self.semantic_data}
        )
        if self.semantic_projection_digest != expected_projection:
            raise EvidenceManifestError("semantic projection digest mismatch")

    @classmethod
    def from_source(cls, record: SourceRecord) -> ManifestSourceRecord:
        return cls(
            source_record_id=record.source_record_id,
            stream_id=record.stream_id,
            source_sequence=record.source_sequence,
            source_type=record.source_type,
            modality=record.modality,
            event_time_start=record.event_time_start,
            event_time_end=record.event_time_end,
            payload_ref=record.payload_ref,
            payload_digest=record.payload_digest,
            semantic_projection_digest=record.semantic_projection_digest,
            semantic_text=record.semantic_text,
            semantic_data=record.semantic_data,
            scene_ref=record.scene_ref,
            track_refs=record.track_refs,
            entity_refs=record.entity_refs,
            capture_state=record.capture_state,
        )

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
            "event_time_start": utc_text(self.event_time_start),
            "event_time_end": utc_text(self.event_time_end),
            "payload_ref": self.payload_ref,
            "payload_digest": self.payload_digest,
            "semantic_projection_digest": self.semantic_projection_digest,
            "semantic_text": self.semantic_text,
            "semantic_data": self.semantic_data,
            "scene_ref": self.scene_ref,
            "track_refs": self.track_refs,
            "entity_refs": self.entity_refs,
            "capture_state": self.capture_state.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> ManifestSourceRecord:
        fields = frozenset(
            {
                "source_record_id",
                "stream_id",
                "source_sequence",
                "source_type",
                "modality",
                "event_time_start",
                "event_time_end",
                "payload_ref",
                "payload_digest",
                "semantic_projection_digest",
                "semantic_text",
                "semantic_data",
                "scene_ref",
                "track_refs",
                "entity_refs",
                "capture_state",
            }
        )
        data = strict_fields(value, "manifest_source_record", fields)
        require_fields(data, "manifest_source_record", fields)
        return cls(
            source_record_id=data["source_record_id"],
            stream_id=data["stream_id"],
            source_sequence=data["source_sequence"],
            source_type=SourceType(data["source_type"]),
            modality=Modality(data["modality"]),
            event_time_start=parse_utc(data["event_time_start"], "event_time_start"),
            event_time_end=parse_utc(data["event_time_end"], "event_time_end"),
            payload_ref=data["payload_ref"],
            payload_digest=data["payload_digest"],
            semantic_projection_digest=data["semantic_projection_digest"],
            semantic_text=data["semantic_text"],
            semantic_data=data["semantic_data"],
            scene_ref=data["scene_ref"],
            track_refs=tuple(data["track_refs"]),
            entity_refs=tuple(data["entity_refs"]),
            capture_state=CaptureState(data["capture_state"]),
        )


@dataclass(frozen=True)
class EvidenceManifest:
    manifest_id: str
    window_id: str
    owner_binding_digest: str
    started_at: datetime
    ended_at: datetime
    sealed_at: datetime
    seal_reason: EvidenceSealReason
    scene_ref: str | None
    track_refs: tuple[str, ...]
    modalities: tuple[Modality, ...]
    coverage_state: EvidenceCoverageState
    blind_intervals: tuple[BlindInterval, ...]
    ordered_source_records: tuple[ManifestSourceRecord, ...]
    total_projection_chars: int
    manifest_digest: str
    schema_version: str = EVIDENCE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_id", identifier(self.manifest_id, "manifest_id"))
        object.__setattr__(self, "window_id", identifier(self.window_id, "window_id"))
        object.__setattr__(self, "owner_binding_digest", sha256_digest(self.owner_binding_digest, "owner_binding_digest"))
        for name in ("started_at", "ended_at", "sealed_at"):
            object.__setattr__(self, name, strict_utc(getattr(self, name), name))
        if self.ended_at < self.started_at or self.sealed_at < self.ended_at:
            raise EvidenceManifestError("manifest time bounds are inconsistent")
        object.__setattr__(self, "seal_reason", EvidenceSealReason(self.seal_reason))
        object.__setattr__(self, "scene_ref", optional_identifier(self.scene_ref, "scene_ref"))
        object.__setattr__(self, "track_refs", identifier_tuple(self.track_refs, "track_refs", maximum_items=10_000))
        modalities = tuple(sorted((Modality(value) for value in self.modalities), key=lambda item: item.value))
        if not modalities or len(set(modalities)) != len(modalities):
            raise EvidenceManifestError("manifest modalities must be a non-empty unique tuple")
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "coverage_state", EvidenceCoverageState(self.coverage_state))
        intervals = tuple(self.blind_intervals)
        if any(not isinstance(item, BlindInterval) for item in intervals):
            raise TypeError("blind_intervals must contain BlindInterval values")
        object.__setattr__(self, "blind_intervals", intervals)
        records = tuple(self.ordered_source_records)
        if not records or any(not isinstance(item, ManifestSourceRecord) for item in records):
            raise EvidenceManifestError("manifest must contain SourceRecord snapshots")
        if tuple(sorted(records, key=lambda item: item.stable_sort_key)) != records:
            raise EvidenceManifestError("manifest SourceRecord snapshots are not stably sorted")
        record_ids = tuple(item.source_record_id for item in records)
        if len(set(record_ids)) != len(record_ids):
            raise EvidenceManifestError("manifest SourceRecord identities must be unique")
        object.__setattr__(self, "ordered_source_records", records)
        object.__setattr__(self, "total_projection_chars", non_negative_int(self.total_projection_chars, "total_projection_chars"))
        object.__setattr__(self, "manifest_digest", sha256_digest(self.manifest_digest, "manifest_digest"))
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))
        expected_digest = canonical_digest(self._digest_payload())
        if self.manifest_digest != expected_digest:
            raise EvidenceManifestError("manifest digest mismatch")
        if self.manifest_id != "manifest_" + expected_digest:
            raise EvidenceManifestError("manifest_id does not match deterministic identity")

    @classmethod
    def seal(
        cls,
        window: EvidenceWindow,
        records: Sequence[SourceRecord],
        *,
        reason: EvidenceSealReason,
        max_blind_intervals: int,
    ) -> EvidenceManifest:
        if not isinstance(window, EvidenceWindow):
            raise TypeError("window must be EvidenceWindow")
        if window.state is not EvidenceWindowState.SEALED:
            raise EvidenceManifestError("only a SEALED EvidenceWindow can publish a Manifest")
        ordered = tuple(sorted(records, key=lambda item: item.stable_sort_key))
        if not ordered:
            raise EvidenceManifestError("empty windows cannot create manifests")
        if tuple(item.source_record_id for item in ordered) != window.ordered_source_record_ids:
            raise EvidenceManifestError("window membership does not match SourceRecord snapshots")
        snapshots = tuple(ManifestSourceRecord.from_source(record) for record in ordered)
        blind = tuple(
            BlindInterval(record.event_time_start, record.event_time_end, record.source_record_id)
            for record in ordered
            if record.capture_state is CaptureState.BLIND
        )
        if len(blind) > max_blind_intervals:
            raise EvidenceManifestError("explicit blind intervals exceed their configured boundary")
        capture_states = {record.capture_state for record in ordered}
        if capture_states == {CaptureState.BLIND}:
            coverage = EvidenceCoverageState.BLIND
        elif capture_states <= {CaptureState.COMPLETE, CaptureState.STREAM_END}:
            coverage = EvidenceCoverageState.COMPLETE
        else:
            coverage = EvidenceCoverageState.PARTIAL
        sealed_at = max(max(record.ingested_at for record in ordered), window.ended_at)
        payload: dict[str, Any] = {
            "window_id": window.window_id,
            "owner_binding_digest": window.owner_binding_digest,
            "started_at": utc_text(window.started_at),
            "ended_at": utc_text(window.ended_at),
            "sealed_at": utc_text(sealed_at),
            "seal_reason": EvidenceSealReason(reason).value,
            "scene_ref": window.scene_ref,
            "track_refs": tuple(sorted({track for record in ordered for track in record.track_refs})),
            "modalities": tuple(sorted({record.modality.value for record in ordered})),
            "coverage_state": coverage.value,
            "blind_intervals": tuple(item.to_dict() for item in blind),
            "ordered_source_records": tuple(item.to_dict() for item in snapshots),
            "total_projection_chars": sum(record.semantic_projection_chars for record in ordered),
            "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        }
        digest = canonical_digest(payload)
        return cls(
            manifest_id="manifest_" + digest,
            manifest_digest=digest,
            window_id=window.window_id,
            owner_binding_digest=window.owner_binding_digest,
            started_at=window.started_at,
            ended_at=window.ended_at,
            sealed_at=sealed_at,
            seal_reason=reason,
            scene_ref=window.scene_ref,
            track_refs=tuple(payload["track_refs"]),
            modalities=tuple(Modality(value) for value in payload["modalities"]),
            coverage_state=coverage,
            blind_intervals=blind,
            ordered_source_records=snapshots,
            total_projection_chars=payload["total_projection_chars"],
        )

    def _digest_payload(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "owner_binding_digest": self.owner_binding_digest,
            "started_at": utc_text(self.started_at),
            "ended_at": utc_text(self.ended_at),
            "sealed_at": utc_text(self.sealed_at),
            "seal_reason": self.seal_reason.value,
            "scene_ref": self.scene_ref,
            "track_refs": self.track_refs,
            "modalities": tuple(item.value for item in self.modalities),
            "coverage_state": self.coverage_state.value,
            "blind_intervals": tuple(item.to_dict() for item in self.blind_intervals),
            "ordered_source_records": tuple(item.to_dict() for item in self.ordered_source_records),
            "total_projection_chars": self.total_projection_chars,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {"manifest_id": self.manifest_id, **self._digest_payload(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_dict(cls, value: object) -> EvidenceManifest:
        fields = frozenset(
            {
                "manifest_id",
                "window_id",
                "owner_binding_digest",
                "started_at",
                "ended_at",
                "sealed_at",
                "seal_reason",
                "scene_ref",
                "track_refs",
                "modalities",
                "coverage_state",
                "blind_intervals",
                "ordered_source_records",
                "total_projection_chars",
                "manifest_digest",
                "schema_version",
            }
        )
        data = strict_fields(value, "evidence_manifest", fields)
        require_fields(data, "evidence_manifest", fields)
        return cls(
            manifest_id=data["manifest_id"],
            window_id=data["window_id"],
            owner_binding_digest=data["owner_binding_digest"],
            started_at=parse_utc(data["started_at"], "started_at"),
            ended_at=parse_utc(data["ended_at"], "ended_at"),
            sealed_at=parse_utc(data["sealed_at"], "sealed_at"),
            seal_reason=EvidenceSealReason(data["seal_reason"]),
            scene_ref=data["scene_ref"],
            track_refs=tuple(data["track_refs"]),
            modalities=tuple(Modality(value) for value in data["modalities"]),
            coverage_state=EvidenceCoverageState(data["coverage_state"]),
            blind_intervals=tuple(BlindInterval.from_dict(item) for item in data["blind_intervals"]),
            ordered_source_records=tuple(
                ManifestSourceRecord.from_dict(item) for item in data["ordered_source_records"]
            ),
            total_projection_chars=data["total_projection_chars"],
            manifest_digest=data["manifest_digest"],
            schema_version=data["schema_version"],
        )


__all__ = [
    "BlindInterval",
    "EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "EvidenceCoverageState",
    "EvidenceManifest",
    "ManifestSourceRecord",
]
