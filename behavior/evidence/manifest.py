"""封存 Owner-scoped 语义记录快照与显式 Coverage。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from behavior._validation import (
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
from behavior.errors import EvidenceManifestError
from behavior.evidence.bundle import (
    EvidenceBundleState,
    EvidenceSealReason,
    SemanticEvidenceBundle,
)
from behavior.ingress.evidence_ref import EvidenceReference
from behavior.ingress.model import OwnerScopedSemanticRecord, SemanticModality, SemanticRecordKind
from behavior.ingress.payloads import CoverageIntervalPayload, CoverageStatus
from foundation.integrity import canonical_digest, canonical_json

EVIDENCE_MANIFEST_SCHEMA_VERSION = "3"


class CoverageSummary(str, Enum):
    COVERED = "COVERED"
    BLIND = "BLIND"
    UNKNOWN = "UNKNOWN"
    MIXED = "MIXED"


@dataclass(frozen=True)
class CoverageInterval:
    modality: SemanticModality
    coverage_scope_ref: str | None
    event_time_start: datetime
    event_time_end: datetime
    semantic_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", SemanticModality(self.modality))
        object.__setattr__(
            self,
            "coverage_scope_ref",
            optional_identifier(self.coverage_scope_ref, "coverage.coverage_scope_ref"),
        )
        object.__setattr__(self, "event_time_start", strict_utc(self.event_time_start, "coverage.event_time_start"))
        object.__setattr__(self, "event_time_end", strict_utc(self.event_time_end, "coverage.event_time_end"))
        if self.event_time_end < self.event_time_start:
            raise EvidenceManifestError("Coverage interval end cannot precede start")
        ids = identifier_tuple(self.semantic_record_ids, "coverage.semantic_record_ids", maximum_items=10_000)
        object.__setattr__(self, "semantic_record_ids", ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "modality": self.modality.value,
            "coverage_scope_ref": self.coverage_scope_ref,
            "event_time_start": utc_text(self.event_time_start),
            "event_time_end": utc_text(self.event_time_end),
            "semantic_record_ids": self.semantic_record_ids,
        }

    @classmethod
    def from_dict(cls, value: object) -> CoverageInterval:
        fields = frozenset(
            {"modality", "coverage_scope_ref", "event_time_start", "event_time_end", "semantic_record_ids"}
        )
        data = strict_fields(value, "coverage_interval", fields)
        require_fields(data, "coverage_interval", fields)
        return cls(
            modality=SemanticModality(data["modality"]),
            coverage_scope_ref=data["coverage_scope_ref"],
            event_time_start=parse_utc(data["event_time_start"], "coverage.event_time_start"),
            event_time_end=parse_utc(data["event_time_end"], "coverage.event_time_end"),
            semantic_record_ids=tuple(data["semantic_record_ids"]),
        )


@dataclass(frozen=True)
class ManifestSemanticRecordSnapshot:
    semantic_record_id: str
    semantic_record_digest: str
    stream_id: str
    source_sequence: int
    record_kind: SemanticRecordKind
    modality: SemanticModality
    event_time_start: datetime
    event_time_end: datetime
    scene_ref: str | None
    evidence_refs: tuple[EvidenceReference, ...]
    payload_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_record_id", identifier(self.semantic_record_id, "semantic_record_id"))
        object.__setattr__(
            self, "semantic_record_digest", sha256_digest(self.semantic_record_digest, "semantic_record_digest")
        )
        object.__setattr__(self, "stream_id", identifier(self.stream_id, "stream_id"))
        object.__setattr__(self, "source_sequence", non_negative_int(self.source_sequence, "source_sequence"))
        object.__setattr__(self, "record_kind", SemanticRecordKind(self.record_kind))
        object.__setattr__(self, "modality", SemanticModality(self.modality))
        object.__setattr__(self, "event_time_start", strict_utc(self.event_time_start, "event_time_start"))
        object.__setattr__(self, "event_time_end", strict_utc(self.event_time_end, "event_time_end"))
        if self.event_time_end < self.event_time_start:
            raise EvidenceManifestError("Manifest record end cannot precede start")
        object.__setattr__(self, "scene_ref", optional_identifier(self.scene_ref, "scene_ref"))
        refs = tuple(self.evidence_refs)
        if any(not isinstance(item, EvidenceReference) for item in refs):
            raise TypeError("evidence_refs must contain EvidenceReference values")
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "payload_digest", sha256_digest(self.payload_digest, "payload_digest"))

    @classmethod
    def from_record(cls, record: OwnerScopedSemanticRecord) -> ManifestSemanticRecordSnapshot:
        value = record.semantic_input
        return cls(
            semantic_record_id=record.semantic_record_id,
            semantic_record_digest=record.semantic_digest,
            stream_id=value.stream_id,
            source_sequence=value.source_sequence,
            record_kind=value.record_kind,
            modality=value.modality,
            event_time_start=value.event_time_start,
            event_time_end=value.event_time_end,
            scene_ref=value.scene_ref,
            evidence_refs=value.evidence_refs,
            payload_digest=record.payload_digest,
        )

    @property
    def stable_sort_key(self) -> tuple[datetime, datetime, str, int, str]:
        return (
            self.event_time_start,
            self.event_time_end,
            self.stream_id,
            self.source_sequence,
            self.semantic_record_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic_record_id": self.semantic_record_id,
            "semantic_record_digest": self.semantic_record_digest,
            "stream_id": self.stream_id,
            "source_sequence": self.source_sequence,
            "record_kind": self.record_kind.value,
            "modality": self.modality.value,
            "event_time_start": utc_text(self.event_time_start),
            "event_time_end": utc_text(self.event_time_end),
            "scene_ref": self.scene_ref,
            "evidence_refs": tuple(item.to_dict() for item in self.evidence_refs),
            "payload_digest": self.payload_digest,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        ingress_config: IngressConfig | None = None,
    ) -> ManifestSemanticRecordSnapshot:
        fields = frozenset(
            {
                "semantic_record_id",
                "semantic_record_digest",
                "stream_id",
                "source_sequence",
                "record_kind",
                "modality",
                "event_time_start",
                "event_time_end",
                "scene_ref",
                "evidence_refs",
                "payload_digest",
            }
        )
        data = strict_fields(value, "manifest_semantic_record", fields)
        require_fields(data, "manifest_semantic_record", fields)
        return cls(
            semantic_record_id=data["semantic_record_id"],
            semantic_record_digest=data["semantic_record_digest"],
            stream_id=data["stream_id"],
            source_sequence=data["source_sequence"],
            record_kind=SemanticRecordKind(data["record_kind"]),
            modality=SemanticModality(data["modality"]),
            event_time_start=parse_utc(data["event_time_start"], "event_time_start"),
            event_time_end=parse_utc(data["event_time_end"], "event_time_end"),
            scene_ref=data["scene_ref"],
            evidence_refs=tuple(
                EvidenceReference.from_dict(item, config=ingress_config) for item in data["evidence_refs"]
            ),
            payload_digest=data["payload_digest"],
        )


@dataclass(frozen=True)
class EvidenceManifest:
    manifest_id: str
    bundle_id: str
    owner_identity_digest: str
    started_at: datetime
    ended_at: datetime
    sealed_at: datetime
    seal_reason: EvidenceSealReason
    ordered_record_snapshots: tuple[ManifestSemanticRecordSnapshot, ...]
    modalities: tuple[SemanticModality, ...]
    scene_refs: tuple[str, ...]
    covered_intervals: tuple[CoverageInterval, ...]
    blind_intervals: tuple[CoverageInterval, ...]
    unknown_intervals: tuple[CoverageInterval, ...]
    coverage_summary: CoverageSummary
    total_projection_chars: int
    manifest_semantic_digest: str
    content_digest: str
    schema_version: str = EVIDENCE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_id", identifier(self.manifest_id, "manifest_id"))
        object.__setattr__(self, "bundle_id", identifier(self.bundle_id, "bundle_id"))
        object.__setattr__(
            self, "owner_identity_digest", sha256_digest(self.owner_identity_digest, "owner_identity_digest")
        )
        for name in ("started_at", "ended_at", "sealed_at"):
            object.__setattr__(self, name, strict_utc(getattr(self, name), name))
        if self.ended_at < self.started_at:
            raise EvidenceManifestError("Manifest end cannot precede start")
        object.__setattr__(self, "seal_reason", EvidenceSealReason(self.seal_reason))
        snapshots = tuple(self.ordered_record_snapshots)
        if not snapshots or any(not isinstance(item, ManifestSemanticRecordSnapshot) for item in snapshots):
            raise EvidenceManifestError("Manifest requires semantic record snapshots")
        if tuple(sorted(snapshots, key=lambda item: item.stable_sort_key)) != snapshots:
            raise EvidenceManifestError("Manifest record snapshots are not stably sorted")
        if len({item.semantic_record_id for item in snapshots}) != len(snapshots):
            raise EvidenceManifestError("Manifest record identities must be unique")
        object.__setattr__(self, "ordered_record_snapshots", snapshots)
        modalities = tuple(sorted((SemanticModality(item) for item in self.modalities), key=lambda item: item.value))
        if not modalities or len(set(modalities)) != len(modalities):
            raise EvidenceManifestError("Manifest modalities must be non-empty and unique")
        object.__setattr__(self, "modalities", modalities)
        scenes = tuple(sorted(identifier_tuple(self.scene_refs, "scene_refs", maximum_items=10_000)))
        object.__setattr__(self, "scene_refs", scenes)
        for name in ("covered_intervals", "blind_intervals", "unknown_intervals"):
            intervals = tuple(getattr(self, name))
            if any(not isinstance(item, CoverageInterval) for item in intervals):
                raise TypeError(f"{name} must contain CoverageInterval values")
            object.__setattr__(self, name, intervals)
        object.__setattr__(self, "coverage_summary", CoverageSummary(self.coverage_summary))
        object.__setattr__(
            self, "total_projection_chars", non_negative_int(self.total_projection_chars, "total_projection_chars")
        )
        object.__setattr__(
            self,
            "manifest_semantic_digest",
            sha256_digest(self.manifest_semantic_digest, "manifest_semantic_digest"),
        )
        object.__setattr__(self, "content_digest", sha256_digest(self.content_digest, "content_digest"))
        object.__setattr__(self, "schema_version", identifier(self.schema_version, "schema_version", maximum=32))
        if self.manifest_semantic_digest != canonical_digest(self._semantic_payload()):
            raise EvidenceManifestError("Manifest semantic digest mismatch")
        if self.content_digest != canonical_digest(self._content_payload()):
            raise EvidenceManifestError("Manifest full content digest mismatch")
        expected_id = "manifest_" + canonical_digest(self._identity_payload())
        if self.manifest_id != expected_id:
            raise EvidenceManifestError("manifest_id does not match deterministic identity")

    @classmethod
    def seal(
        cls,
        bundle: SemanticEvidenceBundle,
        records: tuple[OwnerScopedSemanticRecord, ...],
        *,
        reason: EvidenceSealReason,
        sealed_at: datetime,
        max_coverage_intervals: int,
        max_manifest_encoded_bytes: int,
    ) -> EvidenceManifest:
        if not isinstance(bundle, SemanticEvidenceBundle) or bundle.state is not EvidenceBundleState.SEALED:
            raise EvidenceManifestError("only a SEALED Bundle can publish a Manifest")
        ordered = tuple(sorted(records, key=lambda item: item.stable_sort_key))
        if tuple(item.semantic_record_id for item in ordered) != bundle.ordered_semantic_record_ids:
            raise EvidenceManifestError("Bundle membership does not match semantic records")
        snapshots = tuple(ManifestSemanticRecordSnapshot.from_record(item) for item in ordered)
        covered, blind, unknown, summary = _coverage(
            ordered,
            started_at=bundle.started_at,
            ended_at=bundle.ended_at,
            maximum=max_coverage_intervals,
        )
        identity_payload = {
            "bundle_id": bundle.bundle_id,
            "ordered_record_digests": tuple(item.semantic_record_digest for item in snapshots),
            "seal_reason": EvidenceSealReason(reason).value,
            "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        }
        modalities = tuple(sorted({item.semantic_input.modality.value for item in ordered}))
        scene_refs = tuple(
            sorted({item.semantic_input.scene_ref for item in ordered if item.semantic_input.scene_ref is not None})
        )
        total_projection_chars = sum(item.projection_chars for item in ordered)
        semantic_payload = {
            "bundle_id": bundle.bundle_id,
            "owner_identity_digest": bundle.owner_identity_digest,
            "started_at": utc_text(bundle.started_at),
            "ended_at": utc_text(bundle.ended_at),
            "seal_reason": EvidenceSealReason(reason).value,
            "ordered_record_snapshots": tuple(item.to_dict() for item in snapshots),
            "modalities": modalities,
            "scene_refs": scene_refs,
            "covered_intervals": tuple(item.to_dict() for item in covered),
            "blind_intervals": tuple(item.to_dict() for item in blind),
            "unknown_intervals": tuple(item.to_dict() for item in unknown),
            "coverage_summary": summary.value,
            "total_projection_chars": total_projection_chars,
            "schema_version": EVIDENCE_MANIFEST_SCHEMA_VERSION,
        }
        semantic_digest = canonical_digest(semantic_payload)
        content_payload = {
            "manifest_id": "manifest_" + canonical_digest(identity_payload),
            **semantic_payload,
            "sealed_at": utc_text(sealed_at),
            "manifest_semantic_digest": semantic_digest,
        }
        manifest = cls(
            manifest_id="manifest_" + canonical_digest(identity_payload),
            bundle_id=bundle.bundle_id,
            owner_identity_digest=bundle.owner_identity_digest,
            started_at=bundle.started_at,
            ended_at=bundle.ended_at,
            sealed_at=sealed_at,
            seal_reason=reason,
            ordered_record_snapshots=snapshots,
            modalities=tuple(SemanticModality(item) for item in modalities),
            scene_refs=scene_refs,
            covered_intervals=covered,
            blind_intervals=blind,
            unknown_intervals=unknown,
            coverage_summary=summary,
            total_projection_chars=total_projection_chars,
            manifest_semantic_digest=semantic_digest,
            content_digest=canonical_digest(content_payload),
        )
        if len(canonical_json(manifest.to_dict()).encode("utf-8")) > max_manifest_encoded_bytes:
            raise EvidenceManifestError("Manifest canonical encoding exceeds its configured byte boundary")
        return manifest

    def _identity_payload(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "ordered_record_digests": tuple(item.semantic_record_digest for item in self.ordered_record_snapshots),
            "seal_reason": self.seal_reason.value,
            "schema_version": self.schema_version,
        }

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "owner_identity_digest": self.owner_identity_digest,
            "started_at": utc_text(self.started_at),
            "ended_at": utc_text(self.ended_at),
            "seal_reason": self.seal_reason.value,
            "ordered_record_snapshots": tuple(item.to_dict() for item in self.ordered_record_snapshots),
            "modalities": tuple(item.value for item in self.modalities),
            "scene_refs": self.scene_refs,
            "covered_intervals": tuple(item.to_dict() for item in self.covered_intervals),
            "blind_intervals": tuple(item.to_dict() for item in self.blind_intervals),
            "unknown_intervals": tuple(item.to_dict() for item in self.unknown_intervals),
            "coverage_summary": self.coverage_summary.value,
            "total_projection_chars": self.total_projection_chars,
            "schema_version": self.schema_version,
        }

    def _content_payload(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            **self._semantic_payload(),
            "sealed_at": utc_text(self.sealed_at),
            "manifest_semantic_digest": self.manifest_semantic_digest,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._content_payload(),
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        ingress_config: IngressConfig | None = None,
    ) -> EvidenceManifest:
        fields = frozenset(
            {
                "manifest_id",
                "bundle_id",
                "owner_identity_digest",
                "started_at",
                "ended_at",
                "sealed_at",
                "seal_reason",
                "ordered_record_snapshots",
                "modalities",
                "scene_refs",
                "covered_intervals",
                "blind_intervals",
                "unknown_intervals",
                "coverage_summary",
                "total_projection_chars",
                "manifest_semantic_digest",
                "content_digest",
                "schema_version",
            }
        )
        data = strict_fields(value, "evidence_manifest", fields)
        require_fields(data, "evidence_manifest", fields)
        return cls(
            manifest_id=data["manifest_id"],
            bundle_id=data["bundle_id"],
            owner_identity_digest=data["owner_identity_digest"],
            started_at=parse_utc(data["started_at"], "started_at"),
            ended_at=parse_utc(data["ended_at"], "ended_at"),
            sealed_at=parse_utc(data["sealed_at"], "sealed_at"),
            seal_reason=EvidenceSealReason(data["seal_reason"]),
            ordered_record_snapshots=tuple(
                ManifestSemanticRecordSnapshot.from_dict(item, ingress_config=ingress_config)
                for item in data["ordered_record_snapshots"]
            ),
            modalities=tuple(SemanticModality(item) for item in data["modalities"]),
            scene_refs=tuple(data["scene_refs"]),
            covered_intervals=tuple(CoverageInterval.from_dict(item) for item in data["covered_intervals"]),
            blind_intervals=tuple(CoverageInterval.from_dict(item) for item in data["blind_intervals"]),
            unknown_intervals=tuple(CoverageInterval.from_dict(item) for item in data["unknown_intervals"]),
            coverage_summary=CoverageSummary(data["coverage_summary"]),
            total_projection_chars=data["total_projection_chars"],
            manifest_semantic_digest=data["manifest_semantic_digest"],
            content_digest=data["content_digest"],
            schema_version=data["schema_version"],
        )


def _coverage(
    records: tuple[OwnerScopedSemanticRecord, ...],
    *,
    started_at: datetime,
    ended_at: datetime,
    maximum: int,
) -> tuple[
    tuple[CoverageInterval, ...],
    tuple[CoverageInterval, ...],
    tuple[CoverageInterval, ...],
    CoverageSummary,
]:
    explicit: dict[
        tuple[SemanticModality, str | None],
        list[tuple[datetime, datetime, CoverageStatus, str]],
    ] = {}
    for record in records:
        if record.semantic_input.record_kind is not SemanticRecordKind.COVERAGE_INTERVAL:
            continue
        payload = record.semantic_input.payload
        if not isinstance(payload, CoverageIntervalPayload):
            raise EvidenceManifestError("Coverage record has an invalid Payload")
        explicit.setdefault((payload.modality, payload.coverage_scope_ref), []).append(
            (
                record.semantic_input.event_time_start,
                record.semantic_input.event_time_end,
                payload.coverage_status,
                record.semantic_record_id,
            )
        )
    if not explicit:
        unknown: tuple[CoverageInterval, ...] = (
            CoverageInterval(SemanticModality.MULTIMODAL, None, started_at, ended_at, ()),
        )
        return (), (), unknown, CoverageSummary.UNKNOWN
    by_status: dict[CoverageStatus, list[CoverageInterval]] = {
        CoverageStatus.COVERED: [],
        CoverageStatus.BLIND: [],
        CoverageStatus.UNKNOWN: [],
    }
    priority = (CoverageStatus.BLIND, CoverageStatus.UNKNOWN, CoverageStatus.COVERED)
    for modality, scope in sorted(explicit, key=lambda item: (item[0].value, item[1] or "")):
        source = explicit[(modality, scope)]
        points = sorted({started_at, ended_at, *(item[0] for item in source), *(item[1] for item in source)})
        segments = tuple(zip(points, points[1:], strict=False))
        if not segments:
            statuses = {item[2] for item in source if item[0] <= started_at <= item[1]}
            status = next((item for item in priority if item in statuses), CoverageStatus.UNKNOWN)
            ids = tuple(sorted(item[3] for item in source if item[2] is status))
            by_status[status].append(CoverageInterval(modality, scope, started_at, ended_at, ids))
            continue
        for start, end in segments:
            covering = tuple(item for item in source if item[0] <= start and item[1] >= end)
            statuses = {item[2] for item in covering}
            status = next((item for item in priority if item in statuses), CoverageStatus.UNKNOWN)
            ids = tuple(sorted(item[3] for item in covering if item[2] is status))
            interval = CoverageInterval(modality, scope, start, end, ids)
            target = by_status[status]
            if (
                target
                and target[-1].modality == modality
                and target[-1].coverage_scope_ref == scope
                and target[-1].event_time_end == start
            ):
                previous = target.pop()
                target.append(
                    CoverageInterval(
                        modality,
                        scope,
                        previous.event_time_start,
                        end,
                        tuple(sorted(set(previous.semantic_record_ids) | set(ids))),
                    )
                )
            else:
                target.append(interval)
    explicit_modalities = {item[0] for item in explicit}
    record_modalities = {item.semantic_input.modality for item in records}
    for modality in sorted(record_modalities - explicit_modalities, key=lambda item: item.value):
        by_status[CoverageStatus.UNKNOWN].append(CoverageInterval(modality, None, started_at, ended_at, ()))
    total = sum(len(items) for items in by_status.values())
    if total > maximum:
        raise EvidenceManifestError("Coverage intervals exceed their configured boundary")
    covered = tuple(by_status[CoverageStatus.COVERED])
    blind = tuple(by_status[CoverageStatus.BLIND])
    unknown = tuple(by_status[CoverageStatus.UNKNOWN])
    if unknown:
        summary = CoverageSummary.UNKNOWN
    elif covered and not blind:
        summary = CoverageSummary.COVERED
    elif blind and not covered:
        summary = CoverageSummary.BLIND
    else:
        summary = CoverageSummary.MIXED
    return covered, blind, unknown, summary


__all__ = [
    "CoverageInterval",
    "CoverageSummary",
    "EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "EvidenceManifest",
    "ManifestSemanticRecordSnapshot",
]
