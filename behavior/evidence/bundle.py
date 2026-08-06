"""Owner-scoped 语义证据 Bundle 的事件时间硬分段。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from behavior._validation import (
    identifier,
    identifier_tuple,
    non_negative_int,
    parse_utc,
    require_fields,
    sha256_digest,
    strict_fields,
    strict_utc,
    utc_text,
)
from behavior.config import EvidenceConfig
from behavior.errors import EvidenceBundleError
from behavior.ingress.model import (
    BoundarySignal,
    ClockSyncStatus,
    IngressDecision,
    OwnerScopedSemanticRecord,
)
from foundation.integrity import canonical_digest

SEMANTIC_EVIDENCE_BUNDLE_SCHEMA_VERSION = "2"


class EvidenceBundleState(str, Enum):
    OPEN = "OPEN"
    SEALED = "SEALED"
    EXPIRED = "EXPIRED"


class EvidenceSealReason(str, Enum):
    MAX_GAP = "MAX_GAP"
    MAX_DURATION = "MAX_DURATION"
    MAX_RECORDS = "MAX_RECORDS"
    MAX_PROJECTION_SIZE = "MAX_PROJECTION_SIZE"
    EXPLICIT = "EXPLICIT"
    UPSTREAM_END = "UPSTREAM_END"


class SemanticIngestStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REPLAYED = "REPLAYED"
    LATE_REJECTED = "LATE_REJECTED"
    CLOCK_SKEW_REJECTED = "CLOCK_SKEW_REJECTED"
    EVENT_TOO_OLD_REJECTED = "EVENT_TOO_OLD_REJECTED"
    CAPACITY_REJECTED = "CAPACITY_REJECTED"


@dataclass(frozen=True)
class SemanticEvidenceBundle:
    bundle_id: str
    grouping_key: str
    generation: int
    owner_identity_digest: str
    correlation_id: str
    state: EvidenceBundleState
    started_at: datetime
    ended_at: datetime
    max_event_time: datetime | None
    watermark: datetime | None
    ordered_semantic_record_ids: tuple[str, ...]
    total_projection_chars: int
    seal_reason: EvidenceSealReason | None = None
    schema_version: str = SEMANTIC_EVIDENCE_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", identifier(self.bundle_id, "bundle_id"))
        object.__setattr__(self, "grouping_key", sha256_digest(self.grouping_key, "grouping_key"))
        object.__setattr__(self, "generation", non_negative_int(self.generation, "generation"))
        object.__setattr__(
            self,
            "owner_identity_digest",
            sha256_digest(self.owner_identity_digest, "owner_identity_digest"),
        )
        object.__setattr__(self, "correlation_id", identifier(self.correlation_id, "correlation_id"))
        object.__setattr__(self, "state", EvidenceBundleState(self.state))
        object.__setattr__(self, "started_at", strict_utc(self.started_at, "started_at"))
        object.__setattr__(self, "ended_at", strict_utc(self.ended_at, "ended_at"))
        if self.ended_at < self.started_at:
            raise EvidenceBundleError("Bundle end cannot precede start")
        if self.max_event_time is not None:
            object.__setattr__(self, "max_event_time", strict_utc(self.max_event_time, "max_event_time"))
        if self.watermark is not None:
            object.__setattr__(self, "watermark", strict_utc(self.watermark, "watermark"))
        if self.watermark is not None and self.max_event_time is None:
            raise EvidenceBundleError("Bundle watermark requires trusted max_event_time")
        if self.max_event_time is not None and self.watermark is not None and self.watermark > self.max_event_time:
            raise EvidenceBundleError("Bundle watermark cannot exceed max_event_time")
        record_ids = identifier_tuple(
            self.ordered_semantic_record_ids,
            "ordered_semantic_record_ids",
            maximum_items=1_000_000,
        )
        if not record_ids:
            raise EvidenceBundleError("Bundle cannot be empty")
        object.__setattr__(self, "ordered_semantic_record_ids", record_ids)
        object.__setattr__(
            self,
            "total_projection_chars",
            non_negative_int(self.total_projection_chars, "total_projection_chars"),
        )
        reason = None if self.seal_reason is None else EvidenceSealReason(self.seal_reason)
        object.__setattr__(self, "seal_reason", reason)
        if (self.state is EvidenceBundleState.OPEN) != (reason is None):
            raise EvidenceBundleError("only an OPEN Bundle may omit seal_reason")
        object.__setattr__(
            self,
            "schema_version",
            identifier(self.schema_version, "schema_version", maximum=32),
        )
        expected_id = "bundle_" + canonical_digest(
            {
                "generation": self.generation,
                "grouping_key": self.grouping_key,
                "schema_version": self.schema_version,
            }
        )
        if self.bundle_id != expected_id:
            raise EvidenceBundleError("bundle_id does not match deterministic identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "grouping_key": self.grouping_key,
            "generation": self.generation,
            "owner_identity_digest": self.owner_identity_digest,
            "correlation_id": self.correlation_id,
            "state": self.state.value,
            "started_at": utc_text(self.started_at),
            "ended_at": utc_text(self.ended_at),
            "max_event_time": None if self.max_event_time is None else utc_text(self.max_event_time),
            "watermark": None if self.watermark is None else utc_text(self.watermark),
            "ordered_semantic_record_ids": self.ordered_semantic_record_ids,
            "total_projection_chars": self.total_projection_chars,
            "seal_reason": None if self.seal_reason is None else self.seal_reason.value,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> SemanticEvidenceBundle:
        fields = frozenset(
            {
                "bundle_id",
                "grouping_key",
                "generation",
                "owner_identity_digest",
                "correlation_id",
                "state",
                "started_at",
                "ended_at",
                "max_event_time",
                "watermark",
                "ordered_semantic_record_ids",
                "total_projection_chars",
                "seal_reason",
                "schema_version",
            }
        )
        data = strict_fields(value, "semantic_evidence_bundle", fields)
        require_fields(data, "semantic_evidence_bundle", fields)
        return cls(
            bundle_id=data["bundle_id"],
            grouping_key=data["grouping_key"],
            generation=data["generation"],
            owner_identity_digest=data["owner_identity_digest"],
            correlation_id=data["correlation_id"],
            state=EvidenceBundleState(data["state"]),
            started_at=parse_utc(data["started_at"], "started_at"),
            ended_at=parse_utc(data["ended_at"], "ended_at"),
            max_event_time=None
            if data["max_event_time"] is None
            else parse_utc(data["max_event_time"], "max_event_time"),
            watermark=None if data["watermark"] is None else parse_utc(data["watermark"], "watermark"),
            ordered_semantic_record_ids=tuple(data["ordered_semantic_record_ids"]),
            total_projection_chars=data["total_projection_chars"],
            seal_reason=None if data["seal_reason"] is None else EvidenceSealReason(data["seal_reason"]),
            schema_version=data["schema_version"],
        )


@dataclass(frozen=True)
class SemanticBundlePartition:
    records: tuple[OwnerScopedSemanticRecord, ...]
    seal_reason: EvidenceSealReason | None


class SemanticEvidenceBundleAssembler:
    def __init__(self, config: EvidenceConfig) -> None:
        if not isinstance(config, EvidenceConfig):
            raise TypeError("config must be EvidenceConfig")
        self.config = config

    @staticmethod
    def grouping_key(record: OwnerScopedSemanticRecord) -> str:
        if not isinstance(record, OwnerScopedSemanticRecord):
            raise TypeError("record must be OwnerScopedSemanticRecord")
        return canonical_digest(
            {
                "owner_identity_digest": record.owner_identity_digest,
                "correlation_id": record.semantic_input.correlation_id,
            }
        )

    @staticmethod
    def advances_watermark(record: OwnerScopedSemanticRecord) -> bool:
        return record.semantic_input.clock_sync_status in {
            ClockSyncStatus.SYNCHRONIZED,
            ClockSyncStatus.OFFSET_ESTIMATED,
        }

    @staticmethod
    def watermark_event_time(record: OwnerScopedSemanticRecord) -> datetime | None:
        status = record.semantic_input.clock_sync_status
        if status is ClockSyncStatus.SYNCHRONIZED:
            return record.semantic_input.event_time_end
        if status is ClockSyncStatus.OFFSET_ESTIMATED:
            return record.semantic_input.event_time_end - timedelta(
                milliseconds=record.semantic_input.event_time_uncertainty_ms
            )
        return None

    def is_late(
        self,
        record: OwnerScopedSemanticRecord,
        *,
        committed_watermark: datetime | None,
    ) -> bool:
        if committed_watermark is None:
            return False
        uncertainty = (
            timedelta(0)
            if record.semantic_input.clock_sync_status is ClockSyncStatus.SYNCHRONIZED
            else timedelta(milliseconds=record.semantic_input.event_time_uncertainty_ms)
        )
        latest_plausible_end = record.semantic_input.event_time_end + uncertainty
        return latest_plausible_end < committed_watermark

    def partition(
        self,
        records: tuple[OwnerScopedSemanticRecord, ...],
    ) -> tuple[SemanticBundlePartition, ...]:
        ordered = tuple(sorted(records, key=lambda item: item.stable_sort_key))
        if not ordered:
            return ()
        if len({item.semantic_record_id for item in ordered}) != len(ordered):
            raise EvidenceBundleError("Bundle input records must have unique identities")
        result: list[SemanticBundlePartition] = []
        current: list[OwnerScopedSemanticRecord] = []

        def seal(reason: EvidenceSealReason) -> None:
            if current:
                result.append(SemanticBundlePartition(tuple(current), reason))
                current.clear()

        for record in ordered:
            duration = (record.semantic_input.event_time_end - record.semantic_input.event_time_start).total_seconds()
            if duration > self.config.max_bundle_duration_seconds:
                raise EvidenceBundleError("one semantic record duration exceeds the Bundle boundary")
            if record.projection_chars > self.config.max_projection_chars_per_bundle:
                raise EvidenceBundleError("one semantic record projection exceeds the Bundle boundary")
            if current:
                previous = current[-1]
                gap = (record.semantic_input.event_time_start - previous.semantic_input.event_time_end).total_seconds()
                if gap > self.config.max_gap_seconds:
                    seal(EvidenceSealReason.MAX_GAP)
                elif (
                    record.semantic_input.event_time_end - current[0].semantic_input.event_time_start
                ).total_seconds() > self.config.max_bundle_duration_seconds:
                    seal(EvidenceSealReason.MAX_DURATION)
                elif len(current) + 1 > self.config.max_records_per_bundle:
                    seal(EvidenceSealReason.MAX_RECORDS)
                elif (
                    sum(item.projection_chars for item in current) + record.projection_chars
                    > self.config.max_projection_chars_per_bundle
                ):
                    seal(EvidenceSealReason.MAX_PROJECTION_SIZE)
            current.append(record)
            if len(current) >= self.config.max_records_per_bundle:
                seal(EvidenceSealReason.MAX_RECORDS)
            elif sum(item.projection_chars for item in current) >= self.config.max_projection_chars_per_bundle:
                seal(EvidenceSealReason.MAX_PROJECTION_SIZE)
            elif record.semantic_input.boundary_signal is BoundarySignal.END:
                seal(EvidenceSealReason.UPSTREAM_END)
        if current:
            result.append(SemanticBundlePartition(tuple(current), None))
        return tuple(result)

    def materialize(
        self,
        partition: SemanticBundlePartition,
        *,
        grouping_key: str,
        generation: int,
        previous_max_event_time: datetime | None,
        previous_watermark: datetime | None,
    ) -> SemanticEvidenceBundle:
        if not partition.records:
            raise EvidenceBundleError("cannot materialize an empty Bundle")
        trusted_ends = tuple(
            event_time for item in partition.records if (event_time := self.watermark_event_time(item)) is not None
        )
        candidates = trusted_ends + (() if previous_max_event_time is None else (previous_max_event_time,))
        max_event_time = None if not candidates else max(candidates)
        candidate_watermark = (
            None if max_event_time is None else max_event_time - timedelta(seconds=self.config.allowed_lateness_seconds)
        )
        watermark_values = tuple(item for item in (previous_watermark, candidate_watermark) if item is not None)
        watermark = None if not watermark_values else max(watermark_values)
        bundle_id = "bundle_" + canonical_digest(
            {
                "generation": generation,
                "grouping_key": grouping_key,
                "schema_version": SEMANTIC_EVIDENCE_BUNDLE_SCHEMA_VERSION,
            }
        )
        reason = partition.seal_reason
        return SemanticEvidenceBundle(
            bundle_id=bundle_id,
            grouping_key=grouping_key,
            generation=generation,
            owner_identity_digest=partition.records[0].owner_identity_digest,
            correlation_id=partition.records[0].semantic_input.correlation_id,
            state=EvidenceBundleState.OPEN if reason is None else EvidenceBundleState.SEALED,
            started_at=min(item.semantic_input.event_time_start for item in partition.records),
            ended_at=max(item.semantic_input.event_time_end for item in partition.records),
            max_event_time=max_event_time,
            watermark=watermark,
            ordered_semantic_record_ids=tuple(item.semantic_record_id for item in partition.records),
            total_projection_chars=sum(item.projection_chars for item in partition.records),
            seal_reason=reason,
        )


@dataclass(frozen=True)
class SemanticIngestResult:
    status: SemanticIngestStatus
    semantic_record_id: str
    decision: IngressDecision
    active_bundle: SemanticEvidenceBundle | None = None
    manifest_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SemanticIngestStatus(self.status))
        object.__setattr__(self, "semantic_record_id", identifier(self.semantic_record_id, "semantic_record_id"))
        if not isinstance(self.decision, IngressDecision):
            raise TypeError("decision must be IngressDecision")
        if self.active_bundle is not None and not isinstance(self.active_bundle, SemanticEvidenceBundle):
            raise TypeError("active_bundle must be SemanticEvidenceBundle or None")
        object.__setattr__(
            self,
            "manifest_ids",
            identifier_tuple(self.manifest_ids, "manifest_ids", maximum_items=1_000),
        )


__all__ = [
    "EvidenceBundleState",
    "EvidenceSealReason",
    "SEMANTIC_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "SemanticBundlePartition",
    "SemanticEvidenceBundle",
    "SemanticEvidenceBundleAssembler",
    "SemanticIngestResult",
    "SemanticIngestStatus",
]
