"""只依据事件时间和硬容量边界的确定性窗口规划。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from behavior.config import EvidenceConfig
from behavior.errors import EvidenceWindowError
from behavior.evidence.model import EvidenceSealReason, EvidenceWindow, EvidenceWindowState
from behavior.source.model import CaptureState, SourceRecord
from foundation.integrity import canonical_digest


@dataclass(frozen=True)
class EvidenceWindowPartition:
    records: tuple[SourceRecord, ...]
    seal_reason: EvidenceSealReason | None

    def __post_init__(self) -> None:
        if not self.records or any(not isinstance(record, SourceRecord) for record in self.records):
            raise EvidenceWindowError("window partition must contain SourceRecord values")
        if tuple(sorted(self.records, key=lambda item: item.stable_sort_key)) != self.records:
            raise EvidenceWindowError("window partition records must be stably sorted")


class EvidenceWindowAssembler:
    """对同一上游 correlation scope 做可重放硬切分。"""

    def __init__(self, config: EvidenceConfig) -> None:
        if not isinstance(config, EvidenceConfig):
            raise TypeError("config must be EvidenceConfig")
        self.config = config

    @staticmethod
    def grouping_key(record: SourceRecord) -> str:
        if not isinstance(record, SourceRecord):
            raise TypeError("record must be SourceRecord")
        return canonical_digest(
            {
                "correlation_key": record.correlation_key,
                "owner_binding_digest": record.owner_binding.binding_digest,
                "primary_track_ref": min(record.track_refs) if record.track_refs else None,
                "scene_ref": record.scene_ref,
            }
        )

    def is_late(
        self,
        record: SourceRecord,
        active: EvidenceWindow | None,
        *,
        committed_watermark: datetime | None = None,
    ) -> bool:
        thresholds: list[datetime] = []
        if active is not None:
            if active.state is not EvidenceWindowState.OPEN:
                raise EvidenceWindowError("only OPEN windows may accept SourceRecord values")
            thresholds.append(active.watermark)
        if committed_watermark is not None:
            thresholds.append(committed_watermark)
        return bool(thresholds) and record.event_time_end < max(thresholds)

    def partition(
        self,
        records: tuple[SourceRecord, ...],
        *,
        terminal_reason: EvidenceSealReason | None = None,
    ) -> tuple[EvidenceWindowPartition, ...]:
        if not records:
            return ()
        unique = {record.source_record_id: record for record in records}
        if len(unique) != len(records):
            raise EvidenceWindowError("window input cannot contain duplicate SourceRecord identities")
        ordered = tuple(sorted(unique.values(), key=lambda item: item.stable_sort_key))
        partitions: list[EvidenceWindowPartition] = []
        current: list[SourceRecord] = []
        projection_chars = 0

        def seal(reason: EvidenceSealReason) -> None:
            nonlocal current, projection_chars
            partitions.append(EvidenceWindowPartition(tuple(current), reason))
            current = []
            projection_chars = 0

        for record in ordered:
            record_projection = record.semantic_projection_chars
            if record_projection > self.config.max_projection_chars_per_window:
                raise EvidenceWindowError("one SourceRecord projection exceeds the window boundary")
            if (
                record.event_time_end - record.event_time_start
            ).total_seconds() > self.config.max_window_duration_seconds:
                raise EvidenceWindowError("one SourceRecord duration exceeds the window boundary")
            if current:
                previous = current[-1]
                gap = (record.event_time_start - previous.event_time_end).total_seconds()
                if gap > self.config.max_gap_seconds:
                    seal(EvidenceSealReason.MAX_GAP)
                elif (
                    record.event_time_end - current[0].event_time_start
                ).total_seconds() > self.config.max_window_duration_seconds:
                    seal(EvidenceSealReason.MAX_DURATION)
                elif len(current) >= self.config.max_records_per_window:
                    seal(EvidenceSealReason.MAX_RECORDS)
                elif projection_chars + record_projection > self.config.max_projection_chars_per_window:
                    seal(EvidenceSealReason.MAX_PROJECTION_SIZE)
            current.append(record)
            projection_chars += record_projection
            if len(current) == self.config.max_records_per_window:
                seal(EvidenceSealReason.MAX_RECORDS)
            elif projection_chars == self.config.max_projection_chars_per_window:
                seal(EvidenceSealReason.MAX_PROJECTION_SIZE)
        if current:
            if terminal_reason is None and current[-1].capture_state is CaptureState.STREAM_END:
                terminal_reason = EvidenceSealReason.STREAM_END
            partitions.append(EvidenceWindowPartition(tuple(current), terminal_reason))
        elif terminal_reason is not None and partitions:
            last = partitions[-1]
            partitions[-1] = EvidenceWindowPartition(last.records, terminal_reason)
        return tuple(partitions)

    def materialize(
        self,
        partition: EvidenceWindowPartition,
        *,
        grouping_key: str,
        generation: int,
        max_event_time: datetime,
        minimum_watermark: datetime | None = None,
    ) -> EvidenceWindow:
        records = partition.records
        observed_max = max(record.event_time_end for record in records)
        committed_max = max(observed_max, max_event_time)
        watermark = committed_max - timedelta(seconds=self.config.allowed_lateness_seconds)
        if minimum_watermark is not None:
            watermark = max(watermark, minimum_watermark)
        return EvidenceWindow.create(
            grouping_key=grouping_key,
            generation=generation,
            owner_binding_digest=records[0].owner_binding.binding_digest,
            correlation_key=records[0].correlation_key,
            scene_ref=records[0].scene_ref,
            primary_track_ref=min(records[0].track_refs) if records[0].track_refs else None,
            state=(
                EvidenceWindowState.SEALED
                if partition.seal_reason is not None
                else EvidenceWindowState.OPEN
            ),
            started_at=min(record.event_time_start for record in records),
            ended_at=max(record.event_time_end for record in records),
            max_event_time=committed_max,
            watermark=watermark,
            ordered_source_record_ids=tuple(record.source_record_id for record in records),
            total_projection_chars=sum(record.semantic_projection_chars for record in records),
        )


__all__ = ["EvidenceWindowAssembler", "EvidenceWindowPartition"]
