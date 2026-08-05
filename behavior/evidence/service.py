"""SourceRecord 准入、窗口推进和 Manifest 封存入口。"""

from __future__ import annotations

import time

from behavior.config import EvidenceConfig
from behavior.evidence.manifest import EvidenceManifest
from behavior.evidence.model import EvidenceSealReason, SourceIngestResult, SourceIngestStatus
from behavior.evidence.window import EvidenceWindowAssembler
from behavior.persistence.contracts import BehaviorEvidenceClaimStore
from behavior.source.model import SourceRecord
from foundation.observability import ObservationEvent, ObservationStatus, Observer


class EvidenceService:
    def __init__(
        self,
        store: BehaviorEvidenceClaimStore,
        *,
        config: EvidenceConfig,
        observer: Observer,
    ) -> None:
        if not isinstance(store, BehaviorEvidenceClaimStore):
            raise TypeError("store must implement BehaviorEvidenceClaimStore")
        if not isinstance(config, EvidenceConfig):
            raise TypeError("config must be EvidenceConfig")
        if not callable(getattr(observer, "record", None)):
            raise TypeError("observer must implement record")
        self.store = store
        self.config = config
        self.observer = observer
        self.assembler = EvidenceWindowAssembler(config)

    def ingest_source(self, record: SourceRecord) -> SourceIngestResult:
        if not isinstance(record, SourceRecord):
            raise TypeError("record must be SourceRecord")
        started = time.monotonic()
        result = self.store.ingest_source(record, self.assembler)
        operation = (
            "source_late_rejected"
            if result.status is SourceIngestStatus.LATE_REJECTED
            else "source_ingest"
        )
        self._observe(
            operation,
            ObservationStatus.SUCCESS,
            started,
            {
                "source_type": record.source_type.value,
                "modality": record.modality.value,
                "result_count": len(result.manifest_ids),
                "reused": result.status is SourceIngestStatus.REPLAYED,
            },
        )
        if result.window_opened:
            self._observe(
                "evidence_window_opened",
                ObservationStatus.SUCCESS,
                started,
                {"record_count": 1},
            )
        if result.manifest_ids:
            self._observe(
                "evidence_window_sealed",
                ObservationStatus.SUCCESS,
                started,
                {"result_count": len(result.manifest_ids)},
            )
            self._observe(
                "manifest_published",
                ObservationStatus.SUCCESS,
                started,
                {"result_count": len(result.manifest_ids)},
            )
        return result

    def seal_window(
        self,
        window_id: str,
        *,
        reason: EvidenceSealReason = EvidenceSealReason.EXPLICIT,
    ) -> EvidenceManifest | None:
        started = time.monotonic()
        manifest = self.store.seal_window(window_id, reason=reason, assembler=self.assembler)
        if manifest is not None:
            self._observe(
                "evidence_window_sealed",
                ObservationStatus.SUCCESS,
                started,
                {"record_count": len(manifest.ordered_source_records)},
            )
            self._observe(
                "manifest_published",
                ObservationStatus.SUCCESS,
                started,
                {"result_count": 1},
            )
        return manifest

    def _observe(
        self,
        operation: str,
        status: ObservationStatus,
        started: float,
        attributes: dict[str, str | int | float | bool],
    ) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category="behavior",
                    operation=operation,
                    status=status,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:
            return


__all__ = ["EvidenceService"]
