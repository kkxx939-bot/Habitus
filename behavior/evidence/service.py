"""语义记录、Bundle 推进与 Manifest 封存入口。"""

from __future__ import annotations

import time

from behavior.config import EvidenceConfig
from behavior.evidence.bundle import (
    EvidenceSealReason,
    SemanticEvidenceBundleAssembler,
    SemanticIngestResult,
    SemanticIngestStatus,
)
from behavior.evidence.manifest import EvidenceManifest
from behavior.ingress.model import IngressDecision, OwnerScopedSemanticRecord
from behavior.ingress.service import Clock, SystemClock
from behavior.persistence.contracts import BehaviorEvidenceClaimStore
from foundation.observability import ObservationEvent, ObservationStatus, Observer


class EvidenceService:
    def __init__(
        self,
        store: BehaviorEvidenceClaimStore,
        *,
        config: EvidenceConfig,
        observer: Observer,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(store, BehaviorEvidenceClaimStore):
            raise TypeError("store must implement BehaviorEvidenceClaimStore")
        if not isinstance(config, EvidenceConfig):
            raise TypeError("config must be EvidenceConfig")
        if not callable(getattr(observer, "record", None)):
            raise TypeError("observer must implement record")
        resolved_clock = clock or SystemClock()
        if not isinstance(resolved_clock, Clock):
            raise TypeError("clock must implement Clock")
        self.store = store
        self.config = config
        self.observer = observer
        self.clock = resolved_clock
        self.assembler = SemanticEvidenceBundleAssembler(config)

    def ingest(
        self,
        record: OwnerScopedSemanticRecord,
        decision: IngressDecision,
    ) -> SemanticIngestResult:
        if not isinstance(record, OwnerScopedSemanticRecord):
            raise TypeError("record must be OwnerScopedSemanticRecord")
        if not isinstance(decision, IngressDecision):
            raise TypeError("decision must be IngressDecision")
        started = time.monotonic()
        result = self.store.ingest_semantic_record(
            record,
            decision,
            self.assembler,
            sealed_at=self.clock.now(),
        )
        operation = (
            "semantic_record_late_rejected"
            if result.status is SemanticIngestStatus.LATE_REJECTED
            else "semantic_record_ingest"
        )
        self._observe(
            operation,
            started,
            {
                "record_kind": record.semantic_input.record_kind.value,
                "modality": record.semantic_input.modality.value,
                "result_count": len(result.manifest_ids),
                "reused": result.status is SemanticIngestStatus.REPLAYED,
            },
        )
        if result.active_bundle is not None:
            self._observe("evidence_bundle_opened", started, {"record_count": 1})
        if result.manifest_ids:
            self._observe("evidence_bundle_sealed", started, {"result_count": len(result.manifest_ids)})
            self._observe("manifest_published", started, {"result_count": len(result.manifest_ids)})
        return result

    def seal_bundle(
        self,
        bundle_id: str,
        *,
        reason: EvidenceSealReason = EvidenceSealReason.EXPLICIT,
    ) -> EvidenceManifest | None:
        started = time.monotonic()
        manifest = self.store.seal_bundle(
            bundle_id,
            reason=reason,
            assembler=self.assembler,
            sealed_at=self.clock.now(),
        )
        if manifest is not None:
            self._observe(
                "evidence_bundle_sealed",
                started,
                {"record_count": len(manifest.ordered_record_snapshots)},
            )
            self._observe("manifest_published", started, {"result_count": 1})
        return manifest

    def _observe(
        self,
        operation: str,
        started: float,
        attributes: dict[str, str | int | float | bool],
    ) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category="behavior",
                    operation=operation,
                    status=ObservationStatus.SUCCESS,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:
            return


__all__ = ["EvidenceService"]
