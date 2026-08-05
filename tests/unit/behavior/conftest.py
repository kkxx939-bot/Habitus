from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from behavior.claim import (
    ClaimPipelineService,
    ClaimProducerRegistry,
    DirectStructuredClaimProducer,
)
from behavior.config import BehaviorConfig
from behavior.evidence import EvidenceService
from behavior.owner import ConfirmedOwnerBinding
from behavior.persistence.sqlite import SQLiteBehaviorEvidenceClaimStore
from behavior.source import CaptureState, Modality, SourceRecord, SourceRecordService, SourceType
from foundation.observability import NullObserver

BASE_TIME = datetime(2026, 8, 5, 1, 2, 3, tzinfo=timezone.utc)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def direct_claim_projection(
    *,
    kind: str = "STATE_ASSERTION",
    predicate: str = "door_open",
    epistemic: str = "DIRECT_SOURCE",
    score: float = 1.0,
    object_refs: list[str] | None = None,
    alternative_group_id: str | None = None,
) -> dict[str, object]:
    activity = "bounded_activity" if kind == "ACTIVITY_PHASE" else None
    phase = "in_progress" if kind == "ACTIVITY_PHASE" else None
    return {
        "claim_kind": kind,
        "subject_role": "OWNER",
        "actor_role": "OWNER",
        "predicate": predicate,
        "semantic_family": "test_family",
        "activity": activity,
        "phase": phase,
        "object_refs": ["track-main"] if object_refs is None else object_refs,
        "location_ref": None,
        "epistemic_class": epistemic,
        "raw_score": score,
        "alternative_group_id": alternative_group_id,
        "semantic_payload": {"value": predicate},
        "human_summary": f"Auditable summary for {predicate}",
    }


@pytest.fixture
def owner() -> ConfirmedOwnerBinding:
    return ConfirmedOwnerBinding(
        "local-owner",
        "owner-router-v1",
        BASE_TIME,
        digest("owner-evidence"),
    )


def make_source(
    owner_binding: ConfirmedOwnerBinding,
    *,
    sequence: int = 0,
    offset_seconds: float = 0.0,
    duration_seconds: float = 0.0,
    source_type: SourceType = SourceType.DEVICE_STATE,
    modality: Modality = Modality.DEVICE_STATE,
    stream_id: str = "stream-main",
    correlation_key: str = "correlation-main",
    scene_ref: str | None = "scene-main",
    track_refs: tuple[str, ...] = ("track-main",),
    entity_refs: tuple[str, ...] = ("entity-main",),
    semantic_text: str | None = None,
    semantic_data: object | None = None,
    capture_state: CaptureState = CaptureState.COMPLETE,
    ingested_offset_seconds: float | None = None,
) -> SourceRecord:
    start = BASE_TIME + timedelta(seconds=offset_seconds)
    end = start + timedelta(seconds=duration_seconds)
    ingested = BASE_TIME + timedelta(
        seconds=offset_seconds if ingested_offset_seconds is None else ingested_offset_seconds
    )
    projection = (
        {"claim": direct_claim_projection()}
        if semantic_data is None and source_type in {
            SourceType.SENSOR_SAMPLE,
            SourceType.SENSOR_WINDOW,
            SourceType.DEVICE_STATE,
            SourceType.ROBOT_ACTION_LOG,
            SourceType.TOOL_RESULT_REFERENCE,
            SourceType.COVERAGE_SIGNAL,
            SourceType.UPSTREAM_SEMANTIC,
        }
        else ({} if semantic_data is None else semantic_data)
    )
    return SourceRecord(
        stream_id=stream_id,
        source_sequence=sequence,
        source_type=source_type,
        modality=modality,
        producer_ref="test-producer",
        event_time_start=start,
        event_time_end=end,
        ingested_at=ingested,
        payload_ref=f"blob://source/{stream_id}/{sequence}",
        payload_digest=digest(f"{stream_id}:{sequence}:{source_type.value}"),
        payload_media_type="application/json",
        payload_size_bytes=32,
        semantic_text=semantic_text,
        semantic_data=projection,
        scene_ref=scene_ref,
        track_refs=track_refs,
        entity_refs=entity_refs,
        correlation_key=correlation_key,
        capture_state=capture_state,
        owner_binding=owner_binding,
        attributes={"quality": "test"},
    )


@pytest.fixture
def behavior_config() -> BehaviorConfig:
    return BehaviorConfig()


@pytest.fixture
def store(tmp_path: Path, behavior_config: BehaviorConfig) -> SQLiteBehaviorEvidenceClaimStore:
    result = SQLiteBehaviorEvidenceClaimStore(tmp_path / "behavior", config=behavior_config)
    result.initialize()
    return result


def make_pipeline(
    store: SQLiteBehaviorEvidenceClaimStore,
    config: BehaviorConfig,
    *,
    registry: ClaimProducerRegistry | None = None,
) -> ClaimPipelineService:
    observer = NullObserver()
    source_service = SourceRecordService(store)
    evidence_service = EvidenceService(store, config=config.evidence, observer=observer)
    resolved_registry = registry or ClaimProducerRegistry()
    if registry is None:
        resolved_registry.register(DirectStructuredClaimProducer())
    return ClaimPipelineService(
        store,
        source_service,
        evidence_service,
        resolved_registry,
        config=config.claim,
        observer=observer,
    )


__all__ = [
    "BASE_TIME",
    "digest",
    "direct_claim_projection",
    "make_pipeline",
    "make_source",
]
