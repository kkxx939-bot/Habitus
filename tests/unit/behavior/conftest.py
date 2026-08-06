from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from behavior.claim import (
    ClaimCompatibilityPolicy,
    ClaimNormalizationRouter,
    ClaimNormalizerKind,
    ClaimNormalizerRegistry,
    ClaimPipelineService,
    ClaimSemanticProposalBatch,
    DeterministicClaimNormalizer,
    NormalizerFingerprint,
)
from behavior.config import BehaviorConfig, IngressConfig
from behavior.evidence.service import EvidenceService
from behavior.ingress import (
    BoundarySignal,
    ClockSyncStatus,
    DeviceStatePayload,
    IngressAdapterCapability,
    IngressDecision,
    IngressDecisionStatus,
    IngressTrustClass,
    OwnerScopedSemanticRecord,
    ProducerFingerprint,
    RecordIntegrity,
    SemanticActorRole,
    SemanticIngressAdapterRegistry,
    SemanticModality,
    SemanticRecordInput,
    SemanticRecordInputBatch,
    SemanticRecordKind,
    SemanticRecordService,
    SemanticSubjectRole,
)
from behavior.ingress.service import AcceptedSemanticIngress
from behavior.owner import ConfirmedOwnerBinding
from behavior.persistence.sqlite import SQLiteBehaviorEvidenceClaimStore
from foundation.integrity import canonical_digest
from foundation.observability import NullObserver
from ModelClient import StructuredChatClient

BASE_TIME = datetime(2026, 8, 5, 1, 2, 3, tzinfo=timezone.utc)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class FakeClock:
    def __init__(self, current: datetime = BASE_TIME) -> None:
        self.current = current

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: float = 1.0) -> None:
        self.current += timedelta(seconds=seconds)


class NoopModelNormalizer:
    name = "model_text"
    kind = ClaimNormalizerKind.MODEL
    allowed_record_kinds = frozenset(
        {SemanticRecordKind.FREE_TEXT_SEMANTIC, SemanticRecordKind.OWNER_UTTERANCE_SEGMENT}
    )

    def __init__(self) -> None:
        self.model_client = object.__new__(StructuredChatClient)
        self.compatibility_policy = ClaimCompatibilityPolicy()
        self.fingerprint = NormalizerFingerprint(
            self.name,
            "noop-test-v3",
            self.kind,
            "test",
            "noop",
            "model",
            "noop-v3",
        )

    async def normalize(self, record: OwnerScopedSemanticRecord) -> ClaimSemanticProposalBatch:
        del record
        return ClaimSemanticProposalBatch(True, ())


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def owner() -> ConfirmedOwnerBinding:
    return ConfirmedOwnerBinding(
        "local-owner",
        "owner-router-v2",
        BASE_TIME,
        digest("owner-evidence"),
    )


def make_input(
    *,
    sequence: int = 0,
    offset_seconds: float = 0.0,
    duration_seconds: float = 0.0,
    kind: SemanticRecordKind = SemanticRecordKind.DEVICE_STATE,
    payload: object | None = None,
    subject_role: SemanticSubjectRole = SemanticSubjectRole.ENVIRONMENT,
    actor_role: SemanticActorRole = SemanticActorRole.SYSTEM,
    modality: SemanticModality = SemanticModality.DEVICE,
    stream_id: str = "semantic-stream",
    correlation_id: str = "correlation-main",
    boundary_signal: BoundarySignal = BoundarySignal.CONTINUE,
    clock_sync_status: ClockSyncStatus = ClockSyncStatus.SYNCHRONIZED,
    uncertainty_ms: int = 0,
    scene_ref: str | None = "scene-main",
    upstream_subject_ref: str | None = None,
    object_refs: tuple[str, ...] = (),
    entity_refs: tuple[str, ...] = (),
    location_ref: str | None = None,
    evidence_refs: tuple[object, ...] = (),
    source_confidence: float = 1.0,
) -> SemanticRecordInput:
    start = BASE_TIME + timedelta(seconds=offset_seconds)
    return SemanticRecordInput(
        stream_id=stream_id,
        source_sequence=sequence,
        record_kind=kind,
        subject_role=subject_role,
        actor_role=actor_role,
        modality=modality,
        event_time_start=start,
        event_time_end=start + timedelta(seconds=duration_seconds),
        event_time_uncertainty_ms=uncertainty_ms,
        clock_domain="upstream-clock",
        clock_sync_status=clock_sync_status,
        correlation_id=correlation_id,
        boundary_signal=boundary_signal,
        scene_ref=scene_ref,
        upstream_subject_ref=upstream_subject_ref,
        object_refs=object_refs,
        entity_refs=entity_refs,
        location_ref=location_ref,
        payload=payload or DeviceStatePayload("device-main", "power", "on"),
        evidence_refs=evidence_refs,
        source_confidence=source_confidence,
        integrity=RecordIntegrity.COMPLETE,
    )


def producer_fingerprint(name: str = "fake-device") -> ProducerFingerprint:
    return ProducerFingerprint(name, "2", "semantic-v2", "none", "none", "none", "2")


def bind_record(
    owner_binding: ConfirmedOwnerBinding,
    semantic_input: SemanticRecordInput | None = None,
    *,
    trust: IngressTrustClass = IngressTrustClass.DIRECT_DEVICE_FACT,
    ingested_at: datetime = BASE_TIME,
    producer: ProducerFingerprint | None = None,
) -> OwnerScopedSemanticRecord:
    return OwnerScopedSemanticRecord(
        semantic_input=semantic_input or make_input(),
        owner_binding=owner_binding,
        producer_fingerprint=producer or producer_fingerprint(),
        ingress_trust_class=trust,
        ingested_at=ingested_at,
    )


def accepted_decision(record: OwnerScopedSemanticRecord, *, decided_at: datetime = BASE_TIME) -> IngressDecision:
    return IngressDecision(
        status=IngressDecisionStatus.ACCEPTED,
        reason_code="semantic_record_clock_accepted",
        record=record,
        decided_at=decided_at,
    )


def accepted_ingress(
    record: OwnerScopedSemanticRecord,
    *,
    decided_at: datetime = BASE_TIME,
    ingress_config: IngressConfig | None = None,
) -> AcceptedSemanticIngress:
    capability = IngressAdapterCapability(
        record.ingress_trust_class,
        (record.semantic_input.record_kind,),
        1,
        owner_speaker_binding=record.ingress_trust_class is IngressTrustClass.OWNER_EXPLICIT,
    )
    registry = SemanticIngressAdapterRegistry()
    adapter = FakeAdapter(
        record.semantic_input,
        name=record.producer_fingerprint.producer_name,
        trust=record.ingress_trust_class,
        allowed=(record.semantic_input.record_kind,),
        owner_speaker_binding=record.ingress_trust_class is IngressTrustClass.OWNER_EXPLICIT,
    )
    adapter.fingerprint = record.producer_fingerprint
    adapter.capabilities = capability
    registry.register(adapter)
    return AcceptedSemanticIngress(
        record=record,
        decision=accepted_decision(record, decided_at=decided_at),
        adapter_name=adapter.name,
        adapter_registry=registry,
        adapter_fingerprint=record.producer_fingerprint.digest,
        capability=capability,
        capability_digest=capability.digest,
        ingress_policy_digest=canonical_digest((ingress_config or IngressConfig()).__dict__),
    )


class FakeAdapter:
    def __init__(
        self,
        records: SemanticRecordInput | SemanticRecordInputBatch,
        *,
        name: str = "fake_semantic",
        trust: IngressTrustClass = IngressTrustClass.DIRECT_DEVICE_FACT,
        allowed: tuple[SemanticRecordKind, ...] = (SemanticRecordKind.DEVICE_STATE,),
        owner_speaker_binding: bool = False,
    ) -> None:
        self.name = name
        self.records = records
        self.fingerprint = producer_fingerprint(name)
        self.capabilities = IngressAdapterCapability(
            trust,
            allowed,
            32,
            owner_speaker_binding=owner_speaker_binding,
        )

    async def adapt(
        self,
        payload: object,
        *,
        owner_binding: ConfirmedOwnerBinding,
    ) -> SemanticRecordInput | SemanticRecordInputBatch:
        del payload, owner_binding
        return self.records


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
    clock: FakeClock | None = None,
) -> ClaimPipelineService:
    resolved_clock = clock or FakeClock()
    observer = NullObserver()
    ingress = SemanticRecordService(
        store,
        SemanticIngressAdapterRegistry(),
        config=config.ingress,
        clock=resolved_clock,
    )
    evidence = EvidenceService(
        store,
        config=config.evidence,
        observer=observer,
        clock=resolved_clock,
        adapters=ingress.adapters,
    )
    normalizers = ClaimNormalizerRegistry()
    normalizers.register(DeterministicClaimNormalizer())
    normalizers.register(NoopModelNormalizer())
    router = ClaimNormalizationRouter(normalizers, config=config.claim)
    return ClaimPipelineService(
        store,
        ingress,
        evidence,
        normalizers,
        router,
        config=config.claim,
        observer=observer,
        clock=resolved_clock,
    )


__all__ = [
    "BASE_TIME",
    "FakeAdapter",
    "FakeClock",
    "NoopModelNormalizer",
    "accepted_decision",
    "accepted_ingress",
    "bind_record",
    "digest",
    "make_input",
    "make_pipeline",
    "producer_fingerprint",
]
