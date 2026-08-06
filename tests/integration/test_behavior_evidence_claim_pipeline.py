from __future__ import annotations

import asyncio

from behavior.claim import (
    ClaimNormalizationRouter,
    ClaimNormalizerRegistry,
    ClaimPipelineService,
    DeterministicClaimNormalizer,
)
from behavior.config import BehaviorConfig
from behavior.evidence import EvidenceService, SemanticIngestStatus
from behavior.ingress import (
    ActivitySegmentPayload,
    BoundarySignal,
    IngressTrustClass,
    SemanticActorRole,
    SemanticIngressAdapterRegistry,
    SemanticModality,
    SemanticRecordKind,
    SemanticRecordService,
    SemanticSubjectRole,
    UtteranceChannel,
    UtteranceSegmentPayload,
)
from behavior.owner import ConfirmedOwnerBinding
from behavior.persistence.sqlite import SQLiteBehaviorEvidenceClaimStore
from foundation.observability import NullObserver
from tests.unit.behavior.conftest import BASE_TIME, FakeAdapter, FakeClock, digest, make_input


def test_owner_scoped_semantic_records_to_atomic_claims_is_replayable(tmp_path) -> None:
    owner = ConfirmedOwnerBinding("local-owner", "resolver-v2", BASE_TIME, digest("owner"))
    config = BehaviorConfig()
    clock = FakeClock()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "behavior", config=config, initialize=True)
    adapters = SemanticIngressAdapterRegistry()
    visual = make_input(
        sequence=0,
        offset_seconds=0,
        kind=SemanticRecordKind.OWNER_ACTIVITY_SEGMENT,
        payload=ActivitySegmentPayload("preparing_tea", "IN_PROGRESS", {"quality": "high"}),
        subject_role=SemanticSubjectRole.OWNER,
        actor_role=SemanticActorRole.OWNER,
        modality=SemanticModality.VISION,
        stream_id="vision-semantics",
        upstream_subject_ref="opaque-track-7",
        source_confidence=0.91,
    )
    audio = make_input(
        sequence=0,
        offset_seconds=10,
        kind=SemanticRecordKind.OWNER_UTTERANCE_SEGMENT,
        payload=UtteranceSegmentPayload("tea is ready", "en", UtteranceChannel.VOICE),
        subject_role=SemanticSubjectRole.OWNER,
        actor_role=SemanticActorRole.OWNER,
        modality=SemanticModality.AUDIO,
        stream_id="audio-semantics",
        upstream_subject_ref="opaque-speaker-track-2",
        boundary_signal=BoundarySignal.END,
        source_confidence=0.99,
    )
    device = make_input(
        sequence=0,
        offset_seconds=5,
        stream_id="device-semantics",
        source_confidence=1.0,
    )
    adapters.register(
        FakeAdapter(
            visual,
            name="fake_visual",
            trust=IngressTrustClass.MULTIMODAL_MODEL_INFERRED,
            allowed=(SemanticRecordKind.OWNER_ACTIVITY_SEGMENT,),
        )
    )
    adapters.register(
        FakeAdapter(
            audio,
            name="fake_audio",
            trust=IngressTrustClass.OWNER_EXPLICIT,
            allowed=(SemanticRecordKind.OWNER_UTTERANCE_SEGMENT,),
            owner_speaker_binding=True,
        )
    )
    adapters.register(FakeAdapter(device, name="fake_device"))

    ingress = SemanticRecordService(store, adapters, config=config.ingress, clock=clock)
    evidence = EvidenceService(
        store,
        config=config.evidence,
        observer=NullObserver(),
        clock=clock,
    )
    normalizers = ClaimNormalizerRegistry()
    normalizers.register(DeterministicClaimNormalizer())
    router = ClaimNormalizationRouter(normalizers, config=config.claim)
    pipeline = ClaimPipelineService(
        store,
        ingress,
        evidence,
        normalizers,
        router,
        config=config.claim,
        observer=NullObserver(),
        clock=clock,
    )

    device_result = asyncio.run(pipeline.ingest_semantic("fake_device", {}, owner_binding=owner))[0]
    visual_result = asyncio.run(pipeline.ingest_semantic("fake_visual", {}, owner_binding=owner))[0]
    audio_result = asyncio.run(pipeline.ingest_semantic("fake_audio", {}, owner_binding=owner))[0]
    assert device_result.bundle_result.status is SemanticIngestStatus.ACCEPTED
    assert visual_result.bundle_result.status is SemanticIngestStatus.ACCEPTED
    assert audio_result.bundle_result.manifest_ids
    manifest = store.read_manifest(audio_result.bundle_result.manifest_ids[0])
    assert tuple(item.record_kind for item in manifest.ordered_record_snapshots) == (
        SemanticRecordKind.OWNER_ACTIVITY_SEGMENT,
        SemanticRecordKind.DEVICE_STATE,
        SemanticRecordKind.OWNER_UTTERANCE_SEGMENT,
    )

    first = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    replay = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    assert len(first.validated_claims) == len(first.accepted_claims) == 3
    assert replay.reused
    assert tuple(item.claim_id for item in replay.validated_claims) == tuple(
        item.claim_id for item in first.validated_claims
    )
    accepted = store.list_accepted_claims(
        start=manifest.started_at,
        end=clock.now(),
        limit=10,
    )
    assert {item.claim_id for item in accepted} == {item.claim_id for item in first.accepted_claims}
    assert all(type(item).__name__ == "Claim" for item in first.validated_claims)
