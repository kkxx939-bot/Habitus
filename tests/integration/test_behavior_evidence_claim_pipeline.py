from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace

from behavior.claim import (
    ClaimPipelineService,
    ClaimProducerRegistry,
    ClaimProposal,
    ClaimProposalBatch,
    DirectStructuredClaimProducer,
    StructuredSemanticClaimProducer,
)
from behavior.config import BehaviorConfig
from behavior.evidence import EvidenceService
from behavior.owner import ConfirmedOwnerBinding
from behavior.persistence.sqlite import SQLiteBehaviorEvidenceClaimStore
from behavior.source import Modality, SourceRecordBatch, SourceRecordService, SourceType
from foundation.observability import NullObserver
from ModelClient.config import ChatModelConfig, ProviderConfig
from ModelClient.structured import StructuredChatClient
from tests.unit.behavior.conftest import BASE_TIME, digest, direct_claim_projection, make_source


def _structured_client(output: ClaimProposalBatch):
    client = object.__new__(StructuredChatClient)
    client.client = SimpleNamespace(
        config=ChatModelConfig(
            route=ProviderConfig(provider="test", adapter="fake_chat", model="fake-model"),
            structured_output_mode="json_schema",
        )
    )
    calls = []

    async def complete_model_async(self, request, **kwargs):
        calls.append((request, kwargs))
        return SimpleNamespace(value=output)

    client.complete_model_async = MethodType(complete_model_async, client)
    return client, calls


def test_multimodal_out_of_order_evidence_to_claim_flow_is_idempotent(tmp_path) -> None:
    owner = ConfirmedOwnerBinding(
        "local-owner",
        "owner-router-v1",
        BASE_TIME,
        digest("owner-evidence"),
    )
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "behavior", config=config)
    store.initialize()
    visual = make_source(
        owner,
        sequence=0,
        offset_seconds=2,
        stream_id="vision-stream",
        source_type=SourceType.VLM_OUTPUT,
        modality=Modality.VISION,
        semantic_text="The owner may be interacting with the sink area.",
        semantic_data={"candidate": "sink_interaction"},
    )
    audio = make_source(
        owner,
        sequence=0,
        offset_seconds=1,
        stream_id="audio-stream",
        source_type=SourceType.AUDIO_SEMANTIC,
        modality=Modality.AUDIO,
        semantic_text="Running water is detected.",
        semantic_data={"sound": "running_water"},
    )
    device = make_source(
        owner,
        sequence=0,
        offset_seconds=0,
        stream_id="device-stream",
        source_type=SourceType.DEVICE_STATE,
        modality=Modality.DEVICE_STATE,
        semantic_data={
            "claim": direct_claim_projection(
                predicate="faucet_on",
                object_refs=["track-main"],
            )
        },
    )
    model_proposal = ClaimProposal.model_validate(
        {
            **direct_claim_projection(
                kind="ACTIVITY_PHASE",
                predicate="sink_activity_in_progress",
                epistemic="MULTIMODAL_MODEL_INFERRED",
                score=0.82,
            ),
            "scene_ref": "scene-main",
            "time_start": BASE_TIME.isoformat().replace("+00:00", "Z"),
            "time_end": visual.event_time_end.isoformat().replace("+00:00", "Z"),
            "time_uncertainty_ms": 500,
            "source_record_ids": [visual.source_record_id, audio.source_record_id],
        }
    )
    client, calls = _structured_client(ClaimProposalBatch(False, (model_proposal,)))
    registry = ClaimProducerRegistry()
    registry.register(DirectStructuredClaimProducer())
    registry.register(StructuredSemanticClaimProducer(client, config=config.claim))
    source_service = SourceRecordService(store)
    evidence_service = EvidenceService(store, config=config.evidence, observer=NullObserver())
    pipeline = ClaimPipelineService(
        store,
        source_service,
        evidence_service,
        registry,
        config=config.claim,
        observer=NullObserver(),
    )
    results = pipeline.ingest_source_batch(SourceRecordBatch((visual, device, audio)))
    active = next(result.active_window for result in reversed(results) if result.active_window is not None)
    manifest = pipeline.seal_window(active.window_id)
    assert manifest is not None
    assert tuple(item.source_record_id for item in manifest.ordered_source_records) == (
        device.source_record_id,
        audio.source_record_id,
        visual.source_record_id,
    )
    first = asyncio.run(
        pipeline.process_manifest(
            manifest.manifest_id,
            ("direct_structured", "structured_semantic"),
        )
    )
    replay = asyncio.run(
        pipeline.process_manifest(
            manifest.manifest_id,
            ("direct_structured", "structured_semantic"),
        )
    )
    assert len(first.accepted_claims) == 2
    assert replay.reused
    assert replay.accepted_claims == first.accepted_claims
    assert len(calls) == 1
    assert all(claim.__class__.__name__ == "Claim" for claim in first.accepted_claims)
    forbidden_outputs = {
        "CanonicalEvent",
        "EventResolver",
        "Episode",
        "Opportunity",
        "BehaviorCase",
        "BehaviorPattern",
        "Prediction",
    }
    assert forbidden_outputs.isdisjoint({claim.__class__.__name__ for claim in first.accepted_claims})
