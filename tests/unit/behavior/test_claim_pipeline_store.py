from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from threading import Barrier, Thread

import pytest

from behavior.claim import (
    ClaimAdmissionGate,
    ClaimAdmissionStatus,
    ClaimProducerRegistry,
    ClaimProposal,
    ClaimProposalBatch,
    ClaimValidator,
    DirectStructuredClaimProducer,
    ProducerFingerprint,
)
from behavior.claim.producer import ClaimProducerKind
from behavior.config import BehaviorConfig
from behavior.errors import (
    ClaimValidationError,
    SourceRecordConflictError,
)
from behavior.evidence import EvidenceService, EvidenceWindowAssembler
from behavior.persistence.sqlite import SQLiteBehaviorEvidenceClaimStore
from behavior.source import Modality, SourceType
from foundation.observability import NullObserver
from tests.unit.behavior.conftest import (
    BASE_TIME,
    direct_claim_projection,
    make_pipeline,
    make_source,
)


class StaticProducer:
    kind = ClaimProducerKind.MODEL

    def __init__(self, name: str, batch: ClaimProposalBatch) -> None:
        self.name = name
        self.batch = batch
        self._fingerprint = ProducerFingerprint(
            name,
            "1",
            "test",
            "fake_chat",
            "fake_model",
            "test_prompt_v1",
        )
        self.calls = 0

    @property
    def fingerprint(self) -> ProducerFingerprint:
        return self._fingerprint

    async def produce(self, manifest):
        self.calls += 1
        return self.batch


def _manifest_with_sources(store, owner, records):
    service = EvidenceService(store, config=store.config.evidence, observer=NullObserver())
    active = None
    for record in records:
        result = service.ingest_source(record)
        active = result.active_window
    assert active is not None
    manifest = service.seal_window(active.window_id)
    assert manifest is not None
    return manifest


def _proposal(source_ids, **updates) -> ClaimProposal:
    value = {
        **direct_claim_projection(epistemic="MODEL_INFERRED"),
        "scene_ref": "scene-main",
        "time_start": "2026-08-05T01:02:03Z",
        "time_end": "2026-08-05T01:02:03Z",
        "time_uncertainty_ms": 0,
        "source_record_ids": list(source_ids),
    }
    value.update(updates)
    return ClaimProposal.model_validate(value)


def test_validator_binds_deterministic_claim_and_rejects_scope_forgery(store, owner, monkeypatch) -> None:
    record = make_source(owner, semantic_data={})
    manifest = _manifest_with_sources(store, owner, (record,))
    direct = DirectStructuredClaimProducer()
    proposal = ClaimProposal.model_validate(
        {
            **direct_claim_projection(),
            "scene_ref": "scene-main",
            "time_start": "2026-08-05T01:02:03Z",
            "time_end": "2026-08-05T01:02:03Z",
            "time_uncertainty_ms": 0,
            "source_record_ids": [record.source_record_id],
        }
    )
    validator = ClaimValidator(store, config=store.config.claim)
    first = validator.validate_and_bind(
        manifest=manifest,
        proposal=proposal,
        producer=direct.fingerprint,
        producer_kind=direct.kind,
        claim_batch_id="batch-test",
    )
    second = validator.validate_and_bind(
        manifest=manifest,
        proposal=proposal,
        producer=direct.fingerprint,
        producer_kind=direct.kind,
        claim_batch_id="batch-test",
    )
    assert first == second

    with pytest.raises(ClaimValidationError, match="outside the Manifest"):
        validator.validate_and_bind(
            manifest=manifest,
            proposal=replace(proposal, source_record_ids=("src_" + "f" * 64,)),
            producer=direct.fingerprint,
            producer_kind=direct.kind,
            claim_batch_id="batch-test",
        )
    with pytest.raises(ClaimValidationError, match="time range"):
        validator.validate_and_bind(
            manifest=manifest,
            proposal=replace(
                proposal,
                time_start=BASE_TIME - timedelta(seconds=1),
                time_end=BASE_TIME,
            ),
            producer=direct.fingerprint,
            producer_kind=direct.kind,
            claim_batch_id="batch-test",
        )
    with pytest.raises(ClaimValidationError, match="object_refs"):
        validator.validate_and_bind(
            manifest=manifest,
            proposal=replace(proposal, object_refs=("unknown-object",)),
            producer=direct.fingerprint,
            producer_kind=direct.kind,
            claim_batch_id="batch-test",
        )
    with pytest.raises(ClaimValidationError, match="model Producer"):
        validator.validate_and_bind(
            manifest=manifest,
            proposal=proposal,
            producer=direct.fingerprint,
            producer_kind=ClaimProducerKind.MODEL,
            claim_batch_id="batch-test",
        )
    user_explicit = replace(proposal, epistemic_class=__import__("behavior").EpistemicClass.USER_EXPLICIT)
    with pytest.raises(ClaimValidationError, match="Conversation or ASR"):
        validator.validate_and_bind(
            manifest=manifest,
            proposal=user_explicit,
            producer=direct.fingerprint,
            producer_kind=direct.kind,
            claim_batch_id="batch-test",
        )

    other_record = make_source(owner, sequence=1, offset_seconds=10, semantic_data={})
    other_manifest = _manifest_with_sources(store, owner, (other_record,))
    monkeypatch.setattr(store, "read_manifest", lambda _identity: other_manifest)
    with pytest.raises(ClaimValidationError, match="digest mismatch"):
        validator.validate_and_bind(
            manifest=manifest,
            proposal=proposal,
            producer=direct.fingerprint,
            producer_kind=direct.kind,
            claim_batch_id="batch-test",
        )


def test_pipeline_publishes_atomically_and_reuses_receipt(store, owner, behavior_config) -> None:
    pipeline = make_pipeline(store, behavior_config)
    ingested = pipeline.ingest_source(make_source(owner))
    manifest = pipeline.seal_window(ingested.active_window.window_id)
    first = asyncio.run(pipeline.process_manifest(manifest.manifest_id, ("direct_structured",)))
    second = asyncio.run(pipeline.process_manifest(manifest.manifest_id, ("direct_structured",)))
    assert len(first.accepted_claims) == 1
    assert second.reused
    assert second.processing_identity == first.processing_identity
    assert second.accepted_claims == first.accepted_claims
    assert store.read_receipt(first.processing_identity) is not None
    duplicate = ClaimAdmissionGate(store, config=behavior_config.claim).decide(
        first.accepted_claims[0],
        processing_identity="manual-duplicate-check",
    )
    assert duplicate.status is ClaimAdmissionStatus.EXACT_DUPLICATE


def test_state_repeat_suppression_does_not_suppress_change_kinds(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "repeat", config=config)
    store.initialize()
    pipeline = make_pipeline(store, config)
    first_ingest = pipeline.ingest_source(make_source(owner, sequence=0))
    first_manifest = pipeline.seal_window(first_ingest.active_window.window_id)
    first = asyncio.run(pipeline.process_manifest(first_manifest.manifest_id, ("direct_structured",)))
    assert len(first.accepted_claims) == 1

    repeated = make_source(owner, sequence=1, offset_seconds=10)
    repeated_ingest = pipeline.ingest_source(repeated)
    repeated_manifest = pipeline.seal_window(repeated_ingest.active_window.window_id)
    repeated_result = asyncio.run(
        pipeline.process_manifest(repeated_manifest.manifest_id, ("direct_structured",))
    )
    assert repeated_result.accepted_claims == ()
    assert repeated_result.rejected_decisions[0].status is ClaimAdmissionStatus.REPEATED_STATE_SUPPRESSED

    for sequence, kind in ((2, "STATE_TRANSITION"), (3, "FEEDBACK")):
        record = make_source(
            owner,
            sequence=sequence,
            offset_seconds=10 + sequence,
            semantic_data={"claim": direct_claim_projection(kind=kind)},
        )
        ingested = pipeline.ingest_source(record)
        manifest = pipeline.seal_window(ingested.active_window.window_id)
        result = asyncio.run(pipeline.process_manifest(manifest.manifest_id, ("direct_structured",)))
        assert len(result.accepted_claims) == 1
        assert result.accepted_claims[0].proposal.claim_kind.value == kind


def test_threshold_no_information_gain_and_alternative_candidates(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "admission", config=config)
    store.initialize()
    first_source = make_source(
        owner,
        source_type=SourceType.VLM_OUTPUT,
        modality=Modality.TEXT,
        semantic_data={},
    )
    second_source = make_source(
        owner,
        sequence=1,
        stream_id="stream-second",
        source_type=SourceType.AUDIO_SEMANTIC,
        modality=Modality.AUDIO,
        semantic_data={},
    )
    manifest = _manifest_with_sources(store, owner, (first_source, second_source))
    low = _proposal((first_source.source_record_id,), raw_score=0.1)
    same_a = _proposal((first_source.source_record_id,), predicate="same_semantics", raw_score=0.9)
    same_b = _proposal((second_source.source_record_id,), predicate="same_semantics", raw_score=0.9)
    alternative_a = _proposal(
        (first_source.source_record_id,),
        predicate="candidate_a",
        alternative_group_id="alternative-1",
        raw_score=0.9,
    )
    alternative_b = _proposal(
        (second_source.source_record_id,),
        predicate="candidate_b",
        alternative_group_id="alternative-1",
        raw_score=0.9,
    )
    transition_a = _proposal(
        (first_source.source_record_id,),
        claim_kind="STATE_TRANSITION",
        predicate="changed_state",
        raw_score=0.9,
    )
    transition_b = _proposal(
        (second_source.source_record_id,),
        claim_kind="STATE_TRANSITION",
        predicate="changed_state",
        raw_score=0.9,
    )
    batch = ClaimProposalBatch(
        False,
        (low, same_a, same_b, alternative_a, alternative_b, transition_a, transition_b),
    )
    producer = StaticProducer("static_model", batch)
    registry = ClaimProducerRegistry()
    registry.register(producer)
    pipeline = make_pipeline(store, config, registry=registry)
    result = asyncio.run(pipeline.process_manifest(manifest.manifest_id, (producer.name,)))
    statuses = {decision.status for decision in result.rejected_decisions}
    assert ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD in statuses
    assert ClaimAdmissionStatus.NO_INFORMATION_GAIN in statuses
    alternative_claims = [
        claim for claim in result.accepted_claims if claim.proposal.alternative_group_id == "alternative-1"
    ]
    assert len(alternative_claims) == 2
    transition_claims = [
        claim
        for claim in result.accepted_claims
        if claim.proposal.claim_kind.value == "STATE_TRANSITION"
    ]
    assert len(transition_claims) == 2


def test_concurrent_identical_processing_reuses_the_committed_receipt(store, owner, behavior_config) -> None:
    source = make_source(
        owner,
        source_type=SourceType.VLM_OUTPUT,
        modality=Modality.TEXT,
        semantic_data={},
    )
    manifest = _manifest_with_sources(store, owner, (source,))
    proposal = _proposal((source.source_record_id,), raw_score=0.9)

    class YieldingProducer(StaticProducer):
        async def produce(self, manifest):
            self.calls += 1
            await asyncio.sleep(0)
            return self.batch

    producer = YieldingProducer("yielding_model", ClaimProposalBatch(False, (proposal,)))
    registry = ClaimProducerRegistry()
    registry.register(producer)
    pipeline = make_pipeline(store, behavior_config, registry=registry)

    async def run_both():
        return await asyncio.gather(
            pipeline.process_manifest(manifest.manifest_id, (producer.name,)),
            pipeline.process_manifest(manifest.manifest_id, (producer.name,)),
        )

    first, second = asyncio.run(run_both())
    assert producer.calls == 2
    assert first.processing_identity == second.processing_identity
    assert first.accepted_claims == second.accepted_claims
    assert {first.reused, second.reused} == {False, True}


def test_processing_publication_rolls_back_on_failure(store, owner, behavior_config, monkeypatch) -> None:
    pipeline = make_pipeline(store, behavior_config)
    ingested = pipeline.ingest_source(make_source(owner))
    manifest = pipeline.seal_window(ingested.active_window.window_id)

    def fail_decision(*args, **kwargs):
        raise RuntimeError("simulated decision write failure")

    monkeypatch.setattr(store, "_insert_decision", fail_decision)
    with pytest.raises(RuntimeError, match="simulated"):
        asyncio.run(pipeline.process_manifest(manifest.manifest_id, ("direct_structured",)))
    assert store.claim_count() == 0
    assert store.read_claim("claim_" + "a" * 64) is None


def test_concurrent_source_replay_and_conflict_are_explicit(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "concurrent", config=config)
    store.initialize()
    assembler = EvidenceWindowAssembler(config.evidence)
    record = make_source(owner)
    barrier = Barrier(2)
    results = []
    errors = []

    def write(value):
        try:
            barrier.wait()
            results.append(store.ingest_source(value, assembler).status)
        except Exception as exc:
            errors.append(exc)

    same = make_source(owner)
    threads = [Thread(target=write, args=(record,)), Thread(target=write, args=(same,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert {status.value for status in results} == {"ACCEPTED", "REPLAYED"}

    conflict = make_source(
        owner,
        sequence=0,
        semantic_data={},
        modality=Modality.TEXT,
    )
    with pytest.raises(SourceRecordConflictError):
        store.ingest_source(conflict, assembler)
