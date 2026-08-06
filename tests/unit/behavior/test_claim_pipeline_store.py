from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import timedelta

import pytest

from behavior.claim import (
    ClaimAdmissionStatus,
    ClaimBinder,
    ClaimConfidencePolicy,
    ClaimDerivationClass,
    ClaimKind,
    ClaimSemanticProposal,
    DeterministicClaimNormalizer,
    EpistemicClass,
)
from behavior.config import BehaviorConfig
from behavior.errors import ClaimBindingError, ClaimStoreError
from behavior.evidence.service import EvidenceService
from behavior.ingress import (
    ActionEventPayload,
    ActivitySegmentPayload,
    BoundarySignal,
    IngressTrustClass,
    PhaseHint,
    SemanticActorRole,
    SemanticModality,
    SemanticRecordKind,
    SemanticSubjectRole,
    SensorFactPayload,
    StateTransitionPayload,
    UtteranceChannel,
    UtteranceSegmentPayload,
)
from behavior.persistence.sqlite import SQLiteBehaviorEvidenceClaimStore
from foundation.observability import NullObserver
from tests.unit.behavior.conftest import (
    BASE_TIME,
    FakeClock,
    accepted_ingress,
    bind_record,
    make_input,
    make_pipeline,
)


def publish_manifest(store, owner, semantic_input, *, clock: FakeClock | None = None):
    service = EvidenceService(
        store,
        config=store.config.evidence,
        observer=NullObserver(),
        clock=clock,
    )
    record = bind_record(owner, semantic_input, trust=_trust_for(semantic_input.record_kind))
    result = service.ingest(accepted_ingress(record))
    manifest = (
        store.read_manifest(result.manifest_ids[0])
        if result.manifest_ids
        else service.seal_bundle(result.active_bundle.bundle_id)
    )
    assert manifest is not None
    return manifest, record


def _trust_for(kind: SemanticRecordKind) -> IngressTrustClass:
    if kind is SemanticRecordKind.OWNER_STATE_TRANSITION:
        return IngressTrustClass.SENSOR_INFERRED
    return IngressTrustClass.DIRECT_DEVICE_FACT


def proposal(**updates: object) -> ClaimSemanticProposal:
    values: dict[str, object] = {
        "claim_kind": ClaimKind.STATE_ASSERTION,
        "predicate": "power",
        "semantic_family": "device_state",
        "activity": None,
        "phase": None,
        "object_refs": (),
        "location_ref": None,
        "semantic_payload": {"value": "on"},
        "human_summary": "Device state",
        "local_alternative_group_id": None,
        "normalizer_confidence": 1.0,
    }
    values.update(updates)
    return ClaimSemanticProposal(**values)


def test_binder_owns_role_time_epistemic_and_confidence(store, owner) -> None:
    manifest, record = publish_manifest(store, owner, make_input())
    fingerprint = DeterministicClaimNormalizer().fingerprint
    claim = ClaimBinder(config=store.config.claim, clock=FakeClock(BASE_TIME + timedelta(hours=1))).bind(
        manifest,
        record,
        proposal(),
        fingerprint,
    )
    assert claim.subject_role is record.semantic_input.subject_role
    assert claim.actor_role is record.semantic_input.actor_role
    assert claim.time_start == record.semantic_input.event_time_start
    assert claim.time_end == record.semantic_input.event_time_end
    assert claim.source_epistemic_class is EpistemicClass.DIRECT_SOURCE
    assert claim.effective_confidence == record.semantic_input.source_confidence
    assert claim.created_at == BASE_TIME + timedelta(hours=1)


def test_model_confidence_policy_is_conservative() -> None:
    policy = ClaimConfidencePolicy()
    assert (
        policy.effective(
            source_confidence=0.8,
            normalizer_confidence=0.6,
            derivation_class=ClaimDerivationClass.MODEL,
        )
        == 0.6
    )


@pytest.mark.parametrize(
    ("semantic_input", "trust", "claim_updates", "expected"),
    [
        (
            make_input(),
            IngressTrustClass.DIRECT_DEVICE_FACT,
            {},
            EpistemicClass.DIRECT_SOURCE,
        ),
        (
            make_input(
                kind=SemanticRecordKind.ROBOT_ACTION_EVENT,
                payload=ActionEventPayload("wave", "completed", None, {}),
                subject_role=SemanticSubjectRole.ROBOT,
                actor_role=SemanticActorRole.ROBOT,
                modality=SemanticModality.ROBOT,
            ),
            IngressTrustClass.DIRECT_SYSTEM_LOG,
            {"claim_kind": ClaimKind.ROBOT_ACTION},
            EpistemicClass.DIRECT_SOURCE,
        ),
        (
            make_input(
                kind=SemanticRecordKind.OWNER_UTTERANCE_SEGMENT,
                payload=UtteranceSegmentPayload("hello", "en", UtteranceChannel.VOICE),
                subject_role=SemanticSubjectRole.OWNER,
                actor_role=SemanticActorRole.OWNER,
                modality=SemanticModality.AUDIO,
            ),
            IngressTrustClass.OWNER_EXPLICIT,
            {"claim_kind": ClaimKind.UTTERANCE},
            EpistemicClass.USER_EXPLICIT,
        ),
        (
            make_input(
                kind=SemanticRecordKind.OWNER_SENSOR_FACT,
                payload=SensorFactPayload("heart_rate", 70, "bpm", None, {}),
                subject_role=SemanticSubjectRole.OWNER,
                actor_role=SemanticActorRole.SYSTEM,
                modality=SemanticModality.SENSOR,
            ),
            IngressTrustClass.SENSOR_INFERRED,
            {},
            EpistemicClass.SENSOR_INFERRED,
        ),
        (
            make_input(
                kind=SemanticRecordKind.OWNER_ACTIVITY_SEGMENT,
                payload=ActivitySegmentPayload("walking", PhaseHint.IN_PROGRESS, {}),
                subject_role=SemanticSubjectRole.OWNER,
                actor_role=SemanticActorRole.OWNER,
                modality=SemanticModality.MULTIMODAL,
            ),
            IngressTrustClass.MULTIMODAL_MODEL_INFERRED,
            {
                "claim_kind": ClaimKind.ACTIVITY_PHASE,
                "activity": "walking",
                "phase": "in_progress",
            },
            EpistemicClass.MULTIMODAL_MODEL_INFERRED,
        ),
    ],
)
def test_binder_maps_system_trust_to_epistemic_class(
    store,
    owner,
    semantic_input,
    trust,
    claim_updates,
    expected,
) -> None:
    service = EvidenceService(store, config=store.config.evidence, observer=NullObserver())
    record = bind_record(owner, semantic_input, trust=trust)
    result = service.ingest(accepted_ingress(record))
    manifest = service.seal_bundle(result.active_bundle.bundle_id)
    assert manifest is not None
    claim = ClaimBinder(config=store.config.claim).bind(
        manifest,
        record,
        proposal(**claim_updates),
        DeterministicClaimNormalizer().fingerprint,
    )
    assert claim.source_epistemic_class is expected


def test_binder_rejects_cross_record_references_and_wrong_kind(store, owner) -> None:
    semantic_input = make_input(object_refs=("object-a",), location_ref="kitchen")
    manifest, record = publish_manifest(store, owner, semantic_input)
    binder = ClaimBinder(config=store.config.claim)
    fingerprint = DeterministicClaimNormalizer().fingerprint
    with pytest.raises(ClaimBindingError):
        binder.bind(manifest, record, proposal(object_refs=("object-b",)), fingerprint)
    with pytest.raises(ClaimBindingError):
        binder.bind(manifest, record, proposal(location_ref="garage"), fingerprint)
    with pytest.raises(ClaimBindingError):
        binder.bind(manifest, record, proposal(claim_kind=ClaimKind.ROBOT_ACTION), fingerprint)


def test_claim_identity_excludes_created_at(store, owner) -> None:
    manifest, record = publish_manifest(store, owner, make_input())
    fingerprint = DeterministicClaimNormalizer().fingerprint
    first = ClaimBinder(config=store.config.claim, clock=FakeClock(BASE_TIME)).bind(
        manifest, record, proposal(), fingerprint
    )
    second = ClaimBinder(
        config=store.config.claim,
        clock=FakeClock(BASE_TIME + timedelta(days=1)),
    ).bind(manifest, record, proposal(), fingerprint)
    assert first.claim_id == second.claim_id
    assert first.created_at != second.created_at


def test_pipeline_routes_automatically_publishes_and_reuses(store, owner, behavior_config) -> None:
    manifest, _ = publish_manifest(
        store,
        owner,
        make_input(boundary_signal=BoundarySignal.END),
    )
    pipeline = make_pipeline(store, behavior_config)
    first = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    replay = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    assert len(first.core_result.validated_claims) == len(first.core_result.accepted_claims) == 1
    assert replay.core_result.reused
    assert replay.core_result.processing_identity == first.core_result.processing_identity
    assert replay.core_result.accepted_claims[0].claim_id == first.core_result.accepted_claims[0].claim_id
    assert (
        store.read_claim(first.core_result.accepted_claims[0].claim_id)
        == first.core_result.accepted_claims[0]
    )


def test_below_threshold_claim_is_still_durable_and_not_accepted(tmp_path, owner) -> None:
    config = BehaviorConfig(claim=replace(BehaviorConfig().claim, min_direct_confidence=0.9))
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "threshold", config=config, initialize=True)
    manifest, _ = publish_manifest(
        store,
        owner,
        make_input(source_confidence=0.2, boundary_signal=BoundarySignal.END),
    )
    pipeline = make_pipeline(store, config)
    result = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    assert result.core_result.accepted_claims == ()
    assert result.core_result.rejected_decisions[0].status is ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD
    claim = result.core_result.validated_claims[0]
    assert store.read_claim(claim.claim_id) == claim
    assert store.read_claim_decision(claim.claim_id).status is ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD
    assert (
        pipeline.list_accepted_claims(
            start=BASE_TIME - timedelta(days=1),
            end=BASE_TIME + timedelta(days=1),
            limit=10,
        )
        == ()
    )


def test_repeated_state_is_suppressed_but_transition_is_not(store, owner, behavior_config) -> None:
    pipeline = make_pipeline(store, behavior_config)
    first_manifest, _ = publish_manifest(
        store,
        owner,
        make_input(sequence=0, boundary_signal=BoundarySignal.END),
    )
    first = asyncio.run(pipeline.process_manifest(first_manifest.manifest_id))
    second_manifest, _ = publish_manifest(
        store,
        owner,
        make_input(sequence=1, offset_seconds=10, boundary_signal=BoundarySignal.END),
    )
    second = asyncio.run(pipeline.process_manifest(second_manifest.manifest_id))
    assert first.core_result.accepted_claims
    assert second.core_result.rejected_decisions[0].status is ClaimAdmissionStatus.REPEATED_STATE_SUPPRESSED

    transition_input = make_input(
        sequence=2,
        offset_seconds=11,
        kind=SemanticRecordKind.OWNER_STATE_TRANSITION,
        payload=StateTransitionPayload("presence", False, True),
        subject_role=SemanticSubjectRole.OWNER,
        actor_role=SemanticActorRole.OWNER,
        boundary_signal=BoundarySignal.END,
    )
    transition_manifest, _ = publish_manifest(store, owner, transition_input)
    transition = asyncio.run(pipeline.process_manifest(transition_manifest.manifest_id))
    assert transition.core_result.accepted_claims[0].proposal.claim_kind is ClaimKind.STATE_TRANSITION


def test_alternative_claims_are_not_resolved_by_first_layer(store, owner, behavior_config) -> None:
    manifest, record = publish_manifest(store, owner, make_input())
    binder = ClaimBinder(config=behavior_config.claim)
    fingerprint = DeterministicClaimNormalizer().fingerprint
    first = binder.bind(
        manifest,
        record,
        proposal(predicate="candidate_a", local_alternative_group_id="alternatives"),
        fingerprint,
    )
    second = binder.bind(
        manifest,
        record,
        proposal(predicate="candidate_b", local_alternative_group_id="alternatives"),
        fingerprint,
    )
    assert first.claim_id != second.claim_id
    assert first.proposal.local_alternative_group_id == second.proposal.local_alternative_group_id
    assert first.alternative_group_key == second.alternative_group_key


def test_claim_and_receipt_publication_is_atomic_and_query_bounded(store, owner, behavior_config) -> None:
    manifest, _ = publish_manifest(store, owner, make_input(boundary_signal=BoundarySignal.END))
    result = asyncio.run(make_pipeline(store, behavior_config).process_manifest(manifest.manifest_id))
    receipt = store.read_receipt(result.core_result.processing_identity)
    assert receipt is not None
    assert receipt.claim_ids == tuple(item.claim_id for item in result.core_result.validated_claims)
    assert (
        store.list_claims_by_processing(result.core_result.processing_identity, limit=10)
        == result.core_result.validated_claims
    )
    with pytest.raises(ValueError):
        store.list_claims_by_processing(result.core_result.processing_identity, limit=0)
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM claim_batches").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM claim_processing_receipts").fetchone()[0] == 1


def test_claim_time_query_uses_a_stable_composite_cursor(store, owner, behavior_config) -> None:
    first_manifest, _ = publish_manifest(
        store,
        owner,
        make_input(sequence=40, correlation_id="claim-cursor-a", boundary_signal=BoundarySignal.END),
    )
    second_manifest, _ = publish_manifest(
        store,
        owner,
        make_input(
            sequence=41,
            offset_seconds=1,
            correlation_id="claim-cursor-b",
            boundary_signal=BoundarySignal.END,
        ),
    )
    asyncio.run(
        make_pipeline(store, behavior_config, clock=FakeClock(BASE_TIME + timedelta(seconds=5))).process_manifest(
            first_manifest.manifest_id
        )
    )
    asyncio.run(
        make_pipeline(store, behavior_config, clock=FakeClock(BASE_TIME + timedelta(seconds=6))).process_manifest(
            second_manifest.manifest_id
        )
    )
    page_one = store.list_claims(
        start=BASE_TIME,
        end=BASE_TIME + timedelta(seconds=10),
        limit=1,
    )
    page_two = store.list_claims(
        start=BASE_TIME,
        end=BASE_TIME + timedelta(seconds=10),
        limit=1,
        cursor=page_one[0].claim_id,
    )
    assert len(page_one) == len(page_two) == 1
    assert page_one[0].claim_id != page_two[0].claim_id


def test_concurrent_same_processing_reuses_one_atomic_result(store, owner, behavior_config) -> None:
    manifest, _ = publish_manifest(store, owner, make_input(boundary_signal=BoundarySignal.END))

    def process():
        return asyncio.run(make_pipeline(store, behavior_config).process_manifest(manifest.manifest_id))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: process(), range(2)))
    assert results[0].core_result.processing_identity == results[1].core_result.processing_identity
    assert sum(item.core_result.reused for item in results) >= 1
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM claim_processing_receipts").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1


def test_concurrent_distinct_processing_suppresses_one_repeated_state(
    store,
    owner,
    behavior_config,
) -> None:
    first_manifest, _ = publish_manifest(
        store,
        owner,
        make_input(sequence=10, offset_seconds=0, boundary_signal=BoundarySignal.END),
    )
    second_manifest, _ = publish_manifest(
        store,
        owner,
        make_input(sequence=11, offset_seconds=1, boundary_signal=BoundarySignal.END),
    )

    def process(manifest_id: str):
        return asyncio.run(make_pipeline(store, behavior_config).process_manifest(manifest_id))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(process, (first_manifest.manifest_id, second_manifest.manifest_id)))
    statuses = []
    for result in results:
        statuses.extend(
            [ClaimAdmissionStatus.ACCEPTED] * len(result.core_result.accepted_claims)
            + [item.status for item in result.core_result.rejected_decisions]
        )
    assert statuses.count(ClaimAdmissionStatus.ACCEPTED) == 1
    assert statuses.count(ClaimAdmissionStatus.REPEATED_STATE_SUPPRESSED) == 1


def test_accepted_capacity_rejection_keeps_validated_claim_auditable(tmp_path, owner) -> None:
    defaults = BehaviorConfig()
    config = BehaviorConfig(
        ingress=defaults.ingress,
        evidence=defaults.evidence,
        claim=replace(defaults.claim, max_claims_per_record=1, max_claims_per_batch=1),
        store=replace(defaults.store, max_accepted_claims=1),
    )
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "capacity", config=config, initialize=True)
    first_manifest, _ = publish_manifest(
        store,
        owner,
        make_input(sequence=20, boundary_signal=BoundarySignal.END),
    )
    first = asyncio.run(make_pipeline(store, config).process_manifest(first_manifest.manifest_id))
    assert len(first.core_result.accepted_claims) == 1
    transition_manifest, _ = publish_manifest(
        store,
        owner,
        make_input(
            sequence=21,
            offset_seconds=1,
            kind=SemanticRecordKind.OWNER_STATE_TRANSITION,
            payload=StateTransitionPayload("presence", False, True),
            subject_role=SemanticSubjectRole.OWNER,
            actor_role=SemanticActorRole.OWNER,
            boundary_signal=BoundarySignal.END,
        ),
    )
    second = asyncio.run(make_pipeline(store, config).process_manifest(transition_manifest.manifest_id))
    assert second.core_result.accepted_claims == ()
    assert second.core_result.rejected_decisions[0].status is ClaimAdmissionStatus.CAPACITY_REJECTED
    assert (
        store.read_claim(second.core_result.validated_claims[0].claim_id)
        == second.core_result.validated_claims[0]
    )


@pytest.mark.parametrize("schema_version", ["1", "2"])
def test_old_behavior_schema_is_explicitly_rejected(tmp_path, schema_version) -> None:
    root = tmp_path / "old"
    root.mkdir()
    path = root / "evidence_claims.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE behavior_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO behavior_metadata VALUES('schema_version', ?)",
            (schema_version,),
        )
        connection.commit()
    with pytest.raises(ClaimStoreError, match="migration is not supported"):
        SQLiteBehaviorEvidenceClaimStore(root, config=BehaviorConfig(), initialize=True)
