from __future__ import annotations

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Barrier, Lock

import pytest

import behavior
from behavior.claim import (
    ClaimAdmissionPolicy,
    ClaimAdmissionStatus,
    ClaimBinder,
    ClaimBindingPolicy,
    ClaimCompatibilityPolicy,
    ClaimConfidencePolicy,
    ClaimDerivationClass,
    ClaimKind,
    ClaimNormalizationRouter,
    ClaimNormalizerAttemptStatus,
    ClaimNormalizerKind,
    ClaimNormalizerRegistry,
    ClaimProcessingLane,
    ClaimSemanticProposal,
    ClaimSemanticProposalBatch,
    ClaimSemanticProposalBatchContract,
    DeterministicClaimNormalizer,
    ManifestClaimProcessingStatus,
    NormalizerFingerprint,
)
from behavior.claim.service import ClaimLaneProcessingResult, ClaimPipelineService
from behavior.config import BehaviorConfig, ClaimConfig
from behavior.errors import (
    BehaviorOwnerConflictError,
    ClaimModelContentSafetyError,
    ClaimModelTransportError,
    ClaimProcessingConflictError,
    ClaimProductionError,
    ClaimSchemaError,
    ClaimStoreCapacityError,
    ClaimStoreError,
    EvidenceBundleError,
    SemanticIngressError,
    SemanticRecordConflictError,
)
from behavior.evidence import SemanticIngestStatus
from behavior.evidence.service import EvidenceService
from behavior.ingress import (
    ActionEventPayload,
    BoundarySignal,
    ClockSyncStatus,
    EvidenceKind,
    EvidenceReference,
    FreeTextSemanticPayload,
    IngressDecisionStatus,
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
from behavior.ingress.service import AcceptedSemanticIngress
from behavior.owner import ConfirmedOwnerBinding
from behavior.persistence.sqlite import SQLiteBehaviorEvidenceClaimStore
from foundation.integrity import canonical_digest, canonical_json
from foundation.observability import NullObserver
from ModelClient import StructuredChatClient
from ModelClient.schema_validation import JSONSchemaValidationError, validate_json_schema
from tests.unit.behavior.conftest import (
    BASE_TIME,
    FakeAdapter,
    FakeClock,
    accepted_ingress,
    bind_record,
    digest,
    make_input,
)


def semantic_proposal(
    *,
    claim_kind: ClaimKind = ClaimKind.STATE_ASSERTION,
    predicate: str = "interpreted_state",
    family: str = "model_interpretation",
    confidence: float = 0.8,
    local_group: str | None = None,
    semantic_payload: dict[str, object] | None = None,
) -> ClaimSemanticProposal:
    activity = "interpreted_activity" if claim_kind is ClaimKind.ACTIVITY_PHASE else None
    phase = "in_progress" if claim_kind is ClaimKind.ACTIVITY_PHASE else None
    return ClaimSemanticProposal(
        claim_kind=claim_kind,
        predicate=predicate,
        semantic_family=family,
        activity=activity,
        phase=phase,
        object_refs=(),
        location_ref=None,
        semantic_payload=semantic_payload or {"value": True},
        human_summary="Bounded audit summary",
        local_alternative_group_id=local_group,
        normalizer_confidence=confidence,
    )


class CountingDeterministicNormalizer(DeterministicClaimNormalizer):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def normalize(self, record):
        self.calls += 1
        return await super().normalize(record)


class ManyClaimDeterministicNormalizer(DeterministicClaimNormalizer):
    def __init__(self, count: int) -> None:
        super().__init__(version="many-claim-v3")
        self.count = count
        self.calls = 0

    async def normalize(self, record):
        del record
        self.calls += 1
        return ClaimSemanticProposalBatch(
            False,
            tuple(
                semantic_proposal(
                    predicate=f"state_{index}",
                    family="many_claim_test",
                    confidence=1.0,
                )
                for index in range(self.count)
            ),
        )


class AbstainingCoreNormalizer(DeterministicClaimNormalizer):
    async def normalize(self, record):
        del record
        return ClaimSemanticProposalBatch(True, ())


class ScriptedModelNormalizer:
    name = "model_text"
    kind = ClaimNormalizerKind.MODEL
    allowed_record_kinds = frozenset(
        {SemanticRecordKind.FREE_TEXT_SEMANTIC, SemanticRecordKind.OWNER_UTTERANCE_SEGMENT}
    )

    def __init__(
        self,
        scripts: dict[str, list[ClaimSemanticProposalBatch | Exception]],
        *,
        client: StructuredChatClient | None = None,
        version: str = "scripted-v3",
        compatibility_policy: ClaimCompatibilityPolicy | None = None,
    ) -> None:
        self.model_client = client or object.__new__(StructuredChatClient)
        self.compatibility_policy = compatibility_policy or ClaimCompatibilityPolicy()
        self.scripts = scripts
        self.calls: list[str] = []
        self.fingerprint = NormalizerFingerprint(
            self.name,
            version,
            self.kind,
            "test",
            "scripted",
            "model",
            "prompt-v3",
        )

    async def normalize(self, record):
        record_id = record.semantic_record_id
        self.calls.append(record_id)
        values = self.scripts[record_id]
        result = values.pop(0) if len(values) > 1 else values[0]
        if isinstance(result, Exception):
            raise result
        return result


class ConcurrentMixedModelNormalizer(ScriptedModelNormalizer):
    def __init__(self, record_id: str) -> None:
        super().__init__({record_id: [ClaimSemanticProposalBatch(False, (semantic_proposal(),))]})
        self._barrier = Barrier(2)
        self._lock = Lock()
        self._call_number = 0

    async def normalize(self, record):
        self._barrier.wait(timeout=5)
        with self._lock:
            call_number = self._call_number
            self._call_number += 1
        if call_number == 0:
            raise ClaimModelTransportError("concurrent temporary failure")
        return await super().normalize(record)


def claim_pipeline(
    store: SQLiteBehaviorEvidenceClaimStore,
    config: BehaviorConfig,
    *,
    deterministic: DeterministicClaimNormalizer | None = None,
    model: ScriptedModelNormalizer | None = None,
    claim_config: ClaimConfig | None = None,
    router_version: str = "3",
    binding_policy: ClaimBindingPolicy | None = None,
    confidence_policy: ClaimConfidencePolicy | None = None,
    admission_policy: ClaimAdmissionPolicy | None = None,
    clock: FakeClock | None = None,
) -> ClaimPipelineService:
    resolved_clock = clock or FakeClock()
    resolved_claim = claim_config or config.claim
    ingress = SemanticRecordService(
        store,
        SemanticIngressAdapterRegistry(),
        config=config.ingress,
        clock=resolved_clock,
    )
    evidence = EvidenceService(
        store,
        config=config.evidence,
        observer=NullObserver(),
        clock=resolved_clock,
        adapters=ingress.adapters,
    )
    registry = ClaimNormalizerRegistry()
    resolved_binding = binding_policy or ClaimBindingPolicy()
    registry.register(deterministic or DeterministicClaimNormalizer())
    registry.register(
        model
        or ScriptedModelNormalizer(
            {},
            compatibility_policy=resolved_binding.compatibility,
        )
    )
    router = ClaimNormalizationRouter(
        registry,
        config=resolved_claim,
        routing_policy_version=router_version,
    )
    return ClaimPipelineService(
        store,
        ingress,
        evidence,
        registry,
        router,
        config=resolved_claim,
        observer=NullObserver(),
        clock=resolved_clock,
        binding_policy=resolved_binding,
        confidence_policy=confidence_policy,
        admission_policy=admission_policy,
    )


def publish_records(
    store: SQLiteBehaviorEvidenceClaimStore,
    records: tuple[object, ...],
    *,
    clock: FakeClock | None = None,
):
    evidence = EvidenceService(
        store,
        config=store.config.evidence,
        observer=NullObserver(),
        clock=clock or FakeClock(),
    )
    manifest_ids: list[str] = []
    active = None
    for record in records:
        result = evidence.ingest(accepted_ingress(record))
        manifest_ids.extend(result.manifest_ids)
        active = result.active_bundle
    if manifest_ids:
        manifest = store.read_manifest(manifest_ids[-1])
    else:
        assert active is not None
        manifest = evidence.seal_bundle(active.bundle_id)
    assert manifest is not None
    return manifest


def free_text_record(owner, sequence: int, text: str, *, boundary: BoundarySignal):
    return bind_record(
        owner,
        make_input(
            sequence=sequence,
            offset_seconds=float(sequence),
            kind=SemanticRecordKind.FREE_TEXT_SEMANTIC,
            payload=FreeTextSemanticPayload(text, "en", ("semantic",)),
            modality=SemanticModality.TEXT,
            boundary_signal=boundary,
        ),
        trust=IngressTrustClass.MODEL_INFERRED,
    )


def owner_utterance_record(owner, *, confidence: float = 1.0):
    return bind_record(
        owner,
        make_input(
            sequence=1,
            kind=SemanticRecordKind.OWNER_UTTERANCE_SEGMENT,
            payload=UtteranceSegmentPayload("I feel tired", "en", UtteranceChannel.VOICE),
            subject_role=SemanticSubjectRole.OWNER,
            actor_role=SemanticActorRole.OWNER,
            modality=SemanticModality.AUDIO,
            boundary_signal=BoundarySignal.END,
            source_confidence=confidence,
        ),
        trust=IngressTrustClass.OWNER_EXPLICIT,
    )


def test_optional_model_failure_commits_core_and_retry_runs_only_enhancement(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "optional", config=config, initialize=True)
    device = bind_record(owner, make_input(sequence=0))
    free = free_text_record(owner, 1, "interpret this", boundary=BoundarySignal.END)
    manifest = publish_records(store, (device, free))
    success = ClaimSemanticProposalBatch(False, (semantic_proposal(),))
    model = ScriptedModelNormalizer(
        {free.semantic_record_id: [ClaimModelTransportError("rate limited"), success]}
    )
    deterministic = CountingDeterministicNormalizer()
    pipeline = claim_pipeline(store, config, deterministic=deterministic, model=model)

    first = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    assert first.status is ManifestClaimProcessingStatus.CORE_COMMITTED_ENHANCEMENT_PENDING
    assert len(first.core_result.accepted_claims) == 1
    assert deterministic.calls == 1
    assert first.enhancement_results == ()
    assert first.degradations[0].status is ClaimNormalizerAttemptStatus.FAILED_RETRYABLE
    assert store.read_receipt(first.core_result.processing_identity) is not None
    failed_attempt = store.read_attempts_by_ids((first.degradations[0].attempt_id,))[0]
    assert store.read_receipt(failed_attempt.processing_identity) is None

    replay = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    assert replay.core_result.reused
    assert deterministic.calls == 1
    assert len(model.calls) == 1
    retried = asyncio.run(pipeline.retry_enhancement(manifest.manifest_id, free.semantic_record_id, model.name))
    assert isinstance(retried, ClaimLaneProcessingResult)
    assert retried.processing_lane is ClaimProcessingLane.ENHANCEMENT
    assert len(retried.accepted_claims) == 1
    assert retried.attempts[0].attempt_number == 2
    assert deterministic.calls == 1
    assert len(model.calls) == 2


def test_required_core_abstain_fails_closed_without_publication(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "core-abstain", config=config, initialize=True)
    manifest = publish_records(
        store,
        (bind_record(owner, make_input(boundary_signal=BoundarySignal.END)),),
    )
    with pytest.raises(ClaimProductionError, match="cannot abstain"):
        asyncio.run(
            claim_pipeline(
                store,
                config,
                deterministic=AbstainingCoreNormalizer(),
            ).process_manifest(manifest.manifest_id)
        )
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM claim_processing_receipts").fetchone()[0] == 0


def test_model_routes_are_isolated_and_abstain_has_receipt(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "isolated", config=config, initialize=True)
    failed = free_text_record(owner, 1, "route-a", boundary=BoundarySignal.CONTINUE)
    succeeded = free_text_record(owner, 2, "route-b", boundary=BoundarySignal.CONTINUE)
    abstained = free_text_record(owner, 3, "route-c", boundary=BoundarySignal.END)
    manifest = publish_records(store, (failed, succeeded, abstained))
    model = ScriptedModelNormalizer(
        {
            failed.semantic_record_id: [ClaimModelTransportError("temporary")],
            succeeded.semantic_record_id: [ClaimSemanticProposalBatch(False, (semantic_proposal(),))],
            abstained.semantic_record_id: [ClaimSemanticProposalBatch(True, ())],
        }
    )
    result = asyncio.run(claim_pipeline(store, config, model=model).process_manifest(manifest.manifest_id))

    assert result.status is ManifestClaimProcessingStatus.CORE_COMMITTED_ENHANCEMENT_PENDING
    assert result.core_result.validated_claims == ()
    assert len(result.enhancement_results) == 2
    successful = next(item for item in result.enhancement_results if item.validated_claims)
    abstain = next(item for item in result.enhancement_results if not item.validated_claims)
    assert successful.receipt is not None
    assert abstain.receipt is not None
    assert abstain.attempts[0].status is ClaimNormalizerAttemptStatus.ABSTAINED
    assert result.degradations[0].status is ClaimNormalizerAttemptStatus.FAILED_RETRYABLE


def test_content_safety_is_failed_policy_and_cannot_be_retried(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "safety", config=config, initialize=True)
    record = free_text_record(owner, 1, "sensitive text", boundary=BoundarySignal.END)
    manifest = publish_records(store, (record,))
    model = ScriptedModelNormalizer(
        {record.semantic_record_id: [ClaimModelContentSafetyError("blocked without source text")]}
    )
    pipeline = claim_pipeline(store, config, model=model)
    result = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    degradation = result.degradations[0]
    assert result.status is ManifestClaimProcessingStatus.CORE_COMMITTED_WITH_DEGRADATION
    assert degradation.status is ClaimNormalizerAttemptStatus.FAILED_POLICY
    assert degradation.error_code == "content_safety_blocked"
    assert not degradation.retryable
    with pytest.raises(ClaimProcessingConflictError, match="not retryable"):
        asyncio.run(pipeline.retry_enhancement(manifest.manifest_id, record.semantic_record_id, model.name))


def test_owner_utterance_core_and_model_threshold_are_separate(tmp_path, owner) -> None:
    claim_config = replace(
        BehaviorConfig().claim,
        normalize_owner_utterances=True,
        min_model_confidence=0.55,
    )
    config = BehaviorConfig(claim=claim_config)
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "utterance", config=config, initialize=True)
    record = owner_utterance_record(owner)
    manifest = publish_records(store, (record,))
    model = ScriptedModelNormalizer(
        {
            record.semantic_record_id: [
                ClaimSemanticProposalBatch(False, (semantic_proposal(confidence=0.0),))
            ]
        }
    )
    result = asyncio.run(claim_pipeline(store, config, model=model).process_manifest(manifest.manifest_id))

    assert [item.proposal.claim_kind for item in result.core_result.accepted_claims] == [ClaimKind.UTTERANCE]
    enhancement = result.enhancement_results[0]
    model_claim = enhancement.validated_claims[0]
    assert model_claim.source_epistemic_class.value == "USER_EXPLICIT"
    assert model_claim.derivation_class is ClaimDerivationClass.MODEL
    assert enhancement.rejected_decisions[0].status is ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD
    assert enhancement.rejected_decisions[0].required_threshold == 0.55
    assert model_claim.effective_confidence == 0.0


def test_owner_utterance_model_adds_interpretation_without_duplicate_utterance(tmp_path, owner) -> None:
    claim_config = replace(BehaviorConfig().claim, normalize_owner_utterances=True)
    config = BehaviorConfig(claim=claim_config)
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "utterance-valid", config=config, initialize=True)
    record = owner_utterance_record(owner)
    manifest = publish_records(store, (record,))
    model = ScriptedModelNormalizer(
        {record.semantic_record_id: [ClaimSemanticProposalBatch(False, (semantic_proposal(),))]}
    )
    result = asyncio.run(claim_pipeline(store, config, model=model).process_manifest(manifest.manifest_id))
    all_claims = (*result.core_result.validated_claims, *result.enhancement_results[0].validated_claims)
    assert [item.proposal.claim_kind for item in all_claims].count(ClaimKind.UTTERANCE) == 1
    assert result.enhancement_results[0].accepted_claims[0].proposal.claim_kind is ClaimKind.STATE_ASSERTION


@pytest.mark.parametrize(
    "forbidden",
    [
        ClaimKind.ENVIRONMENT_CHANGE,
        ClaimKind.UTTERANCE,
        ClaimKind.ROBOT_ACTION,
        ClaimKind.COVERAGE,
    ],
)
def test_owner_utterance_model_compatibility_is_mechanical(tmp_path, owner, forbidden) -> None:
    claim_config = replace(BehaviorConfig().claim, normalize_owner_utterances=True)
    config = BehaviorConfig(claim=claim_config)
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / forbidden.value, config=config, initialize=True)
    record = owner_utterance_record(owner)
    manifest = publish_records(store, (record,))
    model = ScriptedModelNormalizer(
        {record.semantic_record_id: [ClaimSemanticProposalBatch(False, (semantic_proposal(claim_kind=forbidden),))]}
    )
    result = asyncio.run(claim_pipeline(store, config, model=model).process_manifest(manifest.manifest_id))
    assert result.core_result.accepted_claims[0].proposal.claim_kind is ClaimKind.UTTERANCE
    assert result.enhancement_results == ()
    assert result.degradations[0].status is ClaimNormalizerAttemptStatus.FAILED_NON_RETRYABLE


def test_alternative_group_key_is_namespaced_by_record_and_normalizer(store, owner) -> None:
    first = free_text_record(owner, 1, "a", boundary=BoundarySignal.CONTINUE)
    second = free_text_record(owner, 2, "b", boundary=BoundarySignal.END)
    manifest = publish_records(store, (first, second))
    proposal = semantic_proposal(local_group="candidate")
    binder = ClaimBinder(config=store.config.claim)
    fingerprint_a = NormalizerFingerprint("model-a", "3", "MODEL", "test", "a", "m", "p")
    fingerprint_b = NormalizerFingerprint("model-b", "3", "MODEL", "test", "b", "m", "p")
    first_a = binder.bind(manifest, first, proposal, fingerprint_a)
    second_a = binder.bind(manifest, second, proposal, fingerprint_a)
    first_b = binder.bind(manifest, first, proposal, fingerprint_b)
    assert first_a.alternative_group_key != second_a.alternative_group_key
    assert first_a.alternative_group_key != first_b.alternative_group_key


def test_static_rejection_cannot_displace_higher_confidence_winner(tmp_path, owner) -> None:
    claim_config = replace(BehaviorConfig().claim, min_model_confidence=0.55)
    config = BehaviorConfig(claim=claim_config)
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "winner", config=config, initialize=True)
    record = free_text_record(owner, 1, "two alternatives", boundary=BoundarySignal.END)
    manifest = publish_records(store, (record,))
    low = semantic_proposal(confidence=0.0)
    high = semantic_proposal(confidence=0.9)
    model = ScriptedModelNormalizer(
        {record.semantic_record_id: [ClaimSemanticProposalBatch(False, (low, high))]}
    )
    result = asyncio.run(claim_pipeline(store, config, model=model).process_manifest(manifest.manifest_id))
    enhancement = result.enhancement_results[0]
    assert len(enhancement.accepted_claims) == 1
    assert enhancement.accepted_claims[0].normalizer_confidence == 0.9
    assert enhancement.rejected_decisions[0].status is ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD


def test_model_semantic_duplicate_of_core_is_no_information_gain(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "core-model-duplicate", config=config, initialize=True)
    device = bind_record(owner, make_input(sequence=0))
    free = free_text_record(owner, 1, "same device state", boundary=BoundarySignal.END)
    manifest = publish_records(store, (device, free))
    duplicate = semantic_proposal(
        predicate="power",
        family="device_state",
        confidence=0.9,
        semantic_payload={"device_ref": "device-main", "state_name": "power", "value": "on"},
    )
    model = ScriptedModelNormalizer(
        {free.semantic_record_id: [ClaimSemanticProposalBatch(False, (duplicate,))]}
    )
    result = asyncio.run(claim_pipeline(store, config, model=model).process_manifest(manifest.manifest_id))
    assert result.core_result.accepted_claims
    decision = result.enhancement_results[0].rejected_decisions[0]
    assert decision.status is ClaimAdmissionStatus.NO_INFORMATION_GAIN
    assert decision.reason_code == "core_semantic_claim_already_accepted_under_policy"


def test_admission_policy_re_evaluates_existing_claim_and_batches_are_processing_scoped(tmp_path, owner) -> None:
    base = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "policy", config=base, initialize=True)
    record = bind_record(owner, make_input(source_confidence=0.5, boundary_signal=BoundarySignal.END))
    manifest = publish_records(store, (record,))
    policy_a_config = replace(base.claim, min_direct_confidence=0.9)
    policy_b_config = replace(base.claim, min_direct_confidence=0.1)
    first = asyncio.run(
        claim_pipeline(store, base, claim_config=policy_a_config).process_manifest(manifest.manifest_id)
    )
    second_pipeline = claim_pipeline(store, base, claim_config=policy_b_config)
    second = asyncio.run(second_pipeline.process_manifest(manifest.manifest_id))
    claim_a = first.core_result.validated_claims[0]
    claim_b = second.core_result.validated_claims[0]
    assert claim_a.claim_id == claim_b.claim_id
    assert first.core_result.rejected_decisions[0].status is ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD
    assert second.core_result.accepted_claims[0].claim_id == claim_a.claim_id
    assert first.core_result.processing_identity != second.core_result.processing_identity
    assert first.core_result.receipt.claim_batch_ids != second.core_result.receipt.claim_batch_ids
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM claim_batches").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM claim_batch_members").fetchone()[0] == 2
    assert second_pipeline.list_accepted_claims(
        start=BASE_TIME - timedelta(days=1),
        end=BASE_TIME + timedelta(days=1),
        limit=10,
    ) == (store.read_claim(claim_a.claim_id),)


def test_current_policy_accepted_query_excludes_claim_rejected_by_new_policy(tmp_path, owner) -> None:
    base = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "active-policy", config=base, initialize=True)
    record = bind_record(owner, make_input(source_confidence=0.5, boundary_signal=BoundarySignal.END))
    manifest = publish_records(store, (record,))
    accepted_pipeline = claim_pipeline(
        store,
        base,
        claim_config=replace(base.claim, min_direct_confidence=0.1),
    )
    rejected_pipeline = claim_pipeline(
        store,
        base,
        claim_config=replace(base.claim, min_direct_confidence=0.9),
    )
    accepted = asyncio.run(accepted_pipeline.process_manifest(manifest.manifest_id))
    rejected = asyncio.run(rejected_pipeline.process_manifest(manifest.manifest_id))
    assert accepted.core_result.accepted_claims
    assert rejected.core_result.rejected_decisions[0].status is ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD
    assert rejected_pipeline.list_accepted_claims(
        start=BASE_TIME - timedelta(days=1),
        end=BASE_TIME + timedelta(days=1),
        limit=10,
    ) == ()


def test_concurrent_same_enhancement_publishes_one_receipt(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "enhancement-concurrent", config=config, initialize=True)
    record = free_text_record(owner, 1, "concurrent", boundary=BoundarySignal.END)
    manifest = publish_records(store, (record,))
    model = ScriptedModelNormalizer(
        {record.semantic_record_id: [ClaimSemanticProposalBatch(False, (semantic_proposal(),))]}
    )
    pipeline = claim_pipeline(store, config, model=model)

    def process():
        return asyncio.run(pipeline.process_manifest(manifest.manifest_id))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: process(), range(2)))
    enhancement_results = tuple(item.enhancement_results[0] for item in results)
    assert enhancement_results[0].processing_identity == enhancement_results[1].processing_identity
    assert sum(item.reused for item in enhancement_results) >= 1
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM claim_processing_receipts WHERE processing_lane='ENHANCEMENT'"
        ).fetchone()[0] == 1


def test_concurrent_failed_enhancement_reuses_one_durable_attempt(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "failed-concurrent", config=config, initialize=True)
    record = free_text_record(owner, 1, "concurrent failure", boundary=BoundarySignal.END)
    manifest = publish_records(store, (record,))
    model = ScriptedModelNormalizer(
        {record.semantic_record_id: [ClaimModelTransportError("temporary failure")]}
    )
    pipeline = claim_pipeline(store, config, model=model)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda _: asyncio.run(pipeline.process_manifest(manifest.manifest_id)),
                range(2),
            )
        )
    attempts = tuple(item.degradations[0] for item in results)
    assert attempts[0].attempt_id == attempts[1].attempt_id
    assert attempts[0].status is ClaimNormalizerAttemptStatus.FAILED_RETRYABLE
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM claim_normalizer_attempts").fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM claim_processing_receipts WHERE processing_lane='ENHANCEMENT'"
        ).fetchone()[0] == 0


def test_concurrent_enhancement_success_wins_over_failed_attempt(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "mixed-concurrent", config=config, initialize=True)
    record = free_text_record(owner, 1, "mixed result", boundary=BoundarySignal.END)
    manifest = publish_records(store, (record,))
    pipeline = claim_pipeline(store, config, model=ConcurrentMixedModelNormalizer(record.semantic_record_id))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda _: asyncio.run(pipeline.process_manifest(manifest.manifest_id)),
                range(2),
            )
        )
    assert any(item.enhancement_results for item in results)
    replay = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    assert replay.enhancement_results[0].reused
    assert replay.enhancement_results[0].accepted_claims
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM claim_processing_receipts WHERE processing_lane='ENHANCEMENT'"
        ).fetchone()[0] == 1
        statuses = {
            row[0]
            for row in connection.execute(
                "SELECT status FROM claim_normalizer_attempts WHERE processing_lane='ENHANCEMENT'"
            )
        }
    assert "COMPLETED" in statuses


def test_dynamic_action_is_not_suppressed_by_historical_semantic_fingerprint(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "dynamic-repeat", config=config, initialize=True)
    pipeline = claim_pipeline(store, config)

    def action(sequence: int, correlation: str):
        return bind_record(
            owner,
            make_input(
                sequence=sequence,
                offset_seconds=float(sequence),
                kind=SemanticRecordKind.ROBOT_ACTION_EVENT,
                payload=ActionEventPayload("wave", "completed", "ok", {}),
                subject_role=SemanticSubjectRole.ROBOT,
                actor_role=SemanticActorRole.ROBOT,
                modality=SemanticModality.ROBOT,
                correlation_id=correlation,
                boundary_signal=BoundarySignal.END,
            ),
            trust=IngressTrustClass.DIRECT_SYSTEM_LOG,
        )

    first = asyncio.run(pipeline.process_manifest(publish_records(store, (action(1, "action-a"),)).manifest_id))
    second = asyncio.run(pipeline.process_manifest(publish_records(store, (action(2, "action-b"),)).manifest_id))
    assert first.core_result.accepted_claims
    assert second.core_result.accepted_claims
    assert second.core_result.rejected_decisions == ()


def test_model_dynamic_claim_is_not_suppressed_across_semantic_records(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "model-dynamic-repeat", config=config, initialize=True)
    first_record = free_text_record(owner, 1, "change one", boundary=BoundarySignal.END)
    second_record = free_text_record(owner, 2, "change two", boundary=BoundarySignal.END)
    proposal = semantic_proposal(
        claim_kind=ClaimKind.ENVIRONMENT_CHANGE,
        predicate="door_changed",
        family="environment_change",
    )
    model = ScriptedModelNormalizer(
        {
            first_record.semantic_record_id: [ClaimSemanticProposalBatch(False, (proposal,))],
            second_record.semantic_record_id: [ClaimSemanticProposalBatch(False, (proposal,))],
        }
    )
    pipeline = claim_pipeline(store, config, model=model)
    first = asyncio.run(
        pipeline.process_manifest(publish_records(store, (first_record,)).manifest_id)
    )
    second = asyncio.run(
        pipeline.process_manifest(publish_records(store, (second_record,)).manifest_id)
    )
    assert first.enhancement_results[0].accepted_claims
    assert second.enhancement_results[0].accepted_claims


def test_store_records_publication_time_inside_transaction_and_replay_reuses_it(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store_clock = FakeClock(BASE_TIME + timedelta(seconds=100))
    pipeline_clock = FakeClock(BASE_TIME + timedelta(seconds=20))
    store = SQLiteBehaviorEvidenceClaimStore(
        tmp_path / "audit-time",
        config=config,
        clock=store_clock,
        initialize=True,
    )
    record = bind_record(owner, make_input(boundary_signal=BoundarySignal.END))
    manifest = publish_records(store, (record,), clock=FakeClock(BASE_TIME + timedelta(seconds=10)))
    pipeline = claim_pipeline(store, config, clock=pipeline_clock)
    first = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    assert first.core_result.receipt.processing_completed_at == pipeline_clock.now()
    assert first.core_result.receipt.publication_recorded_at == store_clock.now()
    store_clock.advance(50)
    replay = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    assert replay.core_result.receipt.publication_recorded_at == first.core_result.receipt.publication_recorded_at


def test_policy_versions_change_processing_identity(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "identities", config=config, initialize=True)
    record = bind_record(owner, make_input(boundary_signal=BoundarySignal.END))
    manifest = publish_records(store, (record,))
    default = asyncio.run(claim_pipeline(store, config).process_manifest(manifest.manifest_id))
    routed = asyncio.run(
        claim_pipeline(store, config, router_version="4").process_manifest(manifest.manifest_id)
    )
    bound = asyncio.run(
        claim_pipeline(
            store,
            config,
            binding_policy=ClaimBindingPolicy(compatibility=ClaimCompatibilityPolicy(version="4")),
        ).process_manifest(manifest.manifest_id)
    )
    confident = asyncio.run(
        claim_pipeline(
            store,
            config,
            confidence_policy=ClaimConfidencePolicy(version="4"),
        ).process_manifest(manifest.manifest_id)
    )
    identities = {
        default.core_result.processing_identity,
        routed.core_result.processing_identity,
        bound.core_result.processing_identity,
        confident.core_result.processing_identity,
    }
    assert len(identities) == 4


def test_admission_capacity_is_part_of_policy_identity() -> None:
    config = BehaviorConfig().claim
    assert ClaimAdmissionPolicy(config=config, max_accepted_claims=10).digest != ClaimAdmissionPolicy(
        config=config,
        max_accepted_claims=11,
    ).digest


def test_receipt_readback_rejects_incomplete_batch_membership(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "receipt-graph", config=config, initialize=True)
    manifest = publish_records(
        store,
        (bind_record(owner, make_input(boundary_signal=BoundarySignal.END)),),
    )
    result = asyncio.run(claim_pipeline(store, config).process_manifest(manifest.manifest_id))
    receipt = result.core_result.receipt
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            "DELETE FROM claim_batch_members WHERE claim_batch_id=?",
            (receipt.claim_batch_ids[0],),
        )
        connection.commit()
    with pytest.raises(ClaimProcessingConflictError, match="BatchMember"):
        store.read_receipt(receipt.processing_identity)


def test_receipt_readback_rejects_batch_member_from_another_route(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "receipt-route", config=config, initialize=True)
    manifest = publish_records(
        store,
        (
            bind_record(owner, make_input(sequence=1)),
            bind_record(
                owner,
                make_input(sequence=2, offset_seconds=1, boundary_signal=BoundarySignal.END),
            ),
        ),
    )
    result = asyncio.run(claim_pipeline(store, config).process_manifest(manifest.manifest_id))
    receipt = result.core_result.receipt
    with closing(sqlite3.connect(store.path)) as connection:
        batches = connection.execute(
            "SELECT claim_batch_id, semantic_record_id FROM claim_batches ORDER BY claim_batch_id"
        ).fetchall()
        claims = dict(
            connection.execute("SELECT semantic_record_id, claim_id FROM claims").fetchall()
        )
        first_batch_id, first_record_id = batches[0]
        other_claim_id = next(claim_id for record_id, claim_id in claims.items() if record_id != first_record_id)
        connection.execute(
            "UPDATE claim_batch_members SET claim_id=? WHERE claim_batch_id=?",
            (other_claim_id, first_batch_id),
        )
        connection.commit()
    with pytest.raises(ClaimProcessingConflictError, match="another normalization route"):
        store.read_receipt(receipt.processing_identity)


def test_config_aware_proposal_schema_validator_and_binder_share_limits(store, owner) -> None:
    config = ClaimConfig(
        max_human_summary_chars=8,
        max_claims_per_record=2,
        max_claims_per_batch=2,
        max_alternative_group_size=2,
    )
    allowed = frozenset({ClaimKind.STATE_ASSERTION})
    schema = ClaimSemanticProposalBatchContract.model_json_schema(config, allowed)
    assert schema["properties"]["claims"]["maxItems"] == 2
    item_schema = schema["properties"]["claims"]["items"]
    assert item_schema["properties"]["human_summary"]["maxLength"] == 8
    mapping = json.loads(canonical_json(semantic_proposal().to_dict()))
    mapping["human_summary"] = "12345678"
    batch_value = {"abstained": False, "claims": [mapping, mapping]}
    validate_json_schema(batch_value, schema)
    assert len(ClaimSemanticProposalBatchContract.model_validate(batch_value, config, allowed).claims) == 2
    mapping["human_summary"] = "123456789"
    with pytest.raises(JSONSchemaValidationError):
        validate_json_schema({"abstained": False, "claims": [mapping]}, schema)
    with pytest.raises(ClaimSchemaError):
        ClaimSemanticProposalBatchContract.model_validate(
            {"abstained": False, "claims": [mapping]}, config, allowed
        )

    manifest = publish_records(store, (bind_record(owner, make_input(boundary_signal=BoundarySignal.END)),))
    record = store.read_semantic_record(manifest.ordered_record_snapshots[0].semantic_record_id)
    assert record is not None
    oversized = semantic_proposal()
    with pytest.raises(ClaimSchemaError):
        ClaimBinder(config=config).bind(
            manifest,
            record,
            oversized,
            DeterministicClaimNormalizer().fingerprint,
        )


def test_receipt_rebuild_is_not_limited_by_external_500_page_size(tmp_path, owner) -> None:
    defaults = BehaviorConfig()
    config = BehaviorConfig(
        ingress=defaults.ingress,
        evidence=replace(defaults.evidence, max_records_per_bundle=1),
        claim=replace(
            defaults.claim,
            max_claims_per_record=501,
            max_claims_per_batch=501,
            max_normalizers_per_processing=1,
            max_claims_per_processing=600,
        ),
        store=replace(
            defaults.store,
            max_validated_claims=1_000,
            max_accepted_claims=1_000,
            max_admission_decisions=1_000,
            max_normalizer_attempts=1_000,
            max_claim_batches=1_000,
            max_processing_receipts=10,
        ),
    )
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "many", config=config, initialize=True)
    manifest = publish_records(
        store,
        (bind_record(owner, make_input(boundary_signal=BoundarySignal.END)),),
    )
    deterministic = ManyClaimDeterministicNormalizer(501)
    pipeline = claim_pipeline(store, config, deterministic=deterministic)
    first = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    replay = asyncio.run(pipeline.process_manifest(manifest.manifest_id))
    assert len(first.core_result.validated_claims) == 501
    assert len(first.core_result.receipt.decision_ids) == 501
    assert len(replay.core_result.validated_claims) == 501
    assert replay.core_result.reused
    assert deterministic.calls == 1
    with pytest.raises(ValueError):
        store.list_claims(
            start=BASE_TIME - timedelta(days=1),
            end=BASE_TIME + timedelta(days=1),
            limit=501,
        )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("max_validated_claims", ClaimStoreCapacityError),
        ("max_admission_decisions", ClaimStoreCapacityError),
        ("max_normalizer_attempts", ClaimStoreCapacityError),
        ("max_claim_batches", ClaimStoreCapacityError),
        ("max_processing_receipts", ClaimStoreCapacityError),
    ],
)
def test_store_capacity_fields_enforce_their_own_tables(tmp_path, owner, field, expected) -> None:
    defaults = BehaviorConfig()
    store_values = {
        "max_validated_claims": 10,
        "max_accepted_claims": 10,
        "max_admission_decisions": 10,
        "max_normalizer_attempts": 10,
        "max_claim_batches": 10,
        "max_processing_receipts": 10,
    }
    store_values[field] = 1
    if field == "max_admission_decisions":
        store_values["max_validated_claims"] = 1
        store_values["max_accepted_claims"] = 1
    if field == "max_validated_claims":
        store_values["max_accepted_claims"] = 1
    config = BehaviorConfig(
        evidence=replace(defaults.evidence, max_records_per_bundle=1),
        claim=replace(
            defaults.claim,
            max_claims_per_record=1,
            max_claims_per_batch=1,
            max_normalizers_per_processing=1,
            max_claims_per_processing=1,
        ),
        store=replace(defaults.store, **store_values),
    )
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / field, config=config, initialize=True)
    pipeline = claim_pipeline(store, config)
    first = publish_records(
        store,
        (bind_record(owner, make_input(sequence=10, boundary_signal=BoundarySignal.END)),),
    )
    asyncio.run(pipeline.process_manifest(first.manifest_id))
    if field == "max_admission_decisions":
        second_pipeline = claim_pipeline(
            store,
            config,
            claim_config=replace(config.claim, min_direct_confidence=0.1),
        )
        with pytest.raises(expected):
            asyncio.run(second_pipeline.process_manifest(first.manifest_id))
    else:
        second = publish_records(
            store,
            (
                bind_record(
                    owner,
                    make_input(
                        sequence=11,
                        offset_seconds=1,
                        correlation_id="second-capacity",
                        boundary_signal=BoundarySignal.END,
                    ),
                ),
            ),
        )
        with pytest.raises(expected):
            asyncio.run(pipeline.process_manifest(second.manifest_id))


def test_ingress_capacity_rejection_is_audited_without_record_or_watermark(tmp_path, owner) -> None:
    defaults = BehaviorConfig()
    config = BehaviorConfig(store=replace(defaults.store, max_semantic_records=1))
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "ingress-capacity", config=config, initialize=True)
    evidence = EvidenceService(store, config=config.evidence, observer=NullObserver())
    first = bind_record(owner, make_input(sequence=0, boundary_signal=BoundarySignal.END))
    assert evidence.ingest(accepted_ingress(first)).status is SemanticIngestStatus.ACCEPTED
    second = bind_record(
        owner,
        make_input(sequence=1, correlation_id="capacity-two", boundary_signal=BoundarySignal.END),
    )
    result = evidence.ingest(accepted_ingress(second))
    assert result.status is SemanticIngestStatus.CAPACITY_REJECTED
    assert result.decision.status is IngressDecisionStatus.CAPACITY_REJECTED
    assert store.read_semantic_record(second.semantic_record_id) is None
    assert store.read_ingress_decision(result.decision.decision_id) == result.decision
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM evidence_watermarks WHERE grouping_key LIKE ?",
            (f"%{second.semantic_input.correlation_id}%",),
        ).fetchone()[0] == 0


def test_first_rejected_ingress_permanently_binds_owner(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "first-owner", config=config, initialize=True)
    future = make_input(offset_seconds=10_000, clock_sync_status=ClockSyncStatus.SYNCHRONIZED)
    adapters = SemanticIngressAdapterRegistry()
    adapters.register(FakeAdapter(future, name="future"))
    service = SemanticRecordService(store, adapters, config=config.ingress, clock=FakeClock())
    result = asyncio.run(service.prepare("future", {}, owner_binding=owner))[0]
    assert result.decision.status is IngressDecisionStatus.CLOCK_SKEW_REJECTED
    other = ConfirmedOwnerBinding("other-owner", "resolver-v3", BASE_TIME, digest("other"))
    with pytest.raises(BehaviorOwnerConflictError):
        asyncio.run(service.prepare("future", {}, owner_binding=other))


def test_manifest_encoded_boundary_counts_evidence_references(tmp_path, owner) -> None:
    defaults = BehaviorConfig()
    config = BehaviorConfig(
        evidence=replace(defaults.evidence, max_manifest_encoded_bytes=4_096),
        store=replace(defaults.store, max_json_bytes=8_192),
    )
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "manifest-size", config=config, initialize=True)
    refs = tuple(
        EvidenceReference(
            reference=f"s3://evidence-bucket/path/{index}-" + "x" * 80,
            evidence_kind=EvidenceKind.IMAGE_FRAME,
            digest=digest(f"ref-{index}"),
            event_time_start=BASE_TIME,
            event_time_end=BASE_TIME,
            media_type="image/jpeg",
            size_bytes=1_024,
            source_system_ref="perception-runtime",
        )
        for index in range(defaults.ingress.max_evidence_refs)
    )
    record = bind_record(
        owner,
        make_input(evidence_refs=refs, boundary_signal=BoundarySignal.END),
    )
    evidence = EvidenceService(store, config=config.evidence, observer=NullObserver())
    with pytest.raises(EvidenceBundleError):
        evidence.ingest(accepted_ingress(record))
    assert store.read_semantic_record(record.semantic_record_id) is None


def _tamper_json(path: Path, table: str, key_name: str, key: str, mutation) -> None:
    with closing(sqlite3.connect(path)) as connection:
        payload = connection.execute(
            f"SELECT content_json FROM {table} WHERE {key_name}=?",
            (key,),
        ).fetchone()[0]
        value = json.loads(payload)
        mutation(value)
        encoded = canonical_json(value)
        connection.execute(
            f"UPDATE {table} SET content_json=?, encoded_digest=? WHERE {key_name}=?",
            (encoded, canonical_digest(value), key),
        )
        connection.commit()


@pytest.mark.parametrize("target", ["ingested_at", "resolver_fingerprint"])
def test_semantic_record_full_digest_detects_audit_tamper(tmp_path, owner, target) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / target, config=config, initialize=True)
    record = bind_record(owner, make_input(boundary_signal=BoundarySignal.END))
    result = EvidenceService(store, config=config.evidence, observer=NullObserver()).ingest(
        accepted_ingress(record)
    )
    assert result.manifest_ids

    def mutate(value):
        if target == "ingested_at":
            value["ingested_at"] = "2026-08-06T01:02:03Z"
        else:
            value["owner_binding"]["resolver_fingerprint"] = "tampered-resolver"

    _tamper_json(store.path, "semantic_records", "semantic_record_id", record.semantic_record_id, mutate)
    with pytest.raises(ClaimStoreError):
        store.read_semantic_record(record.semantic_record_id)


def test_manifest_and_ingress_decision_full_digests_detect_audit_tamper(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "tamper", config=config, initialize=True)
    record = bind_record(owner, make_input(boundary_signal=BoundarySignal.END))
    result = EvidenceService(store, config=config.evidence, observer=NullObserver()).ingest(
        accepted_ingress(record)
    )
    manifest_id = result.manifest_ids[0]
    decision_id = result.decision.decision_id
    _tamper_json(
        store.path,
        "evidence_manifests",
        "manifest_id",
        manifest_id,
        lambda value: value.__setitem__("sealed_at", "2026-08-06T01:02:03Z"),
    )
    with pytest.raises(ClaimStoreError):
        store.read_manifest(manifest_id)

    store2 = SQLiteBehaviorEvidenceClaimStore(tmp_path / "decision-tamper", config=config, initialize=True)
    record2 = bind_record(owner, make_input(boundary_signal=BoundarySignal.END))
    result2 = EvidenceService(store2, config=config.evidence, observer=NullObserver()).ingest(
        accepted_ingress(record2)
    )
    _tamper_json(
        store2.path,
        "semantic_ingress_decisions",
        "decision_id",
        result2.decision.decision_id,
        lambda value: value.__setitem__("decided_at", "2026-08-06T01:02:03Z"),
    )
    with pytest.raises(ClaimStoreError):
        store2.read_ingress_decision(result2.decision.decision_id)
    assert decision_id


def test_readiness_detects_materialized_column_tamper(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "indexed-tamper", config=config, initialize=True)
    record = bind_record(owner, make_input(boundary_signal=BoundarySignal.END))
    EvidenceService(store, config=config.evidence, observer=NullObserver()).ingest(
        accepted_ingress(record)
    )
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            "UPDATE semantic_records SET stream_id='tampered-stream' WHERE semantic_record_id=?",
            (record.semantic_record_id,),
        )
        connection.commit()
    assert store.readiness() == (False, "ClaimStoreError")


def test_accepted_ingress_policy_must_match_store_configuration(tmp_path, owner) -> None:
    defaults = BehaviorConfig()
    config = BehaviorConfig(
        ingress=replace(defaults.ingress, max_payload_chars=defaults.ingress.max_payload_chars + 1),
        evidence=replace(
            defaults.evidence,
            max_projection_chars_per_bundle=defaults.evidence.max_projection_chars_per_bundle + 1,
        ),
    )
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "ingress-policy", config=config, initialize=True)
    record = bind_record(owner, make_input(boundary_signal=BoundarySignal.END))
    evidence = EvidenceService(store, config=config.evidence, observer=NullObserver())
    with pytest.raises(SemanticRecordConflictError, match="policy digest"):
        evidence.ingest(accepted_ingress(record))
    assert store.read_semantic_record(record.semantic_record_id) is None


def test_registry_rejects_normalizer_kind_fingerprint_mismatch() -> None:
    normalizer = ScriptedModelNormalizer({})
    normalizer.fingerprint = NormalizerFingerprint(
        normalizer.name,
        "bad-kind",
        ClaimNormalizerKind.DETERMINISTIC,
        "test",
        "scripted",
        "model",
        "prompt-v3",
    )
    with pytest.raises(ClaimProductionError, match="fingerprint kind"):
        ClaimNormalizerRegistry().register(normalizer)


def test_registry_and_router_require_complete_model_contract() -> None:
    missing_policy = ScriptedModelNormalizer({})
    del missing_policy.compatibility_policy
    with pytest.raises(TypeError, match="ClaimNormalizer"):
        ClaimNormalizerRegistry().register(missing_policy)

    registry = ClaimNormalizerRegistry()
    registry.register(DeterministicClaimNormalizer())
    with pytest.raises(ClaimProductionError, match="unknown Claim Normalizer"):
        ClaimNormalizationRouter(registry, config=BehaviorConfig().claim)


def test_pipeline_rejects_model_and_binder_compatibility_policy_drift(tmp_path) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "policy-drift", config=config, initialize=True)
    model = ScriptedModelNormalizer(
        {},
        compatibility_policy=ClaimCompatibilityPolicy(version="different"),
    )
    with pytest.raises(ValueError, match="compatibility policies must match"):
        claim_pipeline(store, config, model=model)


@pytest.mark.parametrize(
    ("subject", "actor"),
    [
        (SemanticSubjectRole.ENVIRONMENT, SemanticActorRole.SYSTEM),
        (SemanticSubjectRole.ROBOT, SemanticActorRole.ROBOT),
        (SemanticSubjectRole.AGENT, SemanticActorRole.AGENT),
    ],
)
def test_model_activity_requires_owner_roles(subject, actor) -> None:
    result = ClaimCompatibilityPolicy().evaluate(
        record_kind=SemanticRecordKind.FREE_TEXT_SEMANTIC,
        subject_role=subject,
        actor_role=actor,
        derivation_class=ClaimDerivationClass.MODEL,
        claim_kind=ClaimKind.ACTIVITY_PHASE,
    )
    assert not result.allowed
    assert result.reason_code == "model_claim_kind_not_allowed"


def test_registered_adapter_provenance_is_required_for_accepted_ingress(tmp_path, owner) -> None:
    record = bind_record(owner, make_input(boundary_signal=BoundarySignal.END))
    valid = accepted_ingress(record)
    with pytest.raises(SemanticIngressError, match="unknown semantic Adapter"):
        AcceptedSemanticIngress(
            record=valid.record,
            decision=valid.decision,
            adapter_name=valid.adapter_name,
            adapter_registry=SemanticIngressAdapterRegistry(),
            adapter_fingerprint=valid.adapter_fingerprint,
            capability=valid.capability,
            capability_digest=valid.capability_digest,
            ingress_policy_digest=valid.ingress_policy_digest,
        )
    rogue_registry = SemanticIngressAdapterRegistry()
    rogue_adapter = FakeAdapter(
        record.semantic_input,
        name=record.producer_fingerprint.producer_name,
        trust=record.ingress_trust_class,
        allowed=(record.semantic_input.record_kind,),
    )
    rogue_adapter.fingerprint = record.producer_fingerprint
    rogue_adapter.capabilities = valid.capability
    rogue_registry.register(rogue_adapter)
    rogue = AcceptedSemanticIngress(
        record=valid.record,
        decision=valid.decision,
        adapter_name=rogue_adapter.name,
        adapter_registry=rogue_registry,
        adapter_fingerprint=valid.adapter_fingerprint,
        capability=valid.capability,
        capability_digest=valid.capability_digest,
        ingress_policy_digest=valid.ingress_policy_digest,
    )
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "registry-authority", config=config, initialize=True)
    evidence = EvidenceService(
        store,
        config=config.evidence,
        observer=NullObserver(),
        adapters=valid.adapter_registry,
    )
    with pytest.raises(SemanticRecordConflictError, match="authoritative Evidence Registry"):
        evidence.ingest(rogue)
    assert store.read_semantic_record(record.semantic_record_id) is None


def test_materialized_decision_tamper_cannot_create_false_accepted_claim(tmp_path, owner) -> None:
    defaults = BehaviorConfig()
    config = BehaviorConfig(
        claim=replace(defaults.claim, min_direct_confidence=0.9),
    )
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "decision-status-tamper", config=config, initialize=True)
    record = bind_record(
        owner,
        make_input(source_confidence=0.1, boundary_signal=BoundarySignal.END),
    )
    result = asyncio.run(
        claim_pipeline(store, config).process_manifest(
            publish_records(store, (record,)).manifest_id
        )
    )
    assert result.core_result.rejected_decisions[0].status is ClaimAdmissionStatus.BELOW_SCORE_THRESHOLD
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            "UPDATE claim_admission_decisions SET status='ACCEPTED' WHERE decision_id=?",
            (result.core_result.rejected_decisions[0].decision_id,),
        )
        connection.commit()
    assert claim_pipeline(store, config).list_accepted_claims(
        start=BASE_TIME - timedelta(days=1),
        end=BASE_TIME + timedelta(days=1),
        limit=10,
    ) == ()
    assert store.readiness() == (False, "ClaimStoreError")


def test_foreign_key_check_marks_orphaned_store_unready(tmp_path) -> None:
    config = BehaviorConfig()
    root = tmp_path / "foreign-key"
    store = SQLiteBehaviorEvidenceClaimStore(root, config=config, initialize=True)
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            "INSERT INTO claim_batch_members(claim_batch_id, claim_id, member_order) VALUES(?,?,?)",
            ("batch_orphan", "claim_orphan", 0),
        )
        connection.commit()
    ready, detail = store.readiness()
    assert not ready
    assert detail == "ClaimStoreError"
    with pytest.raises(ClaimStoreError, match="foreign key integrity"):
        SQLiteBehaviorEvidenceClaimStore(root, config=config, initialize=True)


def test_top_level_public_api_does_not_export_trusted_internal_writers() -> None:
    assert not hasattr(behavior, "OwnerScopedSemanticRecord")
    assert not hasattr(behavior, "IngressDecision")
    assert not hasattr(behavior, "EvidenceService")
    assert not hasattr(behavior, "ClaimBinder")
    assert not hasattr(behavior, "SQLiteBehaviorEvidenceClaimStore")
