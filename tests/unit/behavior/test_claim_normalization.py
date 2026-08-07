from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from types import MethodType, SimpleNamespace

import pytest

from behavior.claim import (
    BuiltinDeterministicClaimNormalizer,
    ClaimBinder,
    ClaimKind,
    ClaimNormalizationPlanner,
    ClaimNormalizationService,
    ClaimNormalizerRegistry,
    ClaimSemanticProposal,
    ReceiptStatus,
    StructuredModelClaimNormalizer,
)
from behavior.claim.compatibility import ClaimCompatibilityPolicy
from behavior.claim.model import DerivationClass, SourceEpistemicClass
from behavior.claim.receipt import AttemptStatus
from behavior.config import BehaviorConfig
from behavior.errors import (
    BehaviorClaimSchemaError,
    ClaimModelAuthenticationError,
    ClaimModelConfigurationError,
    ClaimModelContentSafetyError,
    ClaimModelInputError,
    ClaimModelPermissionError,
    ClaimModelQuotaError,
    ClaimModelSchemaError,
    ClaimModelTransportError,
    ClaimNormalizationConflictError,
    ClaimNormalizationError,
)
from behavior.evidence import (
    ActionEventPayload,
    ActivitySegmentPayload,
    BehaviorModality,
    BehaviorOriginKind,
    BehaviorRecordKind,
    BehaviorRole,
    BehaviorSemanticAdapterRegistry,
    BehaviorSemanticContent,
    BehaviorSemanticInput,
    BehaviorSourceProvenance,
    BehaviorSourceTrust,
    ClockSyncStatus,
    CommunicationChannel,
    CoverageIntervalPayload,
    CoverageStatus,
    EnvironmentChangePayload,
    EvidenceIntegrity,
    FeedbackPayload,
    FeedbackPolarity,
    FreeTextSemanticPayload,
    InteractionMode,
    InteractionSegmentPayload,
    PhaseHint,
    ProducerFingerprint,
    ProducerImplementationKind,
    StateAssertionPayload,
    StateTransitionPayload,
    ToolCallPayload,
    ToolResultPayload,
    ToolResultStatus,
    UtteranceSegmentPayload,
)
from behavior.evidence.ingress import BehaviorEvidenceIngressService
from behavior.evidence.record import BehaviorEvidenceRecord
from behavior.persistence import BehaviorDatabase, SQLiteBehaviorClaimLedger, SQLiteBehaviorEvidenceLedger
from ModelClient import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelContentSafetyError,
    ModelInputTooLargeError,
    ModelPermissionError,
    ModelQuotaError,
    ModelRateLimitError,
    ModelResponseError,
    ModelStructuredOutputError,
    ModelTransportError,
    StructuredChatClient,
)
from tests.unit.behavior.conftest import (
    BASE_TIME,
    FakeAdapter,
    FakeClock,
    FakeModelNormalizer,
    SQLiteProcessingLock,
    digest,
    source_descriptor,
)
from tests.unit.behavior.test_evidence_contract import content as make_content
from tests.unit.behavior.test_evidence_ingress_ledger import semantic_input


def direct_record(value: BehaviorSemanticInput) -> BehaviorEvidenceRecord:
    producer = ProducerFingerprint("test", "1", "1", "1", ProducerImplementationKind.ADAPTER)
    return BehaviorEvidenceRecord(
        semantic_content=value.content,
        provenance=BehaviorSourceProvenance(value.source, "test", producer, digest("capability")),
        source_trust=BehaviorSourceTrust.MODEL_INFERRED,
        ingested_at=BASE_TIME,
    )


def free_text_input() -> BehaviorSemanticInput:
    return BehaviorSemanticInput(
        BehaviorSemanticContent(
            record_kind=BehaviorRecordKind.FREE_TEXT_SEMANTIC,
            subject_role=BehaviorRole.USER,
            actor_role=BehaviorRole.USER,
            modality=BehaviorModality.TEXT,
            event_time_start=BASE_TIME,
            event_time_end=BASE_TIME,
            event_time_uncertainty_ms=0,
            clock_domain="utc",
            clock_sync_status=ClockSyncStatus.SYNCHRONIZED,
            scene_ref=None,
            location_ref=None,
            object_refs=(),
            entity_refs=(),
            payload=FreeTextSemanticPayload("I am tired", "en", ("state",)),
            evidence_refs=(),
            source_confidence=0.6,
            integrity=EvidenceIntegrity.PARTIAL,
        ),
        source_descriptor(event="free-text"),
    )


def utterance_input() -> BehaviorSemanticInput:
    return BehaviorSemanticInput(
        BehaviorSemanticContent(
            record_kind=BehaviorRecordKind.UTTERANCE_SEGMENT,
            subject_role=BehaviorRole.USER,
            actor_role=BehaviorRole.USER,
            modality=BehaviorModality.AUDIO,
            event_time_start=BASE_TIME,
            event_time_end=BASE_TIME + timedelta(seconds=1),
            event_time_uncertainty_ms=0,
            clock_domain="utc",
            clock_sync_status=ClockSyncStatus.SYNCHRONIZED,
            scene_ref=None,
            location_ref=None,
            object_refs=(),
            entity_refs=(),
            payload=UtteranceSegmentPayload(
                "I started running",
                "en",
                InteractionMode.DIALOGUE,
                CommunicationChannel.VOICE,
            ),
            evidence_refs=(),
            source_confidence=0.9,
            integrity=EvidenceIntegrity.COMPLETE,
        ),
        source_descriptor(
            event="utterance",
            origin=BehaviorOriginKind.DIRECT_AMBIENT_ASR,
        ),
    )


def state_proposal(
    *,
    summary: str | None = "tired",
    confidence: float = 0.7,
    kind: ClaimKind = ClaimKind.STATE_ASSERTION,
) -> ClaimSemanticProposal:
    return ClaimSemanticProposal(
        claim_kind=kind,
        semantic_family="state.user",
        predicate="tired",
        activity=None,
        phase=None,
        semantic_payload={"value": True},
        human_summary=summary,
        local_alternative_group_id=None,
        normalizer_confidence=confidence,
    )


async def ingest_record(database, input_value, *, config, name="adapter"):
    ledger = SQLiteBehaviorEvidenceLedger(database)
    registry = BehaviorSemanticAdapterRegistry()
    if input_value.source.origin_kind is BehaviorOriginKind.DIRECT_AMBIENT_ASR:
        adapter = FakeAdapter(
            input_value,
            name=name,
            trust=BehaviorSourceTrust.USER_EXPLICIT,
            origins=(BehaviorOriginKind.DIRECT_AMBIENT_ASR,),
            kinds=(input_value.content.record_kind,),
            modalities=(input_value.content.modality,),
            role_pairs=((input_value.content.subject_role, input_value.content.actor_role),),
        )
    else:
        adapter = FakeAdapter(
            input_value,
            name=name,
            kinds=(input_value.content.record_kind,),
            modalities=(input_value.content.modality,),
            role_pairs=((input_value.content.subject_role, input_value.content.actor_role),),
        )
    registry.register(adapter)
    ingress = BehaviorEvidenceIngressService(ledger, registry, config=config, clock=FakeClock())
    result = await ingress.ingest(name, {}, delivery_id=digest("delivery-" + name))
    return ledger, result.records[0]


def normalization_service(tmp_path, database, evidence_ledger, config, *model_normalizers):
    claim_ledger = SQLiteBehaviorClaimLedger(database)
    registry = ClaimNormalizerRegistry()
    registry.register(BuiltinDeterministicClaimNormalizer())
    for normalizer in model_normalizers:
        registry.register(normalizer)
    planner = ClaimNormalizationPlanner(registry, config.normalization)
    binder = ClaimBinder(config=config.normalization)
    service = ClaimNormalizationService(
        evidence_ledger,
        claim_ledger,
        registry,
        planner,
        binder,
        SQLiteProcessingLock(tmp_path),
        clock=FakeClock(),
    )
    return claim_ledger, service


def test_deterministic_core_maps_without_model_and_binds_evidence_fields() -> None:
    normalizer = BuiltinDeterministicClaimNormalizer()
    record = direct_record(semantic_input())
    proposals = asyncio.run(normalizer.normalize(record))
    assert len(proposals) == 1 and proposals[0].claim_kind is ClaimKind.ACTIVITY
    binder = ClaimBinder(config=BehaviorConfig().normalization)
    claim = binder.bind(
        record,
        proposals[0],
        normalizer_fingerprint=normalizer.fingerprint.digest,
        normalizer_kind=normalizer.kind,
        derivation_class=DerivationClass.DETERMINISTIC,
        created_at=BASE_TIME,
    )
    assert claim.evidence_record_id == record.evidence_record_id
    assert claim.subject_role is record.semantic_content.subject_role
    assert claim.actor_role is record.semantic_content.actor_role
    assert claim.time_start == record.semantic_content.event_time_start
    assert claim.source_epistemic_class is SourceEpistemicClass.MODEL_INFERRED
    assert claim.effective_confidence == record.semantic_content.source_confidence


def test_binder_rejects_normalizer_kind_derivation_mismatch() -> None:
    normalizer = BuiltinDeterministicClaimNormalizer()
    record = direct_record(semantic_input())
    proposal = asyncio.run(normalizer.normalize(record))[0]
    binder = ClaimBinder(config=BehaviorConfig().normalization)
    with pytest.raises(BehaviorClaimSchemaError, match="derivation class disagree"):
        binder.bind(
            record,
            proposal,
            normalizer_fingerprint=normalizer.fingerprint.digest,
            normalizer_kind=normalizer.kind,
            derivation_class=DerivationClass.MODEL,
            created_at=BASE_TIME,
        )


@pytest.mark.parametrize(
    ("kind", "payload", "subject", "actor", "modality", "expected"),
    [
        (BehaviorRecordKind.ACTIVITY_SEGMENT, ActivitySegmentPayload("walk", PhaseHint.STARTED, {}), BehaviorRole.USER, BehaviorRole.USER, BehaviorModality.VISION, ClaimKind.ACTIVITY),
        (BehaviorRecordKind.UTTERANCE_SEGMENT, UtteranceSegmentPayload("hello", "en", InteractionMode.DIALOGUE, CommunicationChannel.TEXT), BehaviorRole.USER, BehaviorRole.USER, BehaviorModality.TEXT, ClaimKind.UTTERANCE),
        (BehaviorRecordKind.STATE_ASSERTION, StateAssertionPayload("ready", True, {}), BehaviorRole.SYSTEM, None, BehaviorModality.SYSTEM, ClaimKind.STATE_ASSERTION),
        (BehaviorRecordKind.STATE_TRANSITION, StateTransitionPayload("mode", "old", "new", {}), BehaviorRole.TOOL, BehaviorRole.SYSTEM, BehaviorModality.TOOL, ClaimKind.STATE_TRANSITION),
        (BehaviorRecordKind.INTERACTION_SEGMENT, InteractionSegmentPayload("handoff", BehaviorRole.ROBOT, PhaseHint.IN_PROGRESS, {}), BehaviorRole.USER, BehaviorRole.USER, BehaviorModality.VISION, ClaimKind.INTERACTION),
        (BehaviorRecordKind.ACTION_EVENT, ActionEventPayload("move", "STARTED", None, None, None, {}), BehaviorRole.ROBOT, BehaviorRole.ROBOT, BehaviorModality.ROBOT, ClaimKind.ACTION),
        (BehaviorRecordKind.TOOL_CALL_EVENT, ToolCallPayload("search", "call", digest("args"), None, None), BehaviorRole.TOOL, BehaviorRole.AGENT, BehaviorModality.TOOL, ClaimKind.TOOL_CALL),
        (BehaviorRecordKind.TOOL_RESULT_EVENT, ToolResultPayload("search", "call", ToolResultStatus.SUCCESS, None, digest("result"), None), BehaviorRole.TOOL, BehaviorRole.TOOL, BehaviorModality.TOOL, ClaimKind.TOOL_RESULT),
        (BehaviorRecordKind.ENVIRONMENT_CHANGE, EnvironmentChangePayload("light", "off", "on", {}), BehaviorRole.ENVIRONMENT, None, BehaviorModality.SENSOR, ClaimKind.ENVIRONMENT_CHANGE),
        (BehaviorRecordKind.COVERAGE_INTERVAL, CoverageIntervalPayload(BehaviorModality.SYSTEM, CoverageStatus.COVERED, None, None), BehaviorRole.SYSTEM, BehaviorRole.SYSTEM, BehaviorModality.SYSTEM, ClaimKind.COVERAGE),
        (BehaviorRecordKind.FEEDBACK_EVENT, FeedbackPayload("explicit", None, FeedbackPolarity.POSITIVE, None, {}), BehaviorRole.USER, BehaviorRole.USER, BehaviorModality.TEXT, ClaimKind.FEEDBACK),
    ],
)
def test_every_strong_record_kind_has_exactly_one_deterministic_core_claim(
    kind,
    payload,
    subject,
    actor,
    modality,
    expected,
) -> None:
    semantic = BehaviorSemanticInput(
        make_content(kind, payload, subject=subject, actor=actor, modality=modality),
        source_descriptor(event=kind.value.casefold()),
    )
    proposals = asyncio.run(BuiltinDeterministicClaimNormalizer().normalize(direct_record(semantic)))
    assert len(proposals) == 1
    assert proposals[0].claim_kind is expected


def test_ingress_does_not_normalize_and_core_receipt_replays(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, semantic_input(), config=config))
    claim_ledger, service = normalization_service(tmp_path, database, evidence_ledger, config)
    assert claim_ledger.list_after_sequence(0, 10) == ()
    first = asyncio.run(service.normalize(record.evidence_record_id))
    replay = asyncio.run(service.normalize(record.evidence_record_id))
    assert first.core_receipt.status is ReceiptStatus.COMPLETED
    assert replay.core_receipt == first.core_receipt
    claims = claim_ledger.list_after_sequence(0, 10)
    assert len(claims) == 1 and claims[0].claim.claim_kind is ClaimKind.ACTIVITY


def test_free_text_has_no_core_receipt_and_model_success_or_abstain(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "success", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer((state_proposal(),))
    claim_ledger, service = normalization_service(tmp_path / "success", database, evidence_ledger, config, model)
    result = asyncio.run(service.normalize(record.evidence_record_id))
    assert result.core_receipt.status is ReceiptStatus.NO_CORE_REQUIRED
    assert result.core_receipt.attempt_ids == result.core_receipt.claim_ids == ()
    assert result.enhancement_receipts[0].status is ReceiptStatus.COMPLETED
    claim = claim_ledger.read_claim(result.enhancement_receipts[0].claim_ids[0])
    assert claim is not None and claim.effective_confidence == 0.6

    database2 = BehaviorDatabase(tmp_path / "abstain", config=config, initialize=True)
    evidence2, record2 = asyncio.run(ingest_record(database2, free_text_input(), config=config, name="abstain"))
    abstaining = FakeModelNormalizer((), name="abstaining")
    _, service2 = normalization_service(tmp_path / "abstain", database2, evidence2, config, abstaining)
    result2 = asyncio.run(service2.normalize(record2.evidence_record_id))
    assert result2.enhancement_receipts[0].status is ReceiptStatus.ABSTAINED
    assert result2.enhancement_receipts[0].claim_ids == ()


@pytest.mark.parametrize(
    ("error", "status", "retryable"),
    [
        (ClaimModelTransportError("transport"), AttemptStatus.FAILED_RETRYABLE, True),
        (ClaimModelSchemaError("schema"), AttemptStatus.FAILED_NON_RETRYABLE, False),
        (ClaimModelInputError("input"), AttemptStatus.FAILED_NON_RETRYABLE, False),
        (ClaimModelContentSafetyError("safety"), AttemptStatus.FAILED_POLICY, False),
        (ClaimModelAuthenticationError("auth"), AttemptStatus.FAILED_NON_RETRYABLE, False),
        (ClaimModelPermissionError("permission"), AttemptStatus.FAILED_NON_RETRYABLE, False),
        (ClaimModelConfigurationError("config"), AttemptStatus.FAILED_NON_RETRYABLE, False),
        (ClaimModelQuotaError("quota"), AttemptStatus.FAILED_NON_RETRYABLE, False),
    ],
)
def test_model_failure_isolated_from_evidence_and_no_success_receipt(
    tmp_path, error, status, retryable
) -> None:
    config = BehaviorConfig()
    root = tmp_path / type(error).__name__
    database = BehaviorDatabase(root / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(
        ingest_record(database, free_text_input(), config=config, name=type(error).__name__)
    )
    model = FakeModelNormalizer(error=error, name="model_" + type(error).__name__)
    claim_ledger, service = normalization_service(root, database, evidence_ledger, config, model)
    result = asyncio.run(service.normalize(record.evidence_record_id))
    assert result.core_receipt.status is ReceiptStatus.NO_CORE_REQUIRED
    assert evidence_ledger.read(record.evidence_record_id) == record
    assert claim_ledger.list_after_sequence(0, 10) == ()
    assert result.degradations[0].retryable is retryable
    plan = service.planner.plan(record)
    identity = service._route_identity(record, plan, plan.enhancement_routes[0])
    attempt = claim_ledger.read_latest_attempt(identity)
    assert attempt is not None and attempt.status is status
    assert claim_ledger.read_receipt(identity) is None


def test_invalid_model_proposal_sequence_is_non_retryable_schema_failure(tmp_path) -> None:
    class InvalidSequenceNormalizer(FakeModelNormalizer):
        async def normalize(self, record):
            del record
            self.calls += 1
            return [state_proposal()]

    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(
        ingest_record(database, free_text_input(), config=config, name="invalid-sequence")
    )
    model = InvalidSequenceNormalizer(name="invalid_sequence")
    claim_ledger, service = normalization_service(
        tmp_path,
        database,
        evidence_ledger,
        config,
        model,
    )

    result = asyncio.run(service.normalize(record.evidence_record_id))

    assert result.degradations[0].error_code == "CLAIM_SCHEMA"
    assert result.degradations[0].retryable is False
    plan = service.planner.plan(record)
    identity = service._route_identity(record, plan, plan.enhancement_routes[0])
    attempt = claim_ledger.read_latest_attempt(identity)
    assert attempt is not None
    assert attempt.status is AttemptStatus.FAILED_NON_RETRYABLE
    assert claim_ledger.read_receipt(identity) is None


def structured_client(*, value=(), error: Exception | None = None):
    client = object.__new__(StructuredChatClient)
    client.client = SimpleNamespace(provider_name="test", model="test-model")
    calls = []

    async def complete_json_async(self, request, **kwargs):
        del self
        calls.append((request, kwargs))
        if error is not None:
            raise error
        return SimpleNamespace(value=value)

    client.complete_json_async = MethodType(complete_json_async, client)
    return client, calls


@pytest.mark.parametrize(
    ("model_error", "behavior_error"),
    [
        (ModelTransportError("transport"), ClaimModelTransportError),
        (ModelRateLimitError("rate-limit"), ClaimModelTransportError),
        (ModelStructuredOutputError("schema"), ClaimModelSchemaError),
        (ModelResponseError("response"), ClaimModelSchemaError),
        (ModelInputTooLargeError("input"), ClaimModelInputError),
        (ModelAuthenticationError("auth"), ClaimModelAuthenticationError),
        (ModelPermissionError("permission"), ClaimModelPermissionError),
        (ModelConfigurationError("configuration"), ClaimModelConfigurationError),
        (ModelQuotaError("quota"), ClaimModelQuotaError),
        (ModelContentSafetyError("safety"), ClaimModelContentSafetyError),
    ],
)
def test_structured_model_normalizer_preserves_error_classes(model_error, behavior_error) -> None:
    client, _ = structured_client(error=model_error)
    normalizer = StructuredModelClaimNormalizer(client, config=BehaviorConfig().normalization)
    with pytest.raises(behavior_error) as captured:
        asyncio.run(normalizer.normalize(direct_record(free_text_input())))
    assert captured.value.__cause__ is model_error


def test_structured_model_normalizer_uses_dynamic_schema_untrusted_boundary_and_budgets() -> None:
    proposal = state_proposal()
    client, calls = structured_client(value=(proposal,))
    config = replace(
        BehaviorConfig().normalization,
        max_model_input_tokens=777,
        max_model_output_tokens=123,
    )
    normalizer = StructuredModelClaimNormalizer(client, config=config)
    assert asyncio.run(normalizer.normalize(direct_record(free_text_input()))) == (proposal,)
    request, kwargs = calls[0]
    assert request.max_output_tokens == 123
    assert "untrusted data" in request.messages[0].content
    assert kwargs["context"].input_token_limit == 777
    assert kwargs["schema"]["properties"]["proposals"]["maxItems"] == config.max_claims_per_record

    char_client, _ = structured_client()
    with pytest.raises(ClaimModelInputError):
        asyncio.run(
            StructuredModelClaimNormalizer(
                char_client,
                config=replace(BehaviorConfig().normalization, max_model_input_chars=32),
            ).normalize(direct_record(free_text_input()))
        )
    token_client, _ = structured_client()
    with pytest.raises(ClaimModelInputError):
        asyncio.run(
            StructuredModelClaimNormalizer(
                token_client,
                config=replace(BehaviorConfig().normalization, max_model_input_tokens=1),
            ).normalize(direct_record(free_text_input()))
        )


def test_retry_enhancement_only_retries_retryable_failure(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer(error=ClaimModelTransportError("temporary"))
    claim_ledger, service = normalization_service(tmp_path, database, evidence_ledger, config, model)
    first = asyncio.run(service.normalize(record.evidence_record_id))
    assert first.degradations[0].retryable and model.calls == 1
    model.error = None
    model.proposals = (state_proposal(),)
    retried = asyncio.run(service.retry_enhancement(record.evidence_record_id, model.name))
    assert retried.enhancement_receipts and model.calls == 2
    receipt = retried.enhancement_receipts[0]
    assert len(receipt.attempt_ids) == 2
    assert claim_ledger.read_attempt(receipt.attempt_ids[0]).status is AttemptStatus.FAILED_RETRYABLE
    assert claim_ledger.read_attempt(receipt.attempt_ids[1]).status is AttemptStatus.COMPLETED
    assert len(claim_ledger.list_after_sequence(0, 10)) == 1
    with pytest.raises(ClaimNormalizationError):
        asyncio.run(service.retry_enhancement(record.evidence_record_id, "missing"))


def test_retry_enhancement_never_runs_missing_core(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer(error=ClaimModelTransportError("temporary"))
    claim_ledger, service = normalization_service(
        tmp_path,
        database,
        evidence_ledger,
        config,
        model,
    )
    with pytest.raises(ClaimNormalizationConflictError, match="Core must be completed"):
        asyncio.run(service.retry_enhancement(record.evidence_record_id, model.name))
    plan = service.planner.plan(record)
    assert claim_ledger.read_receipt(service._core_identity(record, plan)) is None
    assert claim_ledger.read_latest_attempt(
        service._route_identity(record, plan, plan.enhancement_routes[0])
    ) is None
    assert model.calls == 0


def test_user_utterance_core_survives_forbidden_model_claim(tmp_path) -> None:
    config = BehaviorConfig(
        normalization=replace(BehaviorConfig().normalization, normalize_user_utterances=True)
    )
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, utterance_input(), config=config))
    model = FakeModelNormalizer((state_proposal(kind=ClaimKind.UTTERANCE),))
    claim_ledger, service = normalization_service(tmp_path, database, evidence_ledger, config, model)
    result = asyncio.run(service.normalize(record.evidence_record_id))
    assert result.core_receipt.status is ReceiptStatus.COMPLETED
    assert result.degradations and not result.degradations[0].retryable
    claims = claim_ledger.list_after_sequence(0, 10)
    assert len(claims) == 1 and claims[0].claim.claim_kind is ClaimKind.UTTERANCE


def test_compatibility_policy_forbids_utterance_environment_tool_action_and_coverage() -> None:
    policy = ClaimCompatibilityPolicy()
    for kind in (
        ClaimKind.ENVIRONMENT_CHANGE,
        ClaimKind.TOOL_CALL,
        ClaimKind.TOOL_RESULT,
        ClaimKind.ACTION,
        ClaimKind.COVERAGE,
        ClaimKind.UTTERANCE,
    ):
        assert not policy.evaluate(
            record_kind=BehaviorRecordKind.UTTERANCE_SEGMENT,
            subject_role=BehaviorRole.USER,
            actor_role=BehaviorRole.USER,
            normalizer_kind="MODEL",
            claim_kind=kind,
        ).allowed
    assert policy.evaluate(
        record_kind=BehaviorRecordKind.UTTERANCE_SEGMENT,
        subject_role=BehaviorRole.USER,
        actor_role=BehaviorRole.USER,
        normalizer_kind="MODEL",
        claim_kind=ClaimKind.STATE_ASSERTION,
    ).allowed


def test_claim_identity_summary_fingerprint_confidence_and_alternative_namespace() -> None:
    first_input = free_text_input()
    second_input = replace(
        first_input,
        source=source_descriptor(event="free-text-two", sequence=2),
    )
    record1 = direct_record(first_input)
    record2 = direct_record(second_input)
    binder = ClaimBinder(config=BehaviorConfig().normalization)
    normalizer = FakeModelNormalizer()
    proposal1 = replace(state_proposal(summary="one"), local_alternative_group_id="a")
    proposal2 = replace(state_proposal(summary="two"), local_alternative_group_id="a")
    claim1 = binder.bind(record1, proposal1, normalizer_fingerprint=normalizer.fingerprint.digest, normalizer_kind=normalizer.kind, derivation_class=DerivationClass.MODEL, created_at=BASE_TIME)
    claim1_summary = binder.bind(record1, proposal2, normalizer_fingerprint=normalizer.fingerprint.digest, normalizer_kind=normalizer.kind, derivation_class=DerivationClass.MODEL, created_at=BASE_TIME + timedelta(seconds=1))
    claim2 = binder.bind(record2, proposal1, normalizer_fingerprint=normalizer.fingerprint.digest, normalizer_kind=normalizer.kind, derivation_class=DerivationClass.MODEL, created_at=BASE_TIME)
    assert claim1.claim_id == claim1_summary.claim_id
    assert claim1.content_digest != claim1_summary.content_digest
    assert claim1.semantic_fingerprint == claim2.semantic_fingerprint
    assert claim1.claim_id != claim2.claim_id
    assert claim1.alternative_group_key != claim2.alternative_group_key
    other_normalizer = FakeModelNormalizer(name="other_model")
    other_claim = binder.bind(
        record1,
        proposal1,
        normalizer_fingerprint=other_normalizer.fingerprint.digest,
        normalizer_kind=other_normalizer.kind,
        derivation_class=DerivationClass.MODEL,
        created_at=BASE_TIME,
    )
    assert claim1.alternative_group_key != other_claim.alternative_group_key
    confidence_changed = replace(proposal1, normalizer_confidence=0.2)
    claim3 = binder.bind(record1, confidence_changed, normalizer_fingerprint=normalizer.fingerprint.digest, normalizer_kind=normalizer.kind, derivation_class=DerivationClass.MODEL, created_at=BASE_TIME)
    assert claim3.semantic_fingerprint == claim1.semantic_fingerprint


@pytest.mark.parametrize(
    ("trust", "expected"),
    [
        (BehaviorSourceTrust.DIRECT_SYSTEM_LOG, SourceEpistemicClass.DIRECT_SOURCE),
        (BehaviorSourceTrust.DIRECT_DEVICE_FACT, SourceEpistemicClass.DIRECT_SOURCE),
        (BehaviorSourceTrust.USER_EXPLICIT, SourceEpistemicClass.USER_EXPLICIT),
        (BehaviorSourceTrust.SENSOR_INFERRED, SourceEpistemicClass.SENSOR_INFERRED),
        (BehaviorSourceTrust.MODEL_INFERRED, SourceEpistemicClass.MODEL_INFERRED),
        (
            BehaviorSourceTrust.MULTIMODAL_MODEL_INFERRED,
            SourceEpistemicClass.MULTIMODAL_MODEL_INFERRED,
        ),
    ],
)
def test_all_source_trust_classes_map_mechanically_to_epistemic(trust, expected) -> None:
    record = replace(direct_record(semantic_input()), source_trust=trust)
    normalizer = BuiltinDeterministicClaimNormalizer()
    proposal = asyncio.run(normalizer.normalize(record))[0]
    claim = ClaimBinder(config=BehaviorConfig().normalization).bind(
        record,
        proposal,
        normalizer_fingerprint=normalizer.fingerprint.digest,
        normalizer_kind=normalizer.kind,
        derivation_class=DerivationClass.DETERMINISTIC,
        created_at=BASE_TIME,
    )
    assert claim.source_epistemic_class is expected


def test_compatibility_digest_propagates_to_identity_receipt_and_claim(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer((state_proposal(),))
    claim_ledger, service = normalization_service(
        tmp_path,
        database,
        evidence_ledger,
        config,
        model,
    )
    result = asyncio.run(service.normalize(record.evidence_record_id))
    receipt = result.enhancement_receipts[0]
    claim = claim_ledger.read_claim(receipt.claim_ids[0])
    plan = service.planner.plan(record)
    assert claim is not None
    assert receipt.compatibility_policy_digest == service.binder.compatibility.digest
    assert claim.compatibility_policy_digest == service.binder.compatibility.digest
    assert receipt.processing_identity == service._route_identity(
        record,
        plan,
        plan.enhancement_routes[0],
    )


def test_proposal_semantic_payload_rejects_system_bound_fields_recursively() -> None:
    for field in (
        "subject_role",
        "actor_role",
        "time_start",
        "evidence_record_id",
        "source_trust",
        "created_at",
        "claim_sequence",
        "content_digest",
    ):
        with pytest.raises(ValueError, match="reserved"):
            replace(state_proposal(), semantic_payload={"nested": {field: "forbidden"}})
    normalizer = FakeModelNormalizer()
    record = direct_record(free_text_input())
    claim = ClaimBinder(config=BehaviorConfig().normalization).bind(
        record,
        state_proposal(),
        normalizer_fingerprint=normalizer.fingerprint.digest,
        normalizer_kind=normalizer.kind,
        derivation_class=DerivationClass.MODEL,
        created_at=BASE_TIME,
    )
    with pytest.raises(ValueError, match="reserved"):
        replace(claim, semantic_payload={"nested": {"created_at": "forged"}})


def test_concurrent_same_processing_calls_model_once(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer((state_proposal(),))
    _, service = normalization_service(tmp_path, database, evidence_ledger, config, model)

    async def run_both():
        return await asyncio.gather(
            service.normalize(record.evidence_record_id),
            service.normalize(record.evidence_record_id),
        )

    results = asyncio.run(run_both())
    assert model.calls == 1
    assert results[0].enhancement_receipts[0] == results[1].enhancement_receipts[0]
