from __future__ import annotations

import asyncio
from dataclasses import replace

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
from behavior.ingress import SemanticIngressAdapterRegistry, SemanticRecordKind
from foundation.observability import NullObserver
from ModelClient import StructuredChatClient
from Runtime import RuntimeBehavior, RuntimeState, build_runtime
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies


def test_runtime_behavior_uses_shared_model_store_and_initializes_without_worker(tmp_path) -> None:
    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(config, providers=providers, vector_stores=vectors)
    behavior = runtime.components.behavior
    assert isinstance(behavior, RuntimeBehavior)
    assert behavior.store.root == config.behavior_root
    assert behavior.structured_chat is runtime.components.models.structured_chat
    structured = behavior.claim_normalizers.get("model_text")
    assert structured.model_client is runtime.components.models.structured_chat
    assert behavior.claim_router.registry is behavior.claim_normalizers
    assert behavior.claim_pipeline.ingress_service is behavior.ingress_service
    assert not config.storage_root.exists()
    initialization = runtime.initialize()
    assert initialization.behavior_root == config.behavior_root
    assert behavior.store.path.exists()
    assert runtime.initialize() is initialization
    assert runtime.state is RuntimeState.READY
    report = asyncio.run(runtime.health())
    check = next(item for item in report.checks if item.name == "behavior_store")
    assert "schema=3" in check.detail
    assert "semantic_records=0" in check.detail
    assert "active_bundles=0" in check.detail
    assert "manifests=0" in check.detail
    assert "validated_claims=0" in check.detail
    assert "accepted_claims=0" in check.detail
    assert "normalizer_attempts=0" in check.detail


def test_build_runtime_uses_explicit_behavior_adapter_registry_without_writes(tmp_path) -> None:
    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()
    adapters = SemanticIngressAdapterRegistry()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        behavior_adapters=adapters,
    )
    assert runtime.components.behavior.adapters is adapters
    assert runtime.components.behavior.ingress_service.adapters is adapters
    assert not config.storage_root.exists()


class CustomModelNormalizer:
    name = "model_text"
    kind = ClaimNormalizerKind.MODEL
    allowed_record_kinds = frozenset({SemanticRecordKind.FREE_TEXT_SEMANTIC})

    def __init__(
        self,
        client: StructuredChatClient,
        compatibility_policy: ClaimCompatibilityPolicy | None = None,
    ) -> None:
        self.model_client = client
        self.compatibility_policy = compatibility_policy or ClaimCompatibilityPolicy()
        self.fingerprint = NormalizerFingerprint(
            self.name,
            "custom-v3",
            self.kind,
            "test",
            "custom",
            "model",
            "prompt-v3",
        )

    async def normalize(self, record):
        del record
        return ClaimSemanticProposalBatch(True, ())


def _runtime_behavior_with_custom_model(runtime, client: StructuredChatClient) -> RuntimeBehavior:
    current = runtime.components.behavior
    registry = ClaimNormalizerRegistry()
    registry.register(DeterministicClaimNormalizer())
    registry.register(
        CustomModelNormalizer(
            client,
            current.claim_pipeline.binding_policy.compatibility,
        )
    )
    router = ClaimNormalizationRouter(registry, config=current.store.config.claim)
    pipeline = ClaimPipelineService(
        current.store,
        current.ingress_service,
        current.evidence_service,
        registry,
        router,
        config=current.store.config.claim,
        observer=NullObserver(),
    )
    return RuntimeBehavior(
        store=current.store,
        adapters=current.adapters,
        ingress_service=current.ingress_service,
        evidence_service=current.evidence_service,
        claim_normalizers=registry,
        claim_router=router,
        claim_pipeline=pipeline,
        structured_chat=current.structured_chat,
    )


def test_custom_model_normalizer_must_use_runtime_shared_structured_chat(tmp_path) -> None:
    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(config, providers=providers, vector_stores=vectors)
    shared = runtime.components.models.structured_chat
    assert _runtime_behavior_with_custom_model(runtime, shared).structured_chat is shared
    with pytest.raises(ValueError, match="shared StructuredChatClient"):
        _runtime_behavior_with_custom_model(runtime, object.__new__(StructuredChatClient))


def test_runtime_rejects_router_and_pipeline_claim_config_drift(tmp_path) -> None:
    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(config, providers=providers, vector_stores=vectors)
    current = runtime.components.behavior
    drifted = replace(current.store.config.claim, min_model_confidence=0.75)
    router = ClaimNormalizationRouter(current.claim_normalizers, config=drifted)
    pipeline = ClaimPipelineService(
        current.store,
        current.ingress_service,
        current.evidence_service,
        current.claim_normalizers,
        router,
        config=drifted,
        observer=NullObserver(),
    )
    with pytest.raises(ValueError, match="Store Claim configuration"):
        RuntimeBehavior(
            store=current.store,
            adapters=current.adapters,
            ingress_service=current.ingress_service,
            evidence_service=current.evidence_service,
            claim_normalizers=current.claim_normalizers,
            claim_router=router,
            claim_pipeline=pipeline,
            structured_chat=current.structured_chat,
        )
