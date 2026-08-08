from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest

from behavior.claim import ClaimNormalizerKind
from behavior.evidence import BehaviorSemanticAdapterRegistry
from infrastructure.store.processing_lock import RenewableProcessingLock
from Runtime import RuntimeBehavior, RuntimeState, build_runtime
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies
from tests.unit.behavior.conftest import FakeAdapter, digest
from tests.unit.behavior.test_evidence_ingress_ledger import semantic_input


def test_runtime_behavior_uses_shared_model_database_and_initializes_without_worker(tmp_path) -> None:
    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(config, providers=providers, vector_stores=vectors)
    behavior = runtime.components.behavior
    assert isinstance(behavior, RuntimeBehavior)
    assert behavior.database.root == config.behavior_root
    assert behavior.structured_chat is runtime.components.models.structured_chat
    model_names = behavior.claim_normalizers.names(ClaimNormalizerKind.MODEL)
    assert len(model_names) == 1
    assert behavior.claim_normalizers.get(model_names[0]).model_client is behavior.structured_chat
    assert behavior.claim_normalization.route_executor.processing_lock is behavior.processing_lock
    assert behavior.claim_normalization.evidence_ledger is behavior.evidence_ledger
    assert behavior.claim_normalization.claim_ledger is behavior.claim_ledger
    assert not config.storage_root.exists()
    initialization = runtime.initialize()
    assert initialization.behavior_root == config.behavior_root
    assert behavior.database.connection.path.exists()
    assert runtime.initialize() is initialization
    assert runtime.state is RuntimeState.READY
    report = asyncio.run(runtime.health())
    check = next(item for item in report.checks if item.name == "behavior_store")
    assert "schema=behavior_first_layer_v1" in check.detail
    assert "evidence=0" in check.detail
    assert "claims=0" in check.detail
    assert "attempts=0" in check.detail
    assert "receipts=0" in check.detail


def test_build_runtime_uses_explicit_behavior_adapter_registry_without_writes(tmp_path) -> None:
    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()
    adapters = BehaviorSemanticAdapterRegistry()
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        behavior_adapters=adapters,
    )
    assert runtime.components.behavior.evidence_adapters is adapters
    assert runtime.components.behavior.evidence_ingress.adapters is adapters
    assert not config.storage_root.exists()


def test_runtime_behavior_ingest_normalize_and_retry_entrypoints(tmp_path) -> None:
    config = runtime_config(tmp_path)
    providers, vectors = runtime_dependencies()
    adapters = BehaviorSemanticAdapterRegistry()
    adapter = FakeAdapter(semantic_input())
    adapters.register(adapter)
    runtime = build_runtime(
        config,
        providers=providers,
        vector_stores=vectors,
        behavior_adapters=adapters,
    )
    runtime.initialize()
    ingested = asyncio.run(
        runtime.ingest_behavior_semantic(adapter.name, {}, digest("runtime-delivery"))
    )
    normalized = asyncio.run(
        runtime.normalize_behavior_evidence(ingested.records[0].evidence_record_id)
    )
    assert normalized.core_receipt.claim_ids
    assert runtime.components.behavior.claim_ledger.list_after_sequence(0, 10)


def test_runtime_processing_lock_aborts_body_and_releases_when_renewal_fails() -> None:
    class Guard:
        def checkpoint(self) -> None:
            raise TimeoutError("lease lost")

    class FakePathLock:
        def __init__(self) -> None:
            self.exited = False

        @contextmanager
        def acquire(self, *args, **kwargs):
            del args, kwargs
            try:
                yield Guard()
            finally:
                self.exited = True

    path_lock = FakePathLock()
    lock = RenewableProcessingLock(path_lock, renewal_interval_seconds=0.01)

    async def run() -> None:
        async with lock.acquire("0" * 64):
            await asyncio.sleep(1)

    with pytest.raises(RuntimeError, match="lease renewal failed"):
        asyncio.run(run())
    assert path_lock.exited
