from __future__ import annotations

import asyncio

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
    structured = behavior.claim_producers.get("structured_semantic")
    assert structured.client is runtime.components.models.structured_chat
    assert not config.storage_root.exists()
    initialization = runtime.initialize()
    assert initialization.behavior_root == config.behavior_root
    assert behavior.store.path.exists()
    assert runtime.initialize() is initialization
    assert runtime.state is RuntimeState.READY
    report = asyncio.run(runtime.health())
    check = next(item for item in report.checks if item.name == "behavior_store")
    assert check.detail == "schema=1"
