"""Runtime 五组组件的类型、共享实例和跨领域接线契约矩阵。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from Runtime import build_runtime
from Runtime.components import (
    RuntimeComponents,
    RuntimeConversation,
    RuntimeInfrastructure,
    RuntimeMemory,
    RuntimeModels,
    RuntimeWorkflow,
)
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies

INVALID_OBJECTS = (None, "value", 0, 1, True, False, (), [], {}, object())


@pytest.fixture
def runtime_pair(tmp_path: Path):
    values = []
    for name in ("first", "second"):
        providers, vectors = runtime_dependencies()
        values.append(
            build_runtime(
                runtime_config(tmp_path / name),
                providers=providers,
                vector_stores=vectors,
                path_lock=PathLock(ProcessLocalLockStore()),
                environ={},
            )
        )
    return tuple(values)


@pytest.mark.parametrize("field", ["path_lock", "vector_stores"])
@pytest.mark.parametrize("invalid", INVALID_OBJECTS)
def test_runtime_infrastructure_requires_exact_infrastructure_types(
    runtime_pair,
    field: str,
    invalid: object,
) -> None:
    current = runtime_pair[0].components.infrastructure
    with pytest.raises(TypeError):
        replace(current, **{field: invalid})


class InitializableStore:
    def __init__(self, initialize=...):
        self.calls = 0
        if initialize is not ...:
            self.initialize = initialize


def test_runtime_infrastructure_calls_optional_lock_store_initializer_once_per_request(runtime_pair) -> None:
    store = InitializableStore()

    def initialize() -> None:
        store.calls += 1

    store.initialize = initialize
    infrastructure = replace(
        runtime_pair[0].components.infrastructure,
        path_lock=PathLock(store),
    )
    infrastructure.initialize()
    infrastructure.initialize()
    assert store.calls == 2


def test_runtime_infrastructure_accepts_lock_store_without_initializer(runtime_pair) -> None:
    infrastructure = replace(
        runtime_pair[0].components.infrastructure,
        path_lock=PathLock(InitializableStore()),
    )
    infrastructure.initialize()


@pytest.mark.parametrize("invalid", [0, "initialize", object()])
def test_runtime_infrastructure_rejects_non_callable_initializer(runtime_pair, invalid: object) -> None:
    store = InitializableStore(invalid)
    infrastructure = replace(
        runtime_pair[0].components.infrastructure,
        path_lock=PathLock(store),
    )
    with pytest.raises(TypeError, match="must be callable"):
        infrastructure.initialize()


@pytest.mark.parametrize("field", ["providers", "chat", "structured_chat"])
@pytest.mark.parametrize("invalid", INVALID_OBJECTS)
def test_runtime_models_requires_exact_shared_client_types(
    runtime_pair,
    field: str,
    invalid: object,
) -> None:
    current = runtime_pair[0].components.models
    with pytest.raises(TypeError):
        replace(current, **{field: invalid})


@pytest.mark.parametrize("invalid", INVALID_OBJECTS)
def test_runtime_models_embedder_requires_both_query_and_document_methods(
    runtime_pair,
    invalid: object,
) -> None:
    current = runtime_pair[0].components.models
    with pytest.raises(TypeError, match="Embedder contract"):
        replace(current, embedder=invalid)


@pytest.mark.parametrize("invalid", tuple(value for value in INVALID_OBJECTS if value is not None))
def test_runtime_models_reranker_requires_callable_rerank_or_none(runtime_pair, invalid: object) -> None:
    current = runtime_pair[0].components.models
    with pytest.raises(TypeError, match="Reranker contract"):
        replace(current, reranker=invalid)
    assert replace(current, reranker=None).reranker is None


def test_runtime_models_structured_client_must_wrap_same_chat_instance(runtime_pair) -> None:
    first, second = runtime_pair
    with pytest.raises(ValueError, match="shared chat client"):
        replace(
            first.components.models,
            structured_chat=second.components.models.structured_chat,
        )


@pytest.mark.parametrize(
    "field",
    ["journal", "retention", "summaries", "summary_compactor", "summary_vector_index"],
)
@pytest.mark.parametrize("invalid", INVALID_OBJECTS)
def test_runtime_conversation_requires_each_exact_domain_component(
    runtime_pair,
    field: str,
    invalid: object,
) -> None:
    current = runtime_pair[0].components.conversation
    with pytest.raises(TypeError):
        replace(current, **{field: invalid})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("summaries", "share one root"),
        ("summary_compactor", "shared journal"),
        ("summary_vector_index", "share one Summary compactor"),
    ],
)
def test_runtime_conversation_rejects_valid_component_from_another_runtime(
    runtime_pair,
    field: str,
    message: str,
) -> None:
    first, second = runtime_pair
    with pytest.raises(ValueError, match=message):
        replace(
            first.components.conversation,
            **{field: getattr(second.components.conversation, field)},
        )


@pytest.mark.parametrize(
    "field",
    ["tree", "search", "editor", "semantic_refresher", "vector_index"],
)
@pytest.mark.parametrize("invalid", INVALID_OBJECTS)
def test_runtime_memory_requires_each_exact_domain_component(
    runtime_pair,
    field: str,
    invalid: object,
) -> None:
    current = runtime_pair[0].components.memory
    with pytest.raises(TypeError):
        replace(current, **{field: invalid})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("search", "shared memory tree"),
        ("editor", "shared memory tree"),
        ("semantic_refresher", "shared memory tree"),
        ("vector_index", "shared memory tree"),
    ],
)
def test_runtime_memory_rejects_valid_component_bound_to_another_tree(
    runtime_pair,
    field: str,
    message: str,
) -> None:
    first, second = runtime_pair
    with pytest.raises(ValueError, match=message):
        replace(
            first.components.memory,
            **{field: getattr(second.components.memory, field)},
        )


def test_runtime_memory_search_and_persistent_index_must_share_one_index(runtime_pair) -> None:
    first, second = runtime_pair
    search = first.components.memory.search
    search.semantic_search = second.components.memory.search.semantic_search
    with pytest.raises(ValueError, match="shared persistent vector index"):
        replace(first.components.memory, search=search)


@pytest.mark.parametrize(
    "field",
    ["jobs", "receipts", "enqueuer", "lifecycle", "runner", "worker", "lifecycle_worker"],
)
@pytest.mark.parametrize("invalid", INVALID_OBJECTS)
def test_runtime_workflow_requires_each_exact_domain_component(
    runtime_pair,
    field: str,
    invalid: object,
) -> None:
    current = runtime_pair[0].components.workflow
    with pytest.raises(TypeError):
        replace(current, **{field: invalid})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("jobs", "share one job store"),
        ("receipts", "share workflow stores"),
        ("enqueuer", "share one job store"),
        ("lifecycle", "share workflow stores"),
        ("runner", "share one job store"),
        ("worker", "share one runner"),
        ("lifecycle_worker", "share one lifecycle manager"),
    ],
)
def test_runtime_workflow_rejects_component_from_another_runtime(
    runtime_pair,
    field: str,
    message: str,
) -> None:
    first, second = runtime_pair
    with pytest.raises(ValueError, match=message):
        replace(
            first.components.workflow,
            **{field: getattr(second.components.workflow, field)},
        )


@pytest.mark.parametrize(
    "field",
    ["infrastructure", "models", "conversation", "memory", "workflow"],
)
@pytest.mark.parametrize("invalid", INVALID_OBJECTS)
def test_runtime_components_requires_each_exact_component_group(
    runtime_pair,
    field: str,
    invalid: object,
) -> None:
    current = runtime_pair[0].components
    with pytest.raises(TypeError):
        replace(current, **{field: invalid})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("infrastructure", "share one path lock"),
        ("models", "structured chat client"),
        ("conversation", "conversation journal"),
        ("memory", "memory editor"),
        ("workflow", "conversation journal"),
    ],
)
def test_runtime_components_rejects_cross_runtime_domain_group(
    runtime_pair,
    field: str,
    message: str,
) -> None:
    first, second = runtime_pair
    with pytest.raises(ValueError, match=message):
        replace(
            first.components,
            **{field: getattr(second.components, field)},
        )


@pytest.mark.parametrize(
    "model_type",
    [
        RuntimeInfrastructure,
        RuntimeModels,
        RuntimeConversation,
        RuntimeMemory,
        RuntimeWorkflow,
        RuntimeComponents,
    ],
)
def test_runtime_component_groups_are_frozen(model_type: type[object]) -> None:
    assert model_type.__dataclass_params__.frozen is True


def test_runtime_component_group_rejects_direct_mutation(runtime_pair) -> None:
    current = runtime_pair[0].components
    with pytest.raises(FrozenInstanceError):
        current.memory = runtime_pair[1].components.memory
