"""全部可运维配置字段的类型、嵌套对象和未知字段契约矩阵。"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import pytest

from Config import (
    ConversationConfig,
    ConversationLifecycleConfig,
    MemoryConfig,
    MemoryIntentionReviewConfig,
    MemorySearchServiceConfig,
    StructuredOutputConfig,
    WorkerConfig,
    WorkflowConfig,
)
from Config.loader import ConfigError, construct_config, group_fields, required_field, strict_fields, strict_object
from infrastructure.editor.snapshot import SnapshotReadConfig
from infrastructure.store.sqlite import SQLiteLockStoreConfig
from infrastructure.vector import VectorStoreConfig, VectorStoreRouteConfig
from memory.conversation import (
    ConversationJournalConfig,
    ConversationRangeSummaryCompactionConfig,
    ConversationSegmentationConfig,
    ConversationSegmentSummaryCompactionConfig,
    ConversationSummaryCompactionConfig,
    ConversationSummaryConfig,
    ConversationSummaryVectorIndexConfig,
)
from memory.document import MemoryDocumentConfig
from memory.editor import (
    MemoryCommitConfig,
    MemoryExtractionConfig,
    MemoryRetrievalConfig,
    MemoryTransactionJournalConfig,
)
from memory.indexing import MemoryVectorIndexConfig
from memory.semantic import MemorySemanticConfig
from memory.tree import MemoryTreeConfig
from memory.workflow import (
    MemoryChangeReceiptStoreConfig,
    MemoryJobConfig,
    MemoryWorkflowLifecycleConfig,
)

CONFIG_INSTANCES = (
    ConversationLifecycleConfig(),
    ConversationConfig(),
    MemoryConfig(),
    StructuredOutputConfig(),
    WorkerConfig(),
    WorkflowConfig(),
    ConversationJournalConfig(),
    ConversationSummaryConfig(),
    ConversationSegmentSummaryCompactionConfig(),
    ConversationRangeSummaryCompactionConfig(),
    ConversationSummaryCompactionConfig(),
    ConversationSegmentationConfig(),
    ConversationSummaryVectorIndexConfig(),
    MemoryDocumentConfig(),
    MemoryExtractionConfig(),
    MemoryCommitConfig(),
    MemoryTransactionJournalConfig(),
    MemoryRetrievalConfig(),
    MemoryVectorIndexConfig(),
    MemoryIntentionReviewConfig(),
    MemorySearchServiceConfig(),
    MemorySemanticConfig(),
    MemoryTreeConfig(),
    MemoryJobConfig(),
    MemoryWorkflowLifecycleConfig(),
    MemoryChangeReceiptStoreConfig(),
    SnapshotReadConfig(),
    SQLiteLockStoreConfig(),
    VectorStoreRouteConfig(),
    VectorStoreConfig(),
)


INTEGER_FIELDS = tuple(
    (instance, field.name)
    for instance in CONFIG_INSTANCES
    for field in fields(instance)
    if isinstance(getattr(instance, field.name), int)
    and not isinstance(getattr(instance, field.name), bool)
)
FLOAT_FIELDS = tuple(
    (instance, field.name)
    for instance in CONFIG_INSTANCES
    for field in fields(instance)
    if isinstance(getattr(instance, field.name), float)
)
BOOLEAN_FIELDS = tuple(
    (instance, field.name)
    for instance in CONFIG_INSTANCES
    for field in fields(instance)
    if isinstance(getattr(instance, field.name), bool)
)
NESTED_FIELDS = tuple(
    (instance, field.name)
    for instance in CONFIG_INSTANCES
    for field in fields(instance)
    if is_dataclass(getattr(instance, field.name))
)


@pytest.mark.parametrize(
    ("instance", "field"),
    INTEGER_FIELDS,
    ids=[f"{type(instance).__name__}.{field}" for instance, field in INTEGER_FIELDS],
)
@pytest.mark.parametrize("invalid", [True, False, 1.0, "1", None])
def test_every_integer_config_field_rejects_bool_float_string_and_null(
    instance: object,
    field: str,
    invalid: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(instance, **{field: invalid})


@pytest.mark.parametrize(
    ("instance", "field"),
    FLOAT_FIELDS,
    ids=[f"{type(instance).__name__}.{field}" for instance, field in FLOAT_FIELDS],
)
@pytest.mark.parametrize("invalid", [True, False, "1.0", None, [], {}])
def test_every_numeric_duration_or_score_field_rejects_bool_and_non_numeric_values(
    instance: object,
    field: str,
    invalid: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(instance, **{field: invalid})


@pytest.mark.parametrize(
    ("instance", "field"),
    BOOLEAN_FIELDS,
    ids=[f"{type(instance).__name__}.{field}" for instance, field in BOOLEAN_FIELDS],
)
@pytest.mark.parametrize("invalid", [0, 1, "true", None, [], {}])
def test_every_boolean_config_field_rejects_truthy_or_falsy_coercion(
    instance: object,
    field: str,
    invalid: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(instance, **{field: invalid})


@pytest.mark.parametrize(
    ("instance", "field"),
    NESTED_FIELDS,
    ids=[f"{type(instance).__name__}.{field}" for instance, field in NESTED_FIELDS],
)
@pytest.mark.parametrize("invalid", [None, {}, [], "config", 1, True])
def test_every_nested_config_field_requires_its_exact_domain_config_type(
    instance: object,
    field: str,
    invalid: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(instance, **{field: invalid})


@pytest.mark.parametrize("instance", CONFIG_INSTANCES, ids=lambda item: type(item).__name__)
def test_every_default_config_is_frozen_and_dataclass_declared(instance: object) -> None:
    assert is_dataclass(instance)
    declared = fields(instance)
    assert declared
    with pytest.raises((AttributeError, TypeError)):
        setattr(instance, declared[0].name, getattr(instance, declared[0].name))


@pytest.mark.parametrize("value", [None, [], (), "config", 1, True])
@pytest.mark.parametrize("path", ["config", "config.memory", "config.workflow.jobs"])
def test_strict_object_rejects_every_non_mapping_root(value: object, path: str) -> None:
    with pytest.raises(ConfigError, match="object"):
        strict_object(value, path=path)


@pytest.mark.parametrize("key", ["", None, 1, True, (), object()])
def test_strict_object_rejects_empty_or_non_string_keys(key: object) -> None:
    with pytest.raises(ConfigError, match="string keys"):
        strict_object({key: "value"}, path="config")


@pytest.mark.parametrize(
    ("unknown", "expected_hint"),
    [
        ("memroy", "memory"),
        ("workflwo", "workflow"),
        ("conversaton", "conversation"),
        ("storag", "storage"),
        ("modelsx", "models"),
    ],
)
def test_strict_fields_reports_deterministic_typo_suggestion(unknown: str, expected_hint: str) -> None:
    with pytest.raises(ConfigError, match=expected_hint):
        strict_fields(
            {unknown: {}},
            path="config",
            allowed={"storage", "models", "conversation", "memory", "workflow"},
        )


@pytest.mark.parametrize("name", ["root", "route", "provider", "model", "collection"])
def test_required_field_distinguishes_missing_from_present_falsey_value(name: str) -> None:
    with pytest.raises(ConfigError, match=name):
        required_field({}, name, path="config")
    for value in (None, "", 0, False, [], {}):
        assert required_field({name: value}, name, path="config") is value


@pytest.mark.parametrize("model_type", [object, int, str, dict, list])
def test_group_fields_rejects_non_dataclass_config_types(model_type: type[object]) -> None:
    with pytest.raises(TypeError, match="dataclass"):
        group_fields(model_type, {}, "config")


@pytest.mark.parametrize(
    ("model_type", "field", "value"),
    [
        (MemoryTreeConfig, "max_children_per_directory", 0),
        (SnapshotReadConfig, "max_items", 0),
        (MemoryDocumentConfig, "max_encoded_bytes", 0),
        (WorkerConfig, "poll_interval_seconds", 0),
        (MemoryJobConfig, "max_attempts", 0),
    ],
)
def test_construct_config_wraps_domain_validation_with_config_path(
    model_type: type[object],
    field: str,
    value: object,
) -> None:
    with pytest.raises(ConfigError, match="config.test"):
        construct_config(model_type, {field: value}, "config.test")


@pytest.mark.parametrize(
    ("first", "second", "strong"),
    [
        (1, 2, 3),
        (30, 60, 180),
        (1, 18_250, 36_500),
    ],
)
def test_intention_review_thresholds_accept_strictly_increasing_operational_ranges(
    first: int,
    second: int,
    strong: int,
) -> None:
    config = MemoryIntentionReviewConfig(first, second, strong)
    assert (config.first_review_after_days, config.second_review_after_days, config.strong_review_after_days) == (
        first,
        second,
        strong,
    )


@pytest.mark.parametrize(
    ("first", "second", "strong"),
    [
        (0, 60, 180),
        (30, 30, 180),
        (60, 30, 180),
        (30, 180, 180),
        (30, 181, 180),
        (30, 60, 36_501),
    ],
)
def test_intention_review_thresholds_reject_non_increasing_or_out_of_range_values(
    first: int,
    second: int,
    strong: int,
) -> None:
    with pytest.raises(ValueError):
        MemoryIntentionReviewConfig(first, second, strong)
