from __future__ import annotations

import pytest

from habitus.memory.compaction import (
    SemanticFieldMergePolicy,
    SemanticFieldOperationBatch,
    SemanticFieldOperationError,
    merge_semantic_fields,
)


def batch(*operations: dict[str, object]) -> SemanticFieldOperationBatch:
    return SemanticFieldOperationBatch.model_validate({"operations": list(operations)})


def test_keep_update_and_append_are_merged_by_the_server() -> None:
    result = merge_semantic_fields(
        {"overview": "旧概览", "chronology": "- 第一项", "ending_state": "旧状态"},
        batch(
            {"field": "overview", "operation": "update", "content": "压缩后的概览"},
            {"field": "chronology", "operation": "append", "items": ["第二项", "第一项"]},
            {"field": "ending_state", "operation": "keep"},
        ),
        (
            SemanticFieldMergePolicy("overview", allow_append=False),
            SemanticFieldMergePolicy("chronology"),
            SemanticFieldMergePolicy("ending_state", allow_append=False),
        ),
    )
    assert result == {
        "overview": "压缩后的概览",
        "chronology": "- 第一项\n- 第二项",
        "ending_state": "旧状态",
    }


def test_append_only_update_is_demoted_to_deduplicated_append() -> None:
    result = merge_semantic_fields(
        {"corrections": "- 保留纠正"},
        batch(
            {
                "field": "corrections",
                "operation": "update",
                "content": "- 保留纠正\n- 新纠正",
            }
        ),
        (SemanticFieldMergePolicy("corrections", append_only=True),),
    )
    assert result["corrections"] == "- 保留纠正\n- 新纠正"


def test_missing_operation_keeps_source_and_unknown_field_is_rejected() -> None:
    assert merge_semantic_fields(
        {"overview": "原内容", "ending_state": "保持"},
        batch({"field": "overview", "operation": "keep"}),
        (
            SemanticFieldMergePolicy("overview"),
            SemanticFieldMergePolicy("ending_state"),
        ),
    ) == {"overview": "原内容", "ending_state": "保持"}
    with pytest.raises(SemanticFieldOperationError, match="unknown fields"):
        merge_semantic_fields(
            {"overview": "原内容"},
            batch({"field": "system_revision", "operation": "keep"}),
            (SemanticFieldMergePolicy("overview"),),
        )


@pytest.mark.parametrize(
    "operation",
    [
        {"field": "overview", "operation": "keep", "content": "非法"},
        {"field": "overview", "operation": "update"},
        {"field": "overview", "operation": "append", "items": []},
    ],
)
def test_operation_shape_is_strict(operation: dict[str, object]) -> None:
    with pytest.raises(SemanticFieldOperationError):
        batch(operation)
