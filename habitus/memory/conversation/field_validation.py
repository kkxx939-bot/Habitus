"""把 Summary 模型输出限制为共享的字段操作，并重建严格领域内容。"""

from __future__ import annotations

from habitus.memory.field_ops import (
    SemanticFieldMergePolicy,
    SemanticFieldOperationBatch,
    content_items,
    merge_semantic_fields,
)
from habitus.pre.conversation import ConversationSummaryContent

_SUMMARY_FIELDS = {
    "overview": "",
    "chronology": "",
    "corrections": "",
    "ending_state": "",
    "open_threads": "",
}
_SUMMARY_POLICIES = (
    SemanticFieldMergePolicy("overview", allow_append=False, max_chars=16_000),
    SemanticFieldMergePolicy("chronology", append_only=True, max_chars=256_000),
    SemanticFieldMergePolicy("corrections", append_only=True, max_chars=256_000),
    SemanticFieldMergePolicy("ending_state", allow_append=False, max_chars=4_000),
    SemanticFieldMergePolicy("open_threads", append_only=True, max_chars=256_000),
)


def summary_content_from_operations(
    operations: SemanticFieldOperationBatch,
) -> ConversationSummaryContent:
    """服务端合并白名单字段；缺失操作按 KEEP，最终仍由 Summary Schema 校验。"""

    merged = merge_semantic_fields(_SUMMARY_FIELDS, operations, _SUMMARY_POLICIES)
    return ConversationSummaryContent(
        overview=merged["overview"],
        chronology=content_items(merged["chronology"]),
        corrections=content_items(merged["corrections"]),
        ending_state=merged["ending_state"],
        open_threads=content_items(merged["open_threads"]),
    )


__all__ = ["summary_content_from_operations"]
