"""把 Summary 模型输出限制为共享的字段操作，并重建严格领域内容。"""

from __future__ import annotations

from memory.compaction.field_ops import (
    SemanticFieldMergePolicy,
    SemanticFieldOperationBatch,
    merge_semantic_fields,
)
from pre.conversation import ConversationSummaryContent

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
        chronology=_items(merged["chronology"]),
        corrections=_items(merged["corrections"]),
        ending_state=merged["ending_state"],
        open_threads=_items(merged["open_threads"]),
    )


def _items(value: str) -> tuple[str, ...]:
    items: list[str] = []
    for line in value.splitlines():
        item = line.strip()
        if item.startswith(("- ", "* ")):
            item = item[2:].strip()
        if item and item not in items:
            items.append(item)
    return tuple(items)


__all__ = ["summary_content_from_operations"]
