"""落盘记录的字段名单钉死（W16 后续）：解码器按 ``dataclasses.fields(cls)`` 推出读盘接受的键集。

W16 为压文件尺寸把这几处手写的字段集合换成了从 dataclass 推导（与原手写集合逐个相等）。推导写法的风险是：
给值对象加一个字段，读盘接受范围就跟着悄悄放宽、旧文件里多一个键也照收。这里把每份落盘格式的字段名单写死——
加字段时这条测试变红，逼着同时确认落盘格式要不要跟着变。
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from habitus.memory.conversation.messages import ConversationJournalState
from habitus.memory.workflow.jobs.model import MemoryJob, MemoryJobAbandonment
from habitus.pre.conversation.messages.model import ConversationMessage
from habitus.pre.conversation.summaries.model import (
    ConversationRangeSummary,
    ConversationSegmentSummary,
    ConversationSummaryContent,
)

_SUMMARY_TEXT = ("overview", "chronology", "corrections", "ending_state", "open_threads")

FIELD_SETS: dict[type, tuple[str, ...]] = {
    MemoryJob: (
        "memory_sequence",
        "conversation_id",
        "started_on",
        "segment_id",
        "source_segment_digest",
        "status",
        "attempts",
        "claim_id",
        "claim_generation",
        "worker_id",
        "lease_expires_at",
        "next_attempt_at",
        "last_error",
        "created_at",
        "updated_at",
        "model_waits",
        "cards_written",
        "changed_uris",
        "deleted_uris",
    ),
    MemoryJobAbandonment: (
        "memory_sequence",
        "conversation_id",
        "started_on",
        "segment_id",
        "source_segment_digest",
        "attempts",
        "last_error",
        "reason",
        "abandoned_at",
    ),
    ConversationJournalState: (
        "conversation_id",
        "archived_through_sequence",
        "latest_segment_id",
        "latest_segment_digest",
    ),
    ConversationMessage: (
        "message_id",
        "sequence",
        "role",
        "occurred_at",
        "content",
        "tool_call_id",
        "tool_name",
        "tool_status",
        "content_mode",
        "source_ref",
        "original_size_bytes",
        "original_sha256",
        "logical_message_id",
        "logical_part_index",
        "logical_part_count",
        "mid_turn",
    ),
    ConversationSummaryContent: _SUMMARY_TEXT,
    ConversationSegmentSummary: (
        "conversation_id",
        "segment_id",
        "source_message_digest",
        "start_sequence",
        "end_sequence",
        "started_at",
        "ended_at",
        "generated_at",
        *_SUMMARY_TEXT,
        "starts_mid_turn",
        "ends_mid_turn",
    ),
    ConversationRangeSummary: (
        "conversation_id",
        "range_id",
        "stage",
        "source_refs",
        "start_sequence",
        "end_sequence",
        "started_at",
        "ended_at",
        "generated_at",
        *_SUMMARY_TEXT,
        "starts_mid_turn",
        "ends_mid_turn",
    ),
}


@pytest.mark.parametrize("model", list(FIELD_SETS), ids=lambda model: model.__name__)
def test_disk_record_fields_are_pinned_in_order(model: type) -> None:
    """字段名与顺序都写死：加字段、删字段、调顺序都要先改这里，确认落盘格式随之变化是有意的。"""

    assert tuple(field.name for field in fields(model)) == FIELD_SETS[model]
