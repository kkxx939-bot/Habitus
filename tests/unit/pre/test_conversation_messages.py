"""Conversation 原始消息、工具配对和不可变片段测试。"""

from datetime import datetime

import pytest

from pre.conversation.messages import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationMessageSchemaError,
    ConversationSegment,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from tests.helpers import BASE_TIME, closed_turn, message, tool_turn


def test_prompt_completion_batch_round_trip_preserves_roles_time_and_digest() -> None:
    batch = ConversationBatch("conversation-1", closed_turn())
    restored = ConversationBatch.from_dict(batch.to_dict())

    assert restored == batch
    assert restored.digest == batch.digest
    assert restored.start_sequence == 0
    assert restored.end_sequence == 1


def test_tool_turn_requires_exact_call_name_and_single_terminal_result() -> None:
    batch = ConversationBatch("conversation-1", tool_turn())
    assert [item.role for item in batch.messages] == [
        ConversationMessageRole.PROMPT,
        ConversationMessageRole.TOOL_CALL,
        ConversationMessageRole.TOOL_RESULT,
        ConversationMessageRole.COMPLETION,
    ]

    mismatched = list(tool_turn())
    mismatched[2] = message(
        2,
        ConversationMessageRole.TOOL_RESULT,
        "结果",
        tool_call_id="call-1",
        tool_name="translated.tool",
        tool_status=ConversationToolResultStatus.COMPLETED,
        content_mode=ConversationToolResultContentMode.INLINE,
    )
    with pytest.raises(ConversationMessageSchemaError, match="exactly match"):
        ConversationBatch("conversation-1", tuple(mismatched))

    duplicate_result = (*tool_turn()[:-1], tool_turn()[2])
    with pytest.raises(ConversationMessageSchemaError, match="only one terminal result|sequence"):
        ConversationBatch("conversation-1", duplicate_result)


@pytest.mark.parametrize(
    ("role", "content", "kwargs"),
    [
        (ConversationMessageRole.PROMPT, "", {}),
        (ConversationMessageRole.COMPLETION, {"not": "text"}, {}),
        (ConversationMessageRole.TOOL_CALL, {}, {}),
        (
            ConversationMessageRole.TOOL_RESULT,
            "result",
            {"tool_call_id": "call-1", "tool_name": "tool"},
        ),
    ],
)
def test_message_rejects_role_specific_missing_or_wrong_fields(
    role: ConversationMessageRole,
    content: object,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ConversationMessageSchemaError):
        ConversationMessage("message", 0, role, BASE_TIME, content, **kwargs)


def test_summarized_and_omitted_tool_results_require_bounded_semantics() -> None:
    with pytest.raises(ConversationMessageSchemaError, match="summarized"):
        ConversationMessage(
            "result",
            0,
            ConversationMessageRole.TOOL_RESULT,
            BASE_TIME,
            " ",
            tool_call_id="call-1",
            tool_name="download",
            tool_status=ConversationToolResultStatus.COMPLETED,
            content_mode=ConversationToolResultContentMode.SUMMARIZED,
        )

    omitted = ConversationMessage(
        "result",
        0,
        ConversationMessageRole.TOOL_RESULT,
        BASE_TIME,
        "媒体文件未保存",
        tool_call_id="call-1",
        tool_name="download",
        tool_status=ConversationToolResultStatus.COMPLETED,
        content_mode=ConversationToolResultContentMode.OMITTED,
        original_size_bytes=10_000_000,
        original_sha256="a" * 64,
    )
    assert omitted.content_mode is ConversationToolResultContentMode.OMITTED


@pytest.mark.parametrize(
    "messages",
    [
        (message(0, ConversationMessageRole.PROMPT, "a"), message(2, ConversationMessageRole.COMPLETION, "b")),
        (message(0, ConversationMessageRole.PROMPT, "a"), message(0, ConversationMessageRole.COMPLETION, "b")),
    ],
)
def test_batch_rejects_non_contiguous_or_duplicate_sequences(messages: tuple[ConversationMessage, ...]) -> None:
    with pytest.raises(ConversationMessageSchemaError):
        ConversationBatch("conversation-1", messages)


def test_batch_rejects_non_chronological_messages() -> None:
    first = ConversationMessage("first", 0, ConversationMessageRole.PROMPT, BASE_TIME, "a")
    second = ConversationMessage(
        "second",
        1,
        ConversationMessageRole.COMPLETION,
        BASE_TIME.replace(year=2025),
        "b",
    )
    with pytest.raises(ConversationMessageSchemaError, match="chronological"):
        ConversationBatch("conversation-1", (first, second))


def test_tool_result_without_local_call_is_allowed_for_contiguous_batch_but_segment_round_trip_is_strict() -> None:
    result = message(
        10,
        ConversationMessageRole.TOOL_RESULT,
        "已在之前批次调用",
        tool_call_id="call-before-batch",
        tool_name="workspace.inspect",
        tool_status=ConversationToolResultStatus.COMPLETED,
        content_mode=ConversationToolResultContentMode.INLINE,
    )
    batch = ConversationBatch("conversation-1", (result,))
    assert ConversationBatch.from_dict(batch.to_dict()) == batch


def test_segment_is_immutable_source_with_safe_identity_and_exact_digest() -> None:
    segment = ConversationSegment("conversation-1", "segment-1", tool_turn())
    restored = ConversationSegment.from_dict(segment.to_dict())

    assert restored == segment
    assert restored.message_count == 4
    assert restored.digest == segment.digest

    with pytest.raises(ConversationMessageSchemaError):
        ConversationSegment("../escape", "segment-1", closed_turn())


def test_from_dict_rejects_unknown_fields_schema_drift_and_naive_time() -> None:
    payload = message(0, ConversationMessageRole.PROMPT, "hello").to_dict()
    with pytest.raises(ConversationMessageSchemaError, match="unknown"):
        ConversationMessage.from_dict({**payload, "owner": "unexpected"})
    with pytest.raises(ConversationMessageSchemaError, match="unsupported"):
        ConversationMessage.from_dict({**payload, "schema_version": "v0"})
    with pytest.raises(ConversationMessageSchemaError, match="timezone"):
        ConversationMessage("message", 0, ConversationMessageRole.PROMPT, datetime(2026, 1, 1), "hello")

