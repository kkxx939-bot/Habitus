"""Conversation 消息角色、工具闭合、顺序和序列化的完整契约矩阵。"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

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
from pre.conversation.messages.model import conversation_datetime, require_sha256
from tests.helpers import BASE_TIME, closed_turn, message, tool_turn

UTC = timezone.utc


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (BASE_TIME, BASE_TIME),
        (BASE_TIME.isoformat(), BASE_TIME),
        (BASE_TIME.isoformat().replace("+00:00", "Z"), BASE_TIME),
        (datetime(2026, 7, 1, 16, 0, tzinfo=timezone(timedelta(hours=8))), BASE_TIME),
        ("2026-07-01T16:00:00+08:00", BASE_TIME),
    ],
)
def test_conversation_datetime_normalizes_aware_values_to_utc(value: datetime | str, expected: datetime) -> None:
    assert conversation_datetime(value, "time") == expected


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 7, 1, 8, 0),
        "2026-07-01T08:00:00",
        "not-a-date",
        "",
        None,
        1,
        [],
    ],
)
def test_conversation_datetime_rejects_naive_invalid_or_non_temporal_values(value: object) -> None:
    with pytest.raises(ConversationMessageSchemaError):
        conversation_datetime(value, "time")  # type: ignore[arg-type]


@pytest.mark.parametrize("digest", ["0" * 64, "a" * 64, "0123456789abcdef" * 4])
def test_require_sha256_accepts_lowercase_hex_digest(digest: str) -> None:
    assert require_sha256(digest, "digest") == digest


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "0" * 63 + "-",
        None,
        1,
        b"0" * 64,
    ],
)
def test_require_sha256_rejects_wrong_length_case_alphabet_or_type(digest: object) -> None:
    with pytest.raises(ConversationMessageSchemaError):
        require_sha256(digest, "digest")


def _text_message(role: ConversationMessageRole = ConversationMessageRole.PROMPT, **overrides: object) -> ConversationMessage:
    values: dict[str, object] = {
        "message_id": "message-1",
        "sequence": 0,
        "role": role,
        "occurred_at": BASE_TIME,
        "content": "content",
    }
    values.update(overrides)
    return ConversationMessage(**values)  # type: ignore[arg-type]


def _tool_call(**overrides: object) -> ConversationMessage:
    values: dict[str, object] = {
        "message_id": "call-message",
        "sequence": 0,
        "role": ConversationMessageRole.TOOL_CALL,
        "occurred_at": BASE_TIME,
        "content": {"path": "."},
        "tool_call_id": "call-1",
        "tool_name": "workspace.inspect",
    }
    values.update(overrides)
    return ConversationMessage(**values)  # type: ignore[arg-type]


def _tool_result(**overrides: object) -> ConversationMessage:
    values: dict[str, object] = {
        "message_id": "result-message",
        "sequence": 1,
        "role": ConversationMessageRole.TOOL_RESULT,
        "occurred_at": BASE_TIME + timedelta(seconds=1),
        "content": "result",
        "tool_call_id": "call-1",
        "tool_name": "workspace.inspect",
        "tool_status": ConversationToolResultStatus.COMPLETED,
        "content_mode": ConversationToolResultContentMode.INLINE,
    }
    values.update(overrides)
    return ConversationMessage(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("role", [ConversationMessageRole.PROMPT, ConversationMessageRole.COMPLETION, "prompt", "completion"])
@pytest.mark.parametrize("content", ["a", " ", "\n", "中文", "multi\nline"])
def test_prompt_and_completion_accept_exact_non_empty_text(role: object, content: str) -> None:
    assert _text_message(role, content=content).content == content  # type: ignore[arg-type]


@pytest.mark.parametrize("role", [ConversationMessageRole.PROMPT, ConversationMessageRole.COMPLETION])
@pytest.mark.parametrize("content", ["", None, 1, True, [], {}, ["text"]])
def test_prompt_and_completion_reject_empty_or_non_text_content(role: ConversationMessageRole, content: object) -> None:
    with pytest.raises(ConversationMessageSchemaError):
        _text_message(role, content=content)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tool_call_id", "call-1"),
        ("tool_name", "tool"),
        ("tool_status", ConversationToolResultStatus.COMPLETED),
        ("content_mode", ConversationToolResultContentMode.INLINE),
        ("source_ref", "file"),
        ("original_size_bytes", 1),
        ("original_sha256", "a" * 64),
    ],
)
@pytest.mark.parametrize("role", [ConversationMessageRole.PROMPT, ConversationMessageRole.COMPLETION])
def test_prompt_and_completion_reject_all_tool_result_metadata(
    role: ConversationMessageRole,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ConversationMessageSchemaError, match="cannot carry"):
        _text_message(role, **{field: value})


@pytest.mark.parametrize("message_id", ["message", "中文", "a/b", "name.md", "x" * 512])
def test_message_accepts_clean_non_path_identity(message_id: str) -> None:
    assert _text_message(message_id=message_id).message_id == message_id


@pytest.mark.parametrize("field", ["message_id", "tool_call_id", "tool_name", "source_ref"])
@pytest.mark.parametrize("value", ["", " leading", "trailing ", "line\nbreak", "tab\tvalue", None, 1])
def test_message_rejects_invalid_clean_identifiers(field: str, value: object) -> None:
    if field == "message_id":
        builder = _text_message
    elif field in {"tool_call_id", "tool_name"}:
        builder = _tool_call
    else:
        builder = _tool_result
    if field == "source_ref" and value is None:
        assert builder(source_ref=None).source_ref is None
        return
    with pytest.raises(ConversationMessageSchemaError):
        builder(**{field: value})


@pytest.mark.parametrize("sequence", [0, 1, 999_999_999])
def test_message_accepts_non_negative_sequence(sequence: int) -> None:
    assert _text_message(sequence=sequence).sequence == sequence


@pytest.mark.parametrize("sequence", [-1, True, False, 1.0, "1", None])
def test_message_rejects_negative_boolean_or_non_integer_sequence(sequence: object) -> None:
    with pytest.raises(ConversationMessageSchemaError):
        _text_message(sequence=sequence)


@pytest.mark.parametrize("role", ["assistant", "user", "tool", "", None, 1])
def test_message_rejects_unknown_role(role: object) -> None:
    with pytest.raises(ConversationMessageSchemaError, match="unsupported"):
        _text_message(role=role)


@pytest.mark.parametrize("content", [{}, {"a": 1}, {"nested": [1, True, None]}, "{}", "raw arguments"])
def test_tool_call_accepts_object_or_exact_argument_text(content: object) -> None:
    assert _tool_call(content=content).content is not None


@pytest.mark.parametrize("content", ["", [], [1], 1, True, None])
def test_tool_call_rejects_empty_string_or_non_object_arguments(content: object) -> None:
    with pytest.raises(ConversationMessageSchemaError):
        _tool_call(content=content)


@pytest.mark.parametrize("missing", ["tool_call_id", "tool_name"])
def test_tool_call_requires_call_id_and_exact_tool_name(missing: str) -> None:
    with pytest.raises(ConversationMessageSchemaError, match="requires"):
        _tool_call(**{missing: None})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tool_status", ConversationToolResultStatus.COMPLETED),
        ("content_mode", ConversationToolResultContentMode.INLINE),
        ("source_ref", "file"),
        ("original_size_bytes", 1),
        ("original_sha256", "a" * 64),
    ],
)
def test_tool_call_rejects_result_only_fields(field: str, value: object) -> None:
    with pytest.raises(ConversationMessageSchemaError, match="cannot carry"):
        _tool_call(**{field: value})


@pytest.mark.parametrize("status", list(ConversationToolResultStatus) + [item.value for item in ConversationToolResultStatus])
@pytest.mark.parametrize("mode", list(ConversationToolResultContentMode) + [item.value for item in ConversationToolResultContentMode])
def test_tool_result_accepts_every_declared_terminal_status_and_content_mode(status: object, mode: object) -> None:
    overrides: dict[str, object] = {"tool_status": status, "content_mode": mode}
    if mode in {ConversationToolResultContentMode.OMITTED, "omitted"}:
        overrides["content"] = "payload omitted"
    result = _tool_result(**overrides)
    assert isinstance(result.tool_status, ConversationToolResultStatus)
    assert isinstance(result.content_mode, ConversationToolResultContentMode)


@pytest.mark.parametrize("missing", ["tool_call_id", "tool_name", "tool_status", "content_mode"])
def test_tool_result_requires_call_identity_status_and_content_mode(missing: str) -> None:
    with pytest.raises(ConversationMessageSchemaError, match="requires"):
        _tool_result(**{missing: None})


@pytest.mark.parametrize("status", ["success", "failed", "running", "", 1, True])
def test_tool_result_rejects_unknown_terminal_status(status: object) -> None:
    with pytest.raises(ConversationMessageSchemaError, match="unsupported"):
        _tool_result(tool_status=status)


@pytest.mark.parametrize("mode", ["full", "binary", "reference", "", 1, True])
def test_tool_result_rejects_unknown_content_mode(mode: object) -> None:
    with pytest.raises(ConversationMessageSchemaError, match="unsupported"):
        _tool_result(content_mode=mode)


@pytest.mark.parametrize("content", ["summary", " summary ", "多行\n摘要"])
def test_summarized_tool_result_accepts_non_empty_text(content: str) -> None:
    assert _tool_result(content_mode=ConversationToolResultContentMode.SUMMARIZED, content=content).content == content


@pytest.mark.parametrize("content", ["", " ", None, {}, [], 1])
def test_summarized_tool_result_rejects_empty_or_non_text_content(content: object) -> None:
    with pytest.raises(ConversationMessageSchemaError, match="summarized"):
        _tool_result(content_mode=ConversationToolResultContentMode.SUMMARIZED, content=content)


@pytest.mark.parametrize(
    "metadata",
    [
        {"content": "payload omitted"},
        {"content": "", "source_ref": "https://example.test/result"},
        {"content": None, "original_size_bytes": 0},
        {"content": {}, "original_sha256": "a" * 64},
        {"content": "", "source_ref": "file", "original_size_bytes": 10, "original_sha256": "b" * 64},
    ],
)
def test_omitted_tool_result_requires_description_or_original_metadata(metadata: dict[str, object]) -> None:
    result = _tool_result(content_mode=ConversationToolResultContentMode.OMITTED, **metadata)
    assert result.content_mode is ConversationToolResultContentMode.OMITTED


@pytest.mark.parametrize("content", ["", " ", None, {}, []])
def test_omitted_tool_result_rejects_missing_description_and_metadata(content: object) -> None:
    with pytest.raises(ConversationMessageSchemaError, match="requires a description"):
        _tool_result(content_mode=ConversationToolResultContentMode.OMITTED, content=content)


@pytest.mark.parametrize("size", [0, 1, 10**12])
def test_tool_result_accepts_non_negative_original_size(size: int) -> None:
    assert _tool_result(original_size_bytes=size).original_size_bytes == size


@pytest.mark.parametrize("size", [-1, True, False, 1.0, "1", None])
def test_tool_result_rejects_invalid_original_size_when_present(size: object) -> None:
    if size is None:
        assert _tool_result(original_size_bytes=None).original_size_bytes is None
        return
    with pytest.raises(ConversationMessageSchemaError):
        _tool_result(original_size_bytes=size)


@pytest.mark.parametrize("content", [math.nan, math.inf, -math.inf, {"value": math.nan}, object()])
def test_message_content_must_be_canonical_json(content: object) -> None:
    with pytest.raises(ConversationMessageSchemaError, match="canonical JSON"):
        _tool_result(content=content)


def test_message_content_canonicalizes_set_to_deterministic_immutable_sequence() -> None:
    assert _tool_result(content={3, 1, 2}).content == (1, 2, 3)


@pytest.mark.parametrize(
    "messages",
    [
        closed_turn(),
        tool_turn(),
        tuple(message(index + 10, role, content) for index, (role, content) in enumerate([
            (ConversationMessageRole.PROMPT, "p"),
            (ConversationMessageRole.COMPLETION, "c"),
        ])),
    ],
)
def test_message_collection_accepts_contiguous_unique_chronological_sequences(
    messages: tuple[ConversationMessage, ...],
) -> None:
    batch = ConversationBatch("conversation-1", messages)
    assert batch.start_sequence == messages[0].sequence
    assert batch.end_sequence == messages[-1].sequence
    assert batch.started_at == messages[0].occurred_at
    assert batch.ended_at == messages[-1].occurred_at


@pytest.mark.parametrize("messages", [None, {}, "messages", 1, (), [], (object(),), ["message"]])
def test_message_collection_requires_non_empty_list_or_tuple_of_messages(messages: object) -> None:
    with pytest.raises(ConversationMessageSchemaError):
        ConversationBatch("conversation-1", messages)  # type: ignore[arg-type]


def test_message_collection_rejects_duplicate_message_ids() -> None:
    first, second = closed_turn()
    with pytest.raises(ConversationMessageSchemaError, match="IDs must be unique"):
        ConversationBatch("conversation-1", (first, replace(second, message_id=first.message_id)))


@pytest.mark.parametrize("sequences", [(0, 2), (1, 0), (0, 0), (2, 4, 3)])
def test_message_collection_rejects_gaps_reordering_and_duplicate_sequences(sequences: tuple[int, ...]) -> None:
    messages = tuple(
        _text_message(
            ConversationMessageRole.PROMPT if index % 2 == 0 else ConversationMessageRole.COMPLETION,
            message_id=f"message-{index}",
            sequence=sequence,
            occurred_at=BASE_TIME + timedelta(seconds=index),
        )
        for index, sequence in enumerate(sequences)
    )
    with pytest.raises(ConversationMessageSchemaError, match="contiguous"):
        ConversationBatch("conversation-1", messages)


def test_message_collection_allows_equal_timestamps_but_rejects_reverse_time() -> None:
    first, second = closed_turn()
    assert ConversationBatch("conversation-1", (first, replace(second, occurred_at=first.occurred_at)))
    with pytest.raises(ConversationMessageSchemaError, match="chronological"):
        ConversationBatch("conversation-1", (first, replace(second, occurred_at=first.occurred_at - timedelta(seconds=1))))


def test_message_collection_rejects_duplicate_tool_call_ids() -> None:
    first = _tool_call(sequence=0, message_id="call-1")
    second = _tool_call(sequence=1, message_id="call-2")
    with pytest.raises(ConversationMessageSchemaError, match="tool_call IDs"):
        ConversationBatch("conversation-1", (first, second))


def test_message_collection_rejects_duplicate_terminal_results() -> None:
    first = _tool_result(sequence=0, message_id="result-1")
    second = _tool_result(sequence=1, message_id="result-2")
    with pytest.raises(ConversationMessageSchemaError, match="only one terminal"):
        ConversationBatch("conversation-1", (first, second))


def test_message_collection_rejects_result_before_matching_call() -> None:
    result = _tool_result(sequence=0, occurred_at=BASE_TIME, message_id="result")
    call = _tool_call(sequence=1, occurred_at=BASE_TIME + timedelta(seconds=1), message_id="call")
    with pytest.raises(ConversationMessageSchemaError, match="must follow"):
        ConversationBatch("conversation-1", (result, call))


@pytest.mark.parametrize("result_name", ["Workspace.Inspect", "workspace-inspect", "检查工作区", "workspace.inspect "])
def test_message_collection_requires_result_tool_name_byte_for_byte_match(result_name: str) -> None:
    call = _tool_call(sequence=0, message_id="call")
    if result_name.endswith(" "):
        with pytest.raises(ConversationMessageSchemaError):
            _tool_result(sequence=1, tool_name=result_name)
        return
    result = _tool_result(sequence=1, tool_name=result_name)
    with pytest.raises(ConversationMessageSchemaError, match="exactly match"):
        ConversationBatch("conversation-1", (call, result))


def test_batch_allows_result_for_call_from_previous_append() -> None:
    result = _tool_result(sequence=10, tool_call_id="previous-call")
    assert ConversationBatch("conversation-1", (result,)).messages == (result,)


@pytest.mark.parametrize(
    "identifier",
    ["conversation-1", "中文会话", "a.b-c_1", "000000000000-000000000001"],
)
def test_batch_and_segment_accept_safe_path_identifiers(identifier: str) -> None:
    assert ConversationBatch(identifier, closed_turn()).conversation_id == identifier
    assert ConversationSegment(identifier, identifier, closed_turn()).segment_id == identifier


@pytest.mark.parametrize(
    "identifier",
    ["", ".", "..", "../escape", "a/b", "a\\b", " leading", "trailing ", "line\nbreak", None, 1],
)
@pytest.mark.parametrize("target", ["batch", "segment_conversation", "segment_id"])
def test_batch_and_segment_reject_unsafe_path_identifiers(identifier: object, target: str) -> None:
    with pytest.raises(ConversationMessageSchemaError):
        if target == "batch":
            ConversationBatch(identifier, closed_turn())  # type: ignore[arg-type]
        elif target == "segment_conversation":
            ConversationSegment(identifier, "segment-1", closed_turn())  # type: ignore[arg-type]
        else:
            ConversationSegment("conversation-1", identifier, closed_turn())  # type: ignore[arg-type]


@pytest.mark.parametrize("factory", [ConversationBatch, ConversationSegment])
def test_batch_and_segment_round_trip_are_digest_stable(factory: type[ConversationBatch] | type[ConversationSegment]) -> None:
    value = factory("conversation-1", closed_turn()) if factory is ConversationBatch else factory("conversation-1", "segment-1", closed_turn())
    restored = factory.from_dict(value.to_dict())
    assert restored == value
    assert restored.digest == value.digest


@pytest.mark.parametrize("factory", [ConversationBatch, ConversationSegment])
def test_batch_and_segment_from_dict_reject_unknown_fields(factory: type[ConversationBatch] | type[ConversationSegment]) -> None:
    value = factory("conversation-1", closed_turn()) if factory is ConversationBatch else factory("conversation-1", "segment-1", closed_turn())
    with pytest.raises(ConversationMessageSchemaError, match="unknown"):
        factory.from_dict({**value.to_dict(), "owner": "unexpected"})


@pytest.mark.parametrize("factory", [ConversationBatch, ConversationSegment])
def test_batch_and_segment_from_dict_reject_wrong_schema(factory: type[ConversationBatch] | type[ConversationSegment]) -> None:
    value = factory("conversation-1", closed_turn()) if factory is ConversationBatch else factory("conversation-1", "segment-1", closed_turn())
    with pytest.raises(ConversationMessageSchemaError, match="unsupported"):
        factory.from_dict({**value.to_dict(), "schema_version": "v0"})


@pytest.mark.parametrize("factory", [ConversationBatch, ConversationSegment])
@pytest.mark.parametrize("messages", [None, {}, "messages", ["not-object"], [1]])
def test_batch_and_segment_from_dict_reject_invalid_message_container(
    factory: type[ConversationBatch] | type[ConversationSegment],
    messages: object,
) -> None:
    value = factory("conversation-1", closed_turn()) if factory is ConversationBatch else factory("conversation-1", "segment-1", closed_turn())
    with pytest.raises(ConversationMessageSchemaError):
        factory.from_dict({**value.to_dict(), "messages": messages})


@pytest.mark.parametrize("factory", [ConversationBatch, ConversationSegment])
def test_batch_and_segment_from_dict_reject_each_missing_required_field(
    factory: type[ConversationBatch] | type[ConversationSegment],
) -> None:
    value = factory("conversation-1", closed_turn()) if factory is ConversationBatch else factory("conversation-1", "segment-1", closed_turn())
    payload = value.to_dict()
    required = ["conversation_id", "messages"]
    if factory is ConversationSegment:
        required.append("segment_id")
    for field in required:
        invalid = dict(payload)
        invalid.pop(field)
        with pytest.raises(ConversationMessageSchemaError, match="missing"):
            factory.from_dict(invalid)


def test_message_round_trip_preserves_every_optional_tool_result_field() -> None:
    value = _tool_result(
        content_mode=ConversationToolResultContentMode.OMITTED,
        content="payload omitted",
        source_ref="https://example.test/result",
        original_size_bytes=999,
        original_sha256="c" * 64,
    )
    assert ConversationMessage.from_dict(value.to_dict()) == value


def test_message_from_dict_rejects_each_missing_required_field() -> None:
    payload = _text_message().to_dict()
    for field in ("message_id", "sequence", "role", "occurred_at", "content"):
        invalid = dict(payload)
        invalid.pop(field)
        with pytest.raises(ConversationMessageSchemaError, match="missing"):
            ConversationMessage.from_dict(invalid)


@pytest.mark.parametrize("field", ["role", "tool_status", "content_mode"])
def test_message_from_dict_rejects_unknown_enum_values(field: str) -> None:
    payload = (_tool_result() if field != "role" else _text_message()).to_dict()
    payload[field] = "unknown"
    with pytest.raises(ConversationMessageSchemaError, match="unsupported"):
        ConversationMessage.from_dict(payload)
