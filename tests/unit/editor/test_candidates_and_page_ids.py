"""六类候选、会话证据约束和一次解析周期 page_id 测试。"""


import pytest

from habitus.infrastructure.editor.snapshot import SnapshotBatch
from habitus.memory.editor import (
    MemoryCandidateBatch,
    MemoryCandidateError,
    MemoryPageIdError,
    MemoryPageIdMap,
)
from habitus.memory.model import MemoryKind
from habitus.pre.conversation import (
    ConversationMessageRole,
    ConversationSegment,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from tests.helpers import document, memory_fields, message, segment, snapshot_batch, tool_turn


def empty_output() -> dict[str, object]:
    return {
        "profile": [],
        "preferences": [],
        "entities": [],
        "tools": [],
        "events": [],
        "intentions": [],
        "identity_proposals": [],
        "relations": [],
    }


def candidate_item(kind: MemoryKind, page_id: int, **controls: object) -> dict[str, object]:
    item = {"page_id": page_id, **memory_fields(kind), **controls}
    if kind is MemoryKind.EVENT:
        item["event_date"] = "2026-07-01"
    return item


def test_candidate_schema_requires_every_partition_and_rejects_unknown_or_coerced_fields() -> None:
    output = empty_output()
    output["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 100)]
    batch = MemoryCandidateBatch.model_validate(output)
    assert batch.preferences[0].page_id == 100
    assert batch.to_dict() == output
    assert MemoryCandidateBatch.model_json_schema()["additionalProperties"] is False

    missing = dict(output)
    missing.pop("relations")
    with pytest.raises(MemoryCandidateError, match="missing fields"):
        MemoryCandidateBatch.model_validate(missing)
    with pytest.raises(MemoryCandidateError, match="unknown fields"):
        MemoryCandidateBatch.model_validate({**output, "confidence": 0.9})
    invalid_page = empty_output()
    invalid_page["preferences"] = [{**candidate_item(MemoryKind.PREFERENCE, 100), "page_id": "100"}]
    with pytest.raises(MemoryCandidateError, match="page_id"):
        MemoryCandidateBatch.model_validate(invalid_page)


def test_all_six_memory_kinds_parse_with_their_own_schema_and_intention_confirmation() -> None:
    output = empty_output()
    mapping = {
        "profile": MemoryKind.PROFILE,
        "preferences": MemoryKind.PREFERENCE,
        "entities": MemoryKind.ENTITY,
        "tools": MemoryKind.TOOL,
        "events": MemoryKind.EVENT,
        "intentions": MemoryKind.INTENTION,
    }
    for offset, (field, kind) in enumerate(mapping.items()):
        controls = {"confirmed": True} if kind is MemoryKind.INTENTION else {}
        output[field] = [candidate_item(kind, 100 + offset, **controls)]
    batch = MemoryCandidateBatch.model_validate(output)
    assert tuple(item.kind for item in batch.iter_candidates()) == tuple(mapping.values())

    missing_confirmation = empty_output()
    missing_confirmation["intentions"] = [candidate_item(MemoryKind.INTENTION, 100)]
    with pytest.raises(MemoryCandidateError, match="confirmed"):
        MemoryCandidateBatch.model_validate(missing_confirmation)


def test_page_ids_are_deterministic_and_existing_ids_cannot_be_redirected() -> None:
    profile = document(MemoryKind.PROFILE)
    preference = document(MemoryKind.PREFERENCE)
    snapshots = snapshot_batch(profile, preference)
    page_ids = MemoryPageIdMap.from_snapshots(snapshots)
    assert page_ids.existing_items() == tuple(enumerate((item.identity for item in snapshots.snapshots), start=1))

    old_id, old_uri = page_ids.existing_items()[0]
    with pytest.raises(MemoryPageIdError, match="redirected"):
        page_ids.register_resolved(page_ids.existing_items()[1][1], old_id)
    with pytest.raises(MemoryPageIdError, match="at least 100"):
        page_ids.register_new(old_uri, 99)
    page_ids.register_new(old_uri, 100)
    assert page_ids.page_ids_for(old_uri) == frozenset({old_id, 100})


def test_tool_candidate_requires_exact_successful_tool_name_from_conversation() -> None:
    output = empty_output()
    output["tools"] = [candidate_item(MemoryKind.TOOL, 100)]
    batch = MemoryCandidateBatch.model_validate(output)
    empty_old = SnapshotBatch((), 0)

    batch.validate_context(segment(messages=tool_turn()), empty_old, MemoryPageIdMap())
    wrong_name = empty_output()
    wrong_name["tools"] = [{**candidate_item(MemoryKind.TOOL, 100), "tool_name": "translated.inspect"}]
    with pytest.raises(MemoryCandidateError, match="successful tool_call"):
        MemoryCandidateBatch.model_validate(wrong_name).validate_context(
            segment(messages=tool_turn()), empty_old, MemoryPageIdMap()
        )


def test_tool_failure_recovery_requires_failure_before_later_success() -> None:
    messages = (
        message(0, ConversationMessageRole.PROMPT, "检查"),
        message(1, ConversationMessageRole.TOOL_CALL, {}, tool_call_id="failed", tool_name="workspace.inspect"),
        message(
            2,
            ConversationMessageRole.TOOL_RESULT,
            "失败",
            tool_call_id="failed",
            tool_name="workspace.inspect",
            tool_status=ConversationToolResultStatus.ERROR,
            content_mode=ConversationToolResultContentMode.INLINE,
        ),
        message(3, ConversationMessageRole.TOOL_CALL, {}, tool_call_id="recovered", tool_name="workspace.inspect"),
        message(
            4,
            ConversationMessageRole.TOOL_RESULT,
            "成功",
            tool_call_id="recovered",
            tool_name="workspace.inspect",
            tool_status=ConversationToolResultStatus.COMPLETED,
            content_mode=ConversationToolResultContentMode.INLINE,
        ),
        message(5, ConversationMessageRole.COMPLETION, "已恢复"),
    )
    output = empty_output()
    output["tools"] = [
        {
            "page_id": 100,
            "tool_name": "workspace.inspect",
            "failure_recovery": "失败后修正参数并重试。",
        }
    ]
    MemoryCandidateBatch.model_validate(output).validate_context(
        ConversationSegment("conversation-1", "segment-1", messages),
        SnapshotBatch((), 0),
        MemoryPageIdMap(),
    )

    no_failure = segment(messages=tool_turn())
    with pytest.raises(MemoryCandidateError, match="failure followed"):
        MemoryCandidateBatch.model_validate(output).validate_context(
            no_failure, SnapshotBatch((), 0), MemoryPageIdMap()
        )


def test_completed_intention_must_reuse_existing_page_and_be_explicitly_confirmed() -> None:
    old = document(MemoryKind.INTENTION)
    old_batch = snapshot_batch(old)
    page_ids = MemoryPageIdMap.from_snapshots(old_batch)
    output = empty_output()
    output["intentions"] = [
        {
            "page_id": 1,
            "intent_name": old.fields["intent_name"],
            "status": "completed",
            "confirmed": True,
        }
    ]
    MemoryCandidateBatch.model_validate(output).validate_context(segment(), old_batch, page_ids)

    output["intentions"][0]["confirmed"] = False
    with pytest.raises(MemoryCandidateError, match="explicit confirmation"):
        MemoryCandidateBatch.model_validate(output).validate_context(segment(), old_batch, page_ids)


def test_event_candidate_uses_schema_date_and_rejects_only_future_date_invariant() -> None:
    output = empty_output()
    output["events"] = [candidate_item(MemoryKind.EVENT, 100)]
    batch = MemoryCandidateBatch.model_validate(output)
    batch.validate_context(segment(), SnapshotBatch((), 0), MemoryPageIdMap())

    output["events"][0]["event_date"] = "2099-01-01"
    with pytest.raises(MemoryCandidateError, match="future event date"):
        MemoryCandidateBatch.model_validate(output).validate_context(
            segment(),
            SnapshotBatch((), 0),
            MemoryPageIdMap(),
        )
