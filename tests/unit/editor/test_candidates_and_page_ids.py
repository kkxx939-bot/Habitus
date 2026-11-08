"""六类候选、会话证据约束和一次解析周期 page_id 测试。"""

from datetime import UTC

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

    batch.validate_context(segment(messages=tool_turn()), empty_old, MemoryPageIdMap(), zone=UTC)
    wrong_name = empty_output()
    wrong_name["tools"] = [{**candidate_item(MemoryKind.TOOL, 100), "tool_name": "translated.inspect"}]
    with pytest.raises(MemoryCandidateError, match="successful tool_call"):
        MemoryCandidateBatch.model_validate(wrong_name).validate_context(
            segment(messages=tool_turn()), empty_old, MemoryPageIdMap()
        , zone=UTC)


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
        MemoryPageIdMap(), zone=UTC,
    )

    no_failure = segment(messages=tool_turn())
    with pytest.raises(MemoryCandidateError, match="failure followed"):
        MemoryCandidateBatch.model_validate(output).validate_context(
            no_failure, SnapshotBatch((), 0), MemoryPageIdMap()
        , zone=UTC)


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
    MemoryCandidateBatch.model_validate(output).validate_context(segment(), old_batch, page_ids, zone=UTC)

    output["intentions"][0]["confirmed"] = False
    with pytest.raises(MemoryCandidateError, match="explicit confirmation"):
        MemoryCandidateBatch.model_validate(output).validate_context(segment(), old_batch, page_ids, zone=UTC)


def test_event_candidate_uses_schema_date_and_rejects_only_future_date_invariant() -> None:
    output = empty_output()
    output["events"] = [candidate_item(MemoryKind.EVENT, 100)]
    batch = MemoryCandidateBatch.model_validate(output)
    batch.validate_context(segment(), SnapshotBatch((), 0), MemoryPageIdMap(), zone=UTC)

    output["events"][0]["event_date"] = "2099-01-01"
    with pytest.raises(MemoryCandidateError, match="future event date"):
        MemoryCandidateBatch.model_validate(output).validate_context(
            segment(),
            SnapshotBatch((), 0),
            MemoryPageIdMap(), zone=UTC,
        )


def test_tool_failure_recovery_carried_over_verbatim_keeps_its_old_source() -> None:
    """主链端到端扩 3 份 Job #15（对话 060）：更新 Bash 时逐值保留旧 failure_recovery，本段只有成功结果。"""

    old_recovery = (
        "在当前 macOS zsh 环境中，若 `timeout` 报 command not found，可去掉该包装命令，改用 Bash 调用本身的"
        "超时限制，并让 `curl` 使用 `--max-time`；随后成功取得远端 GitHub API 数据和 raw 源文件。"
    )
    old_fields = {
        "tool_name": "workspace.inspect",
        "invocation": "在 zsh 中检索 Python 源码时，可使用 `grep -rln \"模式\" 目录 --include='*.py'`。",
        "failure_recovery": old_recovery,
        "verification": "本次调用成功列出了 memory 包目录、源文件清单以及包外运行时调用位置。",
    }
    old_batch = snapshot_batch(document(MemoryKind.TOOL, fields=old_fields))
    page_ids = MemoryPageIdMap.from_snapshots(old_batch)
    only_successes = segment(messages=tool_turn())
    output = empty_output()
    output["tools"] = [
        {
            "page_id": 1,
            **old_fields,
            "constraints": "调用结束后工作目录会重置；依赖特定目录的命令应在同一次调用内显式 `cd`。",
        }
    ]
    MemoryCandidateBatch.model_validate(output).validate_context(only_successes, old_batch, page_ids, zone=UTC)

    output["tools"][0]["failure_recovery"] = old_recovery + "另：本段新写的一句恢复办法。"
    with pytest.raises(MemoryCandidateError, match="failure followed"):
        MemoryCandidateBatch.model_validate(output).validate_context(only_successes, old_batch, page_ids, zone=UTC)

    old_without_recovery = {name: value for name, value in old_fields.items() if name != "failure_recovery"}
    fresh_batch = snapshot_batch(document(MemoryKind.TOOL, fields=old_without_recovery))
    output["tools"][0]["failure_recovery"] = old_recovery
    with pytest.raises(MemoryCandidateError, match="failure followed"):
        MemoryCandidateBatch.model_validate(output).validate_context(
            only_successes, fresh_batch, MemoryPageIdMap.from_snapshots(fresh_batch)
        , zone=UTC)


def test_event_date_is_checked_against_the_subjects_local_day() -> None:
    """他在 UTC+8 早上 07:30 说"我今天早上跑了 5 公里"：消息存 2026-10-25T23:30Z，本地已是 10-26（M6）。"""

    from datetime import datetime
    from zoneinfo import ZoneInfo

    from habitus.pre.conversation import ConversationMessage, ConversationMessageRole

    morning = (
        ConversationMessage(
            message_id="m-0",
            sequence=0,
            role=ConversationMessageRole.PROMPT,
            occurred_at=datetime(2026, 10, 25, 23, 30, tzinfo=UTC),
            content="我今天早上跑了 5 公里",
        ),
        ConversationMessage(
            message_id="m-1",
            sequence=1,
            role=ConversationMessageRole.COMPLETION,
            occurred_at=datetime(2026, 10, 25, 23, 30, 5, tzinfo=UTC),
            content="记下了。",
        ),
    )
    shanghai = ZoneInfo("Asia/Shanghai")
    output = empty_output()
    output["events"] = [candidate_item(MemoryKind.EVENT, 100)]
    output["events"][0]["event_date"] = "2026-10-26"
    batch = MemoryCandidateBatch.model_validate(output)
    batch.validate_context(segment(messages=morning), SnapshotBatch((), 0), MemoryPageIdMap(), zone=shanghai)
    with pytest.raises(MemoryCandidateError, match="future event date"):
        batch.validate_context(segment(messages=morning), SnapshotBatch((), 0), MemoryPageIdMap(), zone=UTC)

    output["events"][0]["event_date"] = "2026-10-27"
    with pytest.raises(MemoryCandidateError, match="future event date"):
        MemoryCandidateBatch.model_validate(output).validate_context(
            segment(messages=morning), SnapshotBatch((), 0), MemoryPageIdMap(), zone=shanghai
        )


@pytest.mark.parametrize(
    ("field", "item", "message"),
    [
        (
            "relations",
            {"action": "add", "from_page_id": "1", "to_page_id": 100, "link_type": "related_to"},
            "invalid relation candidate: page_id must be a positive integer",
        ),
        (
            "relations",
            {"action": "add", "from_page_id": 1, "to_page_id": True, "link_type": "related_to"},
            "invalid relation candidate: page_id must be a positive integer",
        ),
        (
            "relations",
            {"action": ["add"], "from_page_id": 1, "to_page_id": 100, "link_type": "related_to"},
            "relation candidate contains an unsupported action",
        ),
        (
            "relations",
            {"action": "add", "from_page_id": 1, "to_page_id": 100, "link_type": 3},
            "relation candidate contains an unsupported link_type",
        ),
        (
            "identity_proposals",
            {"proposal_type": "same_memory", "source_page_id": 1.0, "target_page_id": 2, "basis": "duplicate_identity"},
            "identity proposal contains an invalid type, basis or source_page_id",
        ),
        (
            "identity_proposals",
            {"proposal_type": 1, "source_page_id": 1, "target_page_id": None, "basis": "explicit_forget"},
            "identity proposal contains an invalid type, basis or source_page_id",
        ),
        (
            "identity_proposals",
            {"proposal_type": "same_memory", "source_page_id": 1, "target_page_id": "2", "basis": "duplicate_identity"},
            "same_memory proposal requires a valid target_page_id",
        ),
        (
            "identity_proposals",
            {"proposal_type": "remove_memory", "source_page_id": 1, "target_page_id": "", "basis": "explicit_forget"},
            "remove_memory proposal cannot contain a target_page_id",
        ),
    ],
)
def test_model_output_relation_and_identity_items_with_wrong_types_are_rejected_by_the_decoder(
    field: str, item: dict[str, object], message: str
) -> None:
    """模型答复里关系与身份提议的类型错在解码器就被拒（W16：类型检查在边界）。"""

    output = empty_output()
    output[field] = [item]
    with pytest.raises(MemoryCandidateError, match=message):
        MemoryCandidateBatch.model_validate(output)


def test_context_violations_keep_their_fixed_check_order_and_wording() -> None:
    """来源校验按固定检查顺序逐条列出违规：修补轮请求、剔除原因与 validate_context 的首条报错都依赖这份顺序与文字。"""

    output = empty_output()
    output["tools"] = [candidate_item(MemoryKind.TOOL, 101)]
    output["events"] = [{**candidate_item(MemoryKind.EVENT, 102), "event_date": "2999-01-01"}]
    output["intentions"] = [{**candidate_item(MemoryKind.INTENTION, 103, confirmed=True), "status": "completed"}]
    output["relations"] = [
        {"action": "remove", "from_page_id": 101, "to_page_id": 102, "link_type": "caused_by"},
        {"action": "add", "from_page_id": 101, "to_page_id": 500, "link_type": "caused_by"},
        {"action": "add", "from_page_id": 1, "to_page_id": 101, "link_type": "caused_by"},
    ]
    output["identity_proposals"] = [
        {"proposal_type": "remove_memory", "source_page_id": 1, "target_page_id": None, "basis": "explicit_forget"}
    ]
    batch = MemoryCandidateBatch.model_validate(output)
    empty_old = SnapshotBatch((), 0)

    violations = batch.context_violations(segment(), empty_old, MemoryPageIdMap(), zone=UTC)

    assert [violation.describe() for violation in violations] == [
        "relations add 101->500 caused_by: relation candidate references an unknown to_page_id",
        "relations add 1->101 caused_by: relation candidate references an unknown from_page_id",
        "identity_proposals remove_memory source_page_id=1: identity proposal source is not a complete old-memory page_id",
        "relations add 1->101 caused_by: identity proposal source cannot also be referenced by a relation candidate",
        "relations remove 101->102 caused_by: relation remove must reference two existing memory page_ids",
        "tools page_id=101: tool candidate requires a matching successful tool_call/tool_result pair",
        "events page_id=102: future event date cannot represent an occurred Event",
        "intentions page_id=103: completed intention must target an existing intention read in this batch",
    ]
    with pytest.raises(MemoryCandidateError) as raised:
        batch.validate_context(segment(), empty_old, MemoryPageIdMap(), zone=UTC)
    assert str(raised.value) == "relation candidate references an unknown to_page_id"
