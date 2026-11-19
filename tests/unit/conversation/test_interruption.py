"""会话中断时的收尾（裁定 107 第 2 条）：没有结果的工具调用补终态结果，没有最终回复的轮补一条说明中断的最终回复。"""

from __future__ import annotations

from datetime import timedelta

from habitus.memory.conversation import ConversationRetentionPlanner, ConversationSegmentationConfig
from habitus.memory.conversation.interruption import (
    INTERRUPTED_COMPLETION,
    INTERRUPTED_TOOL_RESULT,
    interruption_messages,
)
from habitus.pre.conversation import (
    ConversationBatch,
    ConversationMessageRole,
    ConversationSegment,
    ConversationToolResultStatus,
)
from tests.helpers import BASE_TIME, closed_turn, message, tool_turn

AT = BASE_TIME + timedelta(days=2)


def _call(sequence: int, call_id: str):  # type: ignore[no-untyped-def]
    return message(sequence, ConversationMessageRole.TOOL_CALL, {"q": sequence}, tool_call_id=call_id, tool_name="shell.run")


def test_closed_turns_need_no_closing() -> None:
    live = ConversationBatch("c", (*closed_turn(start_sequence=0), *tool_turn(start_sequence=2)))
    assert interruption_messages(live, at=AT) == ()


def test_a_turn_without_final_reply_gets_one_closing_reply() -> None:
    live = ConversationBatch("c", (*closed_turn(start_sequence=0), message(2, ConversationMessageRole.PROMPT, "还在吗")))

    (closing,) = interruption_messages(live, at=AT)

    assert (closing.sequence, closing.role, closing.content, closing.occurred_at) == (
        3,
        ConversationMessageRole.COMPLETION,
        INTERRUPTED_COMPLETION,
        AT,
    )


def test_unanswered_calls_get_failed_terminal_results_then_a_closing_reply() -> None:
    answered = tool_turn(start_sequence=0)[:3]
    live = ConversationBatch("c", (*answered, _call(3, "call-a"), _call(4, "call-b")))

    first, second, reply = interruption_messages(live, at=AT)

    assert [(item.sequence, item.tool_call_id, item.tool_name, item.tool_status) for item in (first, second)] == [
        (5, "call-a", "shell.run", ConversationToolResultStatus.ERROR),
        (6, "call-b", "shell.run", ConversationToolResultStatus.ERROR),
    ]
    assert first.content == INTERRUPTED_TOOL_RESULT
    assert (reply.sequence, reply.role) == (7, ConversationMessageRole.COMPLETION)


def test_after_closing_the_live_is_complete_turns_and_seals_on_the_existing_flush_path() -> None:
    """收尾之后走现有路径：flush 规划认得出每一轮都闭合了，整段在轮尾结束，调用和结果在同一段里。"""

    live = (*closed_turn(start_sequence=0), message(2, ConversationMessageRole.PROMPT, "跑一下"), _call(3, "call-a"))
    closing = interruption_messages(ConversationBatch("c", live), at=AT)
    whole = ConversationBatch("c", (*live, *closing))

    plan = ConversationRetentionPlanner(ConversationSegmentationConfig()).plan(whole, flush=True)

    assert plan.through_sequence == whole.messages[-1].sequence
    assert ConversationSegment("c", "s", plan.archive_messages).ends_mid_turn is False


def test_closing_message_ids_are_deterministic() -> None:
    live = ConversationBatch("c", (message(0, ConversationMessageRole.PROMPT, "x"), _call(1, "call-a")))
    assert interruption_messages(live, at=AT) == interruption_messages(live, at=AT)
