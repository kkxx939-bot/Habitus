"""Codex Desktop 异步作答消息的解析只有一份（``pre/conversation/questions.py``）：会话 lane 与提醒出口按同一个认法。"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from habitus.conversation.source import ConversationSourceEnvelope, conversation_source_request_digest
from habitus.foundation.integrity import canonical_digest
from habitus.pre.conversation import (
    QUESTION_REPLY_TAG,
    ConversationAdapterContext,
    ConversationAdapterRegistry,
    QuestionReply,
    parse_question_replies,
)
from habitus.runtime.session_lane import session_turns


def reply_message(question_item_id: str) -> str:
    body = json.dumps([{"questionItemId": question_item_id, "question": "走哪条路？", "answer": "云助手"}], ensure_ascii=False)
    return f"{QUESTION_REPLY_TAG}\n{body}\n</send_user_message_question_reply>\n"


def test_a_desktop_reply_names_the_call_the_question_and_the_answer() -> None:
    assert parse_question_replies(reply_message('["request_user_input_async","call_1",2]')) == (
        QuestionReply("call_1", 2, "走哪条路？", "云助手"),
    )
    assert parse_question_replies(reply_message('["request_user_input_async","call_1"]'))[0].index == 0


@pytest.mark.parametrize(
    "content",
    [
        reply_message("call_1"),  # questionItemId 不是 JSON 数组：认不出是哪次提问
        f"{QUESTION_REPLY_TAG}\nnot json\n</send_user_message_question_reply>",
        f"{QUESTION_REPLY_TAG}\n{{\"answer\": \"a\"}}\n</send_user_message_question_reply>",
        "随便一句话",
    ],
)
def test_anything_else_is_not_a_reply(content: str) -> None:
    assert parse_question_replies(content) == ()


def _envelope(text: str) -> ConversationSourceEnvelope:
    records = [
        {"type": "event_msg", "timestamp": "2026-10-01T00:00:00Z", "payload": {"type": "task_started"}},
        {"type": "response_item", "timestamp": "2026-10-01T00:00:01Z",
         "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}},
        {"type": "response_item", "timestamp": "2026-10-01T00:00:02Z",
         "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "好。"}]}},
    ]
    adapted = ConversationAdapterRegistry.with_builtins().adapt(
        "codex_rollout", {"records": records}, ConversationAdapterContext("codex-c", 0, datetime(2026, 10, 1, tzinfo=UTC))
    )
    started_on = date(2026, 10, 1)
    request_digest = conversation_source_request_digest(
        conversation_id="codex-c",
        started_on=started_on,
        protocol="codex_rollout",
        batch=adapted.batch,
        after_turn=adapted.after_turn,
        omit_tool_call_ids=frozenset(),
    )
    return ConversationSourceEnvelope.create(
        conversation_id="codex-c",
        started_on=started_on,
        protocol="codex_rollout",
        batch=adapted.batch,
        after_turn=adapted.after_turn,
        omit_tool_call_ids=frozenset(),
        delivery_id=canonical_digest(text),
        request_digest=request_digest,
        recorded_at=datetime(2026, 10, 1, tzinfo=UTC),
    )


def test_the_session_lane_reads_a_reply_the_same_way_the_reminder_outlet_does() -> None:
    """认得出的作答在会话 lane 里是一句"（回答助手的提问）…"；``questionItemId="call_1"`` 这种认不出的，提醒出口不记回话，
    会话 lane 也不当作答，照旧当一句人说的话（原先 lane 当作答、出口给空：一条记录两种解释）。"""

    (answered,) = session_turns(_envelope(reply_message('["request_user_input_async","call_1",0]')))
    assert answered.instruction == "（回答助手的提问）走哪条路？ → 云助手"

    unknown = reply_message("call_1")
    assert parse_question_replies(unknown) == ()
    (plain,) = session_turns(_envelope(unknown))
    assert plain.instruction == unknown.strip()
