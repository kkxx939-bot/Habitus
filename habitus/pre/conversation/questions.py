"""会话协议里 agent 用来停下来问人、给选项的工具（方案 32 第 4.2 条、方案 36 第四节）。

每个协议适配器自己声明它那个宿主的提问工具（没有就是空），提醒出口认提问、会话 lane 认"人对提问的作答"都按协议从适配器注册表取，
不再各自维护一张按宿主名的表——接一个新宿主只加一个适配器。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

#: Codex Desktop 的异步作答：agent 用 ``request_user_input_async`` 问完就接着干活，他作答之后以一条用户消息回来，
#: 标签里是一个 JSON 数组，每题带 ``questionItemId``、``question``、``answer``。
QUESTION_REPLY_TAG = "<send_user_message_question_reply>"


@dataclass(frozen=True)
class QuestionTool:
    """一个宿主的提问工具：宿主名、工具名、题目在参数 ``questions[]`` 里叫什么、作答是不是晚些才以一条用户消息回来
    （Codex Desktop 的 ``request_user_input_async``：工具结果只是"收到了"）。"""

    host: str
    tool: str
    title_key: str
    replies_later: bool = False

    def __post_init__(self) -> None:
        for label in ("host", "tool", "title_key"):
            value = getattr(self, label)
            if not value.strip():
                raise ValueError(f"question tool {label} must be non-empty text")


@dataclass(frozen=True)
class QuestionReply:
    """作答消息里的一题：哪次提问调用（``call_id``）的第几题、题目、他的作答。"""

    call_id: str
    index: int
    question: str
    answer: str


def parse_question_replies(content: str) -> tuple[QuestionReply, ...]:
    """``<send_user_message_question_reply>`` 作答消息 → 每一题的作答。会话 lane 与提醒出口都按这一份认。

    形状不对（没有标签、不是 JSON 数组）给空：那条消息照旧当人说的话。某一题没有文字 ``answer``、或 ``questionItemId``
    认不出提问调用的，那一题不算。
    """

    body = content.strip()
    if not body.startswith(QUESTION_REPLY_TAG):
        return ()
    body = body[len(QUESTION_REPLY_TAG) :]
    end = body.rfind("</")
    body = body[:end] if end >= 0 else body
    try:
        items = json.loads(body)
    except ValueError:
        return ()
    if not isinstance(items, list):
        return ()
    found: list[QuestionReply] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(answer := item.get("answer"), str):
            continue
        identity = _question_item(item.get("questionItemId"))
        if identity is not None:
            found.append(QuestionReply(identity[0], identity[1], str(item.get("question", "")), answer))
    return tuple(found)


def _question_item(value: object) -> tuple[str, int] | None:
    """``questionItemId`` 是一段 JSON 文本：``["request_user_input_async","call_xxx",0]`` —— 工具名、call_id、第几题。"""

    if not isinstance(value, str):
        return None
    try:
        parts = json.loads(value)
    except ValueError:
        return None
    if not (isinstance(parts, list) and len(parts) >= 2 and isinstance(parts[1], str) and parts[1]):
        return None
    index = parts[2] if len(parts) >= 3 and isinstance(parts[2], int) and not isinstance(parts[2], bool) else 0
    return parts[1], index


__all__ = ["QUESTION_REPLY_TAG", "QuestionReply", "QuestionTool", "parse_question_replies"]
