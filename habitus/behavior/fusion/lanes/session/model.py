"""会话 lane 的输入形状：一条消息、一轮对话。

行为侧不认识会话源（依赖方向：行为不 import conversation / pre），所以这里自带一份只含融合用得到的字段的
形状；把会话源信封翻成它是组合根的事。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from habitus.behavior.fusion.errors import BehaviorFusionError


class SessionMessageRole(str, Enum):
    """一条消息是谁的、是什么。"""

    PROMPT = "prompt"  # 人说的话
    COMPLETION = "completion"  # 助手的答复
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True)
class SessionMessage:
    """会话里的一条消息。``failed`` 只对工具结果有意义。``mid_turn``：人说的话插在宿主的一轮中间（助手还没给出最终答复时他作答、
    补一句、叫停），属于这一轮、不开新一轮（由协议适配器按宿主自己的轮次标，与记忆主链认的"一轮"是同一个）。"""

    sequence: int
    occurred_at: datetime
    role: SessionMessageRole
    content: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    failed: bool = False
    mid_turn: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", SessionMessageRole(self.role))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise BehaviorFusionError("session message sequence must be a non-negative integer")
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.utcoffset() is None:
            raise BehaviorFusionError("session message occurred_at must be timezone-aware")
        if not isinstance(self.content, str):
            raise BehaviorFusionError("session message content must be text")
        if not isinstance(self.mid_turn, bool):
            raise BehaviorFusionError("session message mid_turn must be a boolean")


@dataclass(frozen=True)
class SessionTurn:
    """一轮对话 = 宿主的一轮：他开口的那句话，到助手给出最终答复为止助手做的全部事，以及这中间他又说的话。

    助手还在干活时他说的话——对助手提问的作答（翻成"（回答助手的提问）…"）、补一句、叫停——都在这一轮里，不另开一轮
    （用户 2026-10-22：一件事被切成碎片是设计哲学的问题，记忆与行为两边按同一个"一轮"）。``messages`` 里第一条之后的
    人说的话就是这些中途的话。助手停下之后被后台任务叫醒又做的那一段自己算一轮，打头的是叫醒它的那条通知。

    ``instructed_at`` 是人开口的时刻，``completed_at`` 是这一轮最后一条消息的时刻（助手做完的时刻）。
    """

    conversation_id: str
    start_sequence: int
    end_sequence: int
    instructed_at: datetime
    completed_at: datetime
    instruction: str
    messages: tuple[SessionMessage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str) or not self.conversation_id:
            raise BehaviorFusionError("session turn conversation_id must be non-empty text")
        if not self.messages or self.messages[0].role is not SessionMessageRole.PROMPT:
            raise BehaviorFusionError("a session turn starts with the person's instruction")
        if self.completed_at < self.instructed_at:
            raise BehaviorFusionError("a session turn cannot complete before it was instructed")

    @property
    def key(self) -> str:
        """这一轮的身份：哪个会话、第几条到第几条消息。"""

        return f"{self.conversation_id}#{self.start_sequence}-{self.end_sequence}"

    @property
    def last_said_at(self) -> datetime:
        """他在这一轮里最后一次开口的时刻（开口那句、或之后作答 / 补一句 / 叫停的最后一句）。他对提醒的回应只能出在这之前。"""

        return max(item.occurred_at for item in self.messages if item.role is SessionMessageRole.PROMPT)

    @property
    def answered(self) -> bool:
        """助手在这一轮里有没有答复或调用。都没有就是被打断或出错了，不出记录。"""

        return any(
            (item.role is SessionMessageRole.COMPLETION and item.content.strip())
            or item.role is SessionMessageRole.TOOL_CALL
            for item in self.messages[1:]
        )


def encode_turn(turn: SessionTurn) -> dict[str, Any]:
    """一轮编成 JSON 对象（每晚重试的堆里存它，裁定 67）：与输入同形，重试走同一个描述器。"""

    return {
        "conversation_id": turn.conversation_id,
        "start_sequence": turn.start_sequence,
        "end_sequence": turn.end_sequence,
        "instructed_at": turn.instructed_at.isoformat(timespec="microseconds"),
        "completed_at": turn.completed_at.isoformat(timespec="microseconds"),
        "instruction": turn.instruction,
        "messages": [
            {
                "sequence": item.sequence,
                "occurred_at": item.occurred_at.isoformat(timespec="microseconds"),
                "role": item.role.value,
                "content": item.content,
                "tool_name": item.tool_name,
                "tool_call_id": item.tool_call_id,
                "failed": item.failed,
            }
            for item in turn.messages
        ],
    }


def decode_turn(raw: Mapping[str, Any]) -> SessionTurn:
    """``encode_turn`` 的反面；形状不对抛 ``BehaviorFusionError``。"""

    try:
        return SessionTurn(
            conversation_id=str(raw["conversation_id"]),
            start_sequence=int(raw["start_sequence"]),
            end_sequence=int(raw["end_sequence"]),
            instructed_at=datetime.fromisoformat(str(raw["instructed_at"])),
            completed_at=datetime.fromisoformat(str(raw["completed_at"])),
            instruction=str(raw["instruction"]),
            messages=tuple(
                SessionMessage(
                    sequence=int(item["sequence"]),
                    occurred_at=datetime.fromisoformat(str(item["occurred_at"])),
                    role=SessionMessageRole(item["role"]),
                    content=str(item["content"]),
                    tool_name=None if item["tool_name"] is None else str(item["tool_name"]),
                    tool_call_id=None if item["tool_call_id"] is None else str(item["tool_call_id"]),
                    failed=bool(item["failed"]),
                )
                for item in raw["messages"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BehaviorFusionError(f"encoded session turn is invalid: {exc}") from exc


__all__ = ["SessionMessage", "SessionMessageRole", "SessionTurn", "decode_turn", "encode_turn"]
