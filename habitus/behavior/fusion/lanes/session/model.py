"""会话 lane 的输入形状：一条消息、一轮对话。

行为侧不认识会话源（依赖方向：行为不 import conversation / pre），所以这里自带一份只含融合用得到的字段的
形状；把会话源信封翻成它是组合根的事。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from habitus.behavior.fusion.errors import BehaviorFusionError


class SessionMessageRole(str, Enum):
    """一条消息是谁的、是什么。"""

    PROMPT = "prompt"  # 人说的话
    COMPLETION = "completion"  # 助手的答复
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True)
class SessionMessage:
    """会话里的一条消息。``failed`` 只对工具结果有意义。"""

    sequence: int
    occurred_at: datetime
    role: SessionMessageRole
    content: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    failed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", SessionMessageRole(self.role))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise BehaviorFusionError("session message sequence must be a non-negative integer")
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.utcoffset() is None:
            raise BehaviorFusionError("session message occurred_at must be timezone-aware")
        if not isinstance(self.content, str):
            raise BehaviorFusionError("session message content must be text")


@dataclass(frozen=True)
class SessionTurn:
    """一轮对话：人说的一句话，加上到他下一句话之前助手做的全部事。

    "人说的话"也包括他对助手提问的作答（拆轮时翻成一句人说的话）；助手发问时上一轮就到此为止。
    助手停下之后被后台任务叫醒又做的那一段也自己算一轮，打头的是叫醒它的那条通知。

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
    def answered(self) -> bool:
        """助手在这一轮里有没有答复或调用。都没有就是被打断或出错了，不出记录。"""

        return any(
            (item.role is SessionMessageRole.COMPLETION and item.content.strip())
            or item.role is SessionMessageRole.TOOL_CALL
            for item in self.messages[1:]
        )


__all__ = ["SessionMessage", "SessionMessageRole", "SessionTurn"]
