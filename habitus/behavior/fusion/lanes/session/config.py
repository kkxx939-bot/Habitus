"""会话 lane 的保护上限（都是保护闸，不是语义参数）。"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SessionLaneConfig:
    """给模型的材料与每轮写出来的量的上限。

    - ``max_turn_chars``：一轮材料（指令 + 全部答复 + 每次调用一行）的总字数上限。
    - ``max_instruction_chars``：人的那条指令最多留多少字（留头尾）。
    - ``tool_argument_chars`` / ``tool_error_chars``：每次工具调用留参数开头多少字、失败时留报错多少字。
    - ``kept_tool_lines``：超出总上限时工具调用留头尾共几行。
    - ``kept_reply_lines``：超出总上限时助手中途说的话留头尾共几条（逐条各留一点仍会超上限、落到整段盲截）。
    - ``max_steps`` / ``max_summary_chars`` / ``max_step_chars``：每轮写出来的步骤条数、概要与每条步骤的字数上限。
    """

    max_turn_chars: int = 8_000
    max_instruction_chars: int = 4_000
    tool_argument_chars: int = 120
    tool_error_chars: int = 240
    kept_tool_lines: int = 12
    kept_reply_lines: int = 6
    max_steps: int = 3
    max_summary_chars: int = 200
    max_step_chars: int = 120

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value <= 0:
                raise ValueError(f"session lane {item.name} must be a positive integer")
        if self.max_instruction_chars > self.max_turn_chars:
            raise ValueError("session lane max_instruction_chars cannot exceed max_turn_chars")
        if self.kept_tool_lines < 2:
            raise ValueError("session lane kept_tool_lines must keep at least a head and a tail line")
        if self.kept_reply_lines < 2:
            raise ValueError("session lane kept_reply_lines must keep at least a head and a tail reply")


__all__ = ["SessionLaneConfig"]
