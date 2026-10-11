"""会话 lane 的保护上限（都是保护闸，不是语义参数）。"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SessionLaneConfig:
    """给模型的材料与每轮写出来的量的上限。

    - ``max_turn_chars``：一轮材料的总字数上限。依据（本机 5,684 轮）：指令 + 全部答复 + 每次调用一行，
      中位 1,585 字、九成以内 5,128 字，不超过 8,000 字的轮占 95.5%。
    - ``max_instruction_chars``：人的那条指令最多留多少字（留头尾）。
    - ``tool_argument_chars`` / ``tool_error_chars``：每次工具调用留参数开头多少字、失败时留报错多少字。
    - ``kept_tool_lines``：超出总上限时工具调用留头尾共几行。
    - ``kept_reply_lines``：超出总上限时助手中途说的话留头尾共几条（依据：本机最长的一轮 989 个动作，
      中途的话逐条各留一点仍会超上限，落到整段盲截；改为只留头尾几条）。
    - ``question_tools``：助手用来向人提问、等人作答的工具名。人对它的作答算一句人说的话，开新的一轮。
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
    question_tools: tuple[str, ...] = ("AskUserQuestion",)

    def __post_init__(self) -> None:
        tools = tuple(self.question_tools)
        if any(not isinstance(name, str) or not name.strip() for name in tools):
            raise ValueError("session lane question_tools must be non-empty tool names")
        object.__setattr__(self, "question_tools", tools)
        for item in fields(self):
            if item.name == "question_tools":
                continue
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"session lane {item.name} must be a positive integer")
        if self.max_instruction_chars > self.max_turn_chars:
            raise ValueError("session lane max_instruction_chars cannot exceed max_turn_chars")
        if self.kept_tool_lines < 2:
            raise ValueError("session lane kept_tool_lines must keep at least a head and a tail line")
        if self.kept_reply_lines < 2:
            raise ValueError("session lane kept_reply_lines must keep at least a head and a tail reply")


__all__ = ["SessionLaneConfig"]
