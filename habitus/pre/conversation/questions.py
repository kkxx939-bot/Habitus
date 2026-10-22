"""会话协议里 agent 用来停下来问人、给选项的工具（方案 32 第 4.2 条、方案 36 第四节）。

每个协议适配器自己声明它那个宿主的提问工具（没有就是空），提醒出口认提问、会话 lane 认"人对提问的作答"都按协议从适配器注册表取，
不再各自维护一张按宿主名的表——接一个新宿主只加一个适配器。
"""

from __future__ import annotations

from dataclasses import dataclass


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
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"question tool {label} must be non-empty text")


__all__ = ["QuestionTool"]
