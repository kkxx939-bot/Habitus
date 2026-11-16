"""会话 lane 的选词口：每轮"描述这一轮"之后，用同一份材料在初始词表里选一个词（裁定 101 第 2、3 条）。

lane 只认这个形状；实现（词表包里的选词器）由组合根注入——融合包不 import 词表包的模型触点（依赖方向）。
实现失败时抛的错按"描述这一轮"同一组分类（``service`` 的 ``DESCRIBE_*`` / 确定性失败）：选词失败与描述失败走同一条路。

后台通知轮（叫醒通知打头、整轮没有他说的话）不问模型：由代码写一条「非事件」判断，名字与概要是下面两句固定的话。
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from habitus.behavior.fusion.lanes.session.prompt import TurnDescription

NOTICE_DESCRIPTION = TurnDescription(
    name="后台任务回来",
    summary="后台任务或子代理的通知回来，助手接着做；他这一轮没有说话",
    goal=None,
    steps=(),
)


class TurnPicker(Protocol):
    async def pick_turn(self, *, conversation: str, at: datetime, material: str) -> str:
        """这一轮的选词结果（词编号或占位标记）。``conversation`` 是会话的归一身份，``at`` 是他开口的时刻，``material``
        是给"描述这一轮"的那份材料。"""
        ...


__all__ = ["NOTICE_DESCRIPTION", "TurnPicker"]
