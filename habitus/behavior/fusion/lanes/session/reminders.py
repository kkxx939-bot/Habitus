"""会话 lane 对"开着的提醒"的接口（方案 30 第五节；裁定 61 第 5 条、62 第 2 条）：每段会话进融合时把还开着的提醒带给模型，
模型判这段会话在回应哪一条、正 / 中 / 负 / 已做；判出来的回应经这个口写回去。

行为侧对预测层零知识：谁给开着的提醒、回应写到哪里，由组合根注入（``runtime/foresight_responses.py``）；没接时是
``NoReminders``——材料里没有这一节，模型也不被问。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from habitus.behavior.fusion.errors import BehaviorFusionError

#: 回应的方向：模型只判这四种（裁定 61 第 4 条；「已做」= 他说这一次已经做过了，裁定 91 第 3 条②）。
#: 与预测层账本的方向（``foresight.ledger.POLARITIES``）同一组值，由组合根的桥原样写过去（行为侧不认识预测层，这里各写一份）。
POLARITIES = ("正", "中", "负", "已做")


@dataclass(frozen=True)
class PendingReminder:
    """带给模型看的一条还开着的提醒：身份、什么时候说的、说了什么、最晚到什么时候。"""

    reminder_id: str
    said_at: datetime
    text: str
    deadline: datetime

    def __post_init__(self) -> None:
        if not self.reminder_id:
            raise BehaviorFusionError("a pending reminder needs an identity")
        for label in ("said_at", "deadline"):
            value = getattr(self, label)
            if value.utcoffset() is None:
                raise BehaviorFusionError(f"a pending reminder's {label} must be timezone-aware")


@dataclass(frozen=True)
class ReminderResponse:
    """模型判出的一条回应：这一轮回应的是哪条提醒、方向。"""

    reminder_id: str
    polarity: str

    def __post_init__(self) -> None:
        if not self.reminder_id:
            raise BehaviorFusionError("a reminder response needs the reminder's identity")
        if self.polarity not in POLARITIES:
            raise BehaviorFusionError(f"unknown reminder response polarity: {self.polarity!r}")


@runtime_checkable
class ReminderFeedback(Protocol):
    """开着的提醒从哪来、回应写到哪去。"""

    def open(self, at: datetime) -> tuple[PendingReminder, ...]:
        """``at`` 那一刻（这一轮他开口的时刻）还开着的提醒，按截止由近到远，最多几条由实现定。"""
        ...

    def respond(self, turn: str, response: ReminderResponse, *, at: datetime) -> None:
        """这一轮（``turn`` 是轮的身份，重放不重判）对一条提醒的回应。"""
        ...

    def answered(self, calls: frozenset[str], at: datetime) -> frozenset[str]:
        """``calls``（这一轮里出现过的调用身份）里，出口已按固定选项写过回话的那几次提问；``at`` 是这一轮他最后开口的时刻。"""
        ...


class NoReminders:
    """没接预测层时的显式空实现：没有开着的提醒，回应也无处可写。"""

    def open(self, at: datetime) -> tuple[PendingReminder, ...]:
        return ()

    def respond(self, turn: str, response: ReminderResponse, *, at: datetime) -> None:
        return None

    def answered(self, calls: frozenset[str], at: datetime) -> frozenset[str]:
        return frozenset()


__all__ = ["POLARITIES", "NoReminders", "PendingReminder", "ReminderFeedback", "ReminderResponse"]
