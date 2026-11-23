"""检索读"每一轮的记录"用的只读接口与纯查找。

记忆的代码不引用行为的代码：这里只定义收发纯值（日期、支、名字、时刻、文字、不透明的地址字符串）的接口，
实现放在 ``runtime``。行为这一侧没开时接口给空。查找没有索引，按日期范围逐天读，天数有上限。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


@dataclass(frozen=True)
class BehaviorRecord:
    """一条"每一轮的记录"：他做的一件事，字段取自行为树的语义面。"""

    address: str
    lane: str
    day: date
    started_at: datetime
    name: str
    goal: str | None
    summary: str
    steps: tuple[str, ...]
    original_name: str | None = None

    def __post_init__(self) -> None:
        if not self.address or not self.lane or not self.name:
            raise ValueError("behavior record needs an address, a lane and a name")
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("behavior record started_at must include its local offset")

    @property
    def searchable(self) -> str:
        """关键词只搜这些字段，不搜渲染后的正文。"""

        parts = [self.name, self.goal or "", self.summary, *self.steps, self.original_name or ""]
        return "\n".join(part for part in parts if part).casefold()

    def line(self) -> str:
        return f"{self.started_at:%Y-%m-%d %H:%M} {self.name}"

    def text(self) -> str:
        steps = "；".join(self.steps) or "（无）"
        return f"{self.line()}\n  目标：{self.goal or '（无）'}\n  概要：{self.summary}\n  步骤：{steps}" + (
            f"\n  原始名：{self.original_name}" if self.original_name else ""
        )


@dataclass(frozen=True)
class BehaviorRecordBatch:
    """读一段日子的结果：读到的记录与读失败的日子（失败的日子当作没读到，不让整次查询失败）。"""

    records: tuple[BehaviorRecord, ...]
    failed_days: tuple[date, ...] = ()


class BehaviorRecordSource(Protocol):
    """读每一轮记录的只读接口；实现在 ``runtime``，行为侧没开时给空。"""

    def days(self) -> tuple[date, ...]:
        """有记录的日子，从早到晚。"""
        ...

    def read_days(self, days: tuple[date, ...]) -> BehaviorRecordBatch:
        """读这些日子的全部记录（两支都读）；「未观测」等空白不算记录。"""
        ...


class NoBehaviorRecords:
    """行为这一侧没开：没有每一轮的记录。"""

    def days(self) -> tuple[date, ...]:
        return ()

    def read_days(self, days: tuple[date, ...]) -> BehaviorRecordBatch:
        return BehaviorRecordBatch(())


def days_in_range(
    available: tuple[date, ...],
    date_from: date | None,
    date_to: date | None,
    *,
    max_days: int,
) -> tuple[date, ...]:
    """圈出范围内有记录的日子；范围超过上限时只取最晚的 ``max_days`` 天。"""

    selected = tuple(
        day for day in available if (date_from is None or day >= date_from) and (date_to is None or day <= date_to)
    )
    return selected[-max_days:]


def keyword_hits(text: str, keywords: Iterable[str]) -> int:
    """有几个关键词作为子串出现（不分大小写）。"""

    folded = text.casefold()
    return sum(1 for keyword in keywords if keyword and keyword.casefold() in folded)


def search_records(
    records: tuple[BehaviorRecord, ...],
    keywords: tuple[str, ...],
    *,
    limit: int,
) -> tuple[tuple[BehaviorRecord, ...], int]:
    """按命中的关键词数从多到少、同数时近的在前（和"材料矛盾以更晚的为准"同一个方向），取前 ``limit`` 条；同时给出
    总命中数。"""

    scored = [(keyword_hits(record.searchable, keywords), record) for record in records]
    matched = sorted(
        ((hits, record) for hits, record in scored if hits),
        key=lambda item: (-item[0], -item[1].started_at.timestamp(), item[1].address),
    )
    return tuple(record for _hits, record in matched[:limit]), len(matched)


__all__ = [
    "BehaviorRecord",
    "BehaviorRecordBatch",
    "BehaviorRecordSource",
    "NoBehaviorRecords",
    "days_in_range",
    "keyword_hits",
    "search_records",
]
