"""行为类的编号与两个占位标记。

编号是行为类的身份：写进树上 occurrence 的 ``kind_token``，一经分配永不改；名称、判据可以改，编号不跟着变。
形状 ``<lane 前缀>-k<四位以上序号>``（``s-k0007`` 会话 lane、``p-k0003`` 物理 lane）——不用 ``:``，
仓库的安全路径规则不允许它（``foundation/ids.py``）。

两个占位标记不是行为类、不进清单：
- ``<前缀>-待定``：同 lane 的在用类都归不进，进待定池，等每晚新增；
- ``<前缀>-非事件``：不是一件事（小动作、单条命令）；它的比例是离线评价融合质量的指标（裁定 13）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Lane(str, Enum):
    """数据从哪条入口进来；每个行为类只属于一条 lane（裁定 2）。"""

    SESSION = "session"
    PHYSICAL = "physical"

    @property
    def prefix(self) -> str:
        return _PREFIXES[self]


_PREFIXES = {Lane.SESSION: "s", Lane.PHYSICAL: "p"}
_LANE_OF_PREFIX = {prefix: lane for lane, prefix in _PREFIXES.items()}
_PENDING = "待定"
_NOT_EVENT = "非事件"
_CLASS_ID = re.compile(r"^(?P<prefix>[a-z])-k(?P<number>\d{4,})$")
_MARKER = re.compile(rf"^(?P<prefix>[a-z])-(?P<marker>{_PENDING}|{_NOT_EVENT})$")


class KindIdError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class ClassId:
    """一个行为类的编号。"""

    lane: Lane
    number: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane", Lane(self.lane))
        if isinstance(self.number, bool) or not isinstance(self.number, int) or self.number <= 0:
            raise KindIdError("class number must be a positive integer")

    def __str__(self) -> str:
        return f"{self.lane.prefix}-k{self.number:04d}"

    @classmethod
    def parse(cls, text: object) -> ClassId:
        """只认规范写法：``s-k0000``（序号 0）与 ``s-k00007``（多补了 0）都不是编号——前者让下游解析时崩，
        后者会和 ``s-k0007`` 被当成两个类（第二轮评审 P2-5）。"""

        match = _CLASS_ID.match(text) if isinstance(text, str) else None
        if match is None or match["prefix"] not in _LANE_OF_PREFIX:
            raise KindIdError(f"not a behavior class id: {text!r}")
        number = int(match["number"])
        if number <= 0:
            raise KindIdError(f"not a behavior class id: {text!r}")
        found = cls(_LANE_OF_PREFIX[match["prefix"]], number)
        if str(found) != text:
            raise KindIdError(f"behavior class id is not canonically written: {text!r} (expected {str(found)!r})")
        return found


def pending_token(lane: Lane) -> str:
    return f"{Lane(lane).prefix}-{_PENDING}"


def not_event_token(lane: Lane) -> str:
    return f"{Lane(lane).prefix}-{_NOT_EVENT}"


def is_marker(token: object) -> bool:
    """``kind_token`` 是不是两个占位标记之一（任一 lane）。"""

    return isinstance(token, str) and _MARKER.match(token) is not None


def lane_of_token(token: object) -> Lane:
    """任一合法 ``kind_token``（编号或占位标记）属于哪条 lane。"""

    if not isinstance(token, str):
        raise KindIdError(f"not a kind token: {token!r}")
    marker = _MARKER.match(token)
    if marker is not None and marker["prefix"] in _LANE_OF_PREFIX:
        return _LANE_OF_PREFIX[marker["prefix"]]
    return ClassId.parse(token).lane


__all__ = [
    "ClassId",
    "KindIdError",
    "Lane",
    "is_marker",
    "lane_of_token",
    "not_event_token",
    "pending_token",
]
