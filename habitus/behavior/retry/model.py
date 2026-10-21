"""每晚重试的堆里一项（裁定 67）：白天读不懂、先落了「没读懂」的东西，留着它重来所需的那一份，等当晚跑批前重试一次。

- **会话轮**（``TURN``）：会话 lane 一轮写不成（确定性失败），或会话源的重试账放弃了；载荷是这一轮本身（与会话 lane 的输入
  同形，``session.model.encode_turn``），重试走同一个描述器。
- **链**（``CHAIN``）：归约里没有可读名 / 物化失败的链；载荷是它的判断原记录。同一份代码再归约几乎必败，只当"推迟一晚再删"。

``cause`` 只记它为什么进来（放弃 / 确定性失败）；当晚重试的结果另按"模型用不了 / 内容不合格 / 确定性失败 / 成功"计数。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class RetryError(ValueError):
    """重试堆里一项的形状不对。"""


class RetryKind(str, Enum):
    TURN = "turn"
    CHAIN = "chain"


class RetryCause(str, Enum):
    GIVEN_UP = "given_up"  # 重试账放弃了（模型连不上、额度满、输出一直不合格）
    DETERMINISTIC = "deterministic"  # 这一版处理下必然写不成


_KEYS = {"item_id", "kind", "cause", "day", "entered_at", "payload"}
_IDENTITY = re.compile(r"[0-9a-f]{32}")


class Admission(str, Enum):
    """进堆的结果。"""

    KEPT = "kept"
    PRESENT = "present"
    FULL = "full"
    LATE = "late"  # 那一天的夜批已经跑过
    DISABLED = "disabled"  # 没接重试堆

    @property
    def kept(self) -> bool:
        return self in (Admission.KEPT, Admission.PRESENT)


@dataclass(frozen=True)
class RetryItem:
    """一项：身份、种类、为什么进来、属于哪一天（主体本地日）、什么时候进来、载荷（JSON 对象）。"""

    item_id: str
    kind: RetryKind
    cause: RetryCause
    day: date
    entered_at: datetime
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RetryKind(self.kind))
        object.__setattr__(self, "cause", RetryCause(self.cause))
        if not isinstance(self.item_id, str) or not _IDENTITY.fullmatch(self.item_id):
            raise RetryError("retry item_id must be 32 lowercase hexadecimal characters")
        if not isinstance(self.day, date) or isinstance(self.day, datetime):
            raise RetryError("retry day must be a date")
        if not isinstance(self.entered_at, datetime) or self.entered_at.utcoffset() is None:
            raise RetryError("retry entered_at must be timezone-aware")
        if not isinstance(self.payload, Mapping):
            raise RetryError("retry payload must be a JSON object")

    def to_bytes(self) -> bytes:
        payload = {
            "item_id": self.item_id,
            "kind": self.kind.value,
            "cause": self.cause.value,
            "day": self.day.isoformat(),
            "entered_at": self.entered_at.isoformat(timespec="microseconds"),
            "payload": dict(self.payload),
        }
        return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    @classmethod
    def from_mapping(cls, raw: object) -> RetryItem:
        if not isinstance(raw, Mapping) or set(raw) != _KEYS:
            raise RetryError("retry item has an unexpected shape")
        try:
            return cls(
                item_id=str(raw["item_id"]),
                kind=RetryKind(raw["kind"]),
                cause=RetryCause(raw["cause"]),
                day=date.fromisoformat(str(raw["day"])),
                entered_at=datetime.fromisoformat(str(raw["entered_at"])),
                payload=raw["payload"],
            )
        except (TypeError, ValueError) as exc:
            raise RetryError(f"retry item is invalid: {exc}") from exc


def retry_id(kind: RetryKind, key: str) -> str:
    """一项的身份由它是什么定（会话轮按轮的身份，链按链的身份）：同一样东西进两次是同一项。"""

    return hashlib.sha256(f"{RetryKind(kind).value}\n{key}".encode()).hexdigest()[:32]


@runtime_checkable
class RetryBacklog(Protocol):
    """进堆的口：会话 lane 与归约从构造器拿到它，拿回进堆的结果（拒收的由调用方计数、留信号）。"""

    def put(self, item: RetryItem) -> Admission: ...


class NoRetryBacklog:
    """没接重试堆（测试、单独跑一条 lane）：什么都不留，读不懂的就只落「没读懂」。"""

    def put(self, item: RetryItem) -> Admission:
        return Admission.DISABLED


__all__ = [
    "Admission",
    "NoRetryBacklog",
    "RetryBacklog",
    "RetryCause",
    "RetryError",
    "RetryItem",
    "RetryKind",
    "retry_id",
]
