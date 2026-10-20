"""一本账压实了的那一段（裁定 57、61 第 7 条；26 号清单 4.8）：窗外的旧事实文件可以删，只留不按权重算的全历史摘要。

压到哪一笔为止（``through``，按（形成时刻, 预期身份）排）之前的事实文件可以删；读的时候 ``through`` 以内残留的文件一律忽略
（先写累计值、再删文件，中途崩溃也不会数两遍）。存的是读账还要的东西，不多存：

- **自然命中的偏差**：绝对值的直方图（窗外发生的强度拿这一次在全部历史偏差里的位置算，``strength.py``）；
- **机会间隔**：没截尾的间隔有几个、最长的多长、最后一次机会是哪天、这些机会落在哪几种日型（账的久别用）。

不再存两条轴的累计值与窗外的连错（裁定 61 第 7 条）：权重从单元历史的各刻按边数，压了就分不出哪几刻某个条件在场、
组合成立后无法重新归边；窗外一刻的分量已低于 α，只读窗口几乎不影响读数。

本模块对存储零知识；编解码是规范 JSON 的同一份。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from habitus.foresight.errors import ForesightError

COMPACTION_SCHEMA = "foresight_account_compaction_v2"

type FactKey = tuple[datetime, str]


@dataclass(frozen=True)
class AccountCompaction:
    account: str
    through: FactKey
    #: 自然命中偏差的绝对值（分钟）→ 次数，按偏差升序。
    offsets: tuple[tuple[int, int], ...]
    gap_count: int
    longest_gap: float
    last_day: date | None
    day_types: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.account, str) or not self.account:
            raise ForesightError("compaction account must be non-empty text")
        at, identity = self.through
        if not isinstance(at, datetime) or at.utcoffset() is None or not isinstance(identity, str) or not identity:
            raise ForesightError("compaction through must be (aware moment, expectation id)")
        if any(size < 0 or count < 1 for size, count in self.offsets) or [s for s, _ in self.offsets] != sorted(
            {s for s, _ in self.offsets}
        ):
            raise ForesightError("compaction offsets are distinct non-negative sizes with positive counts")
        if self.gap_count < 0 or self.longest_gap < 0 or (self.gap_count == 0) != (self.longest_gap == 0.0):
            raise ForesightError("compaction gaps are counted with their longest length")

    def covers(self, at: datetime, expectation_id: str) -> bool:
        """这一笔（按它那次机会的时刻与预期身份）在压实的那一段里：读的时候忽略它的残留文件。"""

        return (at, expectation_id) <= self.through

    def offset_history(self) -> tuple[int, ...]:
        """压实那一段里自然命中的偏差（绝对值），按次数展开。"""

        return tuple(size for size, count in self.offsets for _ in range(count))


def encode_compaction(item: AccountCompaction) -> dict[str, Any]:
    return {
        "schema_version": COMPACTION_SCHEMA,
        "record": "compaction",
        "account": item.account,
        "through": [item.through[0].isoformat(), item.through[1]],
        "offsets": [[size, count] for size, count in item.offsets],
        "gap_count": item.gap_count,
        "longest_gap": item.longest_gap,
        "last_day": None if item.last_day is None else item.last_day.isoformat(),
        "day_types": list(item.day_types),
    }


def decode_compaction(raw: Mapping[str, Any]) -> AccountCompaction:
    try:
        if raw["schema_version"] != COMPACTION_SCHEMA or raw["record"] != "compaction":
            raise ForesightError(f"unknown compaction schema: {raw.get('schema_version')!r}")
        at, identity = raw["through"]
        return AccountCompaction(
            account=str(raw["account"]),
            through=(_moment(at), str(identity)),
            offsets=tuple((int(size), int(count)) for size, count in raw["offsets"]),
            gap_count=int(raw["gap_count"]),
            longest_gap=float(raw["longest_gap"]),
            last_day=None if raw["last_day"] is None else date.fromisoformat(str(raw["last_day"])),
            day_types=tuple(None if value is None else str(value) for value in raw["day_types"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"account compaction is not valid: {exc}") from exc


def _moment(value: object) -> datetime:
    found = datetime.fromisoformat(str(value))
    if found.utcoffset() is None:
        raise ForesightError("compaction moments must be timezone-aware")
    return found


__all__ = ["COMPACTION_SCHEMA", "AccountCompaction", "decode_compaction", "encode_compaction"]
