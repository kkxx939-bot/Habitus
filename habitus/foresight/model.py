"""预测层里不带语义背景的值对象：此刻、还没封口的判断、"到此刻为止开始过的一件事"。对 scene 零知识。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

from habitus.foresight.errors import ForesightError


@dataclass(frozen=True)
class Moment:
    """ "此刻"：本地时刻、在第几槽、今天的日历条件。

    契约：``at`` 必须是**主体的本地时刻**（与行为树上 occurrence 同一偏移）。今天是哪一天、
    此刻在哪个槽，全部从它的本地时分算——传 UTC 进来会读错一天、也会读错槽。``slot`` 由装配层用
    ``temporal`` 的槽运算映射出来，不在这里重算。``calendar`` 是今天的日历条件（周几、几号、日型；裁定 59）：
    没有放假数据时日型取名义值，只给建预期用。
    """

    at: datetime
    slot: int
    calendar: Mapping[str, str] = field(default_factory=dict)
    #: 上一次看的时刻（上一拍）。时窗在那之后才关的，这一拍还算赶上了：拍点晚了一点不让没做的那次机会从账上消失。
    #: None = 只看此刻。
    looked_since: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.at, datetime) or self.at.utcoffset() is None:
            raise ForesightError("moment must be a timezone-aware local datetime")
        if self.looked_since is not None and (self.looked_since.utcoffset() is None or self.looked_since > self.at):
            raise ForesightError("looked_since must be a timezone-aware datetime no later than the moment")

    @property
    def caught_from(self) -> datetime:
        """时窗终点不早于它的，这一拍还算赶上了。"""

        return self.at if self.looked_since is None else self.looked_since

    @property
    def day(self) -> date:
        return self.at.date()


@dataclass(frozen=True)
class UnsealedRow:
    """还没归约到行为树上的一条判断：最近一段的链在判断存储里，此刻要把它们补上。

    ``name`` 为 None 是融合"没读懂"的那种判断——它在树上会成为一段观测空白；读得懂的行为名经白天归类落到类，
    落不到就 ``kind_token`` 为 None。没有 URI——它还不是树上的 occurrence。
    """

    name: str | None
    kind_token: str | None
    started_at: datetime
    last_observed_at: datetime
    summary: str | None

    def __post_init__(self) -> None:
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ForesightError("unsealed row name must be non-empty text or None")
        if self.name is None and self.kind_token is not None:
            raise ForesightError("an unreadable unsealed row cannot carry a kind")
        for label in ("started_at", "last_observed_at"):
            value = getattr(self, label)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ForesightError(f"unsealed row {label} must be a timezone-aware datetime")
        if self.last_observed_at < self.started_at:
            raise ForesightError("an unsealed row cannot be last seen before it started")

    @property
    def readable(self) -> bool:
        return self.name is not None


@dataclass(frozen=True)
class Seen:
    """最近几天到此刻为止开始过的一件事：已封口的行，或还没封口、现场归上类的判断。``ended_at`` 是最后所见（截到此刻）；
    ``source`` 是已封口记录的 URI，或 ``unsealed:<开始时刻>``（封口后编号可能变，预期上记它，反馈读回时才分得清）。"""

    started_at: datetime
    ended_at: datetime
    kind_token: str
    source: str
    #: 融合给这一次起的名字与一句话概要（模型小包里"今天这次"与"历史那几次"读的就是它们；触发不读）。
    name: str = ""
    summary: str = ""
    #: 开始了但放弃了（树上那一行的 status 是 abandoned）：不算这件事做了——与结算同一种读法（裁定 54：一条记录一种读法）。
    abandoned: bool = False
    #: 地点与同在的人（树上的结构字段，上游没给时为空）：白天判带条件关系的"地点 / 同在的人"条件成不成立要看它们。
    place: str | None = None
    subjects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.summary, str):
            raise ForesightError("seen row name and summary must be text")
        if self.place is not None and (not isinstance(self.place, str) or not self.place):
            raise ForesightError("seen row place must be non-empty text or None")
        if not isinstance(self.subjects, tuple) or any(not isinstance(item, str) or not item for item in self.subjects):
            raise ForesightError("seen row subjects must be a tuple of names")
        if not isinstance(self.abandoned, bool):
            raise ForesightError("seen row abandoned must be a boolean")
        for label in ("started_at", "ended_at"):
            value = getattr(self, label)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ForesightError(f"seen row {label} must be a timezone-aware datetime")
        if self.ended_at < self.started_at:
            raise ForesightError("a seen row cannot end before it started")
        for label in ("kind_token", "source"):
            if not isinstance(getattr(self, label), str) or not getattr(self, label):
                raise ForesightError(f"seen row {label} must be non-empty text")

    @property
    def day(self) -> date:
        return self.started_at.date()


class UnsealedReader(Protocol):
    """ "这段时间里有哪些还没封口的判断"。生产实现读判断存储，住组合根；本包只认这个契约。"""

    async def prepare(self, *, until: datetime) -> None: ...

    def rows(self, *, since: datetime, until: datetime) -> tuple[UnsealedRow, ...]: ...


class NoUnsealed:
    """没有判断存储可读时的显式空实现（脚本、离线重放）：明说"最近一段没补"，不是漏了。"""

    async def prepare(self, *, until: datetime) -> None:
        return None

    def rows(self, *, since: datetime, until: datetime) -> tuple[UnsealedRow, ...]:
        return ()


__all__ = ["Moment", "NoUnsealed", "Seen", "UnsealedReader", "UnsealedRow"]
