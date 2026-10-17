"""机会表的形状（``19-预测层方案.md`` 第九节 N1）：一条 lane 一晚，对每条成立（或候选）关系列出历史每次机会，
外加每个行为类按天的日历。

- **机会** = 前因的一条（合并后的）链：锚那条记录、链里全部记录、做完的时刻；这一段窗口里后果来没来（None = 结果未知：
  人不在场、伸过截止日、或只有"判不了"），来了的话是哪条、几点开始。时窗要从这些"几点开始"取，依据要指向这些 URI。
- **日历** = 一个类历史上发生过的每一天：那天第一次的记录、几点、当天几次。"到了日子 / 离上次够久 / 几号"从这里读，
  不用预测树的复发间隔（会话 lane 一轮一条之后它量的是两句话隔多久）。
- 表是夜里从同一份序列重算出来的派生物，第二晚整张重写；不是账本，不只增。

本模块对 scene、存储零知识；时刻一律带时区。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from habitus.foresight.errors import ForesightError

#: 机会表只列这几段（短跨度；长跨度与带条件的还不进预测层）。
SEGMENTS = ("chain", "rest_of_day", "next_day", "days_2_7")
#: 列进表的关系状态：成立的开口用，候选的只给影子账（``shadow_uses_unestablished`` 开着时）。
STATUSES = ("established", "candidate")


def _aware(value: object, label: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ForesightError(f"{label} must be a timezone-aware datetime")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ForesightError(f"{label} must be non-empty text")
    return value


def _day(value: object, label: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ForesightError(f"{label} must be a date")
    return value


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ForesightError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class Opportunity:
    """历史上的一次机会：前因那条链与这一段窗口里后果的结果。"""

    #: 链头那条记录与链里全部记录（含链头）。
    antecedent_uri: str
    antecedent_uris: tuple[str, ...]
    day: date
    #: 链里最晚做完的那一槽的开头（预测层量"做完之后多久"从这里起）。
    ended_at: datetime
    #: 后果来没来；None = 结果未知。
    came: bool | None
    consequent_uri: str | None = None
    consequent_at: datetime | None = None

    def __post_init__(self) -> None:
        _text(self.antecedent_uri, "opportunity antecedent_uri")
        if not isinstance(self.antecedent_uris, tuple) or self.antecedent_uri not in self.antecedent_uris:
            raise ForesightError("opportunity antecedent_uris must be a tuple containing the anchor")
        for uri in self.antecedent_uris:
            _text(uri, "opportunity antecedent uri")
        _day(self.day, "opportunity day")
        _aware(self.ended_at, "opportunity ended_at")
        if self.came is not None and not isinstance(self.came, bool):
            raise ForesightError("opportunity came must be a boolean or None")
        if (self.came is True) != (self.consequent_uri is not None and self.consequent_at is not None):
            raise ForesightError("an opportunity names the consequent exactly when it came")
        if self.consequent_uri is not None:
            _text(self.consequent_uri, "opportunity consequent_uri")
        if self.consequent_at is not None:
            _aware(self.consequent_at, "opportunity consequent_at")

    @property
    def known(self) -> bool:
        return self.came is not None


@dataclass(frozen=True)
class TreeReading:
    """语义树这一晚对这条关系的读数（``RelationTest``）：有前因时后果的发生率、没前因时的（同一天人在场的对照时刻）、效应够不够大
    （成立时要过的第 3 道）。它回答"是不是原因"；"从它出发预测准不准"由账回答（裁定 56 第 10 条：开不开口看账，依据看它）。"""

    treated: float
    control: float
    large_enough: bool | None = None

    def __post_init__(self) -> None:
        for label in ("treated", "control"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
                raise ForesightError(f"tree reading {label} must be a rate in [0, 1]")
        if self.large_enough is not None and not isinstance(self.large_enough, bool):
            raise ForesightError("tree reading large_enough must be a boolean or None")

    @property
    def lift(self) -> float:
        """有前因比没前因多出来的发生率（压低的关系为负）。"""

        return self.treated - self.control


@dataclass(frozen=True)
class RelationOpportunities:
    """一条关系的全部历史机会。``key`` 是语义树的关系身份；前因、后果是概念身份（基础概念 = 类编号，汇总概念另有身份），
    ``antecedent_classes`` / ``consequent_classes`` 是两头概念对应的类编号（预期与触发都记类编号）。"""

    key: str
    antecedent: str
    consequent: str
    antecedent_label: str
    consequent_label: str
    antecedent_classes: tuple[str, ...]
    consequent_classes: tuple[str, ...]
    segment: str
    status: str
    upward: bool
    items: tuple[Opportunity, ...]
    #: 语义树这一晚的读数（这一晚没检验到的为 None）。
    reading: TreeReading | None = None

    def __post_init__(self) -> None:
        if self.reading is not None and not isinstance(self.reading, TreeReading):
            raise ForesightError("relation reading must be a TreeReading or None")
        for label in ("key", "antecedent", "consequent", "antecedent_label", "consequent_label"):
            _text(getattr(self, label), f"relation {label}")
        for label in ("antecedent_classes", "consequent_classes"):
            classes = getattr(self, label)
            if not isinstance(classes, tuple) or not classes:
                raise ForesightError(f"relation {label} must be a non-empty tuple")
            for token in classes:
                _text(token, f"relation {label} entry")
        if self.segment not in SEGMENTS:
            raise ForesightError(f"unknown relation segment: {self.segment!r}")
        if self.status not in STATUSES:
            raise ForesightError(f"unknown relation status: {self.status!r}")
        if not isinstance(self.upward, bool):
            raise ForesightError("relation upward must be a boolean")
        if not isinstance(self.items, tuple) or any(not isinstance(item, Opportunity) for item in self.items):
            raise ForesightError("relation items must be Opportunity records")
        if [item.ended_at for item in self.items] != sorted(item.ended_at for item in self.items):
            raise ForesightError("relation items must be in time order")

    @property
    def opportunities(self) -> int:
        """结果已知的机会数。"""

        return sum(1 for item in self.items if item.known)

    @property
    def hits(self) -> int:
        return sum(1 for item in self.items if item.came)

    @property
    def unknown(self) -> int:
        return len(self.items) - self.opportunities


@dataclass(frozen=True)
class KindDay:
    """一个类在某一天：第一次的记录与时刻、当天几次、其中几次是自然发生的（开始前没被系统提醒过，裁定 54）。
    "离上次多久"按全部发生算（他确实做了）：``first_uri`` / ``first_at`` 是当天第一次，不管提醒没提醒；"命中"只数自然发生的
    那几天（被提醒才做的不是他自己的规律），时窗与出处取当天第一次**自然**发生的 ``natural_uri`` / ``natural_at``——提醒的
    时刻不许漏回"他往常几点做"。``natural`` 为 0 时这两个是 None；全部自然时不给就是第一次那条。"""

    day: date
    first_uri: str
    first_at: datetime
    count: int
    natural: int
    natural_uri: str | None = None
    natural_at: datetime | None = None

    def __post_init__(self) -> None:
        _day(self.day, "kind day")
        _text(self.first_uri, "kind day first_uri")
        _aware(self.first_at, "kind day first_at")
        _positive(self.count, "kind day count")
        if isinstance(self.natural, bool) or not isinstance(self.natural, int) or not 0 <= self.natural <= self.count:
            raise ForesightError("kind day natural must count between zero and the day's occurrences")
        if self.first_at.date() != self.day:
            raise ForesightError("a kind day's first occurrence starts on that day")
        if (self.natural_uri is None) != (self.natural_at is None):
            raise ForesightError("a kind day's first natural occurrence needs both its uri and its moment")
        if self.natural == 0:
            if self.natural_uri is not None:
                raise ForesightError("a kind day with no natural occurrence has no first natural one")
            return
        if self.natural_uri is None:
            if self.natural != self.count:
                raise ForesightError("a kind day with reminded occurrences must name its first natural one")
            object.__setattr__(self, "natural_uri", self.first_uri)
            object.__setattr__(self, "natural_at", self.first_at)
            return
        _text(self.natural_uri, "kind day natural_uri")
        _aware(self.natural_at, "kind day natural_at")
        assert self.natural_at is not None
        if self.natural_at.date() != self.day or self.natural_at < self.first_at:
            raise ForesightError("a kind day's first natural occurrence starts on that day, not before its first one")


@dataclass(frozen=True)
class KindDays:
    """一个行为类的日历：发生过的每一天（升序）。"""

    kind_token: str
    label: str
    days: tuple[KindDay, ...]

    def __post_init__(self) -> None:
        _text(self.kind_token, "kind token")
        _text(self.label, "kind label")
        if not isinstance(self.days, tuple) or any(not isinstance(item, KindDay) for item in self.days):
            raise ForesightError("kind days must be KindDay records")
        if [item.day for item in self.days] != sorted({item.day for item in self.days}):
            raise ForesightError("kind days must be distinct and ascending")

    @property
    def last(self) -> KindDay | None:
        return self.days[-1] if self.days else None


@dataclass(frozen=True)
class OpportunityTable:
    """一条 lane 第 ``night`` 晚的机会表。``present_days`` 是人在场的日子（"到了日子"的机会分母）；``relation_night`` 是
    用的哪一晚的关系表（夜批正常时就是 ``night``）。"""

    lane: str
    night: date
    relation_night: date | None
    slot_minutes: int
    window_slots: int
    present_days: tuple[date, ...]
    relations: tuple[RelationOpportunities, ...]
    kinds: tuple[KindDays, ...]

    def __post_init__(self) -> None:
        _text(self.lane, "table lane")
        _day(self.night, "table night")
        if self.relation_night is not None:
            _day(self.relation_night, "table relation_night")
        _positive(self.slot_minutes, "table slot_minutes")
        _positive(self.window_slots, "table window_slots")
        if not isinstance(self.present_days, tuple) or list(self.present_days) != sorted(set(self.present_days)):
            raise ForesightError("table present_days must be distinct ascending dates")
        for day in self.present_days:
            _day(day, "table present day")
        if not isinstance(self.relations, tuple) or any(
            not isinstance(item, RelationOpportunities) for item in self.relations
        ):
            raise ForesightError("table relations must be RelationOpportunities records")
        keys = [item.key for item in self.relations]
        if len(set(keys)) != len(keys):
            raise ForesightError("a relation appears twice in the table")
        if not isinstance(self.kinds, tuple) or any(not isinstance(item, KindDays) for item in self.kinds):
            raise ForesightError("table kinds must be KindDays records")
        tokens = [item.kind_token for item in self.kinds]
        if len(set(tokens)) != len(tokens):
            raise ForesightError("a kind appears twice in the table")

    def by_consequent(self, kind_token: str) -> tuple[RelationOpportunities, ...]:
        """以这个类为后果的全部关系（反向看前因用）。"""

        return tuple(item for item in self.relations if kind_token in item.consequent_classes)

    def by_antecedent(self, kind_token: str) -> tuple[RelationOpportunities, ...]:
        """前因概念含这个类的全部关系（有件事刚发生，正向出候选用）。"""

        return tuple(item for item in self.relations if kind_token in item.antecedent_classes)

    def kind(self, kind_token: str) -> KindDays | None:
        return next((item for item in self.kinds if item.kind_token == kind_token), None)


__all__ = [
    "SEGMENTS",
    "STATUSES",
    "KindDay",
    "KindDays",
    "Opportunity",
    "OpportunityTable",
    "RelationOpportunities",
    "TreeReading",
]
