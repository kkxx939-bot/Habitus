"""机会表的形状（``19-预测层方案.md`` 第九节 N1）：一条 lane 一晚，对每条成立（或候选）关系列出历史每次机会，
外加每个行为类按天的日历。

- **机会** = 前因的一条（合并后的）链：锚那条记录、链里全部记录、做完的时刻；这一段窗口里后果来没来（None = 结果未知：
  人不在场、伸过截止日、或只有"判不了"），来了的话是哪条、几点开始。时窗要从这些"几点开始"取，依据要指向这些 URI。
- **日历** = 一个类的复现（语义树登记、由 ``temporal`` 算，裁定 59）：按天的发生（那天第一次的记录、几点、当天几次、几次自然）
  与相邻间隔（漏看的权重、被提醒结尾的截尾）。「离上次够久」与补做强度从这里读。
- **看着的日子** = 这条 lane 每天有多大比例在看（``temporal.Coverage``）与那天的日型：账的久别按它算间隔的有效长度
  （人不在、设备没上传的日子不算长度；只数这本账有过机会的日型，26 号清单 3.3）。
- 表是夜里从同一份序列重算出来的派生物，第二晚整张重写；不是账本，不只增。

本模块对 scene、存储零知识；时刻一律带时区。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime

from habitus.conditions import DAY_TYPE_KEY, DayType
from habitus.foresight.errors import ForesightError
from habitus.temporal import DayInterval, OccurrenceDay

#: 机会表只列这几段：短跨度四段，加上日历情境（日历条件 → 后果，裁定 59）。其余长跨度与带条件的还不进预测层。
SEGMENTS = ("chain", "rest_of_day", "next_day", "days_2_7", "situation")
#: 列进表的关系状态：成立的开口用，候选的只给影子账（``shadow_uses_unestablished`` 开着时）。
STATUSES = ("established", "candidate")
#: 没见过的日型怎么办：不外推 / 按外接数据的官方含义归组（待用户定，默认不外推）。
UNSEEN_DAY_TYPES = ("skip", "official")


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


def _official(day_types: Iterable[str]) -> set[str]:
    """几种日型按官方含义放宽到同一类：上班（工作日、调休上班）/ 休息（周末、节假日）。"""

    classes = {DayType(value).working for value in day_types if value in {kind.value for kind in DayType}}
    return {kind.value for kind in DayType if kind.working in classes} | set(day_types)


@dataclass(frozen=True)
class ConditionSpec:
    """一条关系挂的一个条件，预测层白天判"此刻成不成立"要的东西（语义树的条件文本 + 能在白天判它的材料）：
    时间线条件（"之前出现过某一类"）带那个概念对应的类编号（基础概念就是它自己；汇总概念是成员类；细分概念要映射器，
    白天判不了，为空）；日历情境条件带日历键与成立的取值；地点 / 同在的人直接读记录的字段。"""

    text: str
    classes: tuple[str, ...] = ()
    calendar_key: str | None = None
    calendar_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.text, "condition text")
        if not isinstance(self.classes, tuple) or any(
            not isinstance(token, str) or not token for token in self.classes
        ):
            raise ForesightError("condition classes must be a tuple of class tokens")
        if (self.calendar_key is None) != (not self.calendar_values):
            raise ForesightError("a calendar condition names its key and values together")
        if self.calendar_key is not None:
            _text(self.calendar_key, "condition calendar_key")


@dataclass(frozen=True)
class RelationOpportunities:
    """一条关系的全部历史机会。``key`` 是语义树的关系身份；前因、后果是概念身份（基础概念 = 类编号，汇总概念另有身份），
    ``antecedent_classes`` / ``consequent_classes`` 是两头概念对应的类编号（预期与触发都记类编号）。
    ``conditions`` 是这条关系挂的条件（裁定 61 第 6 条：成立的带条件关系各是单元的一个输入）；机会只列条件全成立的链。"""

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
    #: 日历情境关系（``situation`` 段）盯的日历键与成立的取值；别的段没有。
    calendar_key: str | None = None
    calendar_values: tuple[str, ...] = ()
    #: 周几、几号、月末这类关系是在哪几种日型上见过他做的（检验时日型是固定住的，裁定 59 第 7 条）：建预期时今天的日型也要在里面——
    #: 只在工作日的周三做过，国庆的周三就不建。空 = 历史上日型都答不出（没有放假数据），不约束。日型关系自己不用这个。
    within_day_types: tuple[str, ...] = ()
    #: 这条关系挂的条件（按文本排序；不带条件为空）。关系身份的末尾就是它们拼成的键。
    conditions: tuple[ConditionSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.reading is not None and not isinstance(self.reading, TreeReading):
            raise ForesightError("relation reading must be a TreeReading or None")
        if not isinstance(self.conditions, tuple) or any(
            not isinstance(item, ConditionSpec) for item in self.conditions
        ):
            raise ForesightError("relation conditions must be ConditionSpec records")
        if [item.text for item in self.conditions] != sorted({item.text for item in self.conditions}):
            raise ForesightError("relation conditions must be distinct and sorted by text")
        if self.conditions and not self.key.endswith("|" + self.condition_text):
            raise ForesightError("a conditioned relation's key ends with its condition text")
        if self.conditions and self.segment == "situation":
            raise ForesightError("a calendar situation relation carries no conditions")
        if (self.segment == "situation") != (self.calendar_key is not None and bool(self.calendar_values)):
            raise ForesightError("a situation relation names its calendar key and values, other segments do not")
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
        if self.within_day_types and (self.segment != "situation" or self.calendar_key == DAY_TYPE_KEY):
            raise ForesightError(
                "only a calendar relation on another dimension than the day type names the day types it was seen on"
            )
        if self.status not in STATUSES:
            raise ForesightError(f"unknown relation status: {self.status!r}")
        if not isinstance(self.upward, bool):
            raise ForesightError("relation upward must be a boolean")
        if not isinstance(self.items, tuple) or any(not isinstance(item, Opportunity) for item in self.items):
            raise ForesightError("relation items must be Opportunity records")
        if [item.ended_at for item in self.items] != sorted(item.ended_at for item in self.items):
            raise ForesightError("relation items must be in time order")

    def calendar_holds(self, conditions: Mapping[str, str], *, official: bool = False) -> bool:
        """这条日历关系在这一天成不成立（``conditions`` 是那天的日历条件）。``official`` 开着时，没见过的日型按外接数据的官方含义
        归组（调休上班算上班、节假日算休息；待用户定的那一条，裁定 59 第 7 条）：日型关系的取值、其余关系见过的日型都放宽到同一类。
        那天日型答不出时不拿日型约束（建预期用的条件总有名义日型，统计用的可能没有）。"""

        if self.calendar_key is None:
            raise ForesightError("only a calendar relation holds on a day")
        value = conditions.get(self.calendar_key)
        if value is None:
            return False
        values = (
            _official(self.calendar_values)
            if official and self.calendar_key == DAY_TYPE_KEY
            else set(self.calendar_values)
        )
        if value not in values:
            return False
        day_type = conditions.get(DAY_TYPE_KEY)
        if not self.within_day_types or day_type is None or self.calendar_key == DAY_TYPE_KEY:
            return True
        allowed = _official(self.within_day_types) if official else set(self.within_day_types)
        return day_type in allowed

    @property
    def condition_text(self) -> str:
        """关系身份里的条件键（几个条件的文本用 ``+`` 拼；不带条件为空串）。"""

        return "+".join(item.text for item in self.conditions)

    @property
    def base_key(self) -> str:
        """去掉条件之后的关系身份：同一（前因, 后果, 段）的不带条件的那条与它的各个带条件版本是一族，此刻只取最具体的
        一条（裁定 61 第 6 条）。"""

        return self.key[: -len(self.condition_text) - 1] if self.conditions else self.key

    @property
    def coincidental(self) -> bool:
        """语义树已判它效应不够大（巧合）：不进"同一组"的签名与"同样多"的计数，权重照数（方案第六节第 1 条）。"""

        return self.reading is not None and self.reading.large_enough is False

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
class KindDays:
    """一个行为类的复现（裁定 59）：按天的发生（升序）与相邻两天之间的间隔（第 i 段在第 i 天与第 i+1 天之间）。
    "离上次多久"按全部发生算；命中只数自然发生的那几天，时窗只取自然发生的时刻（提醒的时刻不许漏回"往常几点做"）。"""

    kind_token: str
    label: str
    days: tuple[OccurrenceDay, ...]
    intervals: tuple[DayInterval, ...] = ()

    def __post_init__(self) -> None:
        _text(self.kind_token, "kind token")
        _text(self.label, "kind label")
        if not isinstance(self.days, tuple) or any(not isinstance(item, OccurrenceDay) for item in self.days):
            raise ForesightError("kind days must be OccurrenceDay records")
        if [item.day for item in self.days] != sorted({item.day for item in self.days}):
            raise ForesightError("kind days must be distinct and ascending")
        if not isinstance(self.intervals, tuple) or len(self.intervals) != max(0, len(self.days) - 1):
            raise ForesightError("a kind carries one interval between each pair of neighbouring days")
        for index, item in enumerate(self.intervals):
            if item.previous != self.days[index] or item.following != self.days[index + 1]:
                raise ForesightError("kind intervals must sit between their neighbouring days")

    @property
    def last(self) -> OccurrenceDay | None:
        return self.days[-1] if self.days else None


@dataclass(frozen=True)
class DayWatch:
    """这条 lane 的一天：有多大比例在看（0–1）、那天的日型（没有放假数据时为 None）。"""

    day: date
    share: float
    day_type: str | None = None

    def __post_init__(self) -> None:
        _day(self.day, "watch day")
        if isinstance(self.share, bool) or not isinstance(self.share, int | float) or not 0.0 <= self.share <= 1.0:
            raise ForesightError("watch share must be a number in [0, 1]")
        if self.day_type is not None:
            _text(self.day_type, "watch day_type")


@dataclass(frozen=True)
class OpportunityTable:
    """一条 lane 第 ``night`` 晚的机会表。``relation_night`` 是用的哪一晚的关系表（夜批正常时就是 ``night``）。"""

    lane: str
    night: date
    relation_night: date | None
    slot_minutes: int
    window_slots: int
    relations: tuple[RelationOpportunities, ...]
    kinds: tuple[KindDays, ...]
    #: 这一晚日历情境用的放假数据版本（裁定 59 第 4 条）；没接日历是 None。
    calendar_version: str | None = None
    #: 没见过的日型怎么办（``config.foresight.unseen_day_types``）："skip" 不外推，"official" 按官方含义归组。建预期、重标、结算重核都读它。
    unseen_day_types: str = "skip"
    #: 每天看着的比例与日型（升序、不重）；不在里面的日子按没在看算。
    watch: tuple[DayWatch, ...] = ()
    _watch: Mapping[date, DayWatch] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _text(self.lane, "table lane")
        if self.calendar_version is not None:
            _text(self.calendar_version, "table calendar_version")
        if self.unseen_day_types not in UNSEEN_DAY_TYPES:
            raise ForesightError(f"unseen_day_types must be one of {UNSEEN_DAY_TYPES}")
        _day(self.night, "table night")
        if self.relation_night is not None:
            _day(self.relation_night, "table relation_night")
        _positive(self.slot_minutes, "table slot_minutes")
        _positive(self.window_slots, "table window_slots")
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
        if not isinstance(self.watch, tuple) or any(not isinstance(item, DayWatch) for item in self.watch):
            raise ForesightError("table watch must be DayWatch records")
        watched = [item.day for item in self.watch]
        if watched != sorted(set(watched)):
            raise ForesightError("table watch days must be distinct and ascending")
        object.__setattr__(self, "_watch", {item.day: item for item in self.watch})

    def relation(self, key: str) -> RelationOpportunities | None:
        """按关系身份取一条（没有是 None）。"""

        return next((item for item in self.relations if item.key == key), None)

    def by_consequent(self, kind_token: str) -> tuple[RelationOpportunities, ...]:
        """以这个类为后果的全部关系（反向看前因用）。"""

        return tuple(item for item in self.relations if kind_token in item.consequent_classes)

    def kind(self, kind_token: str) -> KindDays | None:
        return next((item for item in self.kinds if item.kind_token == kind_token), None)

    def watched(self, day: date) -> float:
        """这一天有多大比例在看；表上没有的日子（还没定稿、在记录之前）是 0。"""

        item = self._watch.get(day)
        return 0.0 if item is None else item.share

    def day_type(self, day: date) -> str | None:
        item = self._watch.get(day)
        return None if item is None else item.day_type


__all__ = [
    "SEGMENTS",
    "STATUSES",
    "UNSEEN_DAY_TYPES",
    "ConditionSpec",
    "DayWatch",
    "KindDays",
    "Opportunity",
    "OpportunityTable",
    "RelationOpportunities",
    "TreeReading",
]
