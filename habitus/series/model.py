"""事件序列：行为树每晚被读成的那一份输入，预测树与语义树共用（语义树新方案 ``13`` 第四节）。

**为什么单独一个包**：预测树与语义树互不引用（架构测试钉死），可它们读的必须是**同一份**记录、同一套读法
——「待定」「非事件」、撞车重复、观测空白、被提醒过的记录，两处各写一遍就会写反（第二轮评审 D5：待定在预测树里
算删失、在语义树里算"没来"）。所以读法收在这里一次，两棵树都只认这里的值。

本模块是纯值、零依赖：两棵树 import 它不会连带 import 行为树或词表。读行为树的那一半在 ``series.reader``，
只有组合根用。

**截止日是类型的一部分**：第 N 晚的序列 ``cutoff = N``，只含已封口、早于第 N 天的记录。行为树的日期是每条记录
**自己的本地日期**（``occurred_on == started_at`` 的本地日），所以"第 N 天开始"那一刻也按每条记录自己的偏移算
（``boundary_for``）。跨过这一刻的窗口没看全，消费方记删失；跨过这一刻的观测空白截到这一刻。

``until(day)`` 把一份序列截到更早的截止日；它与"直接按那个截止日读一遍"必须逐值相等（CI 钉住）——回放、回测
因此只读一次行为树，再一天一天往前截。
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum


class SeriesError(ValueError):
    """序列自相矛盾：记录晚于截止日、没排好序、并行对指向不在序列里的记录……"""


class GapKind(str, Enum):
    """观测空白的两种：没读懂（在看、但没看明白）与未观测（没在看）。"""

    UNREADABLE = "没读懂"
    UNOBSERVED = "未观测"

    @property
    def watched(self) -> bool:
        """这段空白期间我们在不在看。两种都进曝光扣减，证伪能力不同（``disproves``）。"""

        return self is GapKind.UNREADABLE


def disproves(kind: GapKind, started_at: datetime, ended_at: datetime, starts: Sequence[datetime]) -> bool:
    """这段空白是不是被证伪了（证伪了就整段作废）。**一条记录一种读法的落点**：序列读空白时用它，预测层的证据卡按天读树时也用它，
    不各写一遍（第四轮评审 E2：预测树、证据卡作废，关系时间线不作废，"同一条链"丢掉的恰好是后果出现的那一次）。

    - **在看（「没读懂」）**：它断言的正是"这段读不出行为"。这段里若真读出了一条行为的起点（叫不出名的「待定」也算：那一刻确实
      看懂了在做一件事），这句断言就被证伪了——**整段作废**。作废是唯一不需要编造宽度的规则：只挖掉"那一瞬"是零测度、等于没挖；
      挖"可见跨度"要最后所见，而瞬时行为的跨度仍然是零。代价：作废之后那段里可能还有没读出来的行为，会被当成"没有"
      （真实数据上很小：DAY1 的 21 段空白全是「没读懂」、非零宽的 4 段共 137.8 秒，被证伪的 1 段宽 21.9 秒）。
    - **没在看（「未观测」）**：里面不可能读出行为。真出现了，是上游把"没在看"与"看见了"同时写进了树——这里**不消解**，
      让消费方（预测树累计曝光时）以明确的矛盾报出来。

    ``starts`` 是这段空白可能盖住的那些记录的开始时刻（升序）。"""

    if not kind.watched:
        return False
    begin, end = started_at.timestamp(), ended_at.timestamp()
    stamps = [moment.timestamp() for moment in starts]
    index = bisect_left(stamps, begin)
    return index < len(stamps) and stamps[index] < end


class SkipReason(str, Enum):
    """行为树上有、却不进序列的记录为什么不进。"""

    #: 撞车消歧的已知重复（``original_name`` 非空）：标记由归约层在写入时打好，这里只认标记、不判重。
    DUPLICATE = "duplicate"
    #: 词表的「非事件」：不是一件事，连先后顺序里也当它不存在。
    NOT_EVENT = "not_event"


@dataclass(frozen=True)
class SkippedRecord:
    """没进序列的一条，只作可观测量（哪天、为什么），不参与任何计算。"""

    uri: str
    day: date
    reason: SkipReason


@dataclass(frozen=True)
class SeriesRecord:
    """行为树上的一条 occurrence，读成序列里的一项。

    ``classified`` 为真 = ``kind_token`` 是类编号；为假 = 「待定」（叫不出名：不是候选，只在先后顺序里占位）。
    「非事件」不进序列，撞车消歧的重复也不进。``reminded`` 原样带着：要不要数进自然率是消费方的事。
    """

    uri: str
    #: 融合层原话（行为树的 ``name`` 字段），不是类名。
    name: str
    day: date
    started_at: datetime
    last_observed_at: datetime
    kind_token: str
    lane: str
    classified: bool
    reminded: bool
    summary: str
    goal: str | None = None
    #: 地点与同在的人（调节条件的结构字段要看；上游没给时为空）。
    place: str | None = None
    subjects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subjects", tuple(self.subjects))
        if any(not isinstance(item, str) or not item for item in self.subjects):
            raise SeriesError("subjects must be non-empty names")
        for label in ("started_at", "last_observed_at"):
            value = getattr(self, label)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise SeriesError(f"{label} must carry its local offset")
        if self.started_at.date() != self.day:
            raise SeriesError("a record's day is the local date it started on")
        if self.last_observed_at < self.started_at:
            raise SeriesError("last_observed_at cannot precede started_at")
        for label in ("uri", "name", "kind_token", "lane"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise SeriesError(f"{label} must be non-empty text")
        if not isinstance(self.classified, bool) or not isinstance(self.reminded, bool):
            raise SeriesError("classified and reminded must be booleans")


@dataclass(frozen=True)
class SeriesGap:
    """一段观测空白。没有 lane：哪条 lane 都看不见这段。"""

    uri: str
    started_at: datetime
    ended_at: datetime
    kind: GapKind

    def __post_init__(self) -> None:
        for label in ("started_at", "ended_at"):
            value = getattr(self, label)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise SeriesError(f"{label} must carry its local offset")
        if self.ended_at < self.started_at:
            raise SeriesError("a gap cannot end before it starts")
        if not isinstance(self.kind, GapKind):
            raise SeriesError("kind must be a GapKind")

    @property
    def watched(self) -> bool:
        return self.kind.watched


@dataclass(frozen=True)
class EventSeries:
    """第 ``cutoff`` 晚的事件序列。

    - ``records`` 按（开始时刻, uri）升序，全部早于截止日；
    - ``gaps`` 按（开始时刻, uri）升序，结束不晚于各自的截止时刻（跨过的已截断）；
    - ``concurrent`` 是行为树声明的并行对（``concurrent_with``），两端都在 ``records`` 里，按 uri 排好；
    - ``skipped`` 是没进序列的记录（重复、非事件），只作可观测量，按（日, uri）排好。
    """

    cutoff: date
    records: tuple[SeriesRecord, ...] = ()
    gaps: tuple[SeriesGap, ...] = ()
    concurrent: tuple[tuple[str, str], ...] = ()
    skipped: tuple[SkippedRecord, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.cutoff, datetime) or not isinstance(self.cutoff, date):
            raise SeriesError("cutoff must be a date")
        records, gaps = tuple(self.records), tuple(self.gaps)
        if any(record.day >= self.cutoff for record in records):
            raise SeriesError("a series only holds records from before its cutoff")
        if any(gap.ended_at > self.boundary_for(gap.started_at) for gap in gaps):
            raise SeriesError("a series only holds gaps that end by its cutoff")
        if [_order(item) for item in records] != sorted(_order(item) for item in records):
            raise SeriesError("records must be sorted by start time")
        if [_order(item) for item in gaps] != sorted(_order(item) for item in gaps):
            raise SeriesError("gaps must be sorted by start time")
        uris = {record.uri for record in records}
        if len(uris) != len(records):
            raise SeriesError("a record appears twice")
        pairs = tuple(self.concurrent)
        if any(left >= right or left not in uris or right not in uris for left, right in pairs):
            raise SeriesError("concurrent pairs are ordered uri pairs of records in the series")
        if list(pairs) != sorted(set(pairs)):
            raise SeriesError("concurrent pairs must be sorted and distinct")
        skipped = tuple(self.skipped)
        if any(item.day >= self.cutoff for item in skipped):
            raise SeriesError("a series only reports skips from before its cutoff")
        if [(item.day, item.uri) for item in skipped] != sorted((item.day, item.uri) for item in skipped):
            raise SeriesError("skipped records must be sorted by day and uri")
        object.__setattr__(self, "skipped", skipped)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "concurrent", pairs)

    @property
    def last_day(self) -> date:
        """最后一个已封口的日子（截止日前一天）：衰减与"从最早到今天每一天都算机会"的基准日。"""

        return self.cutoff - timedelta(days=1)

    def boundary_for(self, moment: datetime) -> datetime:
        """截止日开始那一刻，按 ``moment`` 自己的偏移算（行为树的日期是记录自己的本地日期）。"""

        return datetime.combine(self.cutoff, time(), tzinfo=moment.tzinfo)

    def on(self, day: date) -> tuple[SeriesRecord, ...]:
        return tuple(record for record in self.records if record.day == day)

    def skipped_on(self, day: date, reason: SkipReason) -> int:
        return sum(1 for item in self.skipped if item.day == day and item.reason is reason)

    def skipped_count(self, reason: SkipReason) -> int:
        return sum(1 for item in self.skipped if item.reason is reason)

    def gaps_on(self, day: date) -> tuple[SeriesGap, ...]:
        """开始在这一天的空白（行为树按开始日存空白）。"""

        return tuple(gap for gap in self.gaps if gap.started_at.date() == day)

    def until(self, cutoff: date) -> EventSeries:
        """截到更早（或同一个）截止日：等于直接按那个截止日读一遍行为树。"""

        if cutoff > self.cutoff:
            raise SeriesError("a series cannot be extended past its own cutoff")
        earlier = EventSeries(cutoff=cutoff)
        records = tuple(record for record in self.records if record.day < cutoff)
        kept = {record.uri for record in records}
        return EventSeries(
            cutoff=cutoff,
            records=records,
            gaps=tuple(
                clip_gap(gap, earlier.boundary_for(gap.started_at))
                for gap in self.gaps
                if gap.started_at.date() < cutoff
            ),
            concurrent=tuple(pair for pair in self.concurrent if pair[0] in kept and pair[1] in kept),
            skipped=tuple(item for item in self.skipped if item.day < cutoff),
        )


def clip_gap(gap: SeriesGap, boundary: datetime) -> SeriesGap:
    """跨过截止时刻的空白截到那一刻（读行为树与 ``until`` 共用这一条）。"""

    if gap.ended_at <= boundary:
        return gap
    return SeriesGap(uri=gap.uri, started_at=gap.started_at, ended_at=max(gap.started_at, boundary), kind=gap.kind)


def _order(item: SeriesRecord | SeriesGap) -> tuple[datetime, str]:
    return (item.started_at, item.uri)


__all__ = [
    "EventSeries",
    "GapKind",
    "SeriesError",
    "SeriesGap",
    "SeriesRecord",
    "SkipReason",
    "SkippedRecord",
    "clip_gap",
    "disproves",
]
