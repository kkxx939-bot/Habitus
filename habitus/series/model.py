"""事件序列：行为树每晚被读成的那一份输入，语义树、``temporal`` 与预测层的机会表共用（语义树新方案 ``13`` 第四节）。

**为什么单独一个包**：读它的几处互不引用（架构测试钉死），可它们读的必须是**同一份**记录、同一套读法
——「待定」「非事件」、撞车重复、观测空白、被提醒过的记录，两处各写一遍就会写反（第二轮评审 D5：待定在旧预测树里
算删失、在语义树里算"没来"）。所以读法收在这里一次，读者都只认这里的值。

本模块是纯值、零依赖：读者 import 它不会连带 import 行为树或词表。读行为树的那一半在 ``series.reader``，
只有组合根用。

**截止日是类型的一部分**：第 N 晚的序列 ``cutoff = N``，只含已封口、早于第 N 天的记录。行为树的日期是每条记录
**自己的本地日期**（``occurred_on == started_at`` 的本地日），所以"第 N 天开始"那一刻也按每条记录自己的偏移算
（``boundary_for``）。跨过这一刻的窗口没看全，消费方记删失；观测空白按本地日切开，只留截止日之前那几天的段。

``until(day)`` 把一份序列截到更早的截止日；它与"直接按那个截止日读一遍"必须逐值相等（CI 钉住）——回放、回测
因此只读一次行为树，再一天一天往前截。
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum

from habitus.foundation.intervals import standing_pieces


class SeriesError(ValueError):
    """序列自相矛盾：记录晚于截止日、没排好序、并行对指向不在序列里的记录……"""


class GapKind(str, Enum):
    """观测空白的三种：没读懂（在看、模型读了给不出结论）、错误（在看、系统出了问题没读成，裁定 94 第 3 条）与未观测（没在看）。"""

    UNREADABLE = "没读懂"
    ERROR = "错误"
    UNOBSERVED = "未观测"

    @property
    def watched(self) -> bool:
        """这段空白期间我们在不在看。三种都进曝光扣减，证伪能力不同（``standing_parts``）：「错误」与「没读懂」一样可被作废。"""

        return self is not GapKind.UNOBSERVED


def standing_parts(gap: SeriesGap, starts_on: Callable[[date], Sequence[datetime]]) -> tuple[SeriesGap, ...]:
    """一段观测空白读进序列的样子：按本地日切开，「没读懂」的每一天那段被那一天读出的行为起点证伪就作废
    （规则与理由在 ``foundation.intervals.standing_pieces``，按天读树的视图、行为层的当日叙述用的是同一条）。

    - **在看（「没读懂」）**：它断言的正是"这段读不出行为"。那一天那段里若真读出了一条行为的起点（叫不出名的「待定」也算：
      那一刻确实看懂了在做一件事），这句断言就被证伪了，那一段作废。代价：作废之后那段里可能还有没读出来的行为，会被当成
      "没有"（真实数据上很小：DAY1 的 21 段空白全是「没读懂」、非零宽的 4 段共 137.8 秒，被证伪的 1 段宽 21.9 秒）。
    - **没在看（「未观测」）**：里面不可能读出行为。真出现了，是上游把"没在看"与"看见了"同时写进了树——这里不消解：
      那段照样算没在看（覆盖扣掉），那条记录照样进序列；目前没有哪一处把这种矛盾单独报出来。

    ``starts_on(某一天)`` 给那一天算数的记录的开始时刻（升序）。"""

    return tuple(
        SeriesGap(uri=gap.uri, started_at=begin, ended_at=end, kind=gap.kind)
        for begin, end in standing_pieces(gap.started_at, gap.ended_at, voidable=gap.watched, starts_on=starts_on)
    )


class SkipReason(str, Enum):
    """行为树上有、却不进序列的记录为什么不进。"""

    #: 撞车消歧的已知重复（``original_name`` 非空）：标记由归约层在写入时打好，这里只认标记、不判重。
    DUPLICATE = "duplicate"
    #: 词表的「非事件」：不是一件事，连先后顺序里也当它不存在。
    NOT_EVENT = "not_event"
    #: 词表的「续话」：接着之前某一条（追问、叫停、改口、补一句），与「非事件」同一种待遇；单列是为了看续话判得准不准。
    CONTINUED = "continued"


@dataclass(frozen=True)
class SkippedRecord:
    """没进序列的一条，只作可观测量（哪天、为什么），不参与任何计算；只说明那天有记录（``EventSeries.active_days``）。"""

    uri: str
    day: date
    reason: SkipReason


@dataclass(frozen=True)
class SeriesRecord:
    """行为树上的一条 occurrence，读成序列里的一项。

    ``classified`` 为真 = ``kind_token`` 是词编号；为假 = 「表外」或「待定」（一件没有归类的事：算人在场，没有概念，
    只在先后顺序里占位）。「非事件」「续话」不进序列，撞车消歧的重复也不进。``reminded`` 原样带着：要不要数进自然率是消费方的事。
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
    #: 谁发起的（行为树 ``prompted_by``，裁定 54）：system / host_agent / self；不知道是 None。
    prompted_by: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subjects", tuple(self.subjects))
        if any(not item for item in self.subjects):
            raise SeriesError("subjects must be non-empty names")
        for label in ("started_at", "last_observed_at"):
            value = getattr(self, label)
            if value.utcoffset() is None:
                raise SeriesError(f"{label} must carry its local offset")
        if self.started_at.date() != self.day:
            raise SeriesError("a record's day is the local date it started on")
        if self.last_observed_at < self.started_at:
            raise SeriesError("last_observed_at cannot precede started_at")
        for label in ("uri", "name", "kind_token", "lane"):
            value = getattr(self, label)
            if not value:
                raise SeriesError(f"{label} must be non-empty text")


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
            if value.utcoffset() is None:
                raise SeriesError(f"{label} must carry its local offset")
        if self.ended_at < self.started_at:
            raise SeriesError("a gap cannot end before it starts")

    @property
    def watched(self) -> bool:
        return self.kind.watched


@dataclass(frozen=True)
class EventSeries:
    """第 ``cutoff`` 晚的事件序列。

    - ``records`` 按（开始时刻, uri）升序，全部早于截止日；
    - ``gaps`` 按（开始时刻, uri）升序，每段落在一个本地日里（跨午夜的空白按天切成几段、同一个 uri），只有截止日之前的日子；
    - ``skipped`` 是没进序列的记录（重复、非事件），只作可观测量，按（日, uri）排好。
    """

    cutoff: date
    records: tuple[SeriesRecord, ...] = ()
    gaps: tuple[SeriesGap, ...] = ()
    skipped: tuple[SkippedRecord, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.cutoff, datetime):
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
        skipped = tuple(self.skipped)
        if any(item.day >= self.cutoff for item in skipped):
            raise SeriesError("a series only reports skips from before its cutoff")
        if [(item.day, item.uri) for item in skipped] != sorted((item.day, item.uri) for item in skipped):
            raise SeriesError("skipped records must be sorted by day and uri")
        object.__setattr__(self, "skipped", skipped)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "gaps", gaps)

    @property
    def last_day(self) -> date:
        """最后一个已封口的日子（截止日前一天）：衰减与"从最早到今天每一天都算机会"的基准日。"""

        return self.cutoff - timedelta(days=1)

    def boundary_for(self, moment: datetime) -> datetime:
        """截止日开始那一刻，按 ``moment`` 自己的偏移算（行为树的日期是记录自己的本地日期）。"""

        return datetime.combine(self.cutoff, time(), tzinfo=moment.tzinfo)

    def records_of(self, kind_tokens: Collection[str]) -> tuple[SeriesRecord, ...]:
        """几类合起来的时间线：编号在 ``kind_tokens`` 里（已转正的类编号）的记录按开始时刻排好。一个在用节点的记录范围是它自己
        加全部下级（裁定 101 第 8 条）。待定的不在任何一类里（裁定 59）。"""

        return tuple(record for record in self.records if record.classified and record.kind_token in kind_tokens)

    def on(self, day: date) -> tuple[SeriesRecord, ...]:
        return tuple(record for record in self.records if record.day == day)

    def active_days(self) -> frozenset[date]:
        """有记录的日子：进了序列的、没进的（「非事件」、撞车重复）都算。不在里面的日子一整天一条记录都没有，算那天没在看
        （裁定 89 第 2 条；两条 lane 同一个规矩，用户 2026-11-16：上游一点数据都没送来，就是上游没在看，不能猜他做了没做）。"""

        return frozenset(record.day for record in self.records) | frozenset(item.day for item in self.skipped)

    def skipped_on(self, day: date, reason: SkipReason) -> int:
        return sum(1 for item in self.skipped if item.day == day and item.reason is reason)

    def skipped_count(self, reason: SkipReason) -> int:
        return sum(1 for item in self.skipped if item.reason is reason)

    def gaps_on(self, day: date) -> tuple[SeriesGap, ...]:
        """落在这一天的空白段（跨午夜的空白前一天开始的那段，后一天那半段也在这里）。"""

        return tuple(gap for gap in self.gaps if gap.started_at.date() == day)

    def until(self, cutoff: date) -> EventSeries:
        """截到更早（或同一个）截止日：等于直接按那个截止日读一遍行为树。"""

        if cutoff > self.cutoff:
            raise SeriesError("a series cannot be extended past its own cutoff")
        # 空白已按天切开、每天那段只由那一天的记录判（``standing_parts``）：留下截止日之前那几天的段，就是直接读的样子
        return EventSeries(
            cutoff=cutoff,
            records=tuple(record for record in self.records if record.day < cutoff),
            gaps=tuple(gap for gap in self.gaps if gap.started_at.date() < cutoff),
            skipped=tuple(item for item in self.skipped if item.day < cutoff),
        )


def _order(item: SeriesRecord | SeriesGap) -> tuple[datetime, str]:
    return (item.started_at, item.uri)


#: 读行为树上 ``kind_token`` 的换算口（裁定 101 第 7 条）：调一次给出"此刻"的一份 编号 → 编号 的函数——删掉的节点的编号换成它
#: 最近的活着的上级，别的原样。行为树上的记录不改写，读的时候换；口只有一个，由组合根用词表状态装出来，交给每一个读者
#: （序列、预测层按天读树的索引、未封口那座桥）。读者在一次读开头取一份，整次读用同一份。不给就是"没有删掉的节点"。
KindResolution = Callable[[], Callable[[str], str]]


__all__ = [
    "KindResolution",
    "EventSeries",
    "GapKind",
    "SeriesError",
    "SeriesGap",
    "SeriesRecord",
    "SkipReason",
    "SkippedRecord",
    "standing_parts",
]
