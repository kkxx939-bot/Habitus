"""按日读一次、多次查：一天的 occurrence（按行）、观测空白。

一个 ``DayIndexCache`` 只服务**一次查询**：在它的生命周期里同一天只读一次，不论那天有没有归组——
"今天必须新鲜"靠每次查询新建缓存保证，"历史不会变"靠已归组的日子本来就不再改保证。跨查询复用
同一个缓存会读到过期的今天，所以不要那样做。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.model import BehaviorKind
from habitus.behavior.schema.vocabulary import GAP_KINDS
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.conditions import DayTypeCalendar, LocalCalendar
from habitus.scene.views.model import FlowRow, ObservationGap
from habitus.series.model import GapKind, SeriesGap, standing_parts

#: 空白文档只落在**起始日**目录，而一段"未观测"可以横跨好几天（出差一周）。读某一天的空白时往前
#: 回看这么多天的起始目录，把延续到本日的裁进来；比这更长的空白从它中间那些天看不见。
GAP_LOOKBACK_DAYS = 31


class DayIndex:
    """一天的只读索引。行为侧跳过撞车消歧的重复（与预测夜批同口径）；情景侧只读当前一代。

    ``day_note`` 是当地日历对这一天的说法，读时算、不落盘（见 ``conditions.calendar``）。

    ``gaps`` 是落在这一天里的观测空白：更早开始、延续到这一天的空白也读进来、只取这一天那段（回看 ``GAP_LOOKBACK_DAYS``
    天的起始目录）；"没读懂"在这一天那段里若读出了这一天一条行为的开始则那段作废（与事件序列同一个读法
    ``series.model.standing_parts``——我们在看、只是没读懂，读出来了就证伪了）；零宽度的不进（判"窗里有没有空白"用不上）。

    ``rows`` 是这一天的全部行按时刻排成的 ``FlowRow``：读侧要"一条行为是什么、几点、说了什么"都从这里取，
    行为文档的字段名不出本模块。
    """

    def __init__(
        self,
        behavior_tree: BehaviorTree,
        day: date,
        *,
        subject: str,
        admits: Callable[[BehaviorDocument], bool],
        day_note: str | None = None,
    ) -> None:
        """``admits``：一条 occurrence 算不算数——组合根注入事件序列的那一条规则（``series.reader.admitted``：
        撞车重复、「非事件」都不算），读侧不自己再写一遍。"""

        self.day = day
        self.subject = subject
        # 当地日历对这一天的一句话（补班、节假日……）；没有日历数据时恒为空。由缓存按注入的
        # 日历算好传进来，索引自己不认识日历——它只是把这个事实带给视图。
        self.day_note = day_note
        self.occurrences: dict[str, BehaviorDocument] = {}
        documents = behavior_tree.read_day(BehaviorKind.OCCURRENCE, day)
        #: 这一天一条记录都没有（算不算数的都没有）：这棵树那条 lane 那天没来过东西（会话 lane 就是没开会话，裁定 89 第 2 条）。
        self.silent = not documents
        for document in documents:
            if not admits(document):
                continue
            self.occurrences[str(BehaviorURI.from_address(document.address))] = document
        self.ordered: tuple[str, ...] = tuple(
            sorted(self.occurrences, key=lambda uri: (self.occurrences[uri].address.started_at.astimezone(UTC), uri))
        )
        counts = Counter(str(self.occurrences[uri].fields["kind_token"]) for uri in self.ordered)
        self.rows: tuple[FlowRow, ...] = tuple(self._row(uri, counts) for uri in self.ordered)
        self.by_uri: Mapping[str, FlowRow] = MappingProxyType({row.uri: row for row in self.rows})
        self.gaps: tuple[ObservationGap, ...] = self._read_gaps(behavior_tree)

    def _row(self, uri: str, counts: Mapping[str, int]) -> FlowRow:
        document = self.occurrences[uri]
        fields = document.fields
        kind_token = str(fields["kind_token"])
        place = fields.get("place")
        subjects = fields.get("subjects") or ()
        return FlowRow(
            uri=uri,
            name=str(fields["name"]),
            kind_token=kind_token,
            at=document.address.started_at,
            last_observed_at=datetime.fromisoformat(str(fields["last_observed_at"])),
            summary=str(fields["summary"]),
            day_count=counts[kind_token],
            status=str(fields["status"]),
            prompted_by=str(fields["prompted_by"]) if fields.get("prompted_by") is not None else None,
            place=place if isinstance(place, str) and place else None,
            subjects=tuple(str(item) for item in subjects if isinstance(item, str) and item),
        )

    def _read_gaps(self, behavior_tree: BehaviorTree) -> tuple[ObservationGap, ...]:
        starts = sorted(self.occurrences[uri].address.started_at for uri in self.ordered)
        found: list[ObservationGap] = []
        for offset in range(GAP_LOOKBACK_DAYS, -1, -1):
            for document in behavior_tree.read_day(BehaviorKind.GAP, self.day - timedelta(days=offset)):
                kind = str(document.fields["gap_kind"])
                if kind not in GAP_KINDS:
                    raise ValueError(f"unknown gap kind on the behaviour tree: {kind!r}")
                # 与事件序列同一个读法（``series.model.standing_parts``）：按天切开，这一天那段只由这一天的行为判
                gap = SeriesGap(
                    uri=str(BehaviorURI.from_address(document.address)),
                    started_at=document.address.started_at,
                    ended_at=datetime.fromisoformat(str(document.fields["ended_at"])),
                    kind=GapKind(kind),
                )
                found.extend(
                    ObservationGap(started_at=part.started_at, ended_at=part.ended_at, kind=kind)
                    for part in standing_parts(gap, lambda day: starts if day == self.day else ())
                    if part.started_at.date() == self.day and part.ended_at > part.started_at
                )
        return tuple(
            sorted(found, key=lambda gap: (gap.started_at.astimezone(UTC), gap.ended_at.astimezone(UTC), gap.kind))
        )


class DayIndexCache:
    """一次查询里同一天只读一次。

    ``calendar`` 决定每一天的 ``day_note``（见 ``conditions.calendar``）；不给就是名义日历，
    即"没有当地日历数据"这个显式的零修正。
    """

    def __init__(
        self,
        behavior_tree: BehaviorTree,
        *,
        subject: str,
        admits: Callable[[BehaviorDocument], bool],
        calendar: DayTypeCalendar | None = None,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if not callable(admits):
            raise TypeError("admits must be callable")
        self.admits = admits
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("subject must be non-empty text")
        self.behavior_tree = behavior_tree
        self.subject = subject
        self.calendar: DayTypeCalendar = LocalCalendar() if calendar is None else calendar
        self._days: dict[date, DayIndex] = {}

    def day(self, day: date) -> DayIndex:
        cached = self._days.get(day)
        if cached is None:
            cached = self._days[day] = DayIndex(
                self.behavior_tree, day, subject=self.subject, admits=self.admits, day_note=self._note(day)
            )
        return cached

    def has_gap(self, start: datetime, end: datetime, *, kinds: frozenset[str] | None = None) -> bool:
        """``start`` → ``end`` 这一段里有没有观测空白（被那天的行为作废的「没读懂」已经去掉）：那段没看到，"没做"就说不准。
        结算判"窗里有空白记中性"两种都认；递送判"窗里有空白不说"只认「没读懂」（``kinds``，裁定 68 第 2 条）。"""

        return bool(self.gaps_within(start, end, kinds=kinds))

    def silent_within(self, start: datetime, end: datetime) -> bool:
        """``start`` → ``end`` 这一段碰到的日子里有没有一整天一条记录都没有的（``DayIndex.silent``）。会不会因此算"没在看"
        是 lane 的口径（``series.model.silent_day_unwatched``），由调用方判。"""

        if end <= start:
            return False
        day, last = start.date(), (end - timedelta(microseconds=1)).date()
        while day <= last:
            if self.day(day).silent:
                return True
            day += timedelta(days=1)
        return False

    def gaps_within(
        self, start: datetime, end: datetime, *, kinds: frozenset[str] | None = None
    ) -> tuple[ObservationGap, ...]:
        begin, finish = start.astimezone(UTC), end.astimezone(UTC)
        found: list[ObservationGap] = []
        day = start.date()
        while day <= end.date():
            found.extend(
                gap
                for gap in self.day(day).gaps
                if (kinds is None or gap.kind in kinds)
                and gap.started_at.astimezone(UTC) < finish
                and gap.ended_at.astimezone(UTC) > begin
            )
            day += timedelta(days=1)
        return tuple(found)

    def _note(self, day: date) -> str | None:
        """问一次当地日历。空白串当作"没话说"——判断者读到一个空的注记比读不到更糟。"""

        note = self.calendar.describe(day)
        if note is None:
            return None
        if not isinstance(note, str):
            raise TypeError("a calendar note must be text or None")
        return note.strip() or None


__all__ = ["GAP_LOOKBACK_DAYS", "DayIndex", "DayIndexCache"]
