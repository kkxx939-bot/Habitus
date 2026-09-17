"""按日读一次、多次查：一天的 occurrence（按行）、它们的短程关系、观测空白。

一个 ``DayIndexCache`` 只服务**一次查询**：在它的生命周期里同一天只读一次，不论那天有没有归组——
"今天必须新鲜"靠每次查询新建缓存保证，"历史不会变"靠已归组的日子本来就不再改保证。跨查询复用
同一个缓存会读到过期的今天，所以不要那样做。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.document.link import BehaviorLinkType
from habitus.behavior.model import BehaviorKind
from habitus.behavior.schema.vocabulary import GAP_KINDS
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.scene.calendar import DayTypeCalendar, NominalCalendar
from habitus.scene.views.model import ActionRef, FlowRow, ObservationGap

_WATCHED_GAP_KIND = "没读懂"
#: 空白文档只落在**起始日**目录，而一段"未观测"可以横跨好几天（出差一周）。读某一天的空白时往前
#: 回看这么多天的起始目录，把延续到本日的裁进来；比这更长的空白从它中间那些天看不见。
GAP_LOOKBACK_DAYS = 31


class DayIndex:
    """一天的只读索引。行为侧跳过撞车消歧的重复（与预测夜批同口径）；情景侧只读当前一代。

    ``day_note`` 是当地日历对这一天的说法，读时算、不落盘（见 ``scene.calendar``）。

    ``gaps`` 是落在这一天里的观测空白，与预测树的曝光分母同两步归一化：更早开始、延续到这一天的
    空白也读进来并裁到本日（树的 ``group_gaps_by_day``；回看 ``GAP_LOOKBACK_DAYS`` 天的起始目录）；
    零宽度的丢弃、"没读懂"段里若读出了一条行为的开始则整段作废（树的 ``reconcile_gaps``——我们在看、
    只是没读懂，读出来了就证伪了）。

    ``rows`` 是这一天的全部行按时刻排成的 ``FlowRow``：读侧要"一条行为是什么、几点、说了什么"都从这里取，
    行为文档的字段名不出本模块。
    """

    def __init__(
        self, behavior_tree: BehaviorTree, day: date, *, subject: str, day_note: str | None = None
    ) -> None:
        self.day = day
        self.subject = subject
        # 当地日历对这一天的一句话（补班、节假日……）；没有日历数据时恒为空。由缓存按注入的
        # 日历算好传进来，索引自己不认识日历——它只是把这个事实带给视图。
        self.day_note = day_note
        self.occurrences: dict[str, BehaviorDocument] = {}
        for document in behavior_tree.read_day(BehaviorKind.OCCURRENCE, day):
            if document.fields.get("original_name") is not None:
                continue
            self.occurrences[str(BehaviorURI.from_address(document.address))] = document
        self.ordered: tuple[str, ...] = tuple(
            sorted(self.occurrences, key=lambda uri: (self.occurrences[uri].address.started_at.astimezone(UTC), uri))
        )
        counts = Counter(str(self.occurrences[uri].fields["kind_token"]) for uri in self.ordered)
        self.rows: tuple[FlowRow, ...] = tuple(self._row(uri, counts) for uri in self.ordered)
        self.by_uri: Mapping[str, FlowRow] = MappingProxyType({row.uri: row for row in self.rows})
        self.kind_counts: Mapping[str, int] = MappingProxyType(dict(sorted(counts.items())))
        self.gaps: tuple[ObservationGap, ...] = self._read_gaps(behavior_tree)
        # 行为树的短程关系（只存前向、目标可能在相邻的一天）：按类型分开留原始目标 URI，投影时经 cache
        # 解析并对 concurrent_with 取对称闭包
        self.concurrent_targets: dict[str, tuple[str, ...]] = {}
        self.results_from_targets: dict[str, tuple[str, ...]] = {}
        for uri, document in self.occurrences.items():
            concurrent = [str(link.to_uri) for link in document.links if link.link_type is BehaviorLinkType.CONCURRENT_WITH]
            results = [str(link.to_uri) for link in document.links if link.link_type is BehaviorLinkType.RESULTS_FROM]
            self.concurrent_targets[uri] = tuple(concurrent)
            self.results_from_targets[uri] = tuple(results)

    def ref(self, uri: str) -> ActionRef:
        document = self.occurrences[uri]
        return ActionRef(uri=uri, name=str(document.fields["name"]), kind_token=str(document.fields["kind_token"]))

    def _row(self, uri: str, counts: Mapping[str, int]) -> FlowRow:
        document = self.occurrences[uri]
        fields = document.fields
        kind_token = str(fields["kind_token"])
        return FlowRow(
            uri=uri,
            name=str(fields["name"]),
            kind_token=kind_token,
            at=document.address.started_at,
            last_observed_at=datetime.fromisoformat(str(fields["last_observed_at"])),
            summary=str(fields["summary"]),
            day_count=counts[kind_token],
        )

    def others(self, uri: str) -> tuple[str, ...]:
        """"和谁"：subjects 里主体之外的人（主体总在里面，留着它这一槽就永远对上）。"""

        return tuple(str(item) for item in self.occurrences[uri].fields["subjects"] if str(item) != self.subject)

    def _read_gaps(self, behavior_tree: BehaviorTree) -> tuple[ObservationGap, ...]:
        starts = sorted(self.occurrences[uri].address.started_at.astimezone(UTC) for uri in self.ordered)
        clipped: list[ObservationGap] = []
        for offset in range(GAP_LOOKBACK_DAYS, -1, -1):
            for document in behavior_tree.read_day(BehaviorKind.GAP, self.day - timedelta(days=offset)):
                kind = str(document.fields["gap_kind"])
                if kind not in GAP_KINDS:
                    raise ValueError(f"unknown gap kind on the behaviour tree: {kind!r}")
                gap = _clamp_to_day(document.address.started_at, datetime.fromisoformat(str(document.fields["ended_at"])), self.day)
                if gap is None:
                    continue
                begin, end = gap
                if kind == _WATCHED_GAP_KIND and any(begin.astimezone(UTC) <= start < end.astimezone(UTC) for start in starts):
                    continue
                clipped.append(ObservationGap(started_at=begin, ended_at=end, kind=kind))
        return tuple(sorted(clipped, key=lambda gap: (gap.started_at.astimezone(UTC), gap.ended_at.astimezone(UTC), gap.kind)))


def _clamp_to_day(started_at: datetime, ended_at: datetime, day: date) -> tuple[datetime, datetime] | None:
    """把一段空白裁到 ``day`` 的本地一天里（按空白自己的偏移算日界）；裁完为零宽即不落在这一天。"""

    start_of_day = datetime.combine(day, datetime.min.time(), tzinfo=started_at.tzinfo)
    end_of_day = start_of_day + timedelta(days=1)
    begin = max(started_at, start_of_day)
    end = min(ended_at, end_of_day)
    if end <= begin:
        return None
    return begin, end


class DayIndexCache:
    """一次查询里同一天只读一次。

    ``calendar`` 决定每一天的 ``day_note``（见 ``scene.calendar``）；不给就是名义日历，
    即"没有当地日历数据"这个显式的零修正。

    这里原来还答"这一天归过组没有"。那是按天归组的概念，归组删掉之后它换成了"这个候选这一天
    关联完成了没有"，由组合根从规律级树取事实注入（见 ``foresight.assemble.AssociatedDays``）
    ——按候选比按天准一级：同一天可能这个候选做完了、那个还没做。
    """

    def __init__(
        self,
        behavior_tree: BehaviorTree,
        *,
        subject: str,
        calendar: DayTypeCalendar | None = None,
    ) -> None:
        if not isinstance(behavior_tree, BehaviorTree):
            raise TypeError("behavior_tree must be a BehaviorTree")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("subject must be non-empty text")
        self.behavior_tree = behavior_tree
        self.subject = subject
        self.calendar: DayTypeCalendar = NominalCalendar() if calendar is None else calendar
        self._days: dict[date, DayIndex] = {}

    def day(self, day: date) -> DayIndex:
        cached = self._days.get(day)
        if cached is None:
            cached = self._days[day] = DayIndex(
                self.behavior_tree, day, subject=self.subject, day_note=self._note(day)
            )
        return cached

    def _note(self, day: date) -> str | None:
        """问一次当地日历。空白串当作"没话说"——判断者读到一个空的注记比读不到更糟。"""

        note = self.calendar.describe(day)
        if note is None:
            return None
        if not isinstance(note, str):
            raise TypeError("a calendar note must be text or None")
        return note.strip() or None





__all__ = ["GAP_LOOKBACK_DAYS", "DayIndex", "DayIndexCache"]
