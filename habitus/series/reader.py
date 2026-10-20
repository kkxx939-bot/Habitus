"""把行为树读成第 N 晚的事件序列。行为树的读法**只写在这里**，语义树、``temporal`` 与机会表都不再自己读、自己判。

只有组合根调用本模块（架构测试钉死）：它 import 行为树与词表的编号规则，读者只 import 纯值的 ``series.model``。

读法：

- 只读不写；只读**截止日之前**的日子（第 N 晚读 N 之前，已封口）；
- ``original_name`` 非空的 occurrence 是撞车消歧的已知重复，**机械跳过**——标记在写入时由归约层打好，这里只认标记、
  不做判重（判重只在融合层解决）；
- ``kind_token`` 在这里分流（裁定 18、19）：类编号进序列（``classified``）；「待定」进序列但不是类（只占先后顺序）；
  「非事件」不进序列，连顺序里也当它不存在；既不是编号也不是占位标记的硬拒；lane 取编号前缀；
- 观测空白两种都进（``GapKind``），跨过截止时刻的截到那一刻；空白的类型在这里翻译一次，上游改名时以未知取值硬失败；
- ``concurrent_with`` 链接原样读成无序的 uri 对，只留两端都在序列里的（指向被跳过的一端，那条并行没有两个可用端点）；
- ``reminded`` 原样带着：被提醒过的发生要不要算进自然率，是消费方的事（``temporal`` 算它做了、但不是自然发生，见 ``temporal.episodes``）。

两条 lane 各是一整套（裁定 35）：行为树、语义树各有会话、物理两支，各读各的，这里读到的一棵行为树里只有一条
lane 的记录与空白。序列上的 ``lane`` 字段（取自编号前缀）与下游按 lane 分组的逻辑因此总是只有一组——留着不影响结果，
要删是动行为树与语义树的事。

TODO(BHV-SESSION-LANE-001)：会话 lane 一轮一条之后（裁定 32），同一类相邻两条常常只隔几分钟（三天实测全部轮间隔中位
3.7 分钟）。读法不在这里改：``temporal`` 把同一类开始时刻隔不到 45 分钟的连着并成一次、间隔按日历日量（裁定 59），
关系检验的"同一条链"同一把尺；会话 lane 的切分本身归词表与融合。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

from habitus.behavior.document import BehaviorDocument, BehaviorLinkType
from habitus.behavior.kinds.ids import KindIdError, Lane, lane_of_token, not_event_token, pending_token
from habitus.behavior.schema.vocabulary import PROMPTERS
from habitus.behavior.tree import BehaviorKind, BehaviorTree
from habitus.behavior.tree.dirty_days import DirtyDayLog
from habitus.behavior.uri import BehaviorURI
from habitus.series.model import (
    EventSeries,
    GapKind,
    SeriesError,
    SeriesGap,
    SeriesRecord,
    SkippedRecord,
    SkipReason,
    clip_gap,
    disproves,
)


class SeriesReader:
    """按天缓存行为树文档，每晚在内存里重新拼整份序列（26 号清单 6.7）。

    序列不能按天拆开增量算：空白按截止时刻截断、证伪看全部记录的起点、并行关系跨天——所以缓存的是**解码好的文档**，拼装
    每晚全量重做。哪些天要重读看归约记下的脏日日志（``behavior.tree.dirty_days``）：读到第几号记在这里，读完截掉；
    第一次读（缓存是空的）先记下日志读到哪、再读全部日子，期间新记的日子下一次照常重读。新出现的日子（树上有、缓存里没有）
    当场读。"""

    def __init__(self, tree: BehaviorTree, logs: Sequence[DirtyDayLog] = ()) -> None:
        if not isinstance(tree, BehaviorTree):
            raise SeriesError("tree must be a BehaviorTree")
        self.tree = tree
        self.logs = tuple(logs)
        self._days: dict[tuple[BehaviorKind, date], tuple[BehaviorDocument, ...]] = {}
        self._read_through: list[int] | None = None

    def read(self, *, cutoff: date) -> EventSeries:
        """第 ``cutoff`` 晚的序列：截止日之前每一天的 occurrence 与空白。"""

        if isinstance(cutoff, datetime) or not isinstance(cutoff, date):
            raise SeriesError("cutoff must be a date")
        if not self.logs:
            # 没有脏日日志就不知道哪天变过：不缓存，每次照读（缓存了只会把改过的日子一直读成旧的）
            self._days.clear()
        self._forget_dirty_days()
        return _assemble(
            self._documents(BehaviorKind.OCCURRENCE, cutoff), self._documents(BehaviorKind.GAP, cutoff), cutoff
        )

    def _forget_dirty_days(self) -> None:
        if self._read_through is None:
            self._read_through = [log.last() for log in self.logs]
            return
        for index, log in enumerate(self.logs):
            last, days = log.after(self._read_through[index])
            for day in days:
                for kind in (BehaviorKind.OCCURRENCE, BehaviorKind.GAP):
                    self._days.pop((kind, day), None)
            self._read_through[index] = last
            log.truncate_through(last)

    def _documents(self, kind: BehaviorKind, cutoff: date) -> list[BehaviorDocument]:
        found: list[BehaviorDocument] = []
        for day in self.tree.days(kind):
            if day >= cutoff:
                continue
            key = (kind, day)
            if key not in self._days:
                self._days[key] = self.tree.read_day(kind, day)
            found.extend(self._days[key])
        return found


def read_series(tree: BehaviorTree, *, cutoff: date) -> EventSeries:
    """第 ``cutoff`` 晚的序列，读一次不缓存（夜批用常驻的 ``SeriesReader``）。"""

    return SeriesReader(tree).read(cutoff=cutoff)


def _assemble(
    occurrences: Sequence[BehaviorDocument], gap_documents: Sequence[BehaviorDocument], cutoff: date
) -> EventSeries:
    boundary = EventSeries(cutoff=cutoff)
    records: list[SeriesRecord] = []
    skipped: list[SkippedRecord] = []
    links: list[tuple[str, str]] = []
    for document in occurrences:
        uri = str(BehaviorURI.from_address(document.address))
        day = document.address.occurred_on
        reason = skip_reason(document)
        if reason is not None:
            skipped.append(SkippedRecord(uri, day, reason))
            continue
        token = _text(document.fields.get("kind_token"), "kind_token")
        records.append(_record(document, uri, token, _lane(token)))
        links.extend(
            (uri, str(link.to_uri)) for link in document.links if link.link_type is BehaviorLinkType.CONCURRENT_WITH
        )
    present = {record.uri for record in records}
    # 空白在这里消解一次（``disproves``）：之后 temporal、关系检验读到的是同一份
    starts = sorted(record.started_at for record in records)
    gaps = [
        gap
        for gap in (
            clip_gap(_gap(document), boundary.boundary_for(document.address.started_at)) for document in gap_documents
        )
        if not disproves(gap.kind, gap.started_at, gap.ended_at, starts)
    ]
    return EventSeries(
        cutoff=cutoff,
        records=tuple(sorted(records, key=lambda item: (item.started_at, item.uri))),
        gaps=tuple(sorted(gaps, key=lambda item: (item.started_at, item.uri))),
        concurrent=tuple(
            sorted({_unordered(left, right) for left, right in links if left != right and {left, right} <= present})
        ),
        skipped=tuple(sorted(skipped, key=lambda item: (item.day, item.uri))),
    )


def skip_reason(document: BehaviorDocument) -> SkipReason | None:
    """一条 occurrence 为什么不进序列（不进是 None）：撞车消歧的已知重复、「非事件」。

    这是行为树上一条记录**算不算数**的唯一规则。读今天还没封口那段的读侧（预测层的证据卡）也用它——组合根把它注入
    ``scene.views.DayIndexCache``，不各写一遍（第二轮评审 D5 就是两处各写一遍写反了）。
    """

    if document.fields.get("original_name") is not None:
        return SkipReason.DUPLICATE
    token = _text(document.fields.get("kind_token"), "kind_token")
    return SkipReason.NOT_EVENT if token == not_event_token(_lane(token)) else None


def admitted(document: BehaviorDocument) -> bool:
    """这条 occurrence 算不算数（``skip_reason`` 为 None）。"""

    return skip_reason(document) is None


def _record(document: BehaviorDocument, uri: str, token: str, lane: Lane) -> SeriesRecord:
    fields = document.fields
    goal, place = fields.get("goal"), fields.get("place")
    subjects = fields.get("subjects") or ()
    started_at = _local(fields.get("started_at"), "started_at")
    return SeriesRecord(
        uri=uri,
        name=_text(fields.get("name"), "name"),
        day=document.address.occurred_on,
        started_at=started_at,
        # 统一到开始时刻的偏移：同一条记录的两个时刻用同一种写法，读出来的摘要才不随写入方的偏移变
        last_observed_at=_local(fields.get("last_observed_at"), "last_observed_at").astimezone(started_at.tzinfo),
        kind_token=token,
        lane=lane.value,
        classified=token != pending_token(lane),
        reminded=_flag(fields.get("reminded")),
        summary=_text(fields.get("summary"), "summary"),
        goal=goal if isinstance(goal, str) and goal else None,
        place=place if isinstance(place, str) and place else None,
        subjects=tuple(str(item) for item in subjects if isinstance(item, str) and item),
        prompted_by=_prompter(fields.get("prompted_by")),
    )


def _gap(document: BehaviorDocument) -> SeriesGap:
    raw = document.fields.get("gap_kind")
    try:
        kind = GapKind(raw)
    except ValueError as exc:
        raise SeriesError(f"unknown gap kind: {raw!r}") from exc
    return SeriesGap(
        uri=str(BehaviorURI.from_address(document.address)),
        started_at=_local(document.fields.get("started_at"), "started_at"),
        ended_at=_local(document.fields.get("ended_at"), "ended_at"),
        kind=kind,
    )


def _lane(token: str) -> Lane:
    try:
        return lane_of_token(token)
    except KindIdError as exc:
        raise SeriesError(f"kind_token is neither a class id nor a marker: {token!r}") from exc


def _unordered(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _text(raw: object, field: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise SeriesError(f"{field} must be non-empty text")
    return raw


def _prompter(value: object) -> str | None:
    if value is None:
        return None
    if value not in PROMPTERS:
        raise SeriesError(f"prompted_by must be one of {sorted(PROMPTERS)}")
    return str(value)


def _flag(raw: object) -> bool:
    if not isinstance(raw, bool):
        raise SeriesError("reminded must be a boolean")
    return raw


def _local(raw: object, field: str) -> datetime:
    """行为树上的时间是本地时刻加显式偏移，原样保留。"""

    if isinstance(raw, datetime):
        parsed = raw
    else:
        try:
            parsed = datetime.fromisoformat(_text(raw, field))
        except ValueError as exc:
            raise SeriesError(f"{field} is not a timestamp") from exc
    if parsed.utcoffset() is None:
        raise SeriesError(f"{field} must carry its local offset")
    return parsed


__all__ = ["admitted", "read_series", "skip_reason"]
