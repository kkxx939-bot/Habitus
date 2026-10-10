"""把行为树读成第 N 晚的事件序列。行为树的读法**只写在这里**，预测树与语义树都不再自己读、自己判。

只有组合根调用本模块（架构测试钉死）：它 import 行为树与词表的编号规则，两棵树只 import 纯值的 ``series.model``。

读法：

- 只读不写；只读**截止日之前**的日子（第 N 晚读 N 之前，已封口）；
- ``original_name`` 非空的 occurrence 是撞车消歧的已知重复，**机械跳过**——标记在写入时由归约层打好，这里只认标记、
  不做判重（判重只在融合层解决）；
- ``kind_token`` 在这里分流（裁定 18、19）：类编号进序列（``classified``）；「待定」进序列但不是类（只占先后顺序）；
  「非事件」不进序列，连顺序里也当它不存在；既不是编号也不是占位标记的硬拒；lane 取编号前缀；
- 观测空白两种都进（``GapKind``），跨过截止时刻的截到那一刻；空白的类型在这里翻译一次，上游改名时以未知取值硬失败；
- ``concurrent_with`` 链接原样读成无序的 uri 对，只留两端都在序列里的（指向被跳过的一端，那条并行没有两个可用端点）；
- ``reminded`` 原样带着：被提醒过的发生要不要算进自然率，是消费方的事（预测树现在硬拒，见 ``prediction.source``）。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime

from habitus.behavior.document import BehaviorDocument, BehaviorLinkType
from habitus.behavior.kinds.ids import KindIdError, Lane, lane_of_token, not_event_token, pending_token
from habitus.behavior.tree import BehaviorAddress, BehaviorKind, BehaviorTree
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

_PAGE_LIMIT = 10_000

# TODO(BHV-LIFECYCLE-001·序列读取): 每晚读行为树是全量扫描，日集合枚举在文档数超过分页上限后二次增长。
# - 现状：``_days`` 为了拿到"有哪些天"，用 ``list_addresses(kind, limit=10_000, after=…)`` 分页枚举**全部叶子**；
#   ``BehaviorTree.list_addresses`` 每页都从头重走、按游标跳过，总成本随 页数 × 文档数 增长。树上没有"只列日期"的读接口。
# - 具体场景：实测 9,270 条 occurrence，页大小 10,000（1 页）0.21s、2,000（5 页）1.67s、1,000（10 页）3.20s。
#   按约 400 条/天推算，一年约 146,000 条 = 15 页 → 仅"有哪些天"这一步就约 50 秒，而答案只是 365 个日期。
# - 影响大小：中。夜批离线，失败只报 stale，不影响正确性；成本随历史增长。
# - 改造方案：① 行为树增加只读的 ``list_days(kind)``，只枚举 YYYY/MM/DD 目录；② 生命周期给行为树定下保留期之后，
#   只读保留窗内的日期。
# - 时机：与 BHV-LIFECYCLE-001 统一批次做（用户裁定 2026-09-01：生命周期三棵树统一设计）。在此之前一律全量读——
#   没有统一门槛之前少读任何一天都会静默改变曝光分母。


def read_series(tree: BehaviorTree, *, cutoff: date) -> EventSeries:
    """第 ``cutoff`` 晚的序列：截止日之前每一天的 occurrence 与空白。"""

    if not isinstance(tree, BehaviorTree):
        raise SeriesError("tree must be a BehaviorTree")
    if isinstance(cutoff, datetime) or not isinstance(cutoff, date):
        raise SeriesError("cutoff must be a date")
    boundary = EventSeries(cutoff=cutoff)
    records: list[SeriesRecord] = []
    skipped: list[SkippedRecord] = []
    links: list[tuple[str, str]] = []
    for document in _documents(tree, BehaviorKind.OCCURRENCE, cutoff):
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
    # 空白在这里消解一次（``disproves``）：之后预测树、关系检验读到的是同一份
    starts = sorted(record.started_at for record in records)
    gaps = [
        gap
        for gap in (
            clip_gap(_gap(document), boundary.boundary_for(document.address.started_at))
            for document in _documents(tree, BehaviorKind.GAP, cutoff)
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


def _documents(tree: BehaviorTree, kind: BehaviorKind, cutoff: date) -> Iterator[BehaviorDocument]:
    """按天整块读（逐篇 ``read`` 会让每天的目录被重复枚举、成本随篇数平方增长），只读截止日之前的日子。"""

    days: set[date] = set()
    cursor: BehaviorAddress | None = None
    while True:
        page = tree.list_addresses(kind, limit=_PAGE_LIMIT, after=cursor)
        days.update(address.occurred_on for address in page)
        if len(page) < _PAGE_LIMIT:
            break
        cursor = page[-1]
    for day in sorted(day for day in days if day < cutoff):
        yield from tree.read_day(kind, day)


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
