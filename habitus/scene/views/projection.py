"""把一条行为投影成上下文视图（读时计算，不落盘），以及候选行为在钟面邻域内的历史视图列表。

全部机械：所在的事来自情景文档的成员表；此前步骤 / 之后步骤是同一件事里更早 / 更晚的必要或顺带
成员；前提两路同收（事的 needs 目标、事里更早成员留下的待用前提）；起因是事的 results_from 与
行为树自带的 results_from；上一次是回看窗内同 kind 的上一条；同时在做与和谁来自行为树。角色为
irrelevant 的成员只保留"所在的事"（"发生在这件事期间"是事实），不继承事的前提、起因与前后步骤
（模型自己说它与事无关）。

按预测树的维度另给三个事实：``first_of_day``（这一条是不是当天该 kind 的第一次——树的危险率与
累积率只描述首次）；``preceding`` / ``following``（转移窗口内时间上紧邻的上一条 / 下一条，见
``neighbours``）。窗口与槽宽都由调用方把预测树的配置值原样传进来，两侧不各算各的。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

from habitus.behavior.uri import BehaviorURI
from habitus.scene.views.clock import slot_index as slot_of
from habitus.scene.views.clock import slots_per_day
from habitus.scene.views.index import DayIndex, DayIndexCache
from habitus.scene.views.model import (
    ActionRef,
    ContextView,
    LastTime,
    Neighbour,
    transition_window,
)

_Span = tuple[datetime, datetime]


def context_view(
    occurrence_uri: str,
    cache: DayIndexCache,
    *,
    window_days: int,
    transition_window_seconds: float | None = None,
) -> ContextView:
    """一条已落树行为的上下文；``window_days`` 是"上一次"的回看边界，``transition_window_seconds``
    是紧邻上一条 / 下一条的窗口（None 即不算）。"""

    parsed = BehaviorURI.parse(occurrence_uri)
    address = parsed.to_address()
    index = cache.day(address.occurred_on)
    uri = str(parsed)
    if uri not in index.occurrences:
        raise KeyError(f"occurrence is not on the behaviour tree (or is a disambiguated duplicate): {uri}")
    return _project(uri, index, cache, window_days=window_days, window=transition_window(transition_window_seconds))


def history_contexts(
    kind_token: str,
    cache: DayIndexCache,
    *,
    days: Iterable[date],
    window_days: int,
    slot_minutes: int | None = None,
    slot_index: int | None = None,
    slot_half_width: int = 0,
    transition_window_seconds: float | None = None,
) -> tuple[ContextView, ...]:
    """候选 kind 在给定日子里的历史视图，按时间升序。

    **不再按"语义层处理过没有"筛日子**：那是按天归组的概念，而语义层现在按候选累积，"这个候选
    这一天关联完成了没有"由上游（``foresight`` 的 ``unassociated``）如实报出，不在这里悄悄少取。
    给了 ``slot_minutes`` 与 ``slot_index`` 就只取钟面邻域内的：槽的口径与预测树同一公式（``views.clock``，
    环形距离 ≤ ``slot_half_width``）。不给槽就是该 kind 的全部历史（全天那一层吃的就是这一份）。"""

    if not isinstance(kind_token, str) or not kind_token:
        raise ValueError("kind_token must be non-empty text")
    slots_per_day = _slots_per_day(slot_minutes, slot_index, slot_half_width)
    window = transition_window(transition_window_seconds)
    views: list[ContextView] = []
    for day in sorted(set(days)):
        index = cache.day(day)
        for uri in index.ordered:
            document = index.occurrences[uri]
            if str(document.fields["kind_token"]) != kind_token:
                continue
            if slot_minutes is not None and slot_index is not None and slots_per_day is not None:
                own = slot_of(document.address.started_at, slot_minutes=slot_minutes)
                distance = abs(own - slot_index)
                if min(distance, slots_per_day - distance) > slot_half_width:
                    continue
            views.append(_project(uri, index, cache, window_days=window_days, window=window))
    return tuple(views)


def _slots_per_day(slot_minutes: int | None, slot_index: int | None, slot_half_width: int) -> int | None:
    if (slot_minutes is None) != (slot_index is None):
        raise ValueError("slot_minutes and slot_index must be given together")
    if isinstance(slot_half_width, bool) or not isinstance(slot_half_width, int) or slot_half_width < 0:
        raise ValueError("slot_half_width must be a non-negative integer")
    if slot_minutes is None or slot_index is None:
        return None
    slots = slots_per_day(slot_minutes)
    if isinstance(slot_index, bool) or not isinstance(slot_index, int) or not 0 <= slot_index < slots:
        raise ValueError("slot_index must be an integer within the clock face")
    return slots


def _project(uri: str, index: DayIndex, cache: DayIndexCache, *, window_days: int, window: float | None) -> ContextView:
    """装配一条视图：行为树侧的事实与树维度的三个事实各自算好再拼。

    情景侧（所在的事、角色、事里更早成员留下的前提）已经删掉——那是按天归组的产物。规律级的
    上下文按行为 URI 取记录，读口在 ``views.gloss``，由上层贴到卡上，不在视图里。
    """

    document = index.occurrences[uri]
    fields = document.fields
    kind_token = str(fields["kind_token"])
    started = document.address.started_at
    causes: list[ActionRef] = []
    for cause_uri in index.results_from_targets.get(uri, ()):
        resolved = resolve_target(cause_uri, cache)
        if isinstance(resolved, ActionRef):
            causes.append(resolved)
    partners = concurrent_refs(uri, index, cache)
    preceding: Neighbour | None = None
    following: Neighbour | None = None
    if window is not None:
        preceding, following = neighbours(uri, index, cache, window_seconds=window, partners=frozenset(item.uri for item in partners))
    return ContextView(
        kind_token=kind_token,
        at=started,
        occurrence_uri=uri,
        name=str(fields["name"]),
        causes=tuple(causes),
        last_time=last_time(kind_token, before=started, cache=cache, window_days=window_days),
        concurrent=partners,
        subjects=index.others(uri),
        summary=str(fields["summary"]),
        first_of_day=first_of_day(uri, index),
        preceding=preceding,
        following=following,
        day_note=index.day_note,
    )


def first_of_day(uri: str, index: DayIndex) -> bool:
    """这一条是不是当天该 kind 的第一次。

    "当天"是行为树的本地日历日（目录），"第一"按开始时刻——与预测树 ``first_days`` 的记账同口径；
    唯一差别在夏令时回拨的那一小时（树按本地时分槽比先后，这里按瞬时），已知并接受。"""

    kind_token = str(index.occurrences[uri].fields["kind_token"])
    first = next(candidate for candidate in index.ordered if str(index.occurrences[candidate].fields["kind_token"]) == kind_token)
    return first == uri


def neighbours(
    uri: str,
    index: DayIndex,
    cache: DayIndexCache,
    *,
    window_seconds: float,
    partners: frozenset[str] = frozenset(),
) -> tuple[Neighbour, Neighbour]:
    """转移窗口内时间上紧邻的上一条与下一条，三态（找到 / 确认没有 / 删失），跨午夜补读相邻的一天。

    ``following`` 与预测树 ``edges.pair`` 找后继的三条规矩同口径：窗口内下一条开始的；跳过与它声明
    并行的伙伴（``partners``，由调用方算好传入）；起点到后继之间有观测空洞则**删失**——找到了也删，
    没找到时窗口全程在看才算"确认没有"（树上的 ∅）。前后顺序按 ``(开始瞬时, URI)`` 的全序定，
    同一时刻的两条不会互为前后。

    ``preceding`` 是同一规则**反向**用：它之前最近一条不与它并行的行为。这**不是**树上转移边的严格
    逆——树只从源的一侧跳伙伴，并行簇附近树上可能没有对应的边，读侧只把它当弱证据；模块不做
    树那种"从源侧配对"的重放。
    """

    timeline = _timeline(index, cache)
    position = next(number for number, (_, candidate) in enumerate(timeline) if candidate == uri)
    own = _instant(index, uri)
    holes = _holes(index, cache)
    following = _scan(timeline[position + 1 :], own=own, forward=True, window_seconds=window_seconds, partners=partners, holes=holes)
    preceding = _scan(tuple(reversed(timeline[:position])), own=own, forward=False, window_seconds=window_seconds, partners=partners, holes=holes)
    return preceding, following


def _scan(
    candidates: tuple[tuple[DayIndex, str], ...],
    *,
    own: datetime,
    forward: bool,
    window_seconds: float,
    partners: frozenset[str],
    holes: tuple[_Span, ...],
) -> Neighbour:
    for source, candidate in candidates:
        if candidate in partners:
            continue
        instant = _instant(source, candidate)
        distance = (instant - own).total_seconds() if forward else (own - instant).total_seconds()
        if distance > window_seconds:
            break
        span = (own, instant) if forward else (instant, own)
        if _has_hole(holes, span):
            return Neighbour(None, censored=True)
        return Neighbour(source.ref(candidate))
    edge = own + timedelta(seconds=window_seconds) if forward else own - timedelta(seconds=window_seconds)
    span = (own, edge) if forward else (edge, own)
    return Neighbour(None, censored=_has_hole(holes, span))


def _timeline(index: DayIndex, cache: DayIndexCache) -> tuple[tuple[DayIndex, str], ...]:
    days = (cache.day(index.day - timedelta(days=1)), index, cache.day(index.day + timedelta(days=1)))
    return tuple(sorted(((source, uri) for source in days for uri in source.ordered), key=lambda item: (_instant(item[0], item[1]), item[1])))


def _holes(index: DayIndex, cache: DayIndexCache) -> tuple[_Span, ...]:
    days = (cache.day(index.day - timedelta(days=1)), index, cache.day(index.day + timedelta(days=1)))
    return tuple((gap.started_at.astimezone(UTC), gap.ended_at.astimezone(UTC)) for source in days for gap in source.gaps)


def _has_hole(holes: tuple[_Span, ...], span: _Span) -> bool:
    """与预测树 ``edges._window_fully_observed`` 同一判据：有任何空白与 [起, 止] 相交即有洞。"""

    start, end = span
    return any(hole_start < end and hole_end > start for hole_start, hole_end in holes)


def _instant(index: DayIndex, uri: str) -> datetime:
    return index.occurrences[uri].address.started_at.astimezone(UTC)


def concurrent_refs(uri: str, index: DayIndex, cache: DayIndexCache) -> tuple[ActionRef, ...]:
    """concurrent_with 语义对称、存储单向：链接由**开始更晚**的链指向开始更早的链，跨批时也可能由更早开始、
    更晚封口的链指向已消费的目标。所以取对称闭包要扫自己指出去的，加上前一天、同日、次日指回来的。"""

    found: dict[str, ActionRef] = {}
    for target in index.concurrent_targets.get(uri, ()):
        resolved = resolve_target(target, cache)
        if isinstance(resolved, ActionRef):
            found[resolved.uri] = resolved
    for other_index in (cache.day(index.day - timedelta(days=1)), index, cache.day(index.day + timedelta(days=1))):
        for other, targets in other_index.concurrent_targets.items():
            if uri in targets and other != uri:
                found[other] = other_index.ref(other)
    return tuple(found[key] for key in sorted(found))


def last_time(kind_token: str, *, before: datetime, cache: DayIndexCache, window_days: int) -> LastTime | None:
    """``before`` 之前、回看窗内同 kind 的上一条：距今天数、当时所在的事。"""

    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days <= 0:
        raise ValueError("window_days must be a positive integer")
    today = before.date()
    for offset in range(window_days + 1):
        day = today - timedelta(days=offset)
        index = cache.day(day)
        for uri in reversed(index.ordered):
            document = index.occurrences[uri]
            if str(document.fields["kind_token"]) != kind_token:
                continue
            if document.address.started_at.astimezone(UTC) >= before.astimezone(UTC):
                continue
            return LastTime(uri=uri, days_ago=(today - day).days)
    return None



def resolve_target(target_uri: str, cache: DayIndexCache) -> ActionRef | None:
    """边的目标：行为树上的那条行为；解析不到即悬空，少说一点。

    原来还接 ``scene://`` 的目标（按天归组的"事"）。那种目标已经不存在——规律级的边指的都是
    行为，因为"事"这个中间容器整个删掉了。
    """

    day = BehaviorURI.parse(target_uri).to_address().occurred_on
    index = cache.day(day)
    return index.ref(target_uri) if target_uri in index.occurrences else None


__all__ = [
    "concurrent_refs",
    "context_view",
    "first_of_day",
    "history_contexts",
    "last_time",
    "neighbours",
    "resolve_target",
]
