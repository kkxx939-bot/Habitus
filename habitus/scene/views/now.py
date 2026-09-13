"""此刻的上下文：用同一套槽位从今天的时间线算，在线零额外 LLM。

今天还没有情景文档，"所在的事"就是缺信息——本层不替判决层推断它；判决层从对比表的证据里读到
候选历史上都发生在哪些事里、此前有什么。前提是待用前提清单里未兑现的项（带产生方的 kind，跨日
的前提才对得上）；今日已发生的 kind 另放 ``today_kinds``。此前步骤 / 起因取最近一个时间窗内的
行为（窗口与融合层的上下文窗口同源，调用方传入），跨午夜时补读前一天。"同时在做"按行为树的
status 判（仍在进行、或观测中断）且最后一次观测仍在同一时间窗内——十小时前观测中断的行为不是
此刻在做的。

此刻视图上的"起因"只是近期窗内的最后一条（此刻没有 results_from 这种语义边可读），历史视图的
"起因"才是语义边——对比表的起因行比的是这两样，读它的人要知道。

按预测树维度对齐的两个事实：``preceding``（转移窗口内紧邻的上一条，调用方给了窗口才算；排除
已判入"同时在做"的那几条——同一条不能既是同时在做又是紧邻上一条；起点到此刻之间有观测空洞则
删失，窗口内没有且全程在看才是确认没有）；``gaps``（今天到此刻为止的观测空白，截到此刻）。

契约：``now`` 必须是主体的本地时刻（与行为树上 occurrence 同一偏移）——今天是哪一天、此刻在
哪个槽都从它的本地时分算，传 UTC 进来会读错一天。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from habitus.scene.views.index import DayIndex, DayIndexCache
from habitus.scene.views.model import ActionRef, ContextView, Neighbour, ObservationGap, Precondition, transition_window
from habitus.scene.views.projection import last_time, resolve_target

_ONGOING_STATUSES = frozenset({"ongoing"})
_LOST_BASIS = "observation_lost"


def now_context(
    kind_token: str,
    *,
    now: datetime,
    cache: DayIndexCache,
    window_days: int,
    pending_expiry_days: int,
    recent_window_seconds: float = 3600.0,
    transition_window_seconds: float | None = None,
) -> ContextView:
    """候选 ``kind_token`` 在 ``now`` 这一刻的上下文。"""

    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError("now must be a timezone-aware datetime")
    if isinstance(recent_window_seconds, bool) or not isinstance(recent_window_seconds, int | float) or recent_window_seconds <= 0:
        raise ValueError("recent_window_seconds must be a positive number")
    window = transition_window(transition_window_seconds)
    today = now.date()
    instant = now.astimezone(UTC)
    since = instant - timedelta(seconds=recent_window_seconds)
    index = cache.day(today)
    yesterday = cache.day(today - timedelta(days=1))
    # 昨天全部 + 今天到此刻为止的，按 (开始瞬时, URI) 排成一条流：近期窗与转移窗都跨午夜
    stream = sorted(
        ((source, uri) for source in (yesterday, index) for uri in source.ordered if _started(source, uri) <= instant),
        key=lambda item: (_started(item[0], item[1]), item[1]),
    )
    done = [uri for source, uri in stream if source is index]
    recent_refs: list[ActionRef] = []
    concurrent: dict[str, ActionRef] = {}
    for source, uri in stream:
        if _started(source, uri) >= since:
            recent_refs.append(source.ref(uri))
        document = source.occurrences[uri]
        status, basis = str(document.fields["status"]), str(document.fields["status_basis"])
        last_seen = datetime.fromisoformat(str(document.fields["last_observed_at"])).astimezone(UTC)
        if (status in _ONGOING_STATUSES or basis == _LOST_BASIS) and last_seen >= since:
            concurrent[uri] = source.ref(uri)
    gaps = gaps_until(index.gaps, now)  # 视图上的空白只有今天的；删失检查另把昨天的也算上（窗口跨午夜）
    preceding = None if window is None else _preceding(stream, instant=instant, window_seconds=window, excluded=frozenset(concurrent), gaps=yesterday.gaps + gaps)
    preconditions = []
    for item in cache.pending(today, pending_expiry_days):
        producer = resolve_target(item.producer_uri, cache)
        kinds = (producer.kind_token,) if isinstance(producer, ActionRef) else ()
        preconditions.append(Precondition("pending", item.text, item.producer_uri, kinds))
    subjects: tuple[str, ...] = stream[-1][0].others(stream[-1][1]) if stream else ()
    return ContextView(
        kind_token=kind_token,
        at=now,
        covered=False,
        prior_steps=tuple(recent_refs),
        preconditions=tuple(preconditions),
        causes=(recent_refs[-1],) if recent_refs else (),
        last_time=last_time(kind_token, before=now, cache=cache, window_days=window_days),
        concurrent=tuple(concurrent.values()),
        subjects=subjects,
        today_kinds=frozenset(index.ref(uri).kind_token for uri in done),
        preceding=preceding,
        gaps=gaps,
        day_note=index.day_note,
    )


def _preceding(
    stream: list[tuple[DayIndex, str]],
    *,
    instant: datetime,
    window_seconds: float,
    excluded: frozenset[str],
    gaps: tuple[ObservationGap, ...],
) -> Neighbour:
    """此刻的紧邻上一条：流里最后一条不在 ``excluded``、且落在窗口内的；到此刻之间有洞即删失。"""

    holes = tuple((gap.started_at.astimezone(UTC), gap.ended_at.astimezone(UTC)) for gap in gaps)
    for source, uri in reversed(stream):
        if uri in excluded:
            continue
        started = _started(source, uri)
        if (instant - started).total_seconds() > window_seconds:
            break
        if _has_hole(holes, started, instant):
            return Neighbour(None, censored=True)
        return Neighbour(source.ref(uri))
    return Neighbour(None, censored=_has_hole(holes, instant - timedelta(seconds=window_seconds), instant))


def _has_hole(holes: tuple[tuple[datetime, datetime], ...], start: datetime, end: datetime) -> bool:
    return any(hole_start < end and hole_end > start for hole_start, hole_end in holes)


def _started(index: DayIndex, uri: str) -> datetime:
    return index.occurrences[uri].address.started_at.astimezone(UTC)


def gaps_until(gaps: tuple[ObservationGap, ...], now: datetime) -> tuple[ObservationGap, ...]:
    """到此刻为止的观测空白：还没到的不算，跨过此刻的截到此刻。"""

    instant = now.astimezone(UTC)
    clipped: list[ObservationGap] = []
    for gap in gaps:
        if gap.started_at.astimezone(UTC) >= instant:
            continue
        ended = gap.ended_at if gap.ended_at.astimezone(UTC) <= instant else now
        clipped.append(ObservationGap(started_at=gap.started_at, ended_at=ended, kind=gap.kind))
    return tuple(clipped)


__all__ = ["gaps_until", "now_context"]
