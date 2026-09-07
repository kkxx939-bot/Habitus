"""此刻的上下文：用同一套槽位从今天的时间线算，在线零额外 LLM。

今天还没有情景文档，"所在的事"就是缺信息——本层不替判决层推断它；判决层从对比表的证据里读到
候选历史上都发生在哪些事里、此前有什么。前提是待用前提清单里未兑现的项（带产生方的 kind，跨日
的前提才对得上）；今日已发生的 kind 另放 ``today_kinds``。此前步骤 / 起因取最近一个时间窗内的
行为（窗口与融合层的上下文窗口同源，调用方传入），跨午夜时补读前一天。"同时在做"按行为树的
status 判（仍在进行、或观测中断）且最后一次观测仍在同一时间窗内——十小时前观测中断的行为不是
此刻在做的。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from habitus.scene.views.index import DayIndexCache
from habitus.scene.views.model import ActionRef, ContextView, Precondition
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
) -> ContextView:
    """候选 ``kind_token`` 在 ``now`` 这一刻的上下文。"""

    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError("now must be a timezone-aware datetime")
    if isinstance(recent_window_seconds, bool) or not isinstance(recent_window_seconds, int | float) or recent_window_seconds <= 0:
        raise ValueError("recent_window_seconds must be a positive number")
    today = now.date()
    instant = now.astimezone(UTC)
    since = instant - timedelta(seconds=recent_window_seconds)
    index = cache.day(today)
    done = [uri for uri in index.ordered if index.occurrences[uri].address.started_at.astimezone(UTC) <= instant]
    today_kinds = frozenset(index.ref(uri).kind_token for uri in done)
    # 近期窗跨午夜：补读前一天里落在窗内的行为
    recent_refs: list[ActionRef] = []
    concurrent: list[ActionRef] = []
    yesterday = cache.day(today - timedelta(days=1))
    for source, uris in ((yesterday, yesterday.ordered), (index, tuple(done))):
        for uri in uris:
            document = source.occurrences[uri]
            started = document.address.started_at.astimezone(UTC)
            if started > instant:
                continue
            if started >= since:
                recent_refs.append(source.ref(uri))
            status, basis = str(document.fields["status"]), str(document.fields["status_basis"])
            last_seen = datetime.fromisoformat(str(document.fields["last_observed_at"])).astimezone(UTC)
            if (status in _ONGOING_STATUSES or basis == _LOST_BASIS) and last_seen >= since:
                concurrent.append(source.ref(uri))
    preconditions = []
    for item in cache.pending(today, pending_expiry_days):
        producer = resolve_target(item.producer_uri, cache)
        kinds = (producer.kind_token,) if isinstance(producer, ActionRef) else ()
        preconditions.append(Precondition("pending", item.text, item.producer_uri, kinds))
    subjects: tuple[str, ...] = ()
    if done:
        subjects = index.others(done[-1])
    return ContextView(
        kind_token=kind_token,
        at=now,
        covered=False,
        prior_steps=tuple(recent_refs),
        preconditions=tuple(preconditions),
        causes=(recent_refs[-1],) if recent_refs else (),
        last_time=last_time(kind_token, before=now, cache=cache, window_days=window_days),
        concurrent=tuple(concurrent),
        subjects=subjects,
        today_kinds=today_kinds,
    )


__all__ = ["now_context"]
