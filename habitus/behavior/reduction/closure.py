"""定稿日的规划：哪些本地日历日现在可以记为"定稿"。纯函数，读写由归约 runner 负责。

一天定稿 = 那天没有未封口的链头，且封口视界已过那天在本地偏移下的结束时刻。定稿是事实，之后到达的补发
照常进树、不改变它；派生层（语义树、预测层的结算）只处理定稿的日子。

没有任何记录的日子也要定稿（26 号清单 2.2）：会话 lane 周六没写代码，周五那条最晚时限是周六零点的预期，要等周六
定稿才结算。没有记录就没有偏移可取，用那天之前最近一条记录的偏移（人没换时区，空白的一天也按同一个钟算）。
所以候选不只是"树变了的日子"，还包括上一个定稿日之后、视界之前的每一天。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta


@dataclass
class DayClosureState:
    """``open``：还没定稿的候选；``closed``：定稿日 → 定稿时刻；``last_offset``：最近一个有记录的日子用的本地偏移。"""

    open: set[date] = field(default_factory=set)
    closed: dict[date, str] = field(default_factory=dict)
    last_offset: timedelta | None = None


@dataclass(frozen=True)
class ClosurePlan:
    closed_now: tuple[date, ...]
    still_open: frozenset[date]
    last_offset: timedelta | None


def plan_closure(
    state: DayClosureState,
    touched: Iterable[date],
    *,
    horizon: datetime | None,
    unsealed_days: frozenset[date],
    offsets_of: Callable[[date], tuple[timedelta, ...]],
) -> ClosurePlan:
    """``offsets_of(day)``：那天各条记录的本地偏移（没有记录为空）。"""

    candidates = set(state.open) | {day for day in touched if day not in state.closed}
    last_offset = state.last_offset
    if horizon is None:
        return ClosurePlan((), frozenset(candidates), last_offset)
    horizon_utc = horizon.astimezone(UTC)
    silent = _silent_days(state, candidates, until=horizon_utc.date() + timedelta(days=1))
    closed_now: list[date] = []
    still_open: set[date] = set()
    for day in sorted(candidates | silent):
        offsets = offsets_of(day)
        if offsets:
            last_offset = min(offsets)  # 偏移越小，当地日终越晚：取最晚的那个日终，宁晚不早
        if day in unsealed_days or last_offset is None:
            if day not in silent:
                still_open.add(day)
            continue
        offset = min(offsets) if offsets else last_offset
        end_of_day = datetime.combine(day + timedelta(days=1), datetime.min.time()).replace(tzinfo=UTC) - offset
        if horizon_utc >= end_of_day:
            closed_now.append(day)
        elif day not in silent:
            still_open.add(day)  # 还没过完的空白日不记账：下一轮按视界重新推出来
    return ClosurePlan(tuple(closed_now), frozenset(still_open), last_offset)


def _silent_days(state: DayClosureState, candidates: set[date], *, until: date) -> set[date]:
    """上一个定稿日之后（还没有定稿日时，从最早的候选起）到 ``until`` 的每一天里，既没定稿、也不在候选里的。"""

    if state.closed:
        start = max(state.closed) + timedelta(days=1)
    elif candidates:
        start = min(candidates)
    else:
        return set()
    days: set[date] = set()
    day = start
    while day <= until:
        if day not in state.closed and day not in candidates:
            days.add(day)
        day += timedelta(days=1)
    return days


__all__ = ["ClosurePlan", "DayClosureState", "plan_closure"]
