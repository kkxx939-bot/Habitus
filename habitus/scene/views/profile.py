"""一个行为全部历史的聚合画像：对若干历史视图按槽位计数，不逐条。

对应预测树的跨周几 / 全天维度：树在那两层把该行为的全部发生叠在一起估计，语义侧就把全部发生的
上下文叠在一起数——它在哪些周几 / 几点发生、属于哪几件事、之前最常是什么、前提最常是什么、和谁。
只给计数，不给分数；高频行为几百次发生也只是几行数字，低频行为两三次全在里面。

两条与树同口径的记账纪律：周几与小时按**不同的日子**计（树的 ``occurred_days`` 同日同槽封顶 1，
周一看手机五次不比周日一次"多五倍"），其余槽按"多少次发生带着它"计、一次发生里同 kind 只算一次；
画像带**分母**（``covered_days``：请求范围内情景树覆盖了几天，由调用方传入），"2 次"是 2/2 天还是
2/60 天才读得出来。近期 / 长期两栏由调用方切好视图各算一次（窗长按行为的复发周期缩放，那是树的
口径，不在这里重复）。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from habitus.scene.views.model import ActionRef, ContextView, SceneRef

Tally = tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class HistoryProfile:
    """``occurrences`` 是计入的视图数，``days`` 是它们落在的不同日子数，``covered_days`` 是分母（None 即调用方没给）。

    各项计数按 (值, 次数) 降序排列，同次数按值排序，结果确定。``by_weekday`` / ``by_hour`` 按不同日子计。
    ``preceding`` / ``following`` 里 ∅ 是一个值（"之前 / 之后确认什么都没做"），删失的不计。
    """

    kind_token: str
    occurrences: int
    days: int
    covered_days: int | None
    span: tuple[date, date] | None
    first_of_day: int
    by_weekday: tuple[tuple[int, int], ...]
    by_hour: tuple[tuple[int, int], ...]
    scenes: Tally
    roles: Tally
    prior_steps: Tally
    next_steps: Tally
    preceding: Tally
    following: Tally
    preconditions: Tally
    causes: Tally
    concurrent: Tally
    subjects: Tally

    @property
    def empty(self) -> bool:
        return self.occurrences == 0


def history_profile(kind_token: str, views: Iterable[ContextView], *, covered_days: int | None = None) -> HistoryProfile:
    """把同一 kind 的历史视图叠成一份画像。视图的 kind 不一致是我们自己输入的不自洽，硬失败。"""

    if not isinstance(kind_token, str) or not kind_token:
        raise ValueError("kind_token must be non-empty text")
    if covered_days is not None and (isinstance(covered_days, bool) or not isinstance(covered_days, int) or covered_days < 0):
        raise ValueError("covered_days must be a non-negative integer or None")
    items = tuple(views)
    if any(not isinstance(view, ContextView) for view in items):
        raise TypeError("views must contain ContextView values")
    if any(view.kind_token != kind_token for view in items):
        raise ValueError("history profile views must belong to the same kind_token")
    tallies: dict[str, Counter[str]] = {name: Counter() for name in _TALLIED}
    weekday_days: set[tuple[int, date]] = set()
    hour_days: set[tuple[int, date]] = set()
    first = 0
    for view in items:
        weekday_days.add((view.weekday, view.day))
        hour_days.add((view.at.hour, view.day))
        if view.first_of_day:
            first += 1
        if view.scene is not None:
            tallies["scenes"][view.scene.label] += 1
        if view.role is not None:
            tallies["roles"][view.role] += 1
        tallies["prior_steps"].update(_kinds(view.prior_steps))
        tallies["next_steps"].update(_kinds(view.next_steps))
        for name, neighbour in (("preceding", view.preceding), ("following", view.following)):
            if neighbour is not None and neighbour.value is not None:
                tallies[name][neighbour.value] += 1
        tallies["preconditions"].update({kind for item in view.preconditions for kind in item.target_kinds})
        tallies["causes"].update(_cause_kinds(view.causes))
        tallies["concurrent"].update(_kinds(view.concurrent))
        tallies["subjects"].update(set(view.subjects))
    days = sorted({view.day for view in items})
    return HistoryProfile(
        kind_token=kind_token,
        occurrences=len(items),
        days=len(days),
        covered_days=covered_days,
        span=(days[0], days[-1]) if days else None,
        first_of_day=first,
        by_weekday=_day_tally(weekday_days),
        by_hour=_day_tally(hour_days),
        **{name: _tally(counter) for name, counter in tallies.items()},
    )


_TALLIED = ("scenes", "roles", "prior_steps", "next_steps", "preceding", "following", "preconditions", "causes", "concurrent", "subjects")


def _kinds(refs: Iterable[ActionRef]) -> set[str]:
    """一条视图里同一 kind 出现几次都只算一次：画像数的是"有多少次发生带着它"，不是它出现了几回。"""

    return {ref.kind_token for ref in refs}


def _cause_kinds(causes: Iterable[ActionRef | SceneRef]) -> set[str]:
    found: set[str] = set()
    for cause in causes:
        if isinstance(cause, ActionRef):
            found.add(cause.kind_token)
        else:
            found.update(cause.kinds)
    return found


def _tally(counter: Counter[str]) -> Tally:
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _day_tally(pairs: set[tuple[int, date]]) -> tuple[tuple[int, int], ...]:
    counter: Counter[int] = Counter(key for key, _day in pairs)
    return tuple(sorted(counter.items()))


__all__ = ["HistoryProfile", "Tally", "history_profile"]
