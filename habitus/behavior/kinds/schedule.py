"""定期拆改的跨次状态与"这一晚该不该拆改"（裁定 94 第 9 条：词表的每晚新增与定期拆改都在换代里、读序列之前，
不挂任何钟点）：

- 每晚新增：换代每天一次，不需要跨次状态；
- 定期拆改：每周一次，拆改周期按换代的那一天算——从每周 ``revision_weekday`` 那一天起到下一个那一天之前是一个周期，这个周期里
  换代第一次跑到它就拆改（那一天换代没跑到，之后几天接着补，不累积）。合并要有证据：同一对（a→b）在最近连续
  ``merge_evidence_runs`` 个**拆改周期**里都被提出才放行（裁定 17：按周期计，同一周期里跑几次都只算一次）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

from habitus.behavior.kinds.model import BehaviorKindError

MergePair = tuple[str, str]
PeriodMerges = tuple[str, frozenset[MergePair]]


@dataclass(frozen=True)
class JobState:
    #: 上一次拆改在什么时候（它的本地日就是线的那一晚，"这个周期拆改过没有"按它判）。
    revision_at: datetime | None = None
    merge_history: tuple[PeriodMerges, ...] = ()

    def __post_init__(self) -> None:
        if self.revision_at is not None and self.revision_at.utcoffset() is None:
            raise BehaviorKindError("revision_at must be timezone-aware")
        periods = [period for period, _ in self.merge_history]
        if len(set(periods)) != len(periods):
            raise BehaviorKindError("merge history must have one entry per revision period")

    def merge_supported(self, pair: MergePair, *, runs: int, period: str) -> bool:
        """这一对加上本周期，是否已在此前连续 ``runs - 1`` 个别的周期里都被提出。"""

        earlier = [pairs for key, pairs in self.merge_history if key != period]
        previous = earlier[-(runs - 1) :] if runs > 1 else []
        return len(previous) == runs - 1 and all(pair in proposed for proposed in previous)

    def after_revision(self, at: datetime, period: str, proposed: frozenset[MergePair], *, keep: int) -> JobState:
        history = list(self.merge_history)
        if history and history[-1][0] == period:
            history[-1] = (period, history[-1][1] | proposed)
        else:
            history.append((period, proposed))
        return replace(self, revision_at=at, merge_history=tuple(history[-keep:]))


def revision_start(night: date, *, weekday: int) -> date:
    """这一晚属于哪个拆改周期：最近一个（含这一晚）``weekday`` 那一晚。"""

    return night - timedelta(days=(night.weekday() - weekday) % 7)


def revision_period(night: date, *, weekday: int) -> str:
    """拆改周期的名字（合并证据按它计）：周期起点那一晚的日期。"""

    return revision_start(night, weekday=weekday).isoformat()


def revision_due(night: date, last: date | None, *, weekday: int) -> bool:
    """这一晚所在的拆改周期里还没拆改过。"""

    return last is None or revision_start(last, weekday=weekday) < revision_start(night, weekday=weekday)


__all__ = ["JobState", "MergePair", "PeriodMerges", "revision_due", "revision_period", "revision_start"]
