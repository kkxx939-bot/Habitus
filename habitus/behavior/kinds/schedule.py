"""词表两件定时活的跨次状态与"该不该跑"（时刻都是用户本地时间，时区由调用方给）：

- 每晚新增：每天 ``nightly_hour`` 点之后跑一次；
- 定期拆改：每周 ``revision_weekday`` 的 ``revision_hour`` 点之后跑一次。合并要有证据：同一对（a→b）在最近连续
  ``merge_evidence_runs`` 个**拆改周期**里都被提出才放行（裁定 17：按周期计，同一周期里跑几次都只算一次）。

错过的那一次（机器没开）开机后补跑一次，不累积。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from habitus.behavior.kinds.model import BehaviorKindError

MergePair = tuple[str, str]
PeriodMerges = tuple[str, frozenset[MergePair]]


@dataclass(frozen=True)
class JobState:
    nightly_at: datetime | None = None
    revision_at: datetime | None = None
    merge_history: tuple[PeriodMerges, ...] = ()

    def __post_init__(self) -> None:
        for name in ("nightly_at", "revision_at"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, datetime) or value.utcoffset() is None):
                raise BehaviorKindError(f"{name} must be timezone-aware")
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

    def after_nightly(self, at: datetime) -> JobState:
        return replace(self, nightly_at=at)


def nightly_due(now: datetime, last_run_at: datetime | None, *, hour: int) -> bool:
    """最近一个"每天 ``hour`` 点"已经到了、且那之后还没跑过。"""

    return _due(_latest_slot(now, hour=hour, period_days=1, offset_days=0), last_run_at)


def revision_due(now: datetime, last_run_at: datetime | None, *, weekday: int, hour: int) -> bool:
    """最近一个"每周 ``weekday`` 的 ``hour`` 点"已经到了、且那之后还没跑过。"""

    return _due(revision_slot(now, weekday=weekday, hour=hour), last_run_at)


def revision_period(now: datetime, *, weekday: int, hour: int) -> str:
    """这一刻属于哪个拆改周期：最近一个拆改时刻的本地日期。"""

    return revision_slot(now, weekday=weekday, hour=hour).date().isoformat()


def revision_slot(now: datetime, *, weekday: int, hour: int) -> datetime:
    return _latest_slot(now, hour=hour, period_days=7, offset_days=(now.weekday() - weekday) % 7)


def _latest_slot(now: datetime, *, hour: int, period_days: int, offset_days: int) -> datetime:
    if now.utcoffset() is None:
        raise BehaviorKindError("now must be timezone-aware")
    # fold=0：夏令时回拨那天"几点"出现两次，取第一次，回拨后的同一钟点不会再算一次到期。
    scheduled = now.replace(hour=hour, minute=0, second=0, microsecond=0, fold=0) - timedelta(days=offset_days)
    if scheduled > now:
        scheduled -= timedelta(days=period_days)
    return scheduled


def _due(scheduled: datetime, last_run_at: datetime | None) -> bool:
    return last_run_at is None or last_run_at < scheduled


__all__ = ["JobState", "MergePair", "PeriodMerges", "nightly_due", "revision_due", "revision_period", "revision_slot"]
