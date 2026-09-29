"""预测树的最小夹具：一条只在指定槽位有值的曲线，和一棵只带曲线的树。

机会口与节律口这两座桥的测试共用它（夹具独立于测试模块：两边都从这里取，不互相 import）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime

from habitus.prediction.model import DayCurve, IntervalQuantiles, PredictionTree, RecurrenceStatistics

SLOTS = 48  # 半小时一格；槽 15 = 07:30–08:00
FRIDAY = date(2026, 8, 14)


def curve(peaks: Mapping[int, float]) -> DayCurve:
    """在指定槽位上放值，其余为 0。"""

    marginal = tuple(peaks.get(slot, 0.0) for slot in range(SLOTS))
    return DayCurve(marginal=marginal, hazard=marginal, cumulative=tuple(0.0 for _ in range(SLOTS)), trend=None, trend_n_eff=0.0)


def tree(curves: Mapping[tuple[int, str], DayCurve]) -> PredictionTree:
    return PredictionTree(
        built_at=datetime(2026, 8, 18, 3, 0, tzinfo=UTC),
        reference_day=date(2026, 8, 17),
        config_digest="d" * 64,
        slot_minutes=30,
        nodes={},
        curves=dict(curves),
        weekday_baselines={},
        edges={},
        parallels={},
        parallel_totals={},
        recurrences={},
        exposure={},
        baselines={},
        actions=(),
        observed_days=45,
        censored_transitions=0.0,
    )


def with_recurrences(built: PredictionTree, medians_hours: Mapping[str, float]) -> PredictionTree:
    """给树补上几个 kind 的复发间隔（中位数按小时给，树里存的是秒）。"""

    return replace(
        built,
        recurrences={
            kind: RecurrenceStatistics(IntervalQuantiles(p10=hours * 1800.0, p50=hours * 3600.0, p90=hours * 7200.0, sample_count=12.0))
            for kind, hours in medians_hours.items()
        },
    )


__all__ = ["FRIDAY", "SLOTS", "curve", "tree", "with_recurrences"]
