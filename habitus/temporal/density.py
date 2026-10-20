"""密度：这件事整体上多频繁（裁定 59 第 6 条第 4 步）。和间隔不同，它把没做的日子也算进来。

两个数，都按观测覆盖、扣掉盲日，只看最近 ``recent`` 个自然发生日：从其中最早那天算起（新长出来的类不会被稀释，
老的类读的是它现在的频率，不是全历史平均；裁定 57）：
- 平均多久一次 = 在看的天数 ÷ 自然发生的次数（合并后）；
- 发生日占比 = 有自然发生的天数 ÷ 在看的天数。
例：v0.6 例会材料常见间隔 7 天、平均 10.3 天一次——两个数一比，才看得出"每周做、但常跳过"。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from habitus.temporal.coverage import Coverage, blind_days
from habitus.temporal.episodes import OccurrenceDay


@dataclass(frozen=True)
class Density:
    days_per_occurrence: float
    occurrence_day_share: float
    watched_days: float
    occurrences: int
    occurrence_days: int


def density_of(days: Sequence[OccurrenceDay], coverage: Coverage, *, recent: int) -> Density | None:
    """没有自然发生、或从最近那几次的第一次起一天都没在看，就没有读数（``None``）。"""

    if isinstance(recent, bool) or not isinstance(recent, int) or recent < 1:
        raise ValueError("recent must be a positive integer")
    natural = [item for item in days if item.natural > 0][-recent:]
    if not natural:
        return None
    since = natural[0].day
    watched = coverage.watched_days(since=since, excluding=blind_days(days))
    if watched <= 0.0:
        return None
    occurrences = sum(item.natural for item in natural)
    return Density(
        days_per_occurrence=watched / occurrences,
        occurrence_day_share=min(len(natural) / watched, 1.0),
        watched_days=watched,
        occurrences=occurrences,
        occurrence_days=len(natural),
    )


__all__ = ["Density", "density_of"]
