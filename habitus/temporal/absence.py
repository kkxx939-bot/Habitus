"""判缺失：这一次的空白比它历史上最长的一次间隔还长（裁定 57 原文；裁定 60）。全仓只此一份。

- **两种用法**：``exceeds_longest`` 只比历史最长（语义树的"久别重来"）；``long_absence`` 是账的久别（裁定 88 第 2 条），
  另要空白不短于 H 个典型间隔（典型间隔 = 历史上没截尾间隔的中位数）——天天做的事漏一天比历来都长，但只有 2 个典型间隔，不算久别。

- **间隔的有效长度**：空白里每天"看着的比例"之和（从上一次的次日到这一次那天）。人不在、设备没上传的那几天不算长度，
  所以中间 60 天未观测的那段不会把"历史最长"抬到 91 天。只用观测覆盖，不乘日内分布——覆盖是事实、不随窗口漂，不需要落盘状态。
  和 ``intervals.DayInterval.seen``（"这段没漏看的概率"，连乘）是两个量，不混用。
- **只数某些日型**（可选）：账的久别只数这本账有过机会的日型的日子（裁定 59"默认不外推"）——只在工作日做的事，国庆七天不算超过。
- **截尾**：以"被提醒之后才做"结尾的那段不进"历史最长"（不知道他自己会隔多久做）；但它自己是"这一次"且已经长过门槛时照样算超过。
- **样本不够不判**：账的久别要 2 个间隔（``MIN_HISTORY``，代价只是一次影子）；语义树的"久别重来"判的是一段，要
  ⌈1/α⌉ − 1 个（下一次比历来都长的机会是 1/(n+1)，只有它不大于 α 才算数）。门槛由调用方给。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class AbsenceVerdict(str, Enum):
    EXCEEDS = "exceeds"
    WITHIN = "within"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class Visit:
    """某一天来过一次；``prompted`` = 那天只有被提醒之后才做（以它结尾的间隔截尾）。"""

    day: date
    prompted: bool = False


@dataclass(frozen=True)
class EffectiveGap:
    start: date
    end: date
    length: float
    censored: bool

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("a gap ends after it starts")
        if self.length < 0:
            raise ValueError("a gap's effective length is non-negative")


Watched = Callable[[date], float]
Counts = Callable[[date], bool]


def _always(_day: date) -> bool:
    return True


def _fully(_day: date) -> float:
    return 1.0


def gap_between(
    start: date, end: date, *, watched: Watched = _fully, counts: Counts = _always, censored: bool = False
) -> EffectiveGap:
    """从 ``start`` 的次日到 ``end``（含）每天看着的比例之和；``counts`` 为假的日子不算。"""

    length = 0.0
    day = start + timedelta(days=1)
    while day <= end:
        if counts(day):
            share = watched(day)
            if not 0.0 <= share <= 1.0:
                raise ValueError("a watched share lies in [0, 1]")
            length += share
        day += timedelta(days=1)
    return EffectiveGap(start, end, length, censored)


def effective_gaps(
    visits: Sequence[Visit], *, watched: Watched = _fully, counts: Counts = _always
) -> tuple[EffectiveGap, ...]:
    """相邻两次之间的有效间隔；一天多次只算一天，按日子去重排序。"""

    ordered: dict[date, bool] = {}
    for visit in visits:
        # 同一天既有自然的又有被提醒的：这一天不算"只有被提醒之后才做"
        ordered[visit.day] = ordered.get(visit.day, True) and visit.prompted
    days = sorted(ordered)
    return tuple(
        gap_between(previous, following, watched=watched, counts=counts, censored=ordered[following])
        for previous, following in zip(days, days[1:], strict=False)
    )


def open_gap(last: date, until: date, *, watched: Watched = _fully, counts: Counts = _always) -> EffectiveGap | None:
    """从最后一次到 ``until`` 还没结束的那段空白；``until`` 不晚于最后一次时没有空白。"""

    if until <= last:
        return None
    return gap_between(last, until, watched=watched, counts=counts)


@dataclass(frozen=True)
class GapSummary:
    """一串间隔里没截尾的那些的长度分布：（长度, 个数）按长度升序、长度不重。久别要的个数、最长的、中位数都从它出
    （压实了的那一段只存它；同长的并成一项）。"""

    lengths: tuple[tuple[float, int], ...] = ()

    def __post_init__(self) -> None:
        sizes = [length for length, _count in self.lengths]
        if any(length < 0 or count < 1 for length, count in self.lengths) or sizes != sorted(set(sizes)):
            raise ValueError("a gap summary holds distinct non-negative lengths with positive counts")

    @property
    def count(self) -> int:
        return sum(count for _length, count in self.lengths)

    @property
    def longest(self) -> float:
        return self.lengths[-1][0] if self.lengths else 0.0

    @property
    def typical(self) -> float:
        """典型间隔：中位数（偶数个时取中间两个的平均）；没有间隔时为 0。"""

        total = self.count
        if total == 0:
            return 0.0
        return (self._ranked((total - 1) // 2) + self._ranked(total // 2)) / 2.0

    def _ranked(self, rank: int) -> float:
        """从短到长第 ``rank`` 个（从 0 数）间隔的长度。"""

        for length, count in self.lengths:
            if rank < count:
                return length
            rank -= count
        raise IndexError("rank beyond the summarised gaps")

    def including(self, gaps: Iterable[EffectiveGap]) -> GapSummary:
        added = Counter(gap.length for gap in gaps if not gap.censored)
        if not added:
            return self
        merged = Counter(dict(self.lengths))
        merged.update(added)
        return GapSummary(tuple(sorted(merged.items())))


def _history(prior: Sequence[EffectiveGap], earlier: GapSummary | None, min_samples: int) -> GapSummary | None:
    """``prior`` 里没截尾的连同 ``earlier`` 那一段的；不到 ``min_samples`` 个时为 None（不判）。"""

    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    summary = (earlier if earlier is not None else GapSummary()).including(prior)
    return summary if summary.count >= min_samples else None


def exceeds_longest(
    prior: Sequence[EffectiveGap], current: EffectiveGap, *, min_samples: int, earlier: GapSummary | None = None
) -> AbsenceVerdict:
    """``current`` 是不是比 ``prior`` 里没截尾的那些（连同 ``earlier`` 那一段的）都长；没截尾的不到 ``min_samples`` 个时不判。"""

    summary = _history(prior, earlier, min_samples)
    if summary is None:
        return AbsenceVerdict.UNDECIDED
    return AbsenceVerdict.EXCEEDS if current.length > summary.longest else AbsenceVerdict.WITHIN


def long_absence(
    prior: Sequence[EffectiveGap],
    current: EffectiveGap,
    *,
    min_samples: int,
    typical_multiple: float,
    earlier: GapSummary | None = None,
) -> AbsenceVerdict:
    """账的久别（裁定 88 第 2 条）：``current`` 比历史上没截尾的间隔都长，**且** 不短于 ``typical_multiple`` 个典型间隔
    （中位数）。``typical_multiple`` 由调用方给（账用遗忘半衰期 H：天天做 8 天、隔天做 16 天、每周一次 8 周）。"""

    if (
        isinstance(typical_multiple, bool)
        or not isinstance(typical_multiple, int | float)
        or not math.isfinite(typical_multiple)
        or typical_multiple <= 0
    ):
        raise ValueError("typical_multiple must be a positive finite number")
    summary = _history(prior, earlier, min_samples)
    if summary is None:
        return AbsenceVerdict.UNDECIDED
    longer = current.length > summary.longest
    return (
        AbsenceVerdict.EXCEEDS
        if longer and current.length >= typical_multiple * summary.typical
        else AbsenceVerdict.WITHIN
    )


__all__ = [
    "AbsenceVerdict",
    "EffectiveGap",
    "GapSummary",
    "Visit",
    "effective_gaps",
    "exceeds_longest",
    "gap_between",
    "long_absence",
    "open_gap",
]
