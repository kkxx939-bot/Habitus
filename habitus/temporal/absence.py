"""判缺失：这一次的空白比它历史上最长的一次间隔还长（裁定 57 原文；裁定 60）。全仓只此一份。

- **间隔的有效长度**：空白里每天"看着的比例"之和（从上一次的次日到这一次那天）。人不在、设备没上传的那几天不算长度，
  所以中间 60 天未观测的那段不会把"历史最长"抬到 91 天。只用观测覆盖，不乘日内分布——覆盖是事实、不随窗口漂，不需要落盘状态。
  和 ``intervals.DayInterval.seen``（"这段没漏看的概率"，连乘）是两个量，不混用。
- **只数某些日型**（可选）：账的久别只数这本账有过机会的日型的日子（裁定 59"默认不外推"）——只在工作日做的事，国庆七天不算超过。
- **截尾**：以"被提醒之后才做"结尾的那段不进"历史最长"（不知道他自己会隔多久做）；但它自己是"这一次"且已经长过门槛时照样算超过。
- **样本不够不判**：账的久别要 2 个间隔（``MIN_HISTORY``，代价只是一次影子）；语义树的"久别重来"判的是一段，要
  ⌈1/α⌉ − 1 个（下一次比历来都长的机会是 1/(n+1)，只有它不大于 α 才算数）。门槛由调用方给。
"""

from __future__ import annotations

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
    """一串间隔里没截尾的有几个、最长的多长——"超过历史最长"只要这两个数（压实了的那一段只存它们）。"""

    count: int = 0
    longest: float = 0.0

    def __post_init__(self) -> None:
        if self.count < 0 or self.longest < 0 or (self.count == 0 and self.longest != 0.0):
            raise ValueError("a gap summary counts its gaps with their longest length")

    def including(self, gaps: Iterable[EffectiveGap]) -> GapSummary:
        lengths = [gap.length for gap in gaps if not gap.censored]
        if not lengths:
            return self
        return GapSummary(self.count + len(lengths), max(self.longest, *lengths))


def exceeds_longest(
    prior: Sequence[EffectiveGap], current: EffectiveGap, *, min_samples: int, earlier: GapSummary | None = None
) -> AbsenceVerdict:
    """``current`` 是不是比 ``prior`` 里没截尾的那些（连同 ``earlier`` 那一段的）都长；没截尾的不到 ``min_samples`` 个时不判。"""

    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    summary = (earlier if earlier is not None else GapSummary()).including(prior)
    if summary.count < min_samples:
        return AbsenceVerdict.UNDECIDED
    return AbsenceVerdict.EXCEEDS if current.length > summary.longest else AbsenceVerdict.WITHIN


__all__ = [
    "AbsenceVerdict",
    "EffectiveGap",
    "GapSummary",
    "Visit",
    "effective_gaps",
    "exceeds_longest",
    "gap_between",
    "open_gap",
]
