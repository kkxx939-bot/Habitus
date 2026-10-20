"""间隔：合并以后相邻两次隔多久，以及由它读出的常见间隔、成簇与命中、久别门槛（裁定 59 第 6 条第 2 步）。

- **单位**：按发生的日子算（一类一天取第一次），相邻两个发生日之间隔几个本地日历日——6.99 天是 7 天，也不受时区与夏令时影响。
  一天之内的几次由合并与密度说明，这里不再细分（现在所有读间隔的人都按天：块长、久别、离上次够久、补做强度）。
- **被提醒之后才做**：计时照样从它重新开始（裁定 54：离上次多久按全部发生算），但以它结尾的那段间隔是**右截尾**——
  不知道他自己会隔多久做，只知道"至少这么久还没自己做"：进在险的分母，不算命中，不进中位数与久别门槛。
- **跨了观测空白**：不剔除（剔除会让长间隔更容易被剔、系统性压短）。算这段间隔里漏看一次的概率 q：空白那几天里，
  这一类在没看的那些槽本来会开始的占比（日内分布）× 它的发生日占比（密度）；``seen = 1 − q`` 当权重——q 只进在险、不算命中，
  1 − q 照常进中位数与命中。
- **成簇与命中**：沿用旧「离上次够久」（``foresight/triggers/days.py``）：离上次 e 天，历史上恰好隔 e 天出现过、且不短于 e 的
  在险间隔 ≥2 个，才有读数；从 e 起相差 1 天连着的间隔是一簇；含 e 的整簇的第一天是触发实例的身份。
- **久别**：判法在 ``temporal.absence``（有效长度按观测覆盖算、截尾作下界、样本门槛由调用方给），这里不再另有一份。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from habitus.temporal.coverage import Coverage
from habitus.temporal.episodes import OccurrenceDay
from habitus.temporal.profile import DayProfile
from habitus.temporal.slots import slots_per_day

#: 在险间隔至少几个才有读数（旧「离上次够久」的门槛，"样本少就没有读数"的最小表达）。
MIN_AT_RISK = 2


@dataclass(frozen=True)
class DayInterval:
    """相邻两个发生日之间：隔几天、这段里没漏看的概率、结尾那天是不是只有被提醒之后才做（截尾）。"""

    previous: OccurrenceDay
    following: OccurrenceDay
    days: int
    seen: float
    censored: bool

    def __post_init__(self) -> None:
        if self.days < 1:
            raise ValueError("a day interval is at least one day")
        if not 0.0 <= self.seen <= 1.0:
            raise ValueError("seen must lie in [0, 1]")


def day_intervals(
    days: Sequence[OccurrenceDay],
    coverage: Coverage,
    *,
    profile: DayProfile | None,
    occurrence_day_share: float | None,
) -> tuple[DayInterval, ...]:
    """按天的发生 → 相邻的间隔。``profile`` / ``occurrence_day_share`` 没有时，漏看概率按全天平均、发生日占比按 1 估（偏保守）。"""

    per_day = slots_per_day(coverage.slot_minutes)
    shares = profile.shares if profile is not None else tuple(1.0 / per_day for _ in range(per_day))
    share = 1.0 if occurrence_day_share is None else occurrence_day_share
    found = []
    for previous, following in zip(days, days[1:], strict=False):
        gap = (following.day - previous.day).days
        if gap < 1:
            raise ValueError("occurrence days must be distinct and ascending")
        unseen = 1.0
        day = previous.day + timedelta(days=1)
        while day < following.day:
            watched = coverage.days.get(day)
            if watched is not None:
                missed = sum(weight * (1.0 - looked) for weight, looked in zip(shares, watched, strict=True))
                unseen *= 1.0 - min(share * missed, 1.0)
            day += timedelta(days=1)
        found.append(DayInterval(previous, following, gap, unseen, following.natural == 0))
    return tuple(found)


@dataclass(frozen=True)
class ClusterReading:
    """离上次 ``elapsed`` 天时：从它起连着的那一簇、含它的整簇、簇里的自然命中（几段与按 seen 加权）、在险的机会数。"""

    elapsed: int
    cluster: tuple[int, ...]
    block: tuple[int, ...]
    hits: tuple[DayInterval, ...]
    hit_weight: float
    opportunities: int


def cluster_at(intervals: Sequence[DayInterval], elapsed: int) -> ClusterReading | None:
    """旧「离上次够久」的读法：历史上没恰好隔 ``elapsed`` 天、在险的不到 2 个、簇里没有自然命中，都没有读数。"""

    gaps = [item.days for item in intervals]
    at_risk = [item for item in intervals if item.days >= elapsed]
    if elapsed not in gaps:
        return None
    cluster = _cluster(sorted({item.days for item in at_risk}), elapsed)
    block = _block(sorted(set(gaps)), elapsed)
    inside = [item for item in intervals if item.days in cluster]
    hits = tuple(item for item in inside if not item.censored)
    if not hits:
        return None  # 恰好隔这么久的那几次全是被提醒才做的：没有自然命中
    # 在险按扣掉簇里截尾的那几段之后数：截尾的那段不知道他自己会不会来，不是机会（2026-10-18 四方评审：原来扣之前判，
    # 一段自然一段截尾也给了 1/1 的读数）
    opportunities = len(at_risk) - sum(1 for item in inside if item.censored)
    if opportunities < MIN_AT_RISK:
        return None
    return ClusterReading(
        elapsed=elapsed,
        cluster=cluster,
        block=block,
        hits=hits,
        hit_weight=sum(item.seen for item in hits),
        opportunities=opportunities,
    )


def median_days(intervals: Sequence[DayInterval]) -> float | None:
    """常见间隔：没截尾的间隔按 seen 加权的中位数；一个都没有就是 None。"""

    observed = sorted((item.days, item.seen) for item in intervals if not item.censored and item.seen > 0.0)
    total = sum(weight for _days, weight in observed)
    if total <= 0.0:
        return None
    running = 0.0
    for days, weight in observed:
        running += weight
        if running >= total / 2.0:
            return float(days)
    return float(observed[-1][0])


def gap_days(days: Sequence[date]) -> tuple[int, ...]:
    """一串发生的日子（重复的算一天）相邻之间隔几个日历日。"""

    distinct = sorted(set(days))
    return tuple((later - earlier).days for earlier, later in zip(distinct, distinct[1:], strict=False))


def common_interval_days(days: Sequence[date]) -> float | None:
    """常见间隔的不加权读法：相邻发生日间隔的中位数。给只有日子、没有观测覆盖的调用方用（语义树的块长，概念可以是几个类的汇总）；
    与 ``median_days`` 同一个定义，只是没有漏看与截尾的权重。不到两个发生日就是 None。"""

    gaps = sorted(gap_days(days))
    if not gaps:
        return None
    return float(gaps[(len(gaps) - 1) // 2])  # 下中位数：与 ``median_days`` 同一个定义（7 与 28 → 7，不是 17.5）


def _cluster(distinct_gaps: Sequence[int], elapsed: int) -> tuple[int, ...]:
    """从 ``elapsed`` 起、相差 1 天连着的那一簇（29、30 是一簇；2、3、5 里从 2 起是 2、3）。"""

    found = [elapsed]
    for gap in distinct_gaps:
        if gap > found[-1] and gap - found[-1] <= 1:
            found.append(gap)
    return tuple(found)


def _block(distinct_gaps: Sequence[int], elapsed: int) -> tuple[int, ...]:
    """含 ``elapsed`` 的那一整簇（往前往后都按相差 1 天连着）：同一簇哪天重算都是同一个身份。"""

    found = [elapsed]
    for gap in sorted(distinct_gaps, reverse=True):
        if gap < found[0] and found[0] - gap <= 1:
            found.insert(0, gap)
    for gap in sorted(distinct_gaps):
        if gap > found[-1] and gap - found[-1] <= 1:
            found.append(gap)
    return tuple(found)


__all__ = [
    "MIN_AT_RISK",
    "ClusterReading",
    "DayInterval",
    "cluster_at",
    "common_interval_days",
    "day_intervals",
    "gap_days",
    "median_days",
]
