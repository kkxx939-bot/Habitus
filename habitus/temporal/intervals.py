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
  **簇往后接到哪天为止**（裁定 89 第 1 条）：簇只在"到这天为止他来了的在险间隔没超过线 θ"时才往后接下一天——到 e 那天就来了的
  已经超过 θ，簇就只有 e。θ 就是账的那条线（1 − 可接受的误提醒率）：时窗末提醒，"到这天他还没来"里有多少是他的节奏、本来明天会来，
  就是这一次提醒错的比例；低于可接受的误提醒率，就把没来当漏做、在这天提醒，到了或超过才把下一天并进来（恰好在线上时宁可不提醒）。天天做的事偶尔隔一次 2 天
  因此不再把时窗撑成两天；1、2 天真混着的（只有一半是隔 1 天）照旧连成一簇。这里的"没来"按 ``seen`` 计：跨观测空白的长间隔
  只有看得见的那部分算"没来"，看不见的那天不当成他没做（整天没会话、上游报的未观测都一样），不会造出假的"隔 2 天"；
  截尾的间隔若落在已接进的天数之内，不知道他自己会不会在这之前来，两边都不算。
- **这一天归哪条时窗**（``opened``）：簇往后接被线截住以后，含 e 的整簇（``block``）可能分成先后几段时窗——天天做、漏过几次的
  事，离上次 1 天那条只管当天，离上次 2 天是另一段。从整簇第一天起按每一天自己的读数往后走：哪一天读出的簇盖住了 e，e 就归
  那一天开的时窗（同一段时窗哪天重算都是同一个身份）；盖不住就从它的下一天再开一段。没被截住的（1、2 天真混着）整簇仍是一段。
- **连漏、超过了历史**（裁定 90，``overdue_at``）：离上次 e 天，历史上不短于 e 的间隔已不到 ``MIN_AT_RISK`` 个——常规读法在
  这里再也读不出，他已经比历来（够得上读数的）都隔得久。这时按常见间隔 T（``median_days``）往后继续：从历史读得到的最后一天
  （第 ``MIN_AT_RISK`` 长的间隔）起，每过 T 天一条，每条只管那一天。天天做的每天一条，每周一次的每 7 天一条；
  什么时候不再开口不在这里定（账的"规律断了"与提醒反馈）。
- **久别**：判法在 ``temporal.absence``（有效长度按观测覆盖算、截尾作下界、样本门槛由调用方给），这里不再另有一份。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    #: 盖住 ``elapsed`` 的那段时窗从离上次几天开始（模块说明「这一天归哪条时窗」）：触发实例的身份与时窗起点都从它取。
    opened: int


def cluster_at(intervals: Sequence[DayInterval], elapsed: int, *, line: float) -> ClusterReading | None:
    """旧「离上次够久」的读法：历史上没恰好隔 ``elapsed`` 天、在险的不到 2 个、簇里没有自然命中，都没有读数。
    ``line`` 是账的那条线 θ：簇到这天为止接住的在险间隔已超过 θ，就不再往后接（模块说明「簇往后接到哪天为止」）。"""

    _require_line(line)
    found = _reading(intervals, elapsed, line=line)
    if found is None:
        return None
    return replace(found, opened=_opened(intervals, found, line=line))


def overdue_at(intervals: Sequence[DayInterval], elapsed: int, *, line: float) -> ClusterReading | None:
    """连漏、已超过历史（模块说明「连漏、超过了历史」，裁定 90）：离上次 ``elapsed`` 天时不短于它的间隔已不到 ``MIN_AT_RISK``
    个，且正落在"历史读得到的最后一天 + k × 常见间隔"上，才有读数；这一条只管这一天。命中与机会借常见间隔那一天的读数
    （他平时隔 T 天来的那几次：时窗的钟点从它们取）——那一天本身读不出时也没有读数（样本少就没有读数）。"""

    _require_line(line)
    lengths = sorted((item.days for item in intervals), reverse=True)
    if len(lengths) < MIN_AT_RISK or elapsed <= lengths[MIN_AT_RISK - 1]:
        return None  # 历史还读得到这里（或历史太短）：归 ``cluster_at``
    typical = median_days(intervals)
    if typical is None or (elapsed - lengths[MIN_AT_RISK - 1]) % int(typical):
        return None
    usual = _reading(intervals, int(typical), line=line)
    if usual is None:
        return None
    return replace(usual, elapsed=elapsed, cluster=(elapsed,), block=(elapsed,), opened=elapsed)


def _require_line(line: float) -> None:
    if not 0.0 < line < 1.0:
        raise ValueError("line must lie in (0, 1)")


def _reading(intervals: Sequence[DayInterval], elapsed: int, *, line: float) -> ClusterReading | None:
    gaps = [item.days for item in intervals]
    at_risk = [item for item in intervals if item.days >= elapsed]
    if elapsed not in gaps:
        return None
    cluster = _cluster(sorted({item.days for item in at_risk}), elapsed, at_risk=at_risk, line=line)
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
        opened=block[0],
    )


def _opened(intervals: Sequence[DayInterval], reading: ClusterReading, *, line: float) -> int:
    """从整簇第一天起往后走：哪一天读出的簇盖住了 ``elapsed``，就归那一天开的时窗；那一天读不出（没建过预期）就看下一天。"""

    day = reading.block[0]
    while day < reading.elapsed:
        earlier = _reading(intervals, day, line=line)
        if earlier is not None and earlier.cluster[-1] >= reading.elapsed:
            return day
        day = earlier.cluster[-1] + 1 if earlier is not None else day + 1
    return reading.elapsed


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


def _cluster(
    distinct_gaps: Sequence[int], elapsed: int, *, at_risk: Sequence[DayInterval], line: float
) -> tuple[int, ...]:
    """从 ``elapsed`` 起、相差 1 天连着的那一簇（29、30 是一簇；2、3、5 里从 2 起是 2、3），接到"到这天为止来了的超过 θ"为止。"""

    found = [elapsed]
    for gap in distinct_gaps:
        if gap > found[-1] and gap - found[-1] <= 1 and _came_by(at_risk, found[-1]) <= line:
            found.append(gap)
    return tuple(found)


def _came_by(at_risk: Sequence[DayInterval], end: int) -> float:
    """在险的间隔里，到离上次 ``end`` 天为止他自己来了的占比：来了的（没截尾、不长于 ``end``）记 1，没来的（长于 ``end``）
    按 ``seen`` 记——看不见的那部分不当成没来；截尾且不长于 ``end`` 的不知道，不算。两边都是空的没有读数，当作已接住、不再往后接。"""

    came = sum(1.0 for item in at_risk if item.days <= end and not item.censored)
    missed = sum(item.seen for item in at_risk if item.days > end)
    total = came + missed
    return 1.0 if total <= 0.0 else came / total


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
    "overdue_at",
]
