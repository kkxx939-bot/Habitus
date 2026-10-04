"""常态：这个人平常几点做某件事、做多久——映射时 ``baseline_for`` 要交出来的那张表。

**只算被声明过的键**。概念自己声明要哪些常态（``required_baseline_keys``：规则引用的 + 判据句引用的），
这里就只算那几个。不这么做的话，"每个概念 × 两种统计 × 两个窗"很快撞上命中记录里 32 个条目的上限
（``MAX_BASELINE_ENTRIES``），而多出来的那些没有任何人读。

**源是这个概念自己的命中历史，不是预测树**。两处理由：

1. 要**两个窗**（近期 / 历来，2026-09-27 裁定），而一棵树是按一套配置在一段历史上建出来的，给不出两个窗；
2. 常态是**按概念**问的（「就寝」的常态时刻），而树的键是 kind；概念跨哪几个 kind 由命中记录说，
   那就直接在命中记录上算，省掉一次转译。时长更是只有命中记录有（树上没有时长这个量）。

命中记录本来就带着 ``started_at``（URI 里）与 ``last_observed_at``，所以这一层不读行为树。

**不含当天**。今天这一条要和"今天以前的常态"比；把今天算进去就是拿它和自己比，晚睡那条规则会被自己稀释。

**样本不够就不给这个键**（`m2bos`/Habitus 的老规矩：样本少的时候输出"还没攒够"而不是噪声读数）。
键缺了，映射器会把引用它的候选记成**未决**——"材料没给到"，既不进分子也不进分母，这正是想要的；
编一个值出来会让一条错的判定看上去很确定。

**钟面时刻取环形中位数**：23:40 与 00:20 的常态是 00:00，不是 12:00。做法是先用单位向量求平均方向定一个参照，
再取各点相对参照的偏移（落在 ±12 小时内）的中位数，加回参照——对"跨午夜的就寝"这一类才说得通。

**一天多次的行为按峰各算**（2026-10-01，评审 C-15 / R3-26）：三餐的常态时刻取一个中位数是 12:00——谁都不在那个点吃饭。
有节律的概念把记录先归到它典型一天的峰（``rhythms`` 给的，与账本的窗口同一个定义，含同样的容差），每个峰一份近期 / 历来
常态；``baseline_for`` 按这条 occurrence 落在哪个峰交那一份，峰外的（``0``）与没节律的概念用不分峰的那一份。
**样本下限按独立天数**，不按条数：同一天命中三次只算攒了一天。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo
from types import MappingProxyType

from habitus.scene.concepts.model import (
    MINUTES_PER_DAY,
    BaselineKey,
    BaselineStatistic,
    BaselineWindow,
    ConceptSet,
    circular_offset,
    concept_identity,
    widen_windows,
)
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.occurrences.model import ConceptHits

#: 不分峰的那一层的峰号；有节律的概念峰外的记录也归到这里。
UNSPLIT = 0

#: 近期窗取多少天。14 天 ≈ 两周，够盖住工作日/周末的来回，又不至于把两个月前的作息算进"现在的习惯"。
#: **待定值**，54 天重放时按"判据翻转率"复核（窗太短会把一次熬夜当成新常态，太长就跟不上漂移）。
RECENT_WINDOW_DAYS = 14
#: 少于这么多次就不给这个键。3 与语义树别处的累积门槛同一个数：两次谈不上"常态"。
MIN_BASELINE_SAMPLES = 3
#: 近期与历来差多少才算"在漂"。60 分钟是**待定值**，等真实数据看分布再定。
DRIFT_MINUTES = 60


@dataclass(frozen=True)
class BaselineDrift:
    """一个概念的一种统计在两个窗上的差：近期 − 历来（时刻按环形差，分钟）。

    这是**信号**，不是判据（判据只比近期）：它回答"他的就寝在往后漂，漂了多少"，进 profile 的作息骨架，
    并且是闭环的第二个触发源（漂了就去问触点③ "什么导致的"）。
    """

    concept: str
    statistic: BaselineStatistic
    recent: float
    overall: float
    samples_recent: int
    samples_overall: int
    #: 哪个峰的漂移（``UNSPLIT`` = 不分峰那一层）。
    peak: int = UNSPLIT

    @property
    def minutes(self) -> float:
        if self.statistic is BaselineStatistic.USUAL_START:
            return circular_offset(self.recent, self.overall)
        return self.recent - self.overall

    @property
    def drifting(self) -> bool:
        return abs(self.minutes) >= DRIFT_MINUTES

    def render(self) -> str:
        direction = "晚" if self.minutes > 0 else "早"
        if self.statistic is BaselineStatistic.USUAL_DURATION:
            direction = "长" if self.minutes > 0 else "短"
        where = f"第 {self.peak} 个峰的" if self.peak != UNSPLIT else ""
        return (
            f"{self.concept}{where}{self.statistic.quantity}：近期比历来{direction} {abs(self.minutes):.0f} 分钟"
            f"（近期 {self.samples_recent} 天 / 历来 {self.samples_overall} 天）"
        )


@dataclass(frozen=True)
class BaselineSnapshot:
    """一天的常态表：``values`` 是不分峰那一层（键是 ``BaselineKey.text``）；``by_peak`` 是有节律概念按峰各算的那几份，
    ``values_at(minute)`` 把两层合成 ``baseline_for`` 要交的那一份——落在某个峰里就用那个峰的，否则用不分峰的。

    ``samples`` 记每个键背后几**天**——``missing`` 里是声明过、但样本不够或一次都没命中的键，
    它们**故意不在** ``values`` 里。
    """

    day: date
    values: Mapping[str, str]
    samples: Mapping[str, int]
    missing: tuple[str, ...] = ()
    drifts: tuple[BaselineDrift, ...] = ()
    #: (概念名, 峰号) → 那个峰的常态表（只含样本够的键）。
    by_peak: Mapping[tuple[str, int], Mapping[str, str]] = field(default_factory=dict)
    #: 概念名 → 它的峰（归峰用；含容差已经在 ``peaks`` 的判定里，这里只存形状）。
    peaks: Mapping[str, Rhythm] = field(default_factory=dict)
    slack_minutes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "samples", MappingProxyType(dict(self.samples)))
        object.__setattr__(self, "by_peak", MappingProxyType({key: MappingProxyType(dict(table)) for key, table in self.by_peak.items()}))
        object.__setattr__(self, "peaks", MappingProxyType(dict(self.peaks)))

    def values_at(self, minute_of_day: int) -> Mapping[str, str]:
        """这一刻该用的常态：每个概念落在哪个峰就用那个峰的那份，峰外 / 没节律的用不分峰的。"""

        merged = dict(self.values)
        for (concept, peak), table in self.by_peak.items():
            if peak_of(self.peaks.get(concept), minute_of_day, slack_minutes=self.slack_minutes) == peak:
                merged.update(table)
        return MappingProxyType(merged)

    @property
    def drifting(self) -> tuple[BaselineDrift, ...]:
        return tuple(item for item in self.drifts if item.drifting)


def declared_keys(concepts: ConceptSet) -> tuple[BaselineKey, ...]:
    """概念集里被声明过的常态键，去重、按文本排序。没有概念声明的键不算，也不去算。"""

    if not isinstance(concepts, ConceptSet):
        raise TypeError("concepts must be a ConceptSet")
    texts = {key for identity in concepts for key in concepts[identity].required_baseline_keys}
    return tuple(sorted((BaselineKey.parse(text) for text in texts), key=lambda key: key.text))


def peak_of(rhythm: Rhythm | None, minute_of_day: int, *, slack_minutes: int = 0) -> int:
    """钟面上这一刻落在这个概念的第几个峰（含容差）；峰外或没节律 → ``UNSPLIT``。与账本窗口的判定同一套算法。"""

    if rhythm is None:
        return UNSPLIT
    spans = widen_windows([(peak.start_minute, peak.end_minute) for peak in rhythm.peaks], slack_minutes)
    for peak, (start, end) in zip(rhythm.peaks, spans, strict=True):
        if start <= minute_of_day < end or start <= minute_of_day + MINUTES_PER_DAY < end or start <= minute_of_day - MINUTES_PER_DAY < end:
            return peak.ordinal
    return UNSPLIT


def local_minute(moment: datetime, timezone: tzinfo | None = None) -> int:
    """墙钟分钟（0–1439）。常态与窗口都在本地钟面上算。"""

    local = moment if timezone is None else moment.astimezone(timezone)
    return local.hour * 60 + local.minute


def baseline_table(
    records: Iterable[ConceptHits],
    concepts: ConceptSet,
    *,
    day: date,
    recent_days: int = RECENT_WINDOW_DAYS,
    min_samples: int = MIN_BASELINE_SAMPLES,
    rhythms: Mapping[str, Rhythm] | None = None,
    slack_minutes: int = 0,
    timezone: tzinfo | None = None,
) -> BaselineSnapshot:
    """算出 ``day`` 那天该用的常态表。``records`` 是 ``day`` **之前**的命中记录（含不含都按日期筛）。

    沿 parent 链聚合：命中「修改代码」也算「写代码」一次，所以上级概念的常态有样本可算。
    ``rhythms`` 给了就对有节律的概念按峰各算一份（键是概念名）；``min_samples`` 数的是**独立天数**。
    """

    if isinstance(recent_days, bool) or not isinstance(recent_days, int) or recent_days <= 0:
        raise ValueError("recent_days must be a positive integer")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples <= 0:
        raise ValueError("min_samples must be a positive integer")
    keys = declared_keys(concepts)
    wanted = {key.concept: concept_identity(key.concept) for key in keys}
    if not keys:
        return BaselineSnapshot(day=day, values={}, samples={})
    # 声明了近期窗的键，历来窗由算法**自动陪算**（2026-09-27 裁定"判据用近期，历来常态只读漂移"）：
    # 判据只许比近期，所以没有任何概念会声明 `all`，不陪算的话漂移永远算不出来（评审 B-12 / C-12）。
    # 陪算出来的只进漂移，**不进 values**——不给映射器、不占材料上限。
    shadow = tuple(
        BaselineKey(concept=key.concept, statistic=key.statistic, window=BaselineWindow.ALL)
        for key in keys
        if key.window is BaselineWindow.RECENT and BaselineKey(concept=key.concept, statistic=key.statistic, window=BaselineWindow.ALL) not in keys
    )
    cutoff = day - timedelta(days=recent_days)
    split = {name: rhythm for name, rhythm in (rhythms or {}).items() if name in wanted and rhythm.peaks}
    samples: dict[tuple[str, BaselineWindow, int], list[tuple[float, float, date]]] = {}
    for record in records:
        started = record.started_at if timezone is None else record.started_at.astimezone(timezone)
        if started.date() >= day:
            continue  # 今天不算自己的常态
        windows = (BaselineWindow.ALL,) if started.date() < cutoff else (BaselineWindow.ALL, BaselineWindow.RECENT)
        minute = started.hour * 60 + started.minute
        point = (float(minute), record.duration_minutes, started.date())
        for concept in _concepts_of(record, concepts, wanted):
            strata = [UNSPLIT]
            peak = peak_of(split.get(concept), minute, slack_minutes=slack_minutes)
            if peak != UNSPLIT:
                strata.append(peak)
            for window in windows:
                for stratum in strata:
                    samples.setdefault((concept, window, stratum), []).append(point)
    return _snapshot(keys, samples, day=day, min_samples=min_samples, shadow=shadow, rhythms=split, slack_minutes=slack_minutes)


def _concepts_of(record: ConceptHits, concepts: ConceptSet, wanted: Mapping[str, str]) -> tuple[str, ...]:
    """这条记录该记到哪些被声明过的概念名下：命中的叶子 + 它们的祖先，取交集。"""

    names = [hit.concept for hit in record.hits if hit.concept in concepts]
    if not names:
        return ()
    closed = concepts.with_ancestors(names)
    return tuple(name for name, identity in wanted.items() if identity in closed)


def _snapshot(
    keys: Sequence[BaselineKey],
    samples: Mapping[tuple[str, BaselineWindow, int], Sequence[tuple[float, float, date]]],
    *,
    day: date,
    min_samples: int,
    shadow: Sequence[BaselineKey] = (),
    rhythms: Mapping[str, Rhythm] | None = None,
    slack_minutes: int = 0,
) -> BaselineSnapshot:
    values: dict[str, str] = {}
    counts: dict[str, int] = {}
    missing: list[str] = []
    by_peak: dict[tuple[str, int], dict[str, str]] = {}
    medians: dict[tuple[str, BaselineStatistic, BaselineWindow, int], tuple[float, int]] = {}
    strata = {(concept, stratum) for concept, _window, stratum in samples}
    for key in keys:
        for stratum in sorted(stratum for concept, stratum in strata if concept == key.concept) or [UNSPLIT]:
            points = samples.get((key.concept, key.window, stratum), ())
            days = _days_of(points)
            if days < min_samples:
                if stratum == UNSPLIT:
                    missing.append(key.text)
                continue
            median = _median_of(points, key.statistic)
            medians[(key.concept, key.statistic, key.window, stratum)] = (median, days)
            if stratum == UNSPLIT:
                values[key.text] = _render(key.statistic, median)
                counts[key.text] = days
            else:
                by_peak.setdefault((key.concept, stratum), {})[key.text] = _render(key.statistic, median)
    for key in shadow:
        # 陪算的历来窗：只为漂移，不落 values、不记 missing（它不是谁要的材料）。
        for stratum in sorted(stratum for concept, stratum in strata if concept == key.concept):
            points = samples.get((key.concept, key.window, stratum), ())
            if _days_of(points) >= min_samples:
                medians[(key.concept, key.statistic, key.window, stratum)] = (_median_of(points, key.statistic), _days_of(points))
    return BaselineSnapshot(
        day=day, values=values, samples=counts, missing=tuple(missing), drifts=_drifts(medians),
        by_peak=by_peak, peaks=rhythms or {}, slack_minutes=slack_minutes,
    )


def _days_of(points: Sequence[tuple[float, float, date]]) -> int:
    return len({occurred_on for _start, _duration, occurred_on in points})


def _median_of(points: Sequence[tuple[float, float, date]], statistic: BaselineStatistic) -> float:
    if statistic is BaselineStatistic.USUAL_START:
        return _circular_median([start for start, _duration, _day in points])
    return _median([duration for _start, duration, _day in points])


def _drifts(
    medians: Mapping[tuple[str, BaselineStatistic, BaselineWindow, int], tuple[float, int]],
) -> tuple[BaselineDrift, ...]:
    """两个窗都算出来的才有漂移可说；有节律的概念按峰各报（不分峰那一层的漂移对三餐这类是假的，有峰就不报它）。"""

    found = []
    split = {concept for concept, _statistic, _window, stratum in medians if stratum != UNSPLIT}
    for (concept, statistic, window, stratum), (median, count) in sorted(
        medians.items(), key=lambda item: (item[0][0], item[0][1].value, item[0][3])
    ):
        if window is not BaselineWindow.RECENT or (stratum == UNSPLIT and concept in split):
            continue
        overall = medians.get((concept, statistic, BaselineWindow.ALL, stratum))
        if overall is None:
            continue
        found.append(
            BaselineDrift(
                concept=concept,
                statistic=statistic,
                recent=median,
                overall=overall[0],
                samples_recent=count,
                samples_overall=overall[1],
                peak=stratum,
            )
        )
    return tuple(found)


def _render(statistic: BaselineStatistic, median: float) -> str:
    if statistic is BaselineStatistic.USUAL_START:
        hour, minute = divmod(int(round(median)) % MINUTES_PER_DAY, 60)
        return f"{hour:02d}:{minute:02d}"
    return f"{median:.0f}"


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _circular_median(minutes: Sequence[float]) -> float:
    """钟面时刻的中位数：先用平均方向定参照，再取相对参照偏移的中位数，加回去。"""

    radians = [value / MINUTES_PER_DAY * 2.0 * math.pi for value in minutes]
    mean_angle = math.atan2(
        sum(math.sin(item) for item in radians) / len(radians), sum(math.cos(item) for item in radians) / len(radians)
    )
    reference = (mean_angle / (2.0 * math.pi) * MINUTES_PER_DAY) % MINUTES_PER_DAY
    offsets = [circular_offset(value, reference) for value in minutes]
    return (reference + _median(offsets)) % MINUTES_PER_DAY


__all__ = [
    "DRIFT_MINUTES",
    "MIN_BASELINE_SAMPLES",
    "RECENT_WINDOW_DAYS",
    "BaselineDrift",
    "BaselineSnapshot",
    "UNSPLIT",
    "baseline_table",
    "declared_keys",
    "local_minute",
    "peak_of",
]
