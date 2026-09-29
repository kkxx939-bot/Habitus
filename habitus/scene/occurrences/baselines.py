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
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from types import MappingProxyType

from habitus.scene.concepts.model import (
    MINUTES_PER_DAY,
    BaselineKey,
    BaselineStatistic,
    BaselineWindow,
    ConceptSet,
    circular_offset,
    concept_identity,
)
from habitus.scene.occurrences.model import ConceptHits

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
        return (
            f"{self.concept}的{self.statistic.quantity}：近期比历来{direction} {abs(self.minutes):.0f} 分钟"
            f"（近期 {self.samples_recent} 次 / 历来 {self.samples_overall} 次）"
        )


@dataclass(frozen=True)
class BaselineSnapshot:
    """一天的常态表：``values`` 直接就是 ``baseline_for`` 的返回值（键是 ``BaselineKey.text``）。

    ``samples`` 记每个键背后几次命中——``missing`` 里是声明过、但样本不够或一次都没命中的键，
    它们**故意不在** ``values`` 里。
    """

    day: date
    values: Mapping[str, str]
    samples: Mapping[str, int]
    missing: tuple[str, ...] = ()
    drifts: tuple[BaselineDrift, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "samples", MappingProxyType(dict(self.samples)))

    @property
    def drifting(self) -> tuple[BaselineDrift, ...]:
        return tuple(item for item in self.drifts if item.drifting)


def declared_keys(concepts: ConceptSet) -> tuple[BaselineKey, ...]:
    """概念集里被声明过的常态键，去重、按文本排序。没有概念声明的键不算，也不去算。"""

    if not isinstance(concepts, ConceptSet):
        raise TypeError("concepts must be a ConceptSet")
    texts = {key for identity in concepts for key in concepts[identity].required_baseline_keys}
    return tuple(sorted((BaselineKey.parse(text) for text in texts), key=lambda key: key.text))


def baseline_table(
    records: Iterable[ConceptHits],
    concepts: ConceptSet,
    *,
    day: date,
    recent_days: int = RECENT_WINDOW_DAYS,
    min_samples: int = MIN_BASELINE_SAMPLES,
) -> BaselineSnapshot:
    """算出 ``day`` 那天该用的常态表。``records`` 是 ``day`` **之前**的命中记录（含不含都按日期筛）。

    沿 parent 链聚合：命中「修改代码」也算「写代码」一次，所以上级概念的常态有样本可算。
    """

    if isinstance(recent_days, bool) or not isinstance(recent_days, int) or recent_days <= 0:
        raise ValueError("recent_days must be a positive integer")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples <= 0:
        raise ValueError("min_samples must be a positive integer")
    keys = declared_keys(concepts)
    wanted = {key.concept: concept_identity(key.concept) for key in keys}
    if not keys:
        return BaselineSnapshot(day=day, values={}, samples={})
    cutoff = day - timedelta(days=recent_days)
    samples: dict[tuple[str, BaselineWindow], list[tuple[float, float]]] = {}
    for record in records:
        started = record.started_at
        if started.date() >= day:
            continue  # 今天不算自己的常态
        windows = (BaselineWindow.ALL,) if started.date() < cutoff else (BaselineWindow.ALL, BaselineWindow.RECENT)
        point = (float(started.hour * 60 + started.minute), record.duration_minutes)
        for concept in _concepts_of(record, concepts, wanted):
            for window in windows:
                samples.setdefault((concept, window), []).append(point)
    return _snapshot(keys, samples, day=day, min_samples=min_samples)


def _concepts_of(record: ConceptHits, concepts: ConceptSet, wanted: Mapping[str, str]) -> tuple[str, ...]:
    """这条记录该记到哪些被声明过的概念名下：命中的叶子 + 它们的祖先，取交集。"""

    names = [hit.concept for hit in record.hits if hit.concept in concepts]
    if not names:
        return ()
    closed = concepts.with_ancestors(names)
    return tuple(name for name, identity in wanted.items() if identity in closed)


def _snapshot(
    keys: Sequence[BaselineKey],
    samples: Mapping[tuple[str, BaselineWindow], Sequence[tuple[float, float]]],
    *,
    day: date,
    min_samples: int,
) -> BaselineSnapshot:
    values: dict[str, str] = {}
    counts: dict[str, int] = {}
    missing: list[str] = []
    medians: dict[tuple[str, BaselineStatistic, BaselineWindow], tuple[float, int]] = {}
    for key in keys:
        points = samples.get((key.concept, key.window), ())
        if len(points) < min_samples:
            missing.append(key.text)
            continue
        median = _median_of(points, key.statistic)
        medians[(key.concept, key.statistic, key.window)] = (median, len(points))
        values[key.text] = _render(key.statistic, median)
        counts[key.text] = len(points)
    return BaselineSnapshot(day=day, values=values, samples=counts, missing=tuple(missing), drifts=_drifts(medians))


def _median_of(points: Sequence[tuple[float, float]], statistic: BaselineStatistic) -> float:
    if statistic is BaselineStatistic.USUAL_START:
        return _circular_median([start for start, _duration in points])
    return _median([duration for _start, duration in points])


def _drifts(
    medians: Mapping[tuple[str, BaselineStatistic, BaselineWindow], tuple[float, int]],
) -> tuple[BaselineDrift, ...]:
    """两个窗都算出来的才有漂移可说。"""

    found = []
    for (concept, statistic, window), (median, count) in sorted(
        medians.items(), key=lambda item: (item[0][0], item[0][1].value)
    ):
        if window is not BaselineWindow.RECENT:
            continue
        overall = medians.get((concept, statistic, BaselineWindow.ALL))
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
    "baseline_table",
    "declared_keys",
]
