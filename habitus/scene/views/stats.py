"""读时统计的几个基元：Beta 后验区间、确定性按块 bootstrap 区间、Kaplan–Meier。

不依赖 numpy / scipy（项目锁死的依赖里没有）。正则化不完全 Beta 函数用 Lentz 连分式（Numerical Recipes
的 ``betacf``），分位数用二分。全部确定性：bootstrap 的随机数由调用方给种子，同一本账算两遍是同一个区间。

**按块重采样**：天与天不独立（熬夜成簇、不吃早饭也成簇），把同一块里的机会当独立样本会把区间算窄（评审实测
簇内相关 ρ=0.3 时名义 n=20 的 90% 区间真实覆盖只有 37.5%）。块的定义与累积度门槛共用一个（块长 = max(1 天,
前件复发间隔 p50)），重采样以块为单位。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

_MAX_ITERATIONS = 300
_EPSILON = 3.0e-14
_TINY = 1.0e-300


class StatsError(ValueError):
    """统计基元的输入不合法。"""


@dataclass(frozen=True)
class Interval:
    """点估计 + [low, high]。"""

    point: float
    low: float
    high: float

    def __post_init__(self) -> None:
        for label, value in (("point", self.point), ("low", self.low), ("high", self.high)):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise StatsError(f"interval {label} must be a finite number")
        if not self.low <= self.high:
            raise StatsError("interval low must not exceed high")

    @property
    def contains_zero(self) -> bool:
        return self.low <= 0.0 <= self.high

    @property
    def width(self) -> float:
        return self.high - self.low

    def disjoint_from(self, other: Interval) -> bool:
        return self.high < other.low or other.high < self.low


def beta_cdf(x: float, a: float, b: float) -> float:
    """I_x(a, b)，正则化不完全 Beta。"""

    if a <= 0 or b <= 0:
        raise StatsError("beta parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _continued_fraction(x, a, b) / a
    return 1.0 - math.exp(log_front) * _continued_fraction(1.0 - x, b, a) / b


def _continued_fraction(x: float, a: float, b: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) >= _TINY else _TINY)
    h = d
    for m in range(1, _MAX_ITERATIONS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) >= _TINY else _TINY
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) >= _TINY else _TINY
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            break
    return h


def beta_quantile(q: float, a: float, b: float) -> float:
    """Beta(a, b) 的 q 分位数，二分到 1e-10。"""

    if not 0.0 < q < 1.0:
        raise StatsError("quantile level lies in (0, 1)")
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if beta_cdf(mid, a, b) < q:
            low = mid
        else:
            high = mid
        if high - low < 1e-10:
            break
    return (low + high) / 2.0


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise StatsError("median of nothing")
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def mean(values: Sequence[float]) -> float:
    if not values:
        raise StatsError("mean of nothing")
    return sum(values) / len(values)


def block_bootstrap(
    items: Sequence[T],
    blocks: Sequence[int],
    statistic: Callable[[Sequence[T]], float | None],
    *,
    seed: str,
    samples: int = 1000,
    level: float = 0.90,
) -> Interval | None:
    """按块重采样的确定性 bootstrap：``blocks[i]`` 是 ``items[i]`` 所属的块；每次重采样抽同样多个块、把块里的东西
    全带上，再算 ``statistic``。统计量算不出（返回 None）的那次重采样丢掉。点估计是原样本的统计量；原样本算不出返回 None。
    """

    if len(items) != len(blocks):
        raise StatsError("items and blocks must align")
    if not items:
        raise StatsError("bootstrap of nothing")
    point = statistic(items)
    if point is None or not math.isfinite(point):
        return None
    groups: dict[int, list[T]] = {}
    for item, block in zip(items, blocks, strict=True):
        groups.setdefault(block, []).append(item)
    keys = sorted(groups)
    if len(keys) == 1:
        return Interval(point, point, point)
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        chosen = generator.choices(keys, k=len(keys))
        resampled = [item for key in chosen for item in groups[key]]
        value = statistic(resampled)
        if value is not None and math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return Interval(point, point, point)
    estimates.sort()
    tail = (1.0 - level) / 2.0
    low = estimates[min(len(estimates) - 1, int(tail * len(estimates)))]
    high = estimates[min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))]
    return Interval(point, min(low, point), max(high, point))


@dataclass(frozen=True)
class EffectiveCounts:
    """窗口账的有效计数：到了几条（事件）、看清了没来几条（余数）——按块数折算之后的。

    用来给 p1 配一个不会塌成零宽的区间：二值 + 小 n 上百分位 bootstrap 会退化（六条全到 → 每次重采样都给同一个数
    → 区间宽 0 → 必然"显著"，零效应假显著率 p0=0.88 时实测 50.7%）。
    """

    events: float
    remainder: float


def effective_counts(occurred: int, absent: int, *, blocks: int | None = None) -> EffectiveCounts | None:
    """把 (到了, 没来) 折成一对"等价成败数"：等价样本数缺省取条数；给了 ``blocks`` 就取**块数与条数里小的那个**。

    高频前件一天开很多条承诺、等的是同一个后果窗口（探针：22 条到来背后是 9 次发生），按条数当样本数区间过窄——
    4 天 × 6 条/天、零效应时误报 32.5%，按块数折算后 6.8%（评审 C-6）。块 bootstrap 知道这件事，Jeffreys 得告诉它。
    """

    for label, value in (("occurred", occurred), ("absent", absent)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    total = occurred + absent
    if total <= 0:
        return None
    if blocks is not None:
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 0:
            raise ValueError("blocks must be a non-negative integer")
        if blocks > 0:
            total = min(total, blocks)
    events = total * occurred / (occurred + absent)
    return EffectiveCounts(events=events, remainder=total - events)


def jeffreys_interval(events: float, remainder: float, *, level: float = 0.90) -> Interval:
    """Jeffreys 区间：Beta(events+½, remainder+½) 的中央 ``level``，点估计取 events/(events+remainder)。

    二值小样本的标准做法：六条全到时给 [0.68, 1.0] 而不是 [1.0, 1.0]，不会因为"每条都一样"就宣称确定。
    """

    if events < 0 or remainder < 0:
        raise StatsError("counts are non-negative")
    total = events + remainder
    if total <= 0:
        raise StatsError("jeffreys interval needs at least one observation")
    a, b = events + 0.5, remainder + 0.5
    tail = (1.0 - level) / 2.0
    return Interval(point=events / total, low=beta_quantile(tail, a, b), high=beta_quantile(1.0 - tail, a, b))


def widen(first: Interval | None, second: Interval | None) -> Interval | None:
    """两个区间取并（点估计用第一个的）：谁更宽听谁的，不让任一种方法单独把区间收窄。"""

    if first is None:
        return second
    if second is None:
        return first
    return Interval(first.point, min(first.low, second.low), max(first.high, second.high))


def is_monotonic(values: Sequence[float]) -> bool:
    """非严格单调（全不减或全不增）；**少于三个点不算**。

    两个点之间没有"曲线"——任意两个数必然满足其中一个方向，所以 ``[a, b]`` 恒为真。剂量那条判据
    （七e："混杂能伪造一个差，难伪造一条单调曲线 → 可信度升一档"）在两点上会退化成恒真，连把数据
    反过来摆都照样判"单调"。要三个点才谈得上趋势。
    """

    if len(values) < 3:
        return False
    pairs = list(zip(values, values[1:], strict=False))
    return all(a <= b for a, b in pairs) or all(a >= b for a, b in pairs)


__all__ = [
    "EffectiveCounts",
    "Interval",
    "StatsError",
    "beta_cdf",
    "beta_quantile",
    "block_bootstrap",
    "effective_counts",
    "is_monotonic",
    "jeffreys_interval",
    "mean",
    "median",
    "widen",
]
