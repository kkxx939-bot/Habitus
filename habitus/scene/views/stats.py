"""读时统计的几个基元：Beta 后验区间、确定性 bootstrap 区间（含按块重采样）、Kaplan–Meier、钟面上的环形差。

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

MINUTES_PER_DAY = 24 * 60
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


def bootstrap_interval(values: Sequence[float], statistic: str, *, seed: str, samples: int = 1000, level: float = 0.90) -> Interval:
    """确定性 bootstrap：``statistic`` 是 ``median`` 或 ``mean``；种子来自调用方（同一本账同一区间）。"""

    if not values:
        raise StatsError("bootstrap of nothing")
    if any(not math.isfinite(value) for value in values):
        raise StatsError("bootstrap values must be finite")
    if statistic not in {"median", "mean"}:
        raise StatsError("statistic is 'median' or 'mean'")
    fn = median if statistic == "median" else mean
    point = fn(values)
    if len(values) == 1:
        return Interval(point, point, point)
    generator = random.Random(seed)
    estimates = sorted(fn(generator.choices(values, k=len(values))) for _ in range(samples))
    tail = (1.0 - level) / 2.0
    low = estimates[min(len(estimates) - 1, int(tail * len(estimates)))]
    high = estimates[min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))]
    return Interval(point, min(low, point), max(high, point))


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
class SurvivalStep:
    """离散生存曲线上的一步：到第 ``step`` 步还在等的有几条、这一步来了几条、过了这一步还没来的比例。"""

    step: int
    at_risk: int
    events: int
    survival: float


def kaplan_meier(event_steps: Sequence[int], censored_steps: Sequence[int]) -> tuple[SurvivalStep, ...]:
    """按"第几次机会"走的 Kaplan–Meier。

    ``event_steps``：每条后件到来的那一步（第 k 次机会 → k，从 1 数）；``censored_steps``：每条右删失的记录已经过完的步数
    （等了 c 次机会没来 → c，可以是 0）。第 k 步的风险集 = 事件步 ≥ k 的 + 删失步 ≥ k 的。
    """

    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in event_steps):
        raise StatsError("event steps count from 1")
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in censored_steps):
        raise StatsError("censored steps are non-negative")
    last = max([*event_steps, *censored_steps], default=0)
    survival = 1.0
    curve: list[SurvivalStep] = []
    for step in range(1, last + 1):
        at_risk = sum(1 for k in event_steps if k >= step) + sum(1 for c in censored_steps if c >= step)
        events = sum(1 for k in event_steps if k == step)
        if at_risk > 0:
            survival *= 1.0 - events / at_risk
        curve.append(SurvivalStep(step, at_risk, events, survival))
    return tuple(curve)


def survival_at(curve: Sequence[SurvivalStep], step: int) -> float | None:
    """S(step)：过了第 ``step`` 步还没来的比例；S(0)=1。

    曲线没走到那一步时：末步的 S 已经是 0（人全到了）→ 之后的 S 也是 0，返回 0 而不是 None——不然"六条全在第 1 步就
    到了"问第 2 步会读成"算不出"（评审 A-5/C-10）。末步 S>0 而曲线到此为止（没人等到那一步）才是真的不知道 → None。
    """

    if step <= 0:
        return 1.0
    for item in curve:
        if item.step == step:
            return item.survival if item.at_risk > 0 else None
    if curve and step > curve[-1].step and curve[-1].survival == 0.0:
        return 0.0
    return None


def survival_median(curve: Sequence[SurvivalStep]) -> int | None:
    """S 第一次降到 0.5 或以下的那一步；没降到就 None（中位没到——不要拿到达者的中位冒充）。"""

    for item in curve:
        if item.at_risk > 0 and item.survival <= 0.5:
            return item.step
    return None


@dataclass(frozen=True)
class EffectiveCounts:
    """KM 曲线在第 ``step`` 步为止的有效计数：到了几条（事件）、还没来几条（分母里剩下的）。

    用来给 p1 配一个不会塌成零宽的区间：二值 + 小 n 上百分位 bootstrap 会退化（六条全到 → 每次重采样都给同一个数
    → 区间宽 0 → 必然"显著"，零效应假显著率 p0=0.88 时实测 50.7%）。
    """

    events: float
    remainder: float


def effective_counts(curve: Sequence[SurvivalStep], step: int) -> EffectiveCounts | None:
    """把 KM 的 1−S(step) 折成一对"等价成败数"：分母取第 1 步的风险集（进过账的条数），成数 = 分母 × (1−S)。"""

    survival = survival_at(curve, step)
    if survival is None or not curve:
        return None
    at_risk = curve[0].at_risk
    if at_risk <= 0:
        return None
    occurred = at_risk * (1.0 - survival)
    return EffectiveCounts(events=occurred, remainder=at_risk - occurred)


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
    "SurvivalStep",
    "beta_cdf",
    "beta_quantile",
    "block_bootstrap",
    "bootstrap_interval",
    "effective_counts",
    "is_monotonic",
    "jeffreys_interval",
    "kaplan_meier",
    "mean",
    "median",
    "survival_at",
    "survival_median",
    "widen",
]
