"""一本账的可靠度（裁定 49）：带遗忘、带强度的 Beta 后验，给出点估计与单侧上下界。纯函数、纯 Python。

每条事实带方向 y（好 / 坏）、强度 g ∈ [0, 1]、在它那条时钟上的名次 r；权重 ``w = 0.5^((r_last − r) / H) × g``（H 是半衰的机会数），``k = Σ w·y``，``n = Σ w``，
后验 ``Beta(k + ½, n − k + ½)``（Jeffreys 先验：没有信息时不偏向任何一边）。下界是它的 α 分位数、上界是 1 − α 分位数。
样本少时区间宽，下界低、上界高——"样本少就没有读数"在这里不用门槛，由区间自己说。

正则化不完全 Beta 用连分式（Lentz），分位数用二分；精度到 1e-9 足够做"过不过线"的判断。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from habitus.foresight.errors import ForesightError
from habitus.foundation.lifecycle import decay

#: 二分求分位数的迭代次数（区间每次减半，60 次到 1e-18 量级）。
_BISECTIONS = 60
_CF_ITERATIONS = 300
_CF_EPSILON = 1e-12
_TINY = 1e-300


@dataclass(frozen=True)
class Reliability:
    """一条轴此刻的读数：加权的好数 ``k``、加权的总数 ``n``、点估计与单侧上下界。"""

    k: float
    n: float
    mean: float
    lower: float
    upper: float


def weighted(items: Iterable[tuple[int, float, bool]], *, last_rank: int, half_life: float) -> list[tuple[float, bool]]:
    """``items`` 是 (名次, 强度, 好不好)，名次在调用方那条时钟上编好（整本账一次，不在子集里重编）；权重 =
    强度 × 0.5^((``last_rank`` − 名次) / ``half_life``)：一笔证据只看它之后又来了几次机会，不看隔了几天（裁定 57）。"""

    found = []
    for rank, strength, good in items:
        if not 0.0 <= strength <= 1.0:
            raise ForesightError("strength must lie in [0, 1]")
        if rank > last_rank:
            raise ForesightError("a rank cannot lie after the last one")
        try:
            found.append((strength * decay(last_rank - rank, half_life=half_life), good))
        except ValueError as exc:
            raise ForesightError(str(exc)) from exc
    return found


def estimate(weighted: Iterable[tuple[float, bool]], *, alpha: float) -> Reliability:
    """``weighted`` 是 (权重, 好不好)。"""

    k = n = 0.0
    for w, good in weighted:
        if w < 0:
            raise ForesightError("weights must be non-negative")
        n += w
        if good:
            k += w
    return reading(k, n, alpha=alpha)


def reading(k: float, n: float, *, alpha: float) -> Reliability:
    """加权好数 ``k``、加权总数 ``n`` 的读数（账沿时钟 fold 出这两个数，``estimate`` 逐笔加出这两个数）。"""

    if not 0.0 < alpha < 0.5:
        raise ForesightError("alpha must lie in (0, 0.5)")
    if n < 0 or not 0 <= k <= n + 1e-9:
        raise ForesightError("k and n must satisfy 0 <= k <= n")
    k = min(k, n)
    a, b = k + 0.5, n - k + 0.5
    return Reliability(
        k=k, n=n, mean=a / (a + b), lower=beta_quantile(alpha, a, b), upper=beta_quantile(1.0 - alpha, a, b)
    )


def miss_run_probability(k: float, n: float, run: float) -> float:
    """按这条轴此刻的后验（Jeffreys），接下来连着 ``run`` 次都是坏的概率（Beta-二项；``run`` 可以是分量和，Beta 函数对实数成立）：
    ``B(k + ½, n − k + ½ + run) / B(k + ½, n − k + ½)``。样本少时后验宽，这个概率不会小得离谱。"""

    if isinstance(run, bool) or not isinstance(run, int | float) or run < 0:
        raise ForesightError("run must be a non-negative number")
    if n < 0 or not 0 <= k <= n + 1e-9:
        raise ForesightError("k and n must satisfy 0 <= k <= n")
    a, b = k + 0.5, max(n - k, 0.0) + 0.5
    return math.exp(_log_beta(a, b + run) - _log_beta(a, b))


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def incomplete_beta(x: float, a: float, b: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)。"""

    if a <= 0 or b <= 0:
        raise ForesightError("beta parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_fraction(x, a, b) / a
    return 1.0 - front * _continued_fraction(1.0 - x, b, a) / b


def beta_quantile(q: float, a: float, b: float) -> float:
    """Beta(a, b) 的 q 分位数（二分）。"""

    if not 0.0 <= q <= 1.0:
        raise ForesightError("quantile level must lie in [0, 1]")
    low, high = 0.0, 1.0
    for _ in range(_BISECTIONS):
        middle = (low + high) / 2.0
        if incomplete_beta(middle, a, b) < q:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _continued_fraction(x: float, a: float, b: float) -> float:
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    result = d
    for m in range(1, _CF_ITERATIONS + 1):
        twice = 2 * m
        for numerator in (
            m * (b - m) * x / ((a + twice - 1.0) * (a + twice)),
            -(a + m) * (a + b + m) * x / ((a + twice) * (a + twice + 1.0)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > _TINY else _TINY)
            c = 1.0 + numerator / c
            c = c if abs(c) > _TINY else _TINY
            delta = d * c
            result *= delta
        if abs(delta - 1.0) < _CF_EPSILON:
            break
    return result


__all__ = ["Reliability", "beta_quantile", "estimate", "incomplete_beta", "miss_run_probability", "reading", "weighted"]
