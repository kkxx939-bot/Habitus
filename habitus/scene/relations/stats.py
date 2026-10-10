"""关系检验的统计（语义树新方案 ``13`` ③），纯函数、纯 Python。

**层内精确检验（病例交叉的条件检验）**：每次 A 是一层，层里有 A 那一刻与 ``m`` 个对照时刻，其中阳性共 ``T`` 个。
零假设下 A 那一刻只是这 ``m + 1`` 个时刻里随便一个，它是阳性的概率 ``q = T / (m + 1)``；各层独立，实验组阳性总数
服从以 ``q_i`` 为参数的泊松二项分布。对照因此是样本、不是已知数（第三轮评审 P0-2：把对照当已知数会让 58 天里成立过
23 条、其中 18 条后来失效）。有信息的层（0 < q < 1）不多时 p 值精确算；很多时用正态近似，带连续性校正。

**剔除不可能显著的检验**（Tarone；FDR 版按 Gilbert 2005）：一个检验在它自己的层结构下能达到的最小 p 值，若连
最宽松的线都过不了，就不进当晚的检验族、标"样本不够"、不给区间（用户 09-26：少样本不出数）。

**第 1 道**：剩下的检验做加权 Benjamini–Hochberg（权重是先验，归一到平均 1；没有先验就全是 1）。

**区间**：按块重抽样（块随前因自己的节奏缩放，``blocks``），百分位区间；随机只用在这里，种子由调用方按
（关系身份, 第几晚, 用途）给，可回放。块长 1 天的高频行为相邻几天往往相关，区间偏窄——已知限制。
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from habitus.scene.relations.spans import Stratum


@dataclass(frozen=True)
class Effect:
    """实验组的发生率、对照的发生率（层内对照率的平均）、两者之差。"""

    antecedents: int
    treated_rate: float
    control_rate: float

    @property
    def difference(self) -> float:
        return self.treated_rate - self.control_rate

    @property
    def relative(self) -> float | None:
        """相对差：实验组比对照多出（少了）百分之几；对照率为 0 时没有相对差。"""

        return None if self.control_rate <= 0.0 else self.treated_rate / self.control_rate - 1.0


def effect(strata: Sequence[Stratum]) -> Effect:
    n = len(strata)
    if n == 0:
        return Effect(antecedents=0, treated_rate=0.0, control_rate=0.0)
    return Effect(
        antecedents=n,
        treated_rate=sum(item.treated for item in strata) / n,
        control_rate=sum(item.control_rate for item in strata) / n,
    )


def tails(strata: Sequence[Stratum], *, exact_limit: int) -> tuple[float, float]:
    """（更容易一侧的 p, 更难一侧的 p）：P(S ≥ 实测) 与 P(S ≤ 实测)。"""

    observed = sum(item.treated for item in strata)
    probabilities = [item.null_probability for item in strata]
    fixed = sum(1 for p in probabilities if p >= 1.0)
    informative = [p for p in probabilities if 0.0 < p < 1.0]
    target = observed - fixed
    if not informative:
        return 1.0, 1.0
    if len(informative) <= exact_limit:
        distribution = _poisson_binomial(informative)
        upper = sum(distribution[max(0, target) :]) if target <= len(informative) else 0.0
        lower = sum(distribution[: max(0, target + 1)]) if target >= 0 else 0.0
        return min(1.0, upper), min(1.0, lower)
    mean = sum(informative)
    sd = math.sqrt(sum(p * (1.0 - p) for p in informative))
    upper = _normal_sf((target - 0.5 - mean) / sd)
    lower = _normal_sf((mean - target - 0.5) / sd)
    return min(1.0, upper), min(1.0, lower)


def two_sided(upper: float, lower: float) -> float:
    return min(1.0, 2.0 * min(upper, lower))


def minimum_p(strata: Sequence[Stratum]) -> float:
    """在这个层结构下能达到的最小双侧 p：全部有信息的层都阳性（或都阴性）的概率，取小的一侧再乘 2。"""

    informative = [item.null_probability for item in strata if 0.0 < item.null_probability < 1.0]
    if not informative:
        return 1.0
    all_up = math.prod(informative)
    all_down = math.prod(1.0 - p for p in informative)
    return min(1.0, 2.0 * min(all_up, all_down))


def one_sided(strata: Sequence[Stratum], *, upward: bool, exact_limit: int) -> float:
    """前向验证与维持检验用：只朝发现时的方向检。"""

    upper, lower = tails(strata, exact_limit=exact_limit)
    return upper if upward else lower


def interval(
    strata: Sequence[Stratum], *, level: float, rounds: int, seed: int
) -> tuple[tuple[float, float], tuple[float, float] | None]:
    """（差值区间, 相对差区间）：按块整块重抽样，百分位。对照率在某一轮为 0 的那轮不进相对差。"""

    blocks: dict[int, list[Stratum]] = {}
    for item in strata:
        blocks.setdefault(item.block, []).append(item)
    keys = sorted(blocks)
    generator = random.Random(seed)
    differences: list[float] = []
    relatives: list[float] = []
    for _ in range(rounds):
        drawn = [item for _ in keys for item in blocks[keys[generator.randrange(len(keys))]]]
        result = effect(drawn)
        differences.append(result.difference)
        if result.relative is not None:
            relatives.append(result.relative)
    tail = (1.0 - level) / 2.0
    spread = (_quantile(differences, tail), _quantile(differences, 1.0 - tail))
    if len(relatives) < rounds * 0.9:
        return spread, None  # 对照率常常是 0：相对差说不清
    return spread, (_quantile(relatives, tail), _quantile(relatives, 1.0 - tail))


def heterogeneity(strata: Sequence[Stratum]) -> float:
    """各块效应有没有明显差异（Cochran 分解）：各块（实测 − 期望）² / 方差 之和，减去合起来的那一项，
    剩下的是"块与块之间不一样"的部分，按自由度 块数 − 1 的卡方给 p。只有一块时 1。

    不能只看前一半：那是在检"每块都等于零假设"，一条真关系的每一块都偏离零假设，会被当成"不稳"。"""

    observed: dict[int, float] = {}
    for item in strata:
        observed[item.block] = observed.get(item.block, 0.0) + item.treated
    # 方差按**估出的共同效应**算（裁定 27 第 3 条）：用零假设的 q(1−q) 只在效应接近 0 时对，一条真关系各块同样强也会被判"不稳"
    # （第四轮评审算法 4：效应恒定时 16–20% 判不稳，名义 5%）。共同效应取一个对数优势比 θ（每层在自己的 q 上抬同样多，
    # 永远落在 0–1 之间；差值尺度在 q 高的层会顶出 1），θ 使拟合的阳性数等于实测；再看各块实测与拟合之差。
    items = [(item.block, item.treated, min(1.0 - 1e-9, max(1e-9, item.null_probability))) for item in strata]
    target = sum(treated for _block, treated, _q in items)
    theta = _common_log_odds(items, target)
    fitted: dict[int, float] = {}
    variance: dict[int, float] = {}
    for block, _treated, q in items:
        p = 1.0 / (1.0 + math.exp(-(math.log(q / (1.0 - q)) + theta)))
        fitted[block] = fitted.get(block, 0.0) + p
        variance[block] = variance.get(block, 0.0) + p * (1.0 - p)
    usable = [block for block in observed if variance[block] > 1e-12]
    if len(usable) < 2:
        return 1.0
    residual = sum((observed[block] - fitted[block]) ** 2 / variance[block] for block in usable)
    return _chi2_sf(residual, len(usable) - 1)


def _common_log_odds(items: Sequence[tuple[int, int, float]], target: float) -> float:
    """解 Σ expit(logit(q) + θ) = target（二分；单调）。"""

    def total(theta: float) -> float:
        return sum(1.0 / (1.0 + math.exp(-(math.log(q / (1.0 - q)) + theta))) for _block, _treated, q in items)

    low, high = -30.0, 30.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if total(middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def required_antecedents(
    control: float, effect: float, *, upward: bool, alpha: float, power: float, minimum: int, maximum: int
) -> int:
    """前向验证要攒几次独立的前因：单侧精确二项检验（零假设概率 = 对照率）在真效应为 ``effect`` 时功效到 ``power`` 的最少次数，
    夹在 [minimum, maximum]。效应不朝这个方向（≤ 0）就给 ``maximum``。"""

    if effect <= 0.0:
        return maximum
    q = min(1.0 - 1e-9, max(1e-9, control))
    p = min(1.0 - 1e-9, max(1e-9, control + effect if upward else control - effect))
    for count in range(minimum, maximum + 1):
        null = [math.comb(count, k) * q**k * (1 - q) ** (count - k) for k in range(count + 1)]
        true = [math.comb(count, k) * p**k * (1 - p) ** (count - k) for k in range(count + 1)]
        if upward:
            tail = [sum(null[k:]) for k in range(count + 1)]
            critical = next((k for k in range(count + 1) if tail[k] <= alpha), None)
            reached = 0.0 if critical is None else sum(true[critical:])
        else:
            tail = [sum(null[: k + 1]) for k in range(count + 1)]
            critical = next((k for k in reversed(range(count + 1)) if tail[k] <= alpha), None)
            reached = 0.0 if critical is None else sum(true[: critical + 1])
        if reached >= power:
            return count
    return maximum


def leave_one_block_out(strata: Sequence[Stratum]) -> bool:
    """去掉任何一块，方向都不变（差值同号、不为 0）。只有一块时谈不上，算不稳。"""

    blocks = sorted({item.block for item in strata})
    if len(blocks) < 2:
        return False
    overall = effect(strata).difference
    if overall == 0.0:
        return False
    for block in blocks:
        rest = effect([item for item in strata if item.block != block]).difference
        if rest == 0.0 or (rest > 0.0) != (overall > 0.0):
            return False
    return True


def tarone_bh(
    p_values: Mapping[str, float], minimum: Mapping[str, float], weights: Mapping[str, float], *, q: float
) -> tuple[frozenset[str], frozenset[str]]:
    """（检验族, 过了第 1 道的）。

    先按 Gilbert（2005）的 Tarone 修剪定检验族：找最小的 K，使"最小可达 p / 权重 ≤ q / K"的检验不超过 K 个，
    族就是这些检验；其余的不可能显著，不进族。再在族里做加权 BH：p / 权重 排序，第 i 小的 ≤ q · i / 族大小 的最大 i 以内都过。
    权重在族里重新归一到平均 1（先验只调第 1 道，有界、冻结，由调用方给）。
    """

    keys = sorted(p_values)
    if not keys:
        return frozenset(), frozenset()
    raw = {key: max(weights.get(key, 1.0), 1e-12) for key in keys}
    scale = len(keys) / sum(raw.values())
    weight = {key: raw[key] * scale for key in keys}
    family: list[str] = keys
    for limit in range(1, len(keys) + 1):
        members = [key for key in keys if minimum[key] / weight[key] <= q / limit]
        if len(members) <= limit:
            family = members
            break
    if not family:
        return frozenset(), frozenset()
    scale = len(family) / sum(raw[key] for key in family)
    adjusted = sorted(family, key=lambda key: (p_values[key] / (raw[key] * scale), key))
    size = len(adjusted)
    passed = 0
    for rank, key in enumerate(adjusted, start=1):
        if p_values[key] / (raw[key] * scale) <= q * rank / size:
            passed = rank
    return frozenset(family), frozenset(adjusted[:passed])


def _poisson_binomial(probabilities: Sequence[float]) -> list[float]:
    distribution = [1.0]
    for p in probabilities:
        following = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            following[count] += mass * (1.0 - p)
            following[count + 1] += mass * p
        distribution = following
    return distribution


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _chi2_sf(statistic: float, degrees: int) -> float:
    """卡方分布的右尾：正则化上不完全伽马 Q(k/2, x/2)。"""

    if statistic <= 0.0:
        return 1.0
    return _gamma_q(degrees / 2.0, statistic / 2.0)


def _gamma_q(a: float, x: float) -> float:
    if x < a + 1.0:
        term = total = 1.0 / a
        current = a
        for _ in range(500):
            current += 1.0
            term *= x / current
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for step in range(1, 500):
        an = -step * (step - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h)


__all__ = [
    "Effect",
    "effect",
    "heterogeneity",
    "interval",
    "leave_one_block_out",
    "minimum_p",
    "one_sided",
    "required_antecedents",
    "tails",
    "tarone_bh",
    "two_sided",
]
