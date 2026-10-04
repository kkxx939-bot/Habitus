"""强度：三个方面三套算法（用户裁定，不共用公式），以及从概率强度读出的类型。

- **概率（节律型，窗口账，2026-10-01）**：``p1`` = 来了 ÷（来了 + 看清了没来）；``p0`` = 各承诺窗口的平时概率均值；
  强度 = p1 − p0；两边**同一段**，零效应读 0（评审 A-1 / C-1 修法）。区间 = 按块 bootstrap ∪ 按块数折算的 Jeffreys；
  **块 = 同一个后果窗口**（二-6）。"命运已定" = ``until`` 过了窗口末尾（对称截断只剩这一条）。
- **时刻**：``observed_at − 窗口中心``（两个绝对时刻的差），中位数，按块 bootstrap。
- **次数**：``count − 那几个窗口的平时概率之和``，均值，按块 bootstrap。
- **类型**（只对概率）：区间含零 → 无；负 → 抑制；正 → PN = (p1−p0)/p1、PS = (p1−p0)/(1−p0)。

KM 首达曲线删了（10-01 定）："是不做还是往后推"由同一组前因的几本峰账摆在一起回答（``behaviours`` 那一面）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from habitus.scene.hypotheses.model import Aspect, Direction, Hypothesis
from habitus.scene.ledger.model import Claim, Outcome
from habitus.scene.views.accounts import Account, Accumulation, Pair, accumulation_of, informative_pairs, settled_pairs
from habitus.scene.views.config import ViewsConfig
from habitus.scene.views.stats import (
    Interval,
    block_bootstrap,
    effective_counts,
    jeffreys_interval,
    mean,
    median,
    widen,
)


class TypeReading(str, Enum):
    NONE = "none"
    INHIBITING = "inhibiting"
    ENABLING = "enabling"
    PROMOTING = "promoting"





@dataclass(frozen=True)
class Strength:
    aspect: Aspect
    accumulation: Accumulation
    interval: Interval | None
    control: float | None
    unit: str
    #: 概率（窗口账）：p1 = 来了 ÷（来了 + 看清了没来）；``events`` 来了几条、``absent`` 看清了没来几条、
    #: ``censored`` 没看清几条（不进分子分母，只印出来）。``median_hours`` 是来了那些次从锚到后件的中位间隔。
    p1: float | None = None
    events: int = 0
    censored: int = 0
    median_hours: float | None = None
    #: 时刻：缺席的次数（那次机会已观测地过了、后件没来）。概率方面也用它数"看清了没来"。
    absent: int = 0
    #: 有机会、但那条承诺没有对照快照（新概念、树上还没数）→ 不参与。
    without_control: int = 0
    #: 命运还没定（``now`` 还没过窗口末尾）→ 不参与，等它定了再算（对称截断）。
    immature: int = 0
    #: 次数：快照铺不到 horizon 个窗口 → 不参与。读侧也要能看见条数。
    short_snapshot: int = 0
    #: 结算时"没看清"的窗口数 / 收了的窗口总数（2026-09-30 裁定三：逻辑不动，占比印出来）。
    #: 剔除只发生在"后件没来"的窗口上、来了的从不剔，所以占比高时实际率偏高（评审 C-13）。
    skipped_passes: int = 0
    total_passes: int = 0
    #: 区间算不出宽度（观测值全相同）时仍给的点估计——六次晚睡之后六次都喝了 3 杯，这是最一致的那类效应，
    #: 不能印成"还没攒够"（评审 A-14）。有区间时为 None（点在区间里）。
    degenerate_point: float | None = None
    agrees_with_prior: bool | None = None

    @property
    def sufficient(self) -> bool:
        return self.accumulation.sufficient


def strength_of(
    account: Account,
    hypothesis: Hypothesis,
    config: ViewsConfig,
    *,
    gate: tuple[int, int] | None = None,
    seed: str | None = None,
    now: datetime | None = None,
) -> Strength:
    """按方面算强度。没过累积度门槛 → 区间 None，其余计数照报（三态之"证据不够"）。

    ``now`` 给了就对概率方面做**对称截断**：只收命运已定的承诺（``now`` 过了它的窗口末尾，见 ``settled_by``）。不给就不截断
    （只在调用方确信账里没有"刚开还没到期"的承诺时用）。
    """

    resolved_gate = gate or config.main_gate
    seed_text = seed or account.hypothesis_identity
    aspect = hypothesis.aspect
    if hypothesis.is_open_ended:
        # 无节律型不走强度这条路（不读 p1−p0、不读类型）：这里只给计数，数由 ``fulfilment_of`` 读。留一法与分账
        # 拿区间是 None 当"读不出"，于是无节律型自然不参与稳定性扫与借力。
        pairs = settled_pairs(account)
        return Strength(
            aspect,
            accumulation_of(account, pairs, resolved_gate),
            None,
            None,
            "fulfilment",
            events=sum(1 for _c, item in pairs if item.outcome is Outcome.OCCURRED),
        )
    pairs = informative_pairs(account, aspect)
    without_control = sum(1 for claim, s in account.settled if claim.control is None and s.outcome is not Outcome.CENSORED)
    censored = sum(1 for claim, s in account.settled if claim.control is not None and s.outcome is Outcome.CENSORED)
    accumulation = accumulation_of(account, pairs, resolved_gate)
    if aspect is Aspect.PROBABILITY:
        return _probability_strength(account, hypothesis, pairs, accumulation, config, seed_text, without_control, censored, now)
    if aspect is Aspect.TIMING:
        return _timing_strength(account, hypothesis, pairs, accumulation, config, seed_text, without_control)
    return _count_strength(account, hypothesis, pairs, accumulation, config, seed_text, without_control)


def settled_by(claim: Claim, now: datetime) -> bool:
    """这条概率承诺的命运在 ``now`` 之前已经定了吗：它的后果窗口已经过完。

    与 ``settle_probability`` 的收口点是同一个：到那一刻它要么已经 OCCURRED，要么 ABSENT / CENSORED。
    """

    snapshot = claim.control
    if snapshot is None:
        return False
    return snapshot.opportunities[0].span.end <= now


def _probability_strength(
    account: Account,
    hypothesis: Hypothesis,
    pairs: Sequence[Pair],
    accumulation: Accumulation,
    config: ViewsConfig,
    seed: str,
    without_control: int,
    censored: int,
    now: datetime | None,
) -> Strength:
    """窗口口径：p1 = OCCURRED ÷ (OCCURRED + ABSENT)，p0 = 各承诺窗口的平时概率均值；两边同一段窗口。

    没有平时概率的承诺（窗口里哪天的曲线缺）不参与——"不猜"（二-4）。CENSORED 不进分子分母。
    """

    ready = [(claim, s) for claim, s in pairs if now is None or settled_by(claim, now)]
    immature = len(pairs) - len(ready)
    usable = [(claim, s) for claim, s in ready if claim.control is not None and claim.control.opportunities[0].probability is not None]
    without_control += len(ready) - len(usable)
    events = sum(1 for _c, s in usable if s.outcome is Outcome.OCCURRED)
    absent = len(usable) - events
    accumulation = accumulation_of(account, usable, (accumulation.needed_count, accumulation.needed_blocks))
    total_passes = len(usable) + censored
    latencies = [s.latency_hours for _c, s in usable if s.outcome is Outcome.OCCURRED and s.latency_hours is not None]
    median_hours = median(latencies) if latencies else None
    if not usable:
        return Strength(
            hypothesis.aspect, accumulation, None, None, "probability", events=events, absent=absent, censored=censored,
            without_control=without_control, immature=immature, skipped_passes=censored, total_passes=total_passes,
        )
    p0 = _control_of(usable)
    p1 = events / len(usable)
    if not accumulation.sufficient:
        return Strength(
            hypothesis.aspect, accumulation, None, p0, "probability", p1=p1, events=events, absent=absent, censored=censored,
            median_hours=median_hours, without_control=without_control, immature=immature,
            skipped_passes=censored, total_passes=total_passes,
        )
    blocks = [account.block_of(claim) for claim, _s in usable]
    # 区间 = 分块 bootstrap（治天间成簇）∪ Jeffreys（治二值小样本的塌缩）。只用 bootstrap 的话，六条全到 / 全没到时
    # 每次重采样都给同一个数 → 零宽区间 → 必然"显著"（零效应假显著率 p0=0.88 时实测 50.7%，名义 10%）。
    resampled = block_bootstrap(usable, blocks, _probability_effect, seed=seed, samples=config.bootstrap_samples, level=config.interval_level)
    counts = effective_counts(events, absent, blocks=len(set(blocks)))
    shifted = None
    if counts is not None:
        rate = jeffreys_interval(counts.events, counts.remainder, level=config.interval_level)
        shifted = Interval(p1 - p0, rate.low - p0, rate.high - p0)
    interval = usable_interval(widen(resampled, shifted) if resampled is not None else shifted)
    return Strength(
        hypothesis.aspect,
        accumulation,
        interval,
        p0,
        "probability",
        p1=p1,
        events=events,
        absent=absent,
        censored=censored,
        median_hours=median_hours,
        without_control=without_control,
        immature=immature,
        skipped_passes=censored,
        total_passes=total_passes,
        agrees_with_prior=None if interval is None else _agrees(interval, hypothesis.direction),
    )


def usable_interval(interval: Interval | None) -> Interval | None:
    """零宽区间不是"确定"，是"这个方法在这批数据上算不出宽度"——一律当没有区间（连带 ``unrefuted`` 与 profile 排序）。"""

    if interval is None or interval.width <= 0.0:
        return None
    return interval


def _control_of(pairs: Sequence[Pair]) -> float:
    return mean([claim.control.opportunities[0].probability for claim, _s in pairs])  # type: ignore[union-attr, misc]


def _probability_effect(pairs: Sequence[Pair]) -> float | None:
    if not pairs:
        return None
    p1 = sum(1 for _c, s in pairs if s.outcome is Outcome.OCCURRED) / len(pairs)
    return p1 - _control_of(pairs)


def _timing_strength(account: Account, hypothesis: Hypothesis, pairs: Sequence[Pair], accumulation: Accumulation, config: ViewsConfig, seed: str, without_control: int) -> Strength:
    observed: list[Pair] = []
    values: list[float] = []
    for claim, s in pairs:
        target = None if claim.control is None else claim.control.opportunities[0]
        if s.outcome is Outcome.OBSERVED and s.observed_at is not None and target is not None:
            observed.append((claim, s))
            values.append((s.observed_at - target.at).total_seconds() / 60.0)
    absent = sum(1 for _c, s in pairs if s.outcome is Outcome.ABSENT)
    return _bootstrapped(account, hypothesis, observed, values, median, "minutes", accumulation, config, seed, without_control, absent=absent)


def _count_strength(account: Account, hypothesis: Hypothesis, pairs: Sequence[Pair], accumulation: Accumulation, config: ViewsConfig, seed: str, without_control: int) -> Strength:
    usable: list[Pair] = []
    values: list[float] = []
    short_snapshot = 0
    for claim, s in pairs:
        expected = None if claim.control is None else claim.control.expected_count(1, hypothesis.horizon)
        if expected is None:
            short_snapshot += 1
        if s.count is None or expected is None:
            continue
        usable.append((claim, s))
        values.append(float(s.count) - expected)
    accumulation = accumulation_of(account, usable, (accumulation.needed_count, accumulation.needed_blocks))
    return _bootstrapped(
        account, hypothesis, usable, values, mean, "times", accumulation, config, seed, without_control, short_snapshot=short_snapshot
    )


def _bootstrapped(
    account: Account,
    hypothesis: Hypothesis,
    pairs: Sequence[Pair],
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    unit: str,
    accumulation: Accumulation,
    config: ViewsConfig,
    seed: str,
    without_control: int,
    *,
    absent: int = 0,
    short_snapshot: int = 0,
) -> Strength:
    """``accumulation`` 按进账的全部机会算（时刻的缺席也是攒到的证据），区间只用有数值的那些；有数值的太少也不出区间。"""

    enough_values = len(values) >= config.split_min and len({block for block in (account.block_of(claim) for claim, _s in pairs)}) >= config.split_min_blocks
    if not values or not accumulation.sufficient or not enough_values:
        return Strength(
            hypothesis.aspect, accumulation, None, 0.0 if values else None, unit, absent=absent,
            without_control=without_control, short_snapshot=short_snapshot,
        )
    blocks = [account.block_of(claim) for claim, _s in pairs]
    raw = block_bootstrap(values, blocks, statistic, seed=seed, samples=config.bootstrap_samples, level=config.interval_level)
    interval = usable_interval(raw)
    return Strength(
        hypothesis.aspect,
        accumulation,
        interval,
        0.0,
        unit,
        degenerate_point=raw.point if raw is not None and interval is None else None,
        absent=absent,
        without_control=without_control,
        short_snapshot=short_snapshot,
        agrees_with_prior=None if interval is None else _agrees(interval, hypothesis.direction),
    )


def _agrees(interval: Interval, direction: Direction) -> bool | None:
    if interval.contains_zero:
        return None
    return (interval.point > 0) == (direction is Direction.UP)


# ── 类型 ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TypeReadout:
    """从账读出的类型，连同它依据的两个数。PN/PS 是外生 + 单调假设下的界，不是点识别。"""

    reading: TypeReading
    pn: float | None
    ps: float | None


def read_type(strength: Strength, config: ViewsConfig) -> TypeReadout | None:
    """只对概率方面：区间含零 → 无；负 → 抑制；正 → PN = (p1−p0)/p1、PS = (p1−p0)/(1−p0)。"""

    if strength.aspect is not Aspect.PROBABILITY or strength.unit != "probability":
        return None
    if strength.interval is None or strength.control is None or strength.p1 is None:
        return None
    if strength.interval.contains_zero:
        return TypeReadout(TypeReading.NONE, None, None)
    if strength.interval.high < 0:
        return TypeReadout(TypeReading.INHIBITING, None, None)
    p1, p0 = strength.p1, strength.control
    pn = (p1 - p0) / p1 if p1 > 0 else 0.0
    ps = (p1 - p0) / (1.0 - p0) if p0 < 1.0 else 0.0
    reading = TypeReading.ENABLING if pn >= config.enabling_pn and ps < config.promoting_ps else TypeReading.PROMOTING
    return TypeReadout(reading, pn, ps)


__all__ = ["Strength", "TypeReading", "TypeReadout", "read_type", "settled_by", "strength_of", "usable_interval"]
