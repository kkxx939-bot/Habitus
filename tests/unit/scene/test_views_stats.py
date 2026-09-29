"""⑤ 读时统计基元：Beta 分布函数与分位数（无 scipy）、确定性 bootstrap、环形差、单调性。"""

from __future__ import annotations

import math

import pytest

from habitus.scene.views.stats import (
    Interval,
    StatsError,
    beta_cdf,
    beta_quantile,
    block_bootstrap,
    bootstrap_interval,
    effective_counts,
    is_monotonic,
    jeffreys_interval,
    kaplan_meier,
    mean,
    median,
    survival_at,
    survival_median,
    widen,
)


def test_beta_cdf_matches_closed_forms() -> None:
    assert beta_cdf(0.5, 1, 1) == pytest.approx(0.5)
    for x in (0.1, 0.37, 0.8):
        assert beta_cdf(x, 2, 1) == pytest.approx(x * x, abs=1e-9)  # Beta(2,1) 的 CDF 是 x²
        assert beta_cdf(x, 1, 2) == pytest.approx(1 - (1 - x) ** 2, abs=1e-9)
        assert beta_cdf(x, 3, 5) == pytest.approx(1 - beta_cdf(1 - x, 5, 3), abs=1e-9)  # 对称
    assert beta_cdf(0.0, 2, 3) == 0.0 and beta_cdf(1.0, 2, 3) == 1.0
    with pytest.raises(StatsError):
        beta_cdf(0.5, 0, 1)


def test_beta_quantile_inverts_the_cdf() -> None:
    for a, b in ((1, 1), (3, 8), (12.5, 2.5)):
        for q in (0.05, 0.5, 0.95):
            assert beta_cdf(beta_quantile(q, a, b), a, b) == pytest.approx(q, abs=1e-8)
    with pytest.raises(StatsError):
        beta_quantile(1.0, 1, 1)


def test_bootstrap_is_deterministic_per_seed_and_degenerates_gracefully() -> None:
    values = [3.0, 5.0, 4.0, 8.0, 2.0, 6.0]
    first = bootstrap_interval(values, "median", seed="早餐/晚睡--timing")
    again = bootstrap_interval(values, "median", seed="早餐/晚睡--timing")
    assert first == again and first.point == median(values) == 4.5
    assert first.low <= first.point <= first.high
    # 点估计与种子无关（它算的是原样本），只有区间端点可能随种子动；这组数据上连端点都一样，
    # 所以这里只断言"确定性"，不断言"不同种子给不同区间"——后者在小样本上本来就不成立。
    other = bootstrap_interval(values, "median", seed="another")
    assert other.point == first.point
    single = bootstrap_interval([7.0], "mean", seed="x")
    assert (single.point, single.low, single.high) == (7.0, 7.0, 7.0)
    with pytest.raises(StatsError):
        bootstrap_interval([], "mean", seed="x")
    with pytest.raises(StatsError):
        bootstrap_interval([1.0, math.nan], "mean", seed="x")
    with pytest.raises(StatsError):
        bootstrap_interval([1.0], "mode", seed="x")


def test_intervals_and_monotonicity() -> None:
    assert Interval(0.1, -0.2, 0.3).contains_zero and not Interval(-0.5, -0.8, -0.2).contains_zero
    assert Interval(1, 0, 2).disjoint_from(Interval(5, 3, 6)) and not Interval(1, 0, 2).disjoint_from(Interval(2, 1, 3))
    with pytest.raises(StatsError):
        Interval(1, 2, 0)
    assert is_monotonic([1, 2, 2, 5]) and is_monotonic([5, 3, 1]) and not is_monotonic([1, 3, 2]) and not is_monotonic([1])


def test_kaplan_meier_walks_opportunities_with_right_censoring() -> None:
    """6 条：第 1 步来了 2 条，4 条等过 3 步没来 → S(1)=4/6；第 2 步 1 条来了（风险集 4）→ S(2)=4/6·3/4=0.5 → 中位第 2 步。"""

    curve = kaplan_meier([1, 1, 2], [3, 3, 3])
    assert [(s.step, s.at_risk, s.events) for s in curve] == [(1, 6, 2), (2, 4, 1), (3, 3, 0)]
    assert survival_at(curve, 1) == pytest.approx(4 / 6) and survival_at(curve, 2) == pytest.approx(0.5) and survival_at(curve, 0) == 1.0
    assert survival_median(curve) == 2 and survival_at(curve, 7) is None
    # 全部删失、没人到：S 不降，中位没到；删失 0 步的那条从不进风险集。
    flat = kaplan_meier([], [2, 2, 0])
    assert survival_median(flat) is None and survival_at(flat, 1) == 1.0
    with pytest.raises(StatsError):
        kaplan_meier([0], [])
    with pytest.raises(StatsError):
        kaplan_meier([1], [-1])


def test_block_bootstrap_resamples_whole_blocks_and_skips_undefined_resamples() -> None:
    """同一块里的两次机会一起进一起出：只有两个块时区间只在两种组合之间摆。"""

    values = [1.0, 1.0, 5.0, 5.0]
    blocks = [0, 0, 1, 1]
    interval = block_bootstrap(values, blocks, mean, seed="x")
    assert interval is not None and interval.point == 3.0 and {interval.low, interval.high} <= {1.0, 3.0, 5.0}
    single = block_bootstrap([2.0, 4.0], [7, 7], mean, seed="x")
    assert single == Interval(3.0, 3.0, 3.0)  # 一个块：没有可重采样的独立单元
    # 统计量在某些重采样上算不出（返回 None）→ 那次丢掉，不塌。
    picky = block_bootstrap([1.0, 2.0, 3.0], [0, 1, 2], lambda xs: None if len(set(xs)) == 1 else mean(xs), seed="y", samples=50)
    assert picky is not None and picky.point == 2.0
    assert block_bootstrap([1.0], [0], lambda xs: None, seed="z") is None
    with pytest.raises(StatsError):
        block_bootstrap([1.0, 2.0], [0], mean, seed="w")


def test_survival_past_the_point_everyone_arrived_is_zero_not_unknown() -> None:
    """全部在第 1 步就到了 → S(2) 是 0（p1=1），不是"算不出"（评审 A-5/C-10：读数会印成"还没攒够"）。"""

    arrived = kaplan_meier([1, 1, 1], [])
    assert survival_at(arrived, 1) == 0.0 and survival_at(arrived, 2) == 0.0 and survival_at(arrived, 7) == 0.0
    # 末步 S>0 而曲线到此为止（没人等到那一步）才是真的不知道。
    waiting = kaplan_meier([1, 2], [3])
    assert survival_at(waiting, 3) == pytest.approx(1 / 3) and survival_at(waiting, 4) is None


def test_jeffreys_gives_a_width_where_a_percentile_bootstrap_collapses() -> None:
    """六条全到：百分位 bootstrap 每次重采样都给同一个数（零宽 → 必然"显著"）；Jeffreys（Beta(6.5, 0.5) 的中央 90%）给 [0.74, 1.0]。"""

    counts = effective_counts(kaplan_meier([1] * 6, []), 1)
    assert counts is not None and counts.events == 6.0 and counts.remainder == 0.0
    rate = jeffreys_interval(counts.events, counts.remainder)
    assert rate.point == 1.0 and rate.low == pytest.approx(0.736, abs=0.01) and rate.high == pytest.approx(1.0, abs=0.01)
    assert rate.width > 0
    # 取并：谁更宽听谁的，点估计用第一个的。
    assert widen(Interval(0.1, 0.0, 0.2), Interval(0.1, -0.3, 0.15)) == Interval(0.1, -0.3, 0.2)
    assert widen(None, rate) == rate and widen(rate, None) == rate and widen(None, None) is None
    with pytest.raises(StatsError):
        jeffreys_interval(0, 0)
