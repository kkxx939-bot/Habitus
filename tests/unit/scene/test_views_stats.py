"""⑤ 读时统计基元：Beta 分布函数与分位数（无 scipy）、确定性 bootstrap、按块折算的等价计数、单调性。"""

from __future__ import annotations

import math

import pytest

from habitus.scene.views.stats import (
    Interval,
    StatsError,
    beta_cdf,
    beta_quantile,
    block_bootstrap,
    effective_counts,
    is_monotonic,
    jeffreys_interval,
    mean,
    median,
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


def test_block_bootstrap_is_deterministic_per_seed_and_degenerates_gracefully() -> None:
    """（``bootstrap_interval`` 已删：生产里只剩按块的那一个，评审 A-15 / C-16。）"""

    values = [3.0, 5.0, 4.0, 8.0, 2.0, 6.0]
    blocks = ["a", "b", "c", "d", "e", "f"]
    first = block_bootstrap(values, blocks, median, seed="早餐/晚睡--timing")
    again = block_bootstrap(values, blocks, median, seed="早餐/晚睡--timing")
    assert first is not None and first == again and first.point == median(values) == 4.5
    assert first.low <= first.point <= first.high
    # 点估计与种子无关（它算的是原样本），只有区间端点可能随种子动
    other = block_bootstrap(values, blocks, median, seed="another")
    assert other is not None and other.point == first.point
    single = block_bootstrap([7.0], ["a"], mean, seed="x")
    assert single is not None and (single.point, single.low, single.high) == (7.0, 7.0, 7.0)


def test_intervals_and_monotonicity() -> None:
    assert Interval(0.1, -0.2, 0.3).contains_zero and not Interval(-0.5, -0.8, -0.2).contains_zero
    assert Interval(1, 0, 2).disjoint_from(Interval(5, 3, 6)) and not Interval(1, 0, 2).disjoint_from(Interval(2, 1, 3))
    with pytest.raises(StatsError):
        Interval(1, 2, 0)
    assert is_monotonic([1, 2, 2, 5]) and is_monotonic([5, 3, 1]) and not is_monotonic([1, 3, 2]) and not is_monotonic([1])


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


def test_jeffreys_gives_a_width_where_a_percentile_bootstrap_collapses() -> None:
    """六条全到：百分位 bootstrap 每次重采样都给同一个数（零宽 → 必然"显著"）；Jeffreys（Beta(6.5, 0.5) 的中央 90%）给 [0.74, 1.0]。"""

    counts = effective_counts(6, 0)
    assert counts is not None and counts.events == 6.0 and counts.remainder == 0.0
    # 按块数折算：一天开 6 条、等的是同一个窗口 → 等价样本只有块数那么多（评审 C-6）。
    folded = effective_counts(4, 2, blocks=3)
    assert folded is not None and folded.events == pytest.approx(2.0) and folded.remainder == pytest.approx(1.0)
    assert effective_counts(0, 0) is None
    with pytest.raises(ValueError):
        effective_counts(-1, 0)
    rate = jeffreys_interval(counts.events, counts.remainder)
    assert rate.point == 1.0 and rate.low == pytest.approx(0.736, abs=0.01) and rate.high == pytest.approx(1.0, abs=0.01)
    assert rate.width > 0
    # 取并：谁更宽听谁的，点估计用第一个的。
    assert widen(Interval(0.1, 0.0, 0.2), Interval(0.1, -0.3, 0.15)) == Interval(0.1, -0.3, 0.2)
    assert widen(None, rate) == rate and widen(rate, None) == rate and widen(None, None) is None
    with pytest.raises(StatsError):
        jeffreys_interval(0, 0)
