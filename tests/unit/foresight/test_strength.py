"""可靠度（带遗忘、带强度的 Jeffreys Beta 后验）与反馈强度（这一次在这本账自己历史分布里的位置）。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.reliability import beta_quantile, estimate, incomplete_beta, weight
from habitus.foresight.strength import catch_up_strength, gaps_of, offset_strength

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 9, 1, tzinfo=CST)


def test_the_incomplete_beta_and_its_quantile_match_closed_forms() -> None:
    assert beta_quantile(0.05, 1.0, 1.0) == pytest.approx(0.05)
    assert beta_quantile(0.25, 2.0, 1.0) == pytest.approx(0.5)  # I_x(2, 1) = x²
    assert beta_quantile(1 - 0.8**3, 1.0, 3.0) == pytest.approx(0.2)  # I_x(1, 3) = 1 − (1 − x)³
    assert beta_quantile(0.05, 0.5, 0.5) == pytest.approx(math.sin(math.pi * 0.05 / 2) ** 2)  # 反正弦分布
    assert incomplete_beta(0.1, 0.5, 8.5) == pytest.approx(1 - incomplete_beta(0.9, 8.5, 0.5))
    assert incomplete_beta(0.0, 2.0, 2.0) == 0.0 and incomplete_beta(1.0, 2.0, 2.0) == 1.0
    with pytest.raises(ForesightError):
        incomplete_beta(0.5, 0.0, 1.0)
    with pytest.raises(ForesightError):
        beta_quantile(1.5, 1.0, 1.0)


def test_the_estimate_widens_with_few_samples_and_narrows_with_many() -> None:
    nothing = estimate((), alpha=0.05)
    assert nothing.n == 0 and nothing.lower < 0.01 and nothing.upper > 0.99 and nothing.mean == pytest.approx(0.5)
    two = estimate([(1.0, True)] * 2, alpha=0.05)
    eight = estimate([(1.0, True)] * 8, alpha=0.05)
    assert 0.4 < two.lower < 0.45 and 0.78 < eight.lower < 0.8
    mixed = estimate([(1.0, True), (1.0, False)] * 20, alpha=0.05)
    assert mixed.lower < 0.5 < mixed.upper and mixed.upper - mixed.lower < 0.3
    with pytest.raises(ForesightError):
        estimate((), alpha=0.7)
    with pytest.raises(ForesightError):
        estimate([(-1.0, True)], alpha=0.05)


def test_weight_halves_every_half_life_and_scales_with_strength() -> None:
    assert weight(NOW, NOW, strength=1.0, half_life_days=60.0) == 1.0
    assert weight(NOW - timedelta(days=60), NOW, strength=1.0, half_life_days=60.0) == pytest.approx(0.5)
    assert weight(NOW - timedelta(days=120), NOW, strength=0.5, half_life_days=60.0) == pytest.approx(0.125)
    assert weight(NOW + timedelta(days=1), NOW, strength=1.0, half_life_days=60.0) == 1.0
    with pytest.raises(ForesightError):
        weight(NOW, NOW, strength=1.5, half_life_days=60.0)
    with pytest.raises(ForesightError):
        weight(NOW, NOW, strength=1.0, half_life_days=0.0)


def test_an_off_window_hit_counts_as_much_as_history_has_strayed_that_far() -> None:
    assert offset_strength(0, [40, 50]) == 1.0
    assert offset_strength(20, []) == 1.0  # 样本少：只有方向
    history = [0, 0, 0, 25, -30, 0, 0, 0, 0, 40]
    assert offset_strength(20, history) == pytest.approx(0.3)  # 历史上三成偏这么远
    assert offset_strength(-24 * 60, history) == 0.0  # 从没偏这么远过：几乎不算


def test_a_catch_up_is_bad_only_as_far_as_the_delay_reaches_a_normal_gap() -> None:
    gaps = gaps_of([date.toordinal() for date in (NOW.date() + timedelta(days=d) for d in (0, 5, 10, 16, 21))])
    assert gaps == (5, 5, 6, 5)
    assert catch_up_strength(2.0, gaps) == 0.0  # 隔得比最短的正常间隔还短：不算坏
    assert catch_up_strength(5.0, gaps) == pytest.approx(0.75)
    assert catch_up_strength(9.0, gaps) == 1.0
    assert catch_up_strength(2.0, (5,)) == 1.0
    with pytest.raises(ForesightError):
        catch_up_strength(-1.0, gaps)
