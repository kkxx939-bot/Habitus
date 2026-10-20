"""可靠度（带遗忘、带强度的 Jeffreys Beta 后验）与反馈强度（这一次在这本账自己历史分布里的位置）。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.facts import last_before
from habitus.foresight.ledger import Expectation
from habitus.foresight.opportunities import KindDays, OpportunityTable
from habitus.foresight.reliability import beta_quantile, estimate, incomplete_beta, weighted
from habitus.foresight.strength import catch_up_strength, offset_strength
from habitus.temporal import DayInterval, OccurrenceDay

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


def test_weight_halves_every_half_life_of_opportunities_and_scales_with_strength() -> None:
    """一笔的分量只看它之后又来了几次机会（裁定 57）：最新的满分量，隔 H 次减半，再乘强度。"""

    found = weighted([(8, 1.0, True), (0, 1.0, False), (4, 0.5, True)], last_rank=8, half_life=8.0)
    assert (
        found[0] == (1.0, True)
        and found[1] == (pytest.approx(0.5), False)
        and found[2] == (pytest.approx(0.5 * 2**-0.5), True)
    )
    with pytest.raises(ForesightError, match="strength"):
        weighted([(0, 1.5, True)], last_rank=0, half_life=8.0)
    with pytest.raises(ForesightError, match="after the last"):
        weighted([(3, 1.0, True)], last_rank=2, half_life=8.0)
    with pytest.raises(ForesightError, match="half_life"):
        weighted([(0, 1.0, True)], last_rank=0, half_life=0.0)


def test_an_off_window_hit_counts_as_much_as_history_has_strayed_that_far() -> None:
    assert offset_strength(0, [40, 50]) == 1.0
    assert offset_strength(20, []) == 1.0  # 样本少：只有方向
    history = [0, 0, 0, 25, -30, 0, 0, 0, 0, 40]
    assert offset_strength(20, history) == pytest.approx(0.3)  # 历史上三成偏这么远
    assert offset_strength(-24 * 60, history) == 0.0  # 从没偏这么远过：几乎不算


def test_a_catch_up_is_bad_only_as_far_as_the_delay_reaches_a_normal_gap() -> None:
    gaps = (5, 5, 6, 5)
    assert catch_up_strength(2.0, gaps) == 0.0  # 隔得比最短的正常间隔还短：不算坏
    assert catch_up_strength(5.0, gaps) == pytest.approx(0.75)
    assert catch_up_strength(9.0, gaps) == 1.0
    assert catch_up_strength(2.0, (5,)) == 1.0
    with pytest.raises(ForesightError):
        catch_up_strength(-1.0, gaps)


def test_the_catch_up_gaps_come_from_the_recurrence_without_censored_or_later_intervals() -> None:
    """补做比的间隔读语义树登记的复现（temporal 算的），不自己按天相减：以被提醒结尾的那段截尾不算，这次机会之后的也不算。"""

    def day(offset: int, natural: int = 1) -> OccurrenceDay:
        at = NOW + timedelta(days=offset, hours=9)
        return OccurrenceDay(
            at.date(), f"u{offset}", at, 1, natural, f"u{offset}" if natural else None, at if natural else None
        )

    days = (day(0), day(5), day(7, natural=0), day(13), day(30))
    intervals = tuple(
        DayInterval(a, b, (b.day - a.day).days, 1.0, b.natural == 0) for a, b in zip(days, days[1:], strict=False)
    )
    table = OpportunityTable(
        "session", NOW.date(), NOW.date(), 15, 3, (), (KindDays("s-k0001", "买菜", days, intervals),)
    )
    observed_from = NOW + timedelta(days=20)
    item = Expectation(
        expectation_id="e",
        kind_token="s-k0001",
        day=observed_from.date(),
        formed_at=observed_from,
        source="日子",
        trigger="interval",
        trigger_instance="u13@x",
        window=(observed_from, observed_from + timedelta(hours=1)),
        deadline=observed_from + timedelta(days=1),
        basis=(),
        opportunities=2,
        hits=2,
        observed_from=observed_from,
    )
    last, gaps = last_before(item, table)
    assert last == days[3].day and gaps == (5, 6)  # 0→5 与 7→13；5→7 截尾、13→30 在这次之后
