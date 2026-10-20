"""按名次遗忘：名次稳定、分量只看名次差。"""

import pytest

from habitus.foundation.lifecycle import Ranked, decay, rank_by


def test_rank_by_sorts_and_numbers_from_zero() -> None:
    ranked = rank_by(["c", "a", "b"], key=lambda item: item)
    assert ranked == (Ranked("a", 0), Ranked("b", 1), Ranked("c", 2))


def test_rank_by_keeps_input_order_for_ties() -> None:
    items = [("x", 1), ("y", 0), ("z", 1)]
    ranked = rank_by(items, key=lambda item: item[1])
    assert [entry.item[0] for entry in ranked] == ["y", "x", "z"]


def test_decay_halves_every_half_life() -> None:
    assert decay(0, half_life=8.0) == 1.0
    assert decay(8, half_life=8.0) == pytest.approx(0.5)
    assert decay(16, half_life=8.0) == pytest.approx(0.25)


@pytest.mark.parametrize("gap", [-1, 1.5, True])
def test_decay_rejects_bad_gaps(gap: int) -> None:
    with pytest.raises(ValueError, match="rank_gap"):
        decay(gap, half_life=8.0)


@pytest.mark.parametrize("rank", [-1, True, 1.0])
def test_ranked_rejects_bad_ranks(rank: int) -> None:
    with pytest.raises(ValueError, match="rank"):
        Ranked("item", rank)
