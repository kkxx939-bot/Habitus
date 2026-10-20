"""精确压实：先压成累计值再接着加，和从头逐笔加，结果逐位相同。"""

import random

import pytest

from habitus.foundation.lifecycle import Tally, advance, decay, fold, fold_all, rewind, step_factor

H = 8.0


def _history(seed: int, size: int) -> list[tuple[bool, float]]:
    rng = random.Random(seed)
    return [(rng.random() < 0.7, rng.choice((1.0, 0.0, rng.random()))) for _ in range(size)]


@pytest.mark.parametrize("seed", range(20))
def test_compacted_prefix_then_window_equals_full_read(seed: int) -> None:
    events = _history(seed, 200)
    full = fold_all(events, half_life=H)
    compacted = fold_all(events[:-35], half_life=H)
    assert fold_all(events[-35:], half_life=H, start=compacted) == full


def test_fold_decays_old_evidence_before_adding_the_new_one() -> None:
    tally = fold_all([(True, 1.0), (False, 1.0)], half_life=H)
    factor = step_factor(H)
    assert tally == Tally(k=factor, n=factor + 1.0)


def test_fold_matches_rank_weights() -> None:
    """一笔的分量 = 强度 × decay(它之后来了几次机会)，与按名次求和是同一个量。"""

    events = _history(7, 30)
    tally = fold_all(events, half_life=H)
    last = len(events) - 1
    k = sum(s * decay(last - i, half_life=H) for i, (came, s) in enumerate(events) if came)
    n = sum(s * decay(last - i, half_life=H) for i, (_came, s) in enumerate(events))
    assert tally.k == pytest.approx(k, rel=1e-12)
    assert tally.n == pytest.approx(n, rel=1e-12)


def test_zero_strength_is_still_an_opportunity() -> None:
    before = Tally(k=2.0, n=3.0)
    after = fold(before, came=True, strength=0.0, half_life=H)
    assert after == Tally(k=2.0 * step_factor(H), n=3.0 * step_factor(H))


def test_advance_and_rewind_are_inverse() -> None:
    tally = Tally(k=3.0, n=5.0)
    moved = advance(tally, 12, half_life=H)
    assert moved.n == pytest.approx(5.0 * 0.5 ** (12 / H))
    back = rewind(moved, 12, half_life=H)
    assert back.k == pytest.approx(3.0)
    assert back.n == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": -1.0, "n": 1.0}, "non-negative"),
        ({"k": 2.0, "n": 1.0}, "must not exceed"),
        ({"k": float("nan"), "n": 1.0}, "finite"),
        ({"k": True, "n": 1.0}, "finite"),
    ],
)
def test_tally_rejects_inconsistent_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Tally(**kwargs)


@pytest.mark.parametrize("strength", [-0.1, 1.1, True])
def test_fold_rejects_strength_outside_unit_interval(strength: float) -> None:
    with pytest.raises(ValueError, match="strength"):
        fold(Tally(), came=True, strength=strength, half_life=H)


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_steps_must_be_non_negative_integers(steps: int) -> None:
    with pytest.raises(ValueError, match="steps"):
        advance(Tally(), steps, half_life=H)


@pytest.mark.parametrize("half_life", [0.0, -1.0, float("inf"), True])
def test_half_life_must_be_positive_and_finite(half_life: float) -> None:
    with pytest.raises(ValueError, match="half_life"):
        step_factor(half_life)
