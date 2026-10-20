"""退避：base × 2^(attempt − 1)，封顶 cap；与 memory 扫描原有的公式一致。"""

import pytest

from habitus.foundation.lifecycle import BackoffPolicy, DeterministicFailure


def test_delay_doubles_and_caps() -> None:
    policy = BackoffPolicy(base_seconds=60.0, cap_seconds=86_400.0)
    assert [policy.delay(attempt) for attempt in (1, 2, 3)] == [60.0, 120.0, 240.0]
    assert policy.delay(20) == 86_400.0


def test_max_doublings_only_guards_overflow() -> None:
    policy = BackoffPolicy(base_seconds=1e-6, cap_seconds=1e9, max_doublings=20)
    assert policy.delay(100) == pytest.approx(1e-6 * 2**20)


@pytest.mark.parametrize(
    ("base", "cap", "message"),
    [(0.0, 1.0, "base_seconds"), (2.0, 1.0, "must not exceed"), (1.0, float("inf"), "cap_seconds")],
)
def test_policy_rejects_bad_parameters(base: float, cap: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BackoffPolicy(base_seconds=base, cap_seconds=cap)


@pytest.mark.parametrize("attempt", [0, -1, True])
def test_attempt_starts_at_one(attempt: int) -> None:
    with pytest.raises(ValueError, match="attempt"):
        BackoffPolicy(base_seconds=1.0, cap_seconds=2.0).delay(attempt)


def test_policy_rejects_negative_max_doublings() -> None:
    with pytest.raises(ValueError, match="max_doublings"):
        BackoffPolicy(base_seconds=1.0, cap_seconds=2.0, max_doublings=-1)


def test_deterministic_failure_is_a_marker_base() -> None:
    class PromptTooLong(DeterministicFailure, ValueError):
        pass

    error = PromptTooLong("too long")
    assert isinstance(error, DeterministicFailure)
    assert isinstance(error, ValueError)
