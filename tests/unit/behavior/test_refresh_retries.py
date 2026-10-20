"""L0/L1 刷新的重试账：可重试的退避、确定性的放弃、内容变了就重新给机会。"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from habitus.behavior.reduction.refresh import RefreshRetries, RefreshVerdict
from habitus.behavior.semantic import BehaviorOverviewLimitError
from habitus.foundation.lifecycle import BackoffPolicy
from habitus.infrastructure.lifecycle import RetryLedger

DAY = date(2026, 8, 16)


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 17, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _retries(root: Path, clock: _Clock) -> RefreshRetries:
    ledger = RetryLedger(
        root,
        owner="behavior-semantic-test",
        policy=BackoffPolicy(base_seconds=60.0, cap_seconds=3600.0),
        max_attempts=3,
        clock=clock,
    )
    return RefreshRetries(ledger)


def test_retryable_failures_wait_out_the_backoff(tmp_path: Path) -> None:
    clock = _Clock()
    retries = _retries(tmp_path, clock)
    assert retries.verdict(DAY) is RefreshVerdict.TRY
    assert not retries.failed(DAY, TimeoutError("llm down")).give_up
    assert retries.verdict(DAY) is RefreshVerdict.WAIT
    clock.now += timedelta(seconds=61)
    assert retries.verdict(DAY) is RefreshVerdict.TRY


def test_limits_are_given_up_until_the_day_is_touched(tmp_path: Path) -> None:
    retries = _retries(tmp_path, _Clock())
    assert retries.failed(DAY, BehaviorOverviewLimitError("too long")).give_up
    assert retries.verdict(DAY) is RefreshVerdict.GIVEN_UP
    retries.touched({DAY})
    assert retries.verdict(DAY) is RefreshVerdict.TRY


def test_success_clears_the_entry(tmp_path: Path) -> None:
    retries = _retries(tmp_path, _Clock())
    retries.failed(DAY, TimeoutError())
    retries.succeeded(DAY)
    assert retries.ledger.entry("refresh:2026-08-16") is None
