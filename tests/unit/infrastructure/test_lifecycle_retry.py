"""重试账：确定性失败第一次就放弃；可重试的按退避推迟，次数用完转确定性；账本身会退出。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from habitus.foundation.lifecycle import BackoffPolicy, DeterministicFailure
from habitus.infrastructure.lifecycle import FailureKind, RetryLedger, RetryLedgerFullError


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 19, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class _PromptTooLong(DeterministicFailure, ValueError):
    pass


def _ledger(root: Path, clock: _Clock, **kwargs: int) -> RetryLedger:
    return RetryLedger(
        root,
        owner="behavior-semantic-session",
        policy=BackoffPolicy(base_seconds=60.0, cap_seconds=3600.0),
        max_attempts=kwargs.pop("max_attempts", 3),
        clock=clock,
        **kwargs,
    )


def test_deterministic_failure_gives_up_on_first_attempt(tmp_path: Path) -> None:
    clock = _Clock()
    decision = _ledger(tmp_path, clock).record_failure("day:2026-10-18", _PromptTooLong("prompt too long"))
    assert decision.give_up
    assert decision.entry.kind is FailureKind.DETERMINISTIC
    assert decision.entry.attempts == 1


def test_retryable_failure_backs_off_then_becomes_due(tmp_path: Path) -> None:
    clock = _Clock()
    ledger = _ledger(tmp_path, clock)
    decision = ledger.record_failure("day:2026-10-18", TimeoutError("slow"))
    assert not decision.give_up
    assert ledger.blocked("day:2026-10-18")
    assert ledger.due() == ()
    clock.now += timedelta(seconds=61)
    assert ledger.due() == ("day:2026-10-18",)
    assert not ledger.blocked("day:2026-10-18")


def test_retryable_failure_turns_deterministic_after_max_attempts(tmp_path: Path) -> None:
    clock = _Clock()
    ledger = _ledger(tmp_path, clock, max_attempts=3)
    for _ in range(2):
        assert not ledger.record_failure("k", TimeoutError()).give_up
    assert ledger.record_failure("k", TimeoutError()).give_up
    assert [entry.key for entry in ledger.given_up()] == ["k"]


def test_classification_uses_the_marker_not_the_message(tmp_path: Path) -> None:
    """消息里写着 too long 也不算确定性：分类只看是不是 ``DeterministicFailure``。"""

    decision = _ledger(tmp_path, _Clock()).record_failure("k", ValueError("prompt too long"))
    assert not decision.give_up


def test_clear_and_reap_retire_entries(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, _Clock())
    ledger.record_failure("alive", TimeoutError())
    ledger.record_failure("gone", TimeoutError())
    ledger.record_failure("done", TimeoutError())
    assert ledger.clear("done")
    assert not ledger.clear("done")
    assert ledger.reap(lambda key: key == "alive") == 1
    assert ledger.size() == 1


def test_full_ledger_evicts_earliest_retryable_first(tmp_path: Path) -> None:
    clock = _Clock()
    ledger = _ledger(tmp_path, clock, max_entries=2)
    ledger.record_failure("old", TimeoutError())
    clock.now += timedelta(seconds=10)
    ledger.record_failure("stuck", _PromptTooLong())
    ledger.record_failure("new", TimeoutError())
    assert ledger.entry("old") is None
    assert ledger.entry("stuck") is not None
    assert ledger.entry("new") is not None


def test_full_ledger_of_deterministic_failures_raises(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, _Clock(), max_entries=1)
    ledger.record_failure("a", _PromptTooLong())
    with pytest.raises(RetryLedgerFullError):
        ledger.record_failure("b", _PromptTooLong())


def test_state_survives_a_new_instance(tmp_path: Path) -> None:
    clock = _Clock()
    _ledger(tmp_path, clock).record_failure("k", TimeoutError("x"))
    entry = _ledger(tmp_path, clock).entry("k")
    assert entry is not None and entry.attempts == 1 and entry.message == "x"


@pytest.mark.parametrize("key", ["", "a\nb"])
def test_keys_must_be_single_line_text(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match="retry key"):
        _ledger(tmp_path, _Clock()).record_failure(key, TimeoutError())


def test_naive_clock_is_rejected(tmp_path: Path) -> None:
    ledger = RetryLedger(
        tmp_path,
        owner="o",
        policy=BackoffPolicy(base_seconds=1.0, cap_seconds=2.0),
        max_attempts=2,
        clock=lambda: datetime(2026, 10, 19),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        ledger.record_failure("k", TimeoutError())
