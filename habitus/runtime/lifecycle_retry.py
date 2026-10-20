"""组合根里建重试账的一处：退避与次数上限只从 ``config.lifecycle.retry`` 来，各层不各自拼。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from habitus.config import LifecycleRetryConfig
from habitus.foundation.lifecycle import BackoffPolicy
from habitus.infrastructure.lifecycle import RetryLedger


def retry_ledger(
    root: str | Path,
    *,
    owner: str,
    config: LifecycleRetryConfig,
    clock: Callable[[], datetime] | None = None,
) -> RetryLedger:
    return RetryLedger(
        root,
        owner=owner,
        policy=BackoffPolicy(base_seconds=config.backoff_base_seconds, cap_seconds=config.backoff_cap_seconds),
        max_attempts=config.max_attempts,
        clock=clock or (lambda: datetime.now(UTC)),
    )


__all__ = ["retry_ledger"]
