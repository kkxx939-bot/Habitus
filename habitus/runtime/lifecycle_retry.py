"""组合根里建重试账的一处：退避与次数上限只从 ``config.lifecycle.retry`` 来，各层不各自拼。

``deterministic`` 是这本账在 ``DeterministicFailure`` 之外还要当"第一次就放弃"的异常（例如调模型的那几本账给
``model_client.MODEL_DETERMINISTIC``）。"""

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
    deterministic: tuple[type[BaseException], ...] = (),
) -> RetryLedger:
    return RetryLedger(
        root,
        owner=owner,
        policy=BackoffPolicy(base_seconds=config.backoff_base_seconds, cap_seconds=config.backoff_cap_seconds),
        max_attempts=config.max_attempts,
        clock=clock or (lambda: datetime.now(UTC)),
        deterministic=deterministic,
    )


__all__ = ["retry_ledger"]
