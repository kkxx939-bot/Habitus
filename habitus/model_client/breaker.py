"""每种模型能力一个熔断（裁定 108）：借 OpenViking ``circuit_breaker.py`` 的思路，按 Habitus 的三条规则定开关。

- 关着：照常调用。"等一等可能好"的调用失败（已按路由重试过）连着到阈值就打开；"要人处理"的一次就打开；
  "这一条输入有问题"的不计入。任何一次成功都清零。
- 开着：到点之前的调用直接抛 ``ModelCircuitOpenError``（不发请求）；到点后放**一个**请求试探（半开），其余照旧挡住。
- 试探成功就关上、等待回到起点；试探失败重新打开、等待加倍，封顶。

原来模型队列"连着几项失败整个队列退避"就是这个熔断（并成一个），队列、换代、记忆任务、健康检查都读它。只在进程里，
重启后从关着开始。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from habitus.model_client.config import CallFailureConfig
from habitus.model_client.errors import ModelCircuitOpenError, ModelClientError, ModelErrorKind, ModelErrorRule

logger = logging.getLogger("habitus.model_client.breaker")


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class BreakerSnapshot:
    """给健康检查与观测看的熔断此刻的样子。``reason`` 是打开它的那次失败的类别（关着时是最近一次失败的类别或空）。"""

    state: BreakerState
    reason: ModelErrorKind | None
    consecutive_failures: int
    opened_seconds_ago: float | None
    retry_in_seconds: float | None
    detail: str

    @property
    def open(self) -> bool:
        return self.state is not BreakerState.CLOSED


class CircuitBreaker:
    def __init__(
        self,
        config: CallFailureConfig | None = None,
        *,
        name: str = "model",
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or CallFailureConfig()
        self.name = name
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._reset_seconds = self.config.breaker_reset_base_seconds
        self._opened_at: float | None = None
        self._retry_at = 0.0
        self._probing = False
        self._reason: ModelErrorKind | None = None
        self._detail = ""

    def before_call(self) -> bool:
        """要发请求之前问一声：开着且没到点、或半开时已有试探在路上，就抛 ``ModelCircuitOpenError``。返回这次是不是试探。"""

        with self._lock:
            if self._state is BreakerState.CLOSED:
                return False
            now = self._monotonic()
            if self._state is BreakerState.OPEN and now >= self._retry_at:
                self._state = BreakerState.HALF_OPEN
            if self._state is BreakerState.HALF_OPEN and not self._probing:
                self._probing = True
                return True
            reason = self._reason or ModelErrorKind.UNKNOWN
            wait = max(0.0, self._retry_at - now)
            raise ModelCircuitOpenError(
                f"{self.name} model calls are paused ({reason.value}); next probe in {wait:.0f}s",
                opened_by=reason,
                retry_after_seconds=wait,
            )

    def record_success(self) -> None:
        with self._lock:
            if self._state is not BreakerState.CLOSED:
                logger.info("%s model calls resumed after a successful probe", self.name)
            self._state = BreakerState.CLOSED
            self._failures = 0
            self._reset_seconds = self.config.breaker_reset_base_seconds
            self._opened_at = None
            self._probing = False
            self._reason = None
            self._detail = ""

    def record_failure(self, error: ModelClientError) -> None:
        """记一次调用失败（已按路由重试过）。打开后把离下一次试探还有多久写回 ``error.retry_after_seconds``：记忆任务据它排期。"""

        kind = error.kind or ModelErrorKind.UNKNOWN
        if kind.rule is ModelErrorRule.INPUT:
            self.release_probe()
            return
        with self._lock:
            now = self._monotonic()
            self._failures += 1
            self._reason, self._detail = kind, str(error)[:300]
            if self._state is BreakerState.HALF_OPEN:
                self._open(now, min(self._reset_seconds * 2, self.config.breaker_reset_max_seconds))
            elif kind.rule is ModelErrorRule.OPERATOR or self._failures >= self.config.breaker_failure_threshold:
                if self._state is BreakerState.CLOSED:
                    self._open(now, self.config.breaker_reset_base_seconds)
            if self._state is not BreakerState.CLOSED:
                error.retry_after_seconds = max(error.retry_after_seconds or 0.0, self._retry_at - now)

    def release_probe(self) -> None:
        """试探的那次调用没有结论（被取消、或输入本身有问题）：放掉试探名额，下一次调用接着试探。"""

        with self._lock:
            self._probing = False

    def snapshot(self) -> BreakerSnapshot:
        with self._lock:
            now = self._monotonic()
            opened = None if self._opened_at is None else max(0.0, now - self._opened_at)
            retry = None if self._state is BreakerState.CLOSED else max(0.0, self._retry_at - now)
            return BreakerSnapshot(self._state, self._reason, self._failures, opened, retry, self._detail)

    def is_open(self) -> bool:
        return self.snapshot().open

    def _open(self, now: float, reset_seconds: float) -> None:
        reason = self._reason.value if self._reason is not None else "unknown"
        logger.warning("%s model calls paused reason=%s next_probe_in=%.0fs", self.name, reason, reset_seconds)
        self._state = BreakerState.OPEN
        self._reset_seconds = reset_seconds
        self._retry_at = now + reset_seconds
        self._probing = False
        if self._opened_at is None:
            self._opened_at = now


__all__ = ["BreakerSnapshot", "BreakerState", "CircuitBreaker"]
