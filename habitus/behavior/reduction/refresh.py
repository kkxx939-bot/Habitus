"""L0/L1 刷新失败的记账：按天记一本重试账，替掉"失败就留在待刷新集合里、每轮原样再来"（26 号清单 2.4.1）。

- 可重试的失败按退避推迟；退避期内这一天照旧留在待刷新集合里，只是这一轮不试；
- 确定性失败（超限）第一次就放弃：这一天移出待刷新集合、发 signal，不再每轮调一次模型；
- 那一天的内容真的变了（检查点重放、词表迁移改了 kind_token、发布了新记录），它的账就清掉，重新给机会——
  这等于按"来源摘要"记账：放弃的是"这一版内容"，不是"这一天"。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from enum import Enum

from habitus.infrastructure.lifecycle import FailureKind, RetryDecision, RetryLedger


class RefreshVerdict(str, Enum):
    TRY = "try"
    WAIT = "wait"
    GIVEN_UP = "given_up"


class RefreshRetries:
    def __init__(self, ledger: RetryLedger) -> None:
        if not isinstance(ledger, RetryLedger):
            raise TypeError("ledger must be RetryLedger")
        self.ledger = ledger

    def touched(self, days: Iterable[date]) -> None:
        """这些天的内容变了：清掉它们的账。"""

        for day in days:
            self.ledger.clear(_key(day))

    def verdict(self, day: date) -> RefreshVerdict:
        entry = self.ledger.entry(_key(day))
        if entry is None or not self.ledger.blocked(_key(day)):
            return RefreshVerdict.TRY
        return RefreshVerdict.GIVEN_UP if entry.kind is FailureKind.DETERMINISTIC else RefreshVerdict.WAIT

    def failed(self, day: date, error: BaseException) -> RetryDecision:
        return self.ledger.record_failure(_key(day), error)

    def succeeded(self, day: date) -> None:
        self.ledger.clear(_key(day))


def _key(day: date) -> str:
    return f"refresh:{day.isoformat()}"


__all__ = ["RefreshRetries", "RefreshVerdict"]
