"""「开始前有没有被提醒过」——归约封口时盖 ``reminded`` 的查法（方案第九节第 0 步）。

一条记录这一类在它开始之前被提醒过 = 账本里有一条**说出口的**递送（话已经带给他了），它的预期认的类现在对应着这条记录的类
（与结算同一个 ``current`` 映射：词表拆改之后两边认的是同一件事）、决定在这条记录开始之前、而且这条记录落在那条预期说中的
范围里（形成之后、最晚时限之前）。只看账本里还没结完的日子（``open_days``：说中的范围还没过完的预期都在那里）。
端口由组合根注入归约 runner（``BehaviorReductionRunner.attach_reminder_lookup``），behavior 对 foresight 零知识。

归约每条链问一次，一轮上千次：每一天说出口的那几条（连同它们的预期）读一遍记住，递送目录的文件名没变就不再读（只增）。
这一次读到的坏文件就抛（``ForesightError``）：整轮归约失败、下一轮再来，不许静默记成没提醒；别的读口读到的坏文件不归它管。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime

from habitus.foresight import ForesightError
from habitus.foresight.ledger import Expectation
from habitus.runtime.foresight_ledger import ForesightLedgerStore


class RemindedLookup:
    def __init__(
        self, ledger: ForesightLedgerStore, *, current: Callable[[str], Sequence[str]] = lambda token: (token,)
    ) -> None:
        if not isinstance(ledger, ForesightLedgerStore):
            raise TypeError("ledger must be a ForesightLedgerStore")
        if not callable(current):
            raise TypeError("current must be callable")
        self.ledger = ledger
        self.current = current
        #: 日子 → (递送目录的文件名, 那天说出口的 [(预期, 决定时刻)])。
        self._spoken: dict[date, tuple[frozenset[str], tuple[tuple[Expectation, datetime], ...]]] = {}

    def __call__(self, kind_token: str, started_at: datetime) -> bool:
        if not isinstance(kind_token, str) or not kind_token:
            raise ForesightError("kind_token must be non-empty text")
        if not isinstance(started_at, datetime) or started_at.utcoffset() is None:
            raise ForesightError("started_at must be a timezone-aware datetime")
        open_days = self.ledger.open_days()
        # 结完的日子不会再被问到：它们的缓存淘汰掉（26 号清单 6.13）
        for stale in [day for day in self._spoken if day not in open_days]:
            del self._spoken[stale]
        for day in open_days:
            if day > started_at.date():
                continue
            for item, decided_at in self._spoken_on(day):
                if decided_at < started_at and item.covers(started_at) and kind_token in self.current(item.kind_token):
                    return True
        return False

    def _spoken_on(self, day: date) -> tuple[tuple[Expectation, datetime], ...]:
        names = self.ledger.delivery_names(day)
        cached = self._spoken.get(day)
        if cached is not None and cached[0] == names:
            return cached[1]
        unreadable: list[str] = []
        spoken = [item for item in self.ledger.deliveries_on(day, unreadable=unreadable) if item.spoken]
        found: list[tuple[Expectation, datetime]] = []
        if spoken:
            by_id = {item.expectation_id: item for item in self.ledger.expectations_on(day, unreadable=unreadable)}
            found = [(by_id[d.expectation_id], d.decided_at) for d in spoken if d.expectation_id in by_id]
        if unreadable:
            raise ForesightError(f"ledger files could not be read while checking reminders: {unreadable[0]}")
        result = tuple(found)
        self._spoken[day] = (names, result)
        return result


__all__ = ["RemindedLookup"]
