"""每晚跑批的第一步：把白天读不懂的重试一次（裁定 67、68）。编排在这里，怎么试由每种项的处理器管（``handlers``）。

- **按天走**：这一晚处理属于 ``through`` 那天及以前的项。先把"重试到哪一天"推到 ``through``（之后属于这些天的项不再
  收），再试；比上一次推到的那天还早的项是上一晚中途被杀剩下的——那一天已经跑过，直接丢、不再试（同一轮不会补两遍）。
- **每项试完立刻删**：成了、败了都删；中途被杀，试过的不会被第二次试到。
- **先探**：会话轮排在前面，第一条就是"模型用不了"就把这一支剩下的会话轮都按丢弃删——不逐条等超时拖住这一晚；链的
  放回不调模型，照常做。
- 放回的链交给这一支的归约落树，之后问处理器成没成。
- 四类计数：补回 / 模型用不了 / 内容不合格 / 确定性失败（裁定 67 第 12 条），外加上一晚剩下、直接丢的。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Protocol, runtime_checkable

from habitus.behavior.retry.model import RetryItem, RetryKind
from habitus.behavior.retry.store import RetryBacklogStore


class Attempt(str, Enum):
    """试一项的结果。``RESTORED``：放回去了，归约之后才知道成没成。"""

    RECOVERED = "recovered"
    RESTORED = "restored"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    DETERMINISTIC = "deterministic"


@runtime_checkable
class RetryHandler(Protocol):
    async def attempt(self, item: RetryItem) -> Attempt: ...

    def recovered(self, item: RetryItem) -> bool:
        """放回的那一项，归约之后成了没有（只对 ``RESTORED`` 问）。"""
        ...

    def abandon(self, item: RetryItem) -> None:
        """上一晚中途被杀剩下的一项：不再试，撤回它写到一半的东西（回到重试之前的样子）。"""
        ...


@dataclass(frozen=True)
class RetryBranch:
    """一支：它的重试堆、每种项的处理器、跑一次归约（返回没跑成的原因，跑成了为 None）。"""

    name: str
    backlog: RetryBacklogStore
    handlers: Mapping[RetryKind, RetryHandler]
    reduce: Callable[[], Awaitable[str | None]]


@dataclass(frozen=True)
class RetryReport:
    """这一晚的重试：取出几项、各类几项、上一晚剩下直接丢的几项、归约没跑成的支与原因。"""

    taken: int = 0
    recovered: int = 0
    unavailable: int = 0
    invalid: int = 0
    deterministic: int = 0
    leftover: int = 0
    unreduced: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dropped(self) -> int:
        return self.unavailable + self.invalid + self.deterministic + self.leftover


@dataclass
class _Tally:
    recovered: int = 0
    unavailable: int = 0
    invalid: int = 0
    deterministic: int = 0
    leftover: int = 0
    taken: int = 0


class RetryStage:
    def __init__(self, branches: Sequence[RetryBranch]) -> None:
        self.branches = tuple(branches)
        if any(not isinstance(branch, RetryBranch) for branch in self.branches):
            raise TypeError("branches must be RetryBranch instances")

    async def run_once(self, *, through: date) -> RetryReport:
        tally = _Tally()
        unreduced: list[str] = []
        for branch in self.branches:
            reason = await self._branch(branch, through, tally)
            if reason is not None:
                unreduced.append(reason)
        return RetryReport(
            taken=tally.taken,
            recovered=tally.recovered,
            unavailable=tally.unavailable,
            invalid=tally.invalid,
            deterministic=tally.deterministic,
            leftover=tally.leftover,
            unreduced=tuple(unreduced),
        )

    def clean_leftovers(self) -> int:
        """进程重启时（常驻节拍一起来就调）：上一晚中途被杀剩下的项——属于已经跑过的日子——撤回、删掉，不等到下一晚；
        否则它写到一半的凭据会把这一整天的封口视界钉住。返回清了几项。"""

        cleaned = 0
        for branch in self.branches:
            through = branch.backlog.through()
            if through is None:
                continue
            batch = branch.backlog.snapshot(through=through)
            for item in batch.items:
                handler = branch.handlers.get(item.kind)
                if handler is not None:
                    handler.abandon(item)
            cleaned += branch.backlog.discard(batch.ids)
        return cleaned

    async def _branch(self, branch: RetryBranch, through: date, tally: _Tally) -> str | None:
        previous = branch.backlog.through()
        branch.backlog.advance(through)
        batch = branch.backlog.snapshot(through=through)
        tally.taken += len(batch.ids)
        tally.deterministic += branch.backlog.discard(batch.unreadable)
        restored: list[RetryItem] = []
        stopped = False
        # 会话轮排前面：第一条就探出模型能不能用
        for item in sorted(batch.items, key=lambda item: item.kind is not RetryKind.TURN):
            handler = branch.handlers.get(item.kind)
            if previous is not None and item.day <= previous:
                tally.leftover += 1
                if handler is not None:
                    handler.abandon(item)
            elif handler is None:
                tally.deterministic += 1
            elif stopped and item.kind is RetryKind.TURN:
                tally.unavailable += 1
            else:
                outcome = await handler.attempt(item)
                if outcome is Attempt.RECOVERED:
                    tally.recovered += 1
                elif outcome is Attempt.RESTORED:
                    restored.append(item)
                elif outcome is Attempt.UNAVAILABLE:
                    tally.unavailable += 1
                    stopped = stopped or item.kind is RetryKind.TURN
                elif outcome is Attempt.INVALID:
                    tally.invalid += 1
                else:
                    tally.deterministic += 1
            branch.backlog.discard((item.item_id,))
        reason = None
        if tally.recovered or restored:
            reason = await branch.reduce()
        for item in restored:
            if branch.handlers[item.kind].recovered(item):
                tally.recovered += 1
            else:
                tally.deterministic += 1
        return reason


__all__ = ["Attempt", "RetryBranch", "RetryHandler", "RetryReport", "RetryStage"]
