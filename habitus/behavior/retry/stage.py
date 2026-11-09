"""夜里那条线的前置处理里那一步：白天放一边的错误统一看一次（裁定 67、68；裁定 94 第 3、4、8、9 条）。编排在这里，怎么试由每种项的
处理器管（``RetryHandler``：住在它试的东西那边，会话轮的在会话 lane、链的在归约）。

- **按天走**：这一晚处理属于 ``through`` 那天及以前的项。先把"重试到哪一天"推到 ``through``（之后属于这些天的项不再
  收），再试；比上一次推到的那天还早的项是上一晚中途被杀剩下的——那一天已经跑过，直接丢、不再试（同一轮不会补两遍）。
- **每项试完立刻删**：成了、败了都删；中途被杀，试过的不会被第二次试到。写回了判断、放回了链的那几项例外：等这一支的归约
  落了树、问过成没成才删（归约没跑成时它们还在，进程重启由 ``clean_leftovers`` 撤回）。
- **能重试的才试**（``RetryCause.retried``）：内容安全拦下的、我们自己处理不了的不白试，直接删、按类计数进健康面
  （树上照旧是「错误」）。
- **一项抛了是 bug，只拦这一项**：记成我们自己处理不了的（树上照旧是「错误」、计数报出），后面的项照试、这一支照常归约。
- **模型整个调不通不重试**：会话轮排在前面，试到"模型用不了"就记一项 ``unavailable``、不再试别的（不逐条等超时），这一支
  已写回的照常归约后收工；这一支的归约里碰到模型整个用不了（``Reduction.UNREACHABLE``）也一样。夜里那条线看到
  ``unreachable`` 就停下这一晚（裁定 94 第 4 条），这一晚的计数照样留在报告上。剩下的属于已推过的日子，下一晚按上一晚剩下的处理。
- **锁被另一进程拿着**不是错误、更不是模型调不通，是**还没准备好**（裁定 94 第 9 条）：一项的锁被占（``Attempt.BUSY``）不删、
  这一支归约的锁被占（``Reduction.BUSY``）写回的那几项也不删，报告记下这一支在等（``waiting``），线不往下结算、下一眼再来——
  这一趟从这里接着跑：等归约的那几项不再试、直接归约，锁被占的那几项再试一次，计数接着这一晚之前那一趟的。所以写回的判断一定在
  结算之前落树，不会在白天落进已经结算的日子（裁定 68 第 3 条）。
- **归约抛了是 bug**：原样抛给线，停下这一晚、报出（裁定 94 第 9 条）；写回的那几项留着，下一次先归约它们。
- 写回了判断的会话轮、放回的链交给这一支的归约落树，之后逐项问处理器成没成（写回途中被同时跑的白天归约释放了凭据的
  会话轮落不了树，不算补回）；这一支一项都没写回、没放回就不跑归约，计数也只看这一支的。
- 计数：补回 / 模型用不了 / 内容不合格 / 确定性失败 / 内容安全（裁定 67 第 12 条），外加上一晚剩下、直接丢的。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from typing import Protocol, runtime_checkable

from habitus.behavior.retry.model import RetryCause, RetryItem, RetryKind
from habitus.behavior.retry.store import RetryBacklogStore


class Attempt(str, Enum):
    """试一项的结果。``RECOVERED``：重新描述、写回了判断；``RESTORED``：放回去了。这两种都要等归约之后才知道成没成。"""

    RECOVERED = "recovered"
    RESTORED = "restored"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    DETERMINISTIC = "deterministic"
    BUSY = "busy"  # 锁被另一进程拿着：没试成，不是错误，还没准备好


class Reduction(str, Enum):
    """一支跑一次归约的结局（抛出的是 bug）。"""

    DONE = "done"
    BUSY = "busy"  # sweep 锁被另一进程拿着：还没准备好，下一眼再来
    UNREACHABLE = "unreachable"  # 归类时模型整个用不了：线停下这一晚


@runtime_checkable
class RetryHandler(Protocol):
    async def attempt(self, item: RetryItem) -> Attempt: ...

    def recovered(self, item: RetryItem) -> bool:
        """写回 / 放回的那一项，归约之后成了没有（对 ``RECOVERED``、``RESTORED`` 问）。"""
        ...

    def abandon(self, item: RetryItem) -> None:
        """上一晚中途被杀剩下的一项：不再试，撤回它写到一半的东西（回到重试之前的样子）。"""
        ...


@dataclass(frozen=True)
class RetryBranch:
    """一支：它的重试堆、每种项的处理器、跑一次归约（结局见 ``Reduction``；抛了是 bug）。"""

    name: str
    backlog: RetryBacklogStore
    handlers: Mapping[RetryKind, RetryHandler]
    reduce: Callable[[], Awaitable[Reduction]]


@dataclass(frozen=True)
class RetryReport:
    """这一晚的重试：取出几项、各类几项、上一晚剩下直接丢的几项、锁被占还在等的支。"""

    taken: int = 0
    recovered: int = 0
    unavailable: int = 0
    invalid: int = 0
    deterministic: int = 0
    leftover: int = 0
    #: 内容安全拦下的（不重试，裁定 94 错误分类）。
    refused: int = 0
    #: 锁被另一进程拿着、还没准备好的支（一项的锁或这一支归约的锁）：线据它停在前置处理、下一眼再来。
    waiting: tuple[str, ...] = field(default_factory=tuple)
    #: 某一支的归约碰到模型整个用不了。
    reduction_unreachable: bool = False

    @property
    def dropped(self) -> int:
        return self.unavailable + self.invalid + self.deterministic + self.leftover + self.refused

    @property
    def unreachable(self) -> bool:
        """试到了"模型整个用不了"（一项、或一支的归约）：夜里那条线据它停下这一晚。"""

        return self.unavailable > 0 or self.reduction_unreachable

    @property
    def ready(self) -> bool:
        """这一晚的重试做完了（没有哪一支还在等锁）。"""

        return not self.waiting


@dataclass
class _Tally:
    recovered: int = 0
    unavailable: int = 0
    invalid: int = 0
    deterministic: int = 0
    leftover: int = 0
    taken: int = 0
    refused: int = 0
    #: 试到模型整个用不了：不再试别的项、别的支。
    halted: bool = False
    reduction_unreachable: bool = False

    def report(self, waiting: Sequence[str]) -> RetryReport:
        return RetryReport(
            taken=self.taken,
            recovered=self.recovered,
            unavailable=self.unavailable,
            invalid=self.invalid,
            deterministic=self.deterministic,
            leftover=self.leftover,
            refused=self.refused,
            waiting=tuple(waiting),
            reduction_unreachable=self.reduction_unreachable,
        )


@dataclass(frozen=True)
class RetryCheckpoint:
    """还没准备好时留下的、能落盘的那一份（夜里那条线把它存进前置处理那个 Job，重启或换进程接着跑时交回 ``restore``）：
    哪一晚、各支写回了等归约的项与锁被占的项（只记编号，项本身还在重试堆里）、这一晚已经攒下的计数。"""

    through: date
    #: 支名 → (写回了等归约的项编号, 锁被占的项编号)。
    waiting: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)


_COUNTED = ("recovered", "unavailable", "invalid", "deterministic", "leftover", "taken", "refused")


@dataclass(frozen=True)
class _Waiting:
    """一支还没准备好时留下的：哪一晚的、写回了等归约的那几项（不再试，直接归约）、锁被占的那几项（再试一次）。"""

    through: date
    written: tuple[RetryItem, ...] = ()
    busy: frozenset[str] = frozenset()


class RetryStage:
    def __init__(self, branches: Sequence[RetryBranch]) -> None:
        self.branches = tuple(branches)
        #: 还没准备好的支留下的（只在这个进程里；重启后由 ``clean_leftovers`` 把剩下的撤回）。
        self._waiting: dict[str, _Waiting] = {}
        #: 这一晚没做完时攒下的计数（哪一晚、计数）：下一眼接着这一晚的计数往下数。
        self._carried: tuple[date, _Tally] | None = None

    async def run_once(self, *, through: date) -> RetryReport:
        carried = self._carried
        tally = replace(carried[1]) if carried is not None and carried[0] == through else _Tally()
        self._carried = None
        waiting: list[str] = []
        for branch in self.branches:
            if tally.halted:
                break
            if not await self._branch(branch, through, tally):
                waiting.append(branch.name)
        if waiting and not tally.halted:
            self._carried = (through, replace(tally))
        return tally.report(waiting)

    def checkpoint(self) -> RetryCheckpoint | None:
        """这一晚还没做完时留下的（没有就是 None）：等锁的支与攒下的计数，交给夜里那条线落盘。"""

        if self._carried is None:
            return None
        through, tally = self._carried
        return RetryCheckpoint(
            through=through,
            waiting={
                name: (tuple(item.item_id for item in left.written), tuple(sorted(left.busy)))
                for name, left in sorted(self._waiting.items())
                if left.through == through
            },
            counts={name: getattr(tally, name) for name in _COUNTED},
        )

    def restore(self, checkpoint: RetryCheckpoint | None) -> None:
        """换成落盘的那一份（``None`` 是这一晚没有等锁的）：写回等归约的项按编号从重试堆里取回，堆里已经没有的不再等。"""

        self._waiting.clear()
        self._carried = None
        if checkpoint is None:
            return
        branches = {branch.name: branch for branch in self.branches}
        for name, (written, busy) in checkpoint.waiting.items():
            branch = branches.get(name)
            if branch is None:
                continue
            kept = {item.item_id: item for item in branch.backlog.snapshot(through=checkpoint.through).items}
            items = tuple(kept[identity] for identity in written if identity in kept)
            self._waiting[name] = _Waiting(checkpoint.through, items, frozenset(busy))
        tally = _Tally()
        for name in _COUNTED:
            setattr(tally, name, int(checkpoint.counts.get(name, 0)))
        self._carried = (checkpoint.through, tally)

    def clean_leftovers(self) -> int:
        """进程重启时（常驻节拍一起来就调）：上一晚中途被杀剩下的项——属于已经跑过的日子——撤回、删掉，不等到下一晚；
        否则它写到一半的凭据会把这一整天的封口视界钉住。返回清了几项。"""

        cleaned = 0
        self._waiting.clear()
        self._carried = None
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

    async def _branch(self, branch: RetryBranch, through: date, tally: _Tally) -> bool:
        """这一支试一遍、归约；返回这一支做完了没有（锁被占是没做完，``_waiting`` 记下从哪里接着跑）。"""

        left = self._waiting.pop(branch.name, None)
        # 上一眼写回、等归约的那几项：不再试（会话轮重试会重新描述、再问一次模型），直接归约
        written: list[RetryItem] = [] if left is None else list(left.written)
        resumed = frozenset() if left is None or left.through != through else left.busy
        known = resumed | {item.item_id for item in written}
        previous = branch.backlog.through()
        branch.backlog.advance(through)
        batch = branch.backlog.snapshot(through=through)
        tally.taken += sum(1 for identity in batch.ids if identity not in known)
        tally.deterministic += branch.backlog.discard(batch.unreadable)
        busy: list[str] = []
        # 会话轮排前面：第一条就探出模型能不能用
        for item in sorted(batch.items, key=lambda item: item.kind is not RetryKind.TURN):
            if item.item_id in known and item.item_id not in resumed:
                continue  # 写回了、等归约的：上面已经拿着了
            handler = branch.handlers.get(item.kind)
            if previous is not None and item.day <= previous and item.item_id not in resumed:
                tally.leftover += 1
                if handler is not None:
                    handler.abandon(item)
            elif item.cause is RetryCause.REFUSED:
                tally.refused += 1
            elif handler is None or not item.cause.retried:
                tally.deterministic += 1
            else:
                outcome = await attempt(handler, item)
                if outcome is Attempt.UNAVAILABLE:
                    # 模型整个用不了：这一项不删、别的不再试（剩下的下一晚按上一晚剩下的处理），线据此停下这一晚
                    tally.unavailable += 1
                    tally.halted = True
                    break
                if outcome in (Attempt.RECOVERED, Attempt.RESTORED):
                    written.append(item)
                    continue  # 归约之后问过成没成再删
                if outcome is Attempt.BUSY:
                    busy.append(item.item_id)
                    continue  # 锁被另一进程拿着：还没准备好，不删，下一眼再试
                if outcome is Attempt.INVALID:
                    tally.invalid += 1
                else:
                    tally.deterministic += 1
            branch.backlog.discard((item.item_id,))
        if written:
            # 先记下等归约的那几项：归约抛了（bug，线停下这一晚）它们还在堆里，下一次先归约它们，不当上一晚剩下的撤回
            self._waiting[branch.name] = _Waiting(through, tuple(written), frozenset(busy))
            reduction = await branch.reduce()
            if reduction is Reduction.BUSY:
                return False
            del self._waiting[branch.name]
            if reduction is Reduction.UNREACHABLE:
                tally.halted = tally.reduction_unreachable = True
            for item in written:
                if branch.handlers[item.kind].recovered(item):
                    tally.recovered += 1
                elif reduction is Reduction.UNREACHABLE:
                    tally.unavailable += 1
                else:
                    tally.deterministic += 1
                branch.backlog.discard((item.item_id,))
        if busy and not tally.halted:
            self._waiting[branch.name] = _Waiting(through, (), frozenset(busy))
            return False
        return True


async def attempt(handler: RetryHandler, item: RetryItem) -> Attempt:
    """试一项；处理器抛了是 bug：只拦这一项，记成我们自己处理不了的（裁定 94 第 8 条）。"""

    try:
        return await handler.attempt(item)
    except Exception:  # noqa: BLE001 - 一项的 bug 不中止这一晚剩下的重试；按确定性失败计数报出
        return Attempt.DETERMINISTIC


__all__ = [
    "Attempt",
    "Reduction",
    "RetryBranch",
    "RetryCheckpoint",
    "RetryHandler",
    "RetryReport",
    "RetryStage",
    "attempt",
]
