"""重试堆的两段编排（裁定 67、68；裁定 94 第 3、4、8、9 条；实时化阶段 2 起出错重试就近）。怎么试由每种项的处理器管
（``RetryHandler``：住在它试的东西那边，会话轮的在会话 lane、链的在归约）。

**就近试**（``run_once``，模型队列调）：试属于还没封口的日子（晚于"重试到哪一天"、不晚于 ``through``）的可重试项，不等到
那天封口。

- **每项试完立刻删**：成了、败了都删；中途被杀，试过的不会被第二次试到。写回了判断、放回了链的那几项例外：等这一支的归约
  落了树、问过成没成才删（归约没跑成时它们还在；进程重启后再试一次，会话 lane 接着上一次写到一半的写完、不重问模型）。
- **能重试的才试**（``RetryCause.retried``）：内容安全拦下的、我们自己处理不了的不白试，留在堆里，封口时按类计数进健康面
  （树上照旧是「错误」）。
- **一项抛了是 bug，只拦这一项**：记成我们自己处理不了的（树上照旧是「错误」、计数报出），后面的项照试、这一支照常归约。
- **模型此刻用不了的那一次不算试过**（报告的 ``stuck_items`` 记下卡在它上面的项，模型队列下一遍可以 ``skip`` 掉它们、
  先试别的）：会话轮排在前面，试到"模型用不了"就不再试别的（不逐条等超时），这一项留着；这一支
  已写回的照常归约后收工；这一支的归约里碰到模型整个用不了（``Reduction.UNREACHABLE``）也一样，没补回的项留着。报告的
  ``unreachable`` 为真，模型队列据它整体退避。
- **锁被另一进程拿着**不是错误、更不是模型调不通，是**还没准备好**（裁定 94 第 9 条）：一项的锁被占（``Attempt.BUSY``）不删、
  这一支归约的锁被占（``Reduction.BUSY``）写回的那几项也不删，报告记下这一支在等（``waiting``）；下一次等归约的那几项不再试、
  直接归约。
- **不与时窗末当场重读重复试**：试之前先在堆上占这一项（``RetryBacklogStore.claim``），写回了在等归约的一直占着。
- **试的时候那一天封口了**：这一次写下的撤回（裁定 68 第 3 条：写回的判断不落进已经做过重试收尾的那一天）。
- **归约抛了是 bug**：原样抛给调用方；写回的那几项留着，下一次先归约它们。
- 试出的结果（补回 / 内容不合格 / 确定性失败）按项所属的那一天攒着（只在进程里），那一天封口时并进重试收尾的报告。

**重试收尾**（``close``，换代里结算之前调）：不试，只做推日子、丢弃、撤回。先把"重试到哪一天"推到 ``through``（之后属于
这些天的项不再收），堆里属于这些天的项全部丢掉——能重试而到封口还没补回的记一项 ``unavailable``（原来就是格式不合的记
``invalid``），写到一半的撤回；内容安全拦下的、我们自己处理不了的按类计数；比上一次推到的那天还早的才是 ``leftover``。
就近试的时候中途被杀剩下的项不记 ``leftover``：就近试不先推日子，重起后它们与没试过的项在盘上没有区别、照常再试（会话 lane
接着写到一半的写完），不是"那一天已经跑过、直接丢"；到封口仍没补回才丢，与别的没补回的同一个名目。``leftover`` 现在只剩
重试收尾自己推了日子之后、丢到一半被杀的那几项。不调模型、不跑归约；这些天有写回的
判断还在等归约时（归约正做到一半、或它的锁被占）什么都不动、报告记下在等的支（``waiting``），换代把这一代的结算留到
下次——所以写回的判断要么在封口之前落了树，要么被撤回，要么那一天还没结算，不会落进已经结算的日子。

计数：补回 / 模型用不了 / 内容不合格 / 确定性失败 / 内容安全（裁定 67 第 12 条），外加上一次剩下、直接丢的。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
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
    BUSY = "busy"  # sweep 锁被另一进程拿着：还没准备好，下一次再来
    UNREACHABLE = "unreachable"  # 归类时模型整个用不了：没补回的项留着，模型队列整体退避


@runtime_checkable
class RetryHandler(Protocol):
    async def attempt(self, item: RetryItem) -> Attempt: ...

    def recovered(self, item: RetryItem) -> bool:
        """写回 / 放回的那一项，归约之后成了没有（对 ``RECOVERED``、``RESTORED`` 问）。"""
        ...

    def abandon(self, item: RetryItem) -> None:
        """不再试的一项（那一天封口了、或上一次中途被杀剩下的）：撤回它写到一半的东西（回到重试之前的样子）。"""
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
    """这一次的重试：取出几项、各类几项、上一次剩下直接丢的几项、锁被占还在等的支。"""

    taken: int = 0
    recovered: int = 0
    unavailable: int = 0
    invalid: int = 0
    deterministic: int = 0
    leftover: int = 0
    #: 内容安全拦下的（不重试，裁定 94 错误分类）。
    refused: int = 0
    #: 锁被另一进程拿着、还没准备好的支（一项的锁或这一支归约的锁）：下一次再来。
    waiting: tuple[str, ...] = field(default_factory=tuple)
    #: 某一支的归约碰到模型整个用不了。
    reduction_unreachable: bool = False
    #: 就近试因"模型用不了"停下来时，卡在它上面的那几项（试到的那一项，或归约没落成的那几项）；没停下来是空。模型队列
    #: 据它只让这几项等着（下一遍 ``skip`` 掉），别的照试。
    stuck_items: tuple[str, ...] = field(default_factory=tuple, compare=False)

    @property
    def dropped(self) -> int:
        return self.unavailable + self.invalid + self.deterministic + self.leftover + self.refused

    @property
    def unreachable(self) -> bool:
        """就近试时试到了"模型整个用不了"（一项、或一支的归约）：模型队列据它整体退避。重试收尾的报告里 ``unavailable``
        是到封口还没补回的项数，不看这个属性。"""

        return self.unavailable > 0 or self.reduction_unreachable

    @property
    def ready(self) -> bool:
        """这一次的重试做完了（没有哪一支还在等锁）。"""

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
    stuck_items: tuple[str, ...] = ()

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
            stuck_items=self.stuck_items,
        )


class RetryStage:
    def __init__(self, branches: Sequence[RetryBranch]) -> None:
        self.branches = tuple(branches)
        #: 各支写回了、还在等归约的那几项（只在这个进程里）：下一次不再试，直接归约。
        self._waiting: dict[str, tuple[RetryItem, ...]] = {}
        #: 就近试出的结果按项所属的那一天攒着（只在这个进程里），那一天封口时并进重试收尾的报告。
        self._tried: dict[date, _Tally] = {}
        #: 这一遍不试的项（``run_once`` 的 ``skip``）。
        self._skip: frozenset[str] = frozenset()

    def due(self, *, through: date) -> tuple[RetryItem, ...]:
        """就近要试的项：各支堆里属于还没封口的日子（不晚于 ``through``）、能重试、有处理器的（写回了在等归约的也在里面）。"""

        found: list[RetryItem] = []
        for branch in self.branches:
            closed = branch.backlog.through()
            found.extend(
                item
                for item in branch.backlog.snapshot(through=through).items
                if (closed is None or item.day > closed) and item.cause.retried and item.kind in branch.handlers
            )
        return tuple(found)

    async def run_once(self, *, through: date, skip: Collection[str] = ()) -> RetryReport:
        """就近试一遍（见模块文档串）。``skip`` 里的项这一遍不试（模型队列按重试账退避着的：上一次卡在传输类上的那几项）。"""

        self._skip = frozenset(skip)
        tally = _Tally()
        waiting: list[str] = []
        for branch in self.branches:
            if tally.halted:
                break
            if not await self._branch(branch, through, tally):
                waiting.append(branch.name)
        return tally.report(waiting)

    def close(self, *, through: date) -> RetryReport:
        """重试收尾（见模块文档串）：不试，推日子、丢弃、撤回；返回这些天的计数（含就近试出的）。这些天有写回的判断还在等
        归约时什么都不动，返回的报告 ``ready`` 为假。"""

        reducing = [
            branch.name
            for branch in self.branches
            if any(item.day <= through for item in self._waiting.get(branch.name, ()))
        ]
        if reducing:
            # 这几天有写回的判断还在等归约（归约正做到一半、或它的锁被占）：现在丢弃、结算，那条判断可能落进已经结算的
            # 日子（裁定 68 第 3 条）。什么都不动，报告记下在等的支（换代据此把这一代的结算留到下次）
            return _Tally().report(reducing)
        tally = _Tally()
        for day in sorted(day for day in self._tried if day <= through):
            tried = self._tried.pop(day)
            tally.taken += tried.taken
            tally.recovered += tried.recovered
            tally.invalid += tried.invalid
            tally.deterministic += tried.deterministic
        for branch in self.branches:
            previous = branch.backlog.through()
            branch.backlog.advance(through)
            batch = branch.backlog.snapshot(through=through)
            tally.taken += len(batch.ids)
            tally.deterministic += branch.backlog.discard(batch.unreadable)
            for item in batch.items:
                handler = branch.handlers.get(item.kind)
                attempted = handler is not None and item.cause.retried
                if previous is not None and item.day <= previous:
                    tally.leftover += 1
                    attempted = handler is not None
                elif item.cause is RetryCause.REFUSED:
                    tally.refused += 1
                elif not attempted:
                    tally.deterministic += 1
                elif item.cause is RetryCause.INVALID:
                    tally.invalid += 1
                else:
                    tally.unavailable += 1  # 到封口还没补回：模型一直用不了、没轮到、或试到一半被杀之后没再补成
                if attempted and handler is not None:
                    handler.abandon(item)  # 就近试过、写到一半的撤回
                branch.backlog.discard((item.item_id,))
                branch.backlog.release(item.item_id)
        return tally.report(())

    def clean_leftovers(self) -> int:
        """进程重启时（进程起来后换代的第一眼调）：上一次中途被杀剩下的项——属于已经封口的日子——撤回、删掉，不等到
        下一代；否则它写到一半的凭据会把这一整天的封口视界钉住。返回清了几项。"""

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
                branch.backlog.release(item.item_id)
            cleaned += branch.backlog.discard(batch.ids)
            self._forget_waiting(branch.name, through)
        return cleaned

    def _forget_waiting(self, branch: str, through: date) -> None:
        """这一支等归约的那几项里，属于已经封口的日子的不再等（已丢弃、撤回）。"""

        kept = tuple(item for item in self._waiting.pop(branch, ()) if item.day > through)
        if kept:
            self._waiting[branch] = kept

    def _settled(self, item: RetryItem, outcome: str, tally: _Tally) -> None:
        """一项有了定论（补回 / 内容不合格 / 确定性失败）：这一趟与它所属的那一天各记一次。"""

        tried = self._tried.setdefault(item.day, _Tally())
        tried.taken += 1
        for counts in (tally, tried):
            setattr(counts, outcome, getattr(counts, outcome) + 1)

    async def _branch(self, branch: RetryBranch, through: date, tally: _Tally) -> bool:
        """这一支试一遍、归约；返回这一支做完了没有（锁被占是没做完，``_waiting`` 记下等归约的那几项）。"""

        backlog = branch.backlog
        # 上一次写回、等归约的那几项：不再试（会话轮重试会重新描述、再问一次模型），直接归约
        written: list[RetryItem] = list(self._waiting.pop(branch.name, ()))
        known = {item.item_id for item in written} | self._skip
        closed = backlog.through()
        batch = backlog.snapshot(through=through)
        if batch.unreadable:
            unreadable = backlog.discard(batch.unreadable)
            tried = self._tried.setdefault(through, _Tally())
            for counts in (tally, tried):
                counts.taken += unreadable
                counts.deterministic += unreadable
        busy = False
        # 这一趟要试的（会话轮排前面：第一条就探出模型能不能用）：已封口的日子剩下的留给重试收尾 / ``clean_leftovers``，
        # 不白试的留到封口时按类计数
        due = [
            item
            for item in batch.items
            if item.item_id not in known
            and (closed is None or item.day > closed)
            and item.cause.retried
            and item.kind in branch.handlers
        ]
        tally.taken += len(due)
        for item in sorted(due, key=lambda item: item.kind is not RetryKind.TURN):
            handler = branch.handlers[item.kind]
            if not backlog.claim(item.item_id):
                continue  # 时窗末正在当场重读它
            try:
                outcome = await attempt(handler, item)
            except BaseException:
                backlog.release(item.item_id)  # 被杀 / 取消：不占着，下一次再试
                raise
            if _closed(backlog, item):
                # 试的时候那一天封口了：这一次写下的撤回，不落进已经做过重试收尾的那一天（裁定 68 第 3 条）
                handler.abandon(item)
                backlog.release(item.item_id)
                continue
            if outcome in (Attempt.RECOVERED, Attempt.RESTORED):
                written.append(item)
                continue  # 归约之后问过成没成再删；在那之前一直占着
            backlog.release(item.item_id)
            if outcome is Attempt.UNAVAILABLE:
                # 模型用不了：这一次不算试过、项留着，别的不再试；记下是哪一项，模型队列只让它等着
                tally.unavailable += 1
                tally.halted, tally.stuck_items = True, (item.item_id,)
                break
            if outcome is Attempt.BUSY:
                busy = True
                continue  # 锁被另一进程拿着：还没准备好，不删，下一次再试
            self._settled(item, "invalid" if outcome is Attempt.INVALID else "deterministic", tally)
            backlog.discard((item.item_id,))
        if written and not await self._reduce(branch, written, tally):
            return False
        return not busy or tally.halted

    async def _reduce(self, branch: RetryBranch, written: Sequence[RetryItem], tally: _Tally) -> bool:
        """这一支写回的那几项交归约落树，逐项问成没成；归约的锁被占为假（那几项留着等下一次）。"""

        backlog = branch.backlog
        # 先记下等归约的那几项：归约抛了（bug）它们还在堆里，下一次先归约它们
        self._waiting[branch.name] = tuple(written)
        reduction = await branch.reduce()
        if reduction is Reduction.BUSY:
            return False
        del self._waiting[branch.name]
        down = reduction is Reduction.UNREACHABLE
        if down:
            tally.halted = tally.reduction_unreachable = True
        stuck: list[str] = []
        for item in written:
            backlog.release(item.item_id)
            if _closed(backlog, item):
                continue  # 归约的时候那一天封口了：重试收尾已经把它丢掉、撤回
            if branch.handlers[item.kind].recovered(item):
                self._settled(item, "recovered", tally)
            elif down:
                tally.unavailable += 1  # 归类时模型用不了：这一次不算试过，项留着
                stuck.append(item.item_id)
                continue
            else:
                self._settled(item, "deterministic", tally)
            backlog.discard((item.item_id,))
        tally.stuck_items = (*tally.stuck_items, *stuck)
        return True


def _closed(backlog: RetryBacklogStore, item: RetryItem) -> bool:
    through = backlog.through()
    return through is not None and item.day <= through


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
    "RetryHandler",
    "RetryReport",
    "RetryStage",
    "attempt",
]
