"""会话 lane：一轮一轮地描述并写成判断。

规则（方案第五节）：
- 一轮出一条记录，不判续、不合并；
- 助手在这一轮里既没有答复也没有任何调用的（被打断、出错），不出记录——重发同一句话的情况由这一条覆盖；
- 同一轮重做时不重复写：写完了的跳过；上一次写到一半的（判断落了、回执没落）接着写完，不重问模型。
- 预测层接上时（``reminders``，方案 30 第五节）：每轮把他开口那一刻还开着的提醒带给模型，判出的回应在这一轮记录落定之后
  写回去。一轮只判一次、重放不重判：已经记录过的轮不再问模型，也就不再写回应；记录落定之后、回应写回之前断了，这一轮的回应
  就丢了（结算按"没回应"处理），不为它重问模型。
- 写成「没读懂」的轮（确定性写不成、或重试账放弃了）把这一轮本身放进每晚重试的堆（``backlog``，裁定 67）：当晚跑批前
  ``retry`` 再描述一次，成了就补一条判断（原「没读懂」不删，读的时候被它作废），没成就算了。

存储都是同步的文件读写，放到线程里做，不占着事件循环。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum

from habitus.behavior.fusion.lanes.session.model import SessionTurn, encode_turn
from habitus.behavior.fusion.lanes.session.recorder import SessionTurnRecorder
from habitus.behavior.fusion.lanes.session.reminders import NoReminders, ReminderFeedback, ReminderResponse
from habitus.behavior.fusion.lanes.session.service import SessionTurnDescriber
from habitus.behavior.retry import Admission, NoRetryBacklog, RetryBacklog, RetryCause, RetryItem, RetryKind, retry_id
from habitus.behavior.tree import BehaviorKind, BehaviorTree
from habitus.foundation.lifecycle import DeterministicFailure


class TurnDisposition(str, Enum):
    """一轮的去向。"""

    RECORDED = "recorded"  # 这次写了判断
    ALREADY_RECORDED = "already_recorded"  # 之前写过（重做时），包括上一次写到一半、这次接着写完的
    UNANSWERED = "unanswered"  # 助手没答复也没调用，不出记录
    UNREADABLE = "unreadable"  # 写不成（确定性的，或重试账放弃了）：写成一段没读懂，不再每次重做（26 号清单 2.4.4）


@dataclass(frozen=True)
class TurnOutcome:
    """一轮的处理结果，供回执与可观测用。"""

    turn: SessionTurn
    disposition: TurnDisposition
    judgement_id: str | None = None
    validation_attempts: int = 0


class SessionLane:
    """把若干轮依次处理掉。某一轮的模型调用失败就整体抛出，由调用方（会话源的重做机制）之后重来；
    已经写过的轮重来时会被认出来、不重复写。"""

    def __init__(
        self,
        describer: SessionTurnDescriber,
        recorder: SessionTurnRecorder,
        reminders: ReminderFeedback | None = None,
        *,
        backlog: RetryBacklog | None = None,
        tree: BehaviorTree | None = None,
    ) -> None:
        if not isinstance(describer, SessionTurnDescriber):
            raise TypeError("describer must be SessionTurnDescriber")
        if not isinstance(recorder, SessionTurnRecorder):
            raise TypeError("recorder must be SessionTurnRecorder")
        if reminders is not None and not isinstance(reminders, ReminderFeedback):
            raise TypeError("reminders must implement ReminderFeedback")
        if backlog is not None and not isinstance(backlog, RetryBacklog):
            raise TypeError("backlog must implement RetryBacklog")
        if tree is not None and not isinstance(tree, BehaviorTree):
            raise TypeError("tree must be BehaviorTree")
        self.describer = describer
        self.recorder = recorder
        #: 开着的提醒从哪来、回应写到哪去（组合根在预测层装好之后接上；没接就是空实现）。
        self.reminders: ReminderFeedback = reminders if reminders is not None else NoReminders()
        #: 写成「没读懂」的轮放到哪里等当晚重试（没接就只落「没读懂」）。
        self.backlog: RetryBacklog = backlog if backlog is not None else NoRetryBacklog()
        #: 这一支的行为树：补救时认"这一轮已经落树"（没给就只认覆盖记录与判断存储）。
        self.tree = tree

    def attach_reminders(self, reminders: ReminderFeedback) -> None:
        """预测层装好之后把开着的提醒接进来（行为侧先装、预测层后装）。"""

        if not isinstance(reminders, ReminderFeedback):
            raise TypeError("reminders must implement ReminderFeedback")
        self.reminders = reminders

    async def consume(
        self,
        turns: Sequence[SessionTurn],
        *,
        before_each: Callable[[], None] | None = None,
        sources: tuple[str, ...] = (),
    ) -> tuple[TurnOutcome, ...]:
        """``before_each`` 在每一轮开始前调一次（调用方用它确认自己还该接着做），它抛错就停在这一轮之前。``sources`` 是这些轮
        所属的会话源（覆盖记录带上它，见 ``SessionTurnRecorder.record``）。"""

        return await self._each(turns, before_each, sources, given_up=None)

    async def abandon(
        self,
        turns: Sequence[SessionTurn],
        *,
        reason: str,
        before_each: Callable[[], None] | None = None,
        sources: tuple[str, ...] = (),
    ) -> tuple[TurnOutcome, ...]:
        """重试账对这份源放弃了（怎么失败的都一样，23 号第一节第 6 条）：还没写进树的轮不再问模型，写成一段没读懂——
        不留空白，下游就会把"看了没读懂"读成"没做"。写完了的、写到一半的照常认回来。"""

        return await self._each(turns, before_each, sources, given_up=reason)

    async def _each(
        self,
        turns: Sequence[SessionTurn],
        before_each: Callable[[], None] | None,
        sources: tuple[str, ...],
        *,
        given_up: str | None,
    ) -> tuple[TurnOutcome, ...]:
        # 覆盖记录整批只读一次：每轮都读的话，读的量随已有记录数涨，一份长会话会越做越慢。
        covered = await asyncio.to_thread(self.recorder.covered)
        outcomes: list[TurnOutcome] = []
        for turn in turns:
            if before_each is not None:
                before_each()
            outcomes.append(await self._consume(turn, covered, sources, given_up))
        return tuple(outcomes)

    async def _consume(
        self, turn: SessionTurn, covered: frozenset[str], sources: tuple[str, ...], given_up: str | None
    ) -> TurnOutcome:
        if not turn.answered:
            return TurnOutcome(turn, TurnDisposition.UNANSWERED)
        if self.recorder.is_recorded(turn, covered):
            return TurnOutcome(turn, TurnDisposition.ALREADY_RECORDED)
        try:
            interrupted = await asyncio.to_thread(self.recorder.interrupted, turn)
            if interrupted is not None:
                judgement_id = await asyncio.to_thread(self.recorder.resume, turn, interrupted, sources=sources)
                resumed = TurnDisposition.UNREADABLE if interrupted.unreadable else TurnDisposition.ALREADY_RECORDED
                return TurnOutcome(turn, resumed, judgement_id)
            if given_up is not None:
                return await self._unreadable(turn, RetryCause.GIVEN_UP, sources)
            return await self._describe(turn, sources)
        except DeterministicFailure:
            return await self._unreadable(turn, RetryCause.DETERMINISTIC, sources)

    async def retry(self, turn: SessionTurn) -> TurnOutcome:
        """重试堆里的一轮（每晚跑批前；时窗末用得着时白天先试，裁定 67、68）：不看写没写过（它已经落了「没读懂」，覆盖记录
        会说"写过了"），再描述一次、补一条判断。上一次重试写到一半就断了的（可读判断落了、还没归约），接着写完、不重问模型。
        失败就撤回这一次写下的（``withdraw``），错误照样抛给调用方（它按"模型用不了 / 内容不合格 / 确定性失败"计数）。"""

        unfinished = await asyncio.to_thread(self.recorder.readable_unfinished, turn)
        if unfinished is not None:
            judgement_id = await asyncio.to_thread(self.recorder.resume, turn, unfinished)
            return TurnOutcome(turn, TurnDisposition.RECORDED, judgement_id)
        try:
            return await self._describe(turn, ())
        except Exception:
            # 被杀 / 取消（BaseException）不撤：留给下一次接着写完；上一晚中途被杀剩下的由夜批编排撤回（``abandon``）
            await asyncio.to_thread(self.withdraw, turn)
            raise

    def withdraw(self, turn: SessionTurn) -> None:
        """撤回这一轮写到一半的重试，回到重试之前的样子（原「没读懂」不动）：重试失败时、上一晚中途被杀剩下的那一项丢弃时。"""

        self.recorder.withdraw(turn)

    async def _describe(self, turn: SessionTurn, sources: tuple[str, ...]) -> TurnOutcome:
        pending = await asyncio.to_thread(self.reminders.open, turn.instructed_at)
        described = await self.describer.describe(turn, pending)
        judgement_id = await asyncio.to_thread(
            self.recorder.record,
            turn,
            described.description,
            validation_attempts=described.validation_attempts,
            sources=sources,
        )
        for number, polarity in described.description.responses:
            response = ReminderResponse(pending[number - 1].reminder_id, polarity)
            await asyncio.to_thread(self.reminders.respond, turn.key, response, at=turn.instructed_at)
        return TurnOutcome(turn, TurnDisposition.RECORDED, judgement_id, described.validation_attempts)

    async def _unreadable(self, turn: SessionTurn, cause: RetryCause, sources: tuple[str, ...]) -> TurnOutcome:
        """先落「没读懂」（下游不等它），再把这一轮放进重试堆；堆满了就只是「没读懂」。"""

        unreadable = await asyncio.to_thread(self.recorder.record_unreadable, turn, sources=sources)
        await asyncio.to_thread(self._keep, turn, cause)
        return TurnOutcome(turn, TurnDisposition.UNREADABLE, unreadable)

    def salvage(self, turns: Sequence[SessionTurn]) -> tuple[TurnOutcome, ...]:
        """会话源"处理完了没有"的账坏了（信封完好，裁定 67 第 5、16 条）：逐轮看写没写过，没写过的落「没读懂」、进重试堆
        （那一天的夜批已经跑过的，堆按天拒收，裁定 68 第 3 条）。不问模型，全是同步的读写。

        "写过"三处任一认得出就算：覆盖记录、判断存储里凭据相同的判断（写了没归约；回执没落的照常接着写完）、树上
        （这一支那天有一条从这一轮开口那一刻开始的行为或空白）——覆盖记录可能早在判坏之前就随 Behavior"到终态"
        删掉了，只看它会把已经落树的轮重读一遍。"""

        covered = self.recorder.covered()
        outcomes: list[TurnOutcome] = []
        for turn in turns:
            if not turn.answered:
                outcomes.append(TurnOutcome(turn, TurnDisposition.UNANSWERED))
                continue
            if self.recorder.is_recorded(turn, covered) or self._landed(turn):
                outcomes.append(TurnOutcome(turn, TurnDisposition.ALREADY_RECORDED))
                continue
            interrupted = self.recorder.interrupted(turn)
            if interrupted is not None:
                judgement_id = self.recorder.resume(turn, interrupted)
                outcomes.append(TurnOutcome(turn, TurnDisposition.ALREADY_RECORDED, judgement_id))
                continue
            unreadable = self.recorder.record_unreadable(turn)
            self._keep(turn, RetryCause.DETERMINISTIC)
            outcomes.append(TurnOutcome(turn, TurnDisposition.UNREADABLE, unreadable))
        return tuple(outcomes)

    def _landed(self, turn: SessionTurn) -> bool:
        """这一支的树上，这一轮开口那一天有没有一条从这一刻开始的行为或空白（会话 lane 一轮一条，开口时刻精确到微秒）。"""

        if self.tree is None:
            return False
        return any(
            address.started_at == turn.instructed_at
            for kind in (BehaviorKind.OCCURRENCE, BehaviorKind.GAP)
            for address in self.tree.list_day_addresses(kind, self._day(turn))
        )

    def _keep(self, turn: SessionTurn, cause: RetryCause) -> Admission:
        """这一轮放进每晚重试的堆；拒收的（满了、那天已过）它已经落了「没读懂」，堆自己记数（裁定 68 第 3、4 条）。"""

        return self.backlog.put(
            RetryItem(
                retry_id(RetryKind.TURN, turn.key),
                RetryKind.TURN,
                cause,
                self._day(turn),
                self.recorder.clock(),
                encode_turn(turn),
            )
        )

    def _day(self, turn: SessionTurn) -> date:
        return turn.instructed_at.astimezone(self.recorder.zone).date()


__all__ = ["SessionLane", "TurnDisposition", "TurnOutcome"]
