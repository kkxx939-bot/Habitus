"""会话 lane：一轮一轮地描述并写成判断。

规则（方案第五节）：
- 一轮出一条记录，不判续、不合并；
- 助手在这一轮里既没有答复也没有任何调用的（被打断、出错），不出记录——重发同一句话的情况由这一条覆盖；
- 同一轮重做时不重复写：写完了的跳过；上一次写到一半的（判断落了、回执没落）接着写完，不重问模型。
- 预测层接上时（``reminders``，方案 30 第五节）：每轮把他开口那一刻还开着的提醒带给模型，判出的回应在这一轮记录落定之后
  写回去。一轮只判一次、重放不重判：已经记录过的轮不再问模型，也就不再写回应；记录落定之后、回应写回之前断了，这一轮的回应
  就丢了（结算按"没回应"处理），不为它重问模型。

存储都是同步的文件读写，放到线程里做，不占着事件循环。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from habitus.behavior.fusion.lanes.session.model import SessionTurn
from habitus.behavior.fusion.lanes.session.recorder import SessionTurnRecorder
from habitus.behavior.fusion.lanes.session.reminders import NoReminders, ReminderFeedback, ReminderResponse
from habitus.behavior.fusion.lanes.session.service import SessionTurnDescriber
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
    ) -> None:
        if not isinstance(describer, SessionTurnDescriber):
            raise TypeError("describer must be SessionTurnDescriber")
        if not isinstance(recorder, SessionTurnRecorder):
            raise TypeError("recorder must be SessionTurnRecorder")
        if reminders is not None and not isinstance(reminders, ReminderFeedback):
            raise TypeError("reminders must implement ReminderFeedback")
        self.describer = describer
        self.recorder = recorder
        #: 开着的提醒从哪来、回应写到哪去（组合根在预测层装好之后接上；没接就是空实现）。
        self.reminders: ReminderFeedback = reminders if reminders is not None else NoReminders()

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
                unreadable = await asyncio.to_thread(
                    self.recorder.record_unreadable, turn, reason=given_up, sources=sources
                )
                return TurnOutcome(turn, TurnDisposition.UNREADABLE, unreadable)
            pending = await asyncio.to_thread(self.reminders.open, turn.instructed_at)
            described = await self.describer.describe(turn, pending)
            judgement_id = await asyncio.to_thread(
                self.recorder.record,
                turn,
                described.description,
                validation_attempts=described.validation_attempts,
                sources=sources,
            )
        except DeterministicFailure as exc:
            unreadable = await asyncio.to_thread(
                self.recorder.record_unreadable, turn, reason=str(exc) or type(exc).__name__, sources=sources
            )
            return TurnOutcome(turn, TurnDisposition.UNREADABLE, unreadable)
        for number, polarity in described.description.responses:
            response = ReminderResponse(pending[number - 1].reminder_id, polarity)
            await asyncio.to_thread(self.reminders.respond, turn.key, response, at=turn.instructed_at)
        return TurnOutcome(turn, TurnDisposition.RECORDED, judgement_id, described.validation_attempts)


__all__ = ["SessionLane", "TurnDisposition", "TurnOutcome"]
