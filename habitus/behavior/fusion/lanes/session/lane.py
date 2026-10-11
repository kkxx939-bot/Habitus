"""会话 lane：一轮一轮地描述并写成判断。

规则（方案第五节）：
- 一轮出一条记录，不判续、不合并；
- 助手在这一轮里既没有答复也没有任何调用的（被打断、出错），不出记录——重发同一句话的情况由这一条覆盖；
- 同一轮重做时不重复写：写完了的跳过；上一次写到一半的（判断落了、回执没落）接着写完，不重问模型。

存储都是同步的文件读写，放到线程里做，不占着事件循环。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from habitus.behavior.fusion.lanes.session.model import SessionTurn
from habitus.behavior.fusion.lanes.session.recorder import SessionTurnRecorder
from habitus.behavior.fusion.lanes.session.service import SessionTurnDescriber


class TurnDisposition(str, Enum):
    """一轮的去向。"""

    RECORDED = "recorded"  # 这次写了判断
    ALREADY_RECORDED = "already_recorded"  # 之前写过（重做时），包括上一次写到一半、这次接着写完的
    UNANSWERED = "unanswered"  # 助手没答复也没调用，不出记录


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

    def __init__(self, describer: SessionTurnDescriber, recorder: SessionTurnRecorder) -> None:
        if not isinstance(describer, SessionTurnDescriber):
            raise TypeError("describer must be SessionTurnDescriber")
        if not isinstance(recorder, SessionTurnRecorder):
            raise TypeError("recorder must be SessionTurnRecorder")
        self.describer = describer
        self.recorder = recorder

    async def consume(
        self, turns: Sequence[SessionTurn], *, before_each: Callable[[], None] | None = None
    ) -> tuple[TurnOutcome, ...]:
        """``before_each`` 在每一轮开始前调一次（调用方用它确认自己还该接着做），它抛错就停在这一轮之前。"""

        # 覆盖记录整批只读一次：每轮都读的话，读的量随已有记录数涨，一份长会话会越做越慢。
        covered = await asyncio.to_thread(self.recorder.covered)
        outcomes: list[TurnOutcome] = []
        for turn in turns:
            if before_each is not None:
                before_each()
            outcomes.append(await self._consume(turn, covered))
        return tuple(outcomes)

    async def _consume(self, turn: SessionTurn, covered: frozenset[str]) -> TurnOutcome:
        if not turn.answered:
            return TurnOutcome(turn, TurnDisposition.UNANSWERED)
        if self.recorder.is_recorded(turn, covered):
            return TurnOutcome(turn, TurnDisposition.ALREADY_RECORDED)
        interrupted = await asyncio.to_thread(self.recorder.interrupted, turn)
        if interrupted is not None:
            judgement_id = await asyncio.to_thread(self.recorder.resume, turn, interrupted)
            return TurnOutcome(turn, TurnDisposition.ALREADY_RECORDED, judgement_id)
        described = await self.describer.describe(turn)
        judgement_id = await asyncio.to_thread(
            self.recorder.record, turn, described.description, validation_attempts=described.validation_attempts
        )
        return TurnOutcome(turn, TurnDisposition.RECORDED, judgement_id, described.validation_attempts)


__all__ = ["SessionLane", "TurnDisposition", "TurnOutcome"]
