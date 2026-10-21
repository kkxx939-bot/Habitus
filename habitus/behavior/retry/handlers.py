"""两种项怎么试（裁定 67、68）：会话轮交给会话 lane 再描述一次；链的原判断放回判断存储、等归约。

"模型用不了"与"同一份输入再问也一样"的分类只认 ``model_client`` 具名导出的那两组（``MODEL_UNAVAILABLE`` /
``MODEL_DETERMINISTIC``）；时窗末当场重读（裁定 68 第 1 条）用的也是会话轮这一个处理器。
"""

from __future__ import annotations

from habitus.behavior.fusion.errors import BehaviorFusionError
from habitus.behavior.fusion.lanes.session import SessionLane
from habitus.behavior.fusion.lanes.session.model import decode_turn
from habitus.behavior.reduction.failed import FailedChains
from habitus.behavior.retry.model import RetryItem
from habitus.behavior.retry.stage import Attempt
from habitus.foundation.lifecycle import DeterministicFailure
from habitus.model_client import MODEL_DETERMINISTIC, MODEL_UNAVAILABLE

#: 同一份输入再问也一样：这一版处理下必然写不成，或模型对这份输入必然拒绝。
_DETERMINISTIC: tuple[type[BaseException], ...] = (DeterministicFailure, *MODEL_DETERMINISTIC)
#: 模型此刻用不了：整批作罢（会话轮）、留到当晚（时窗末重读）。
_UNAVAILABLE: tuple[type[BaseException], ...] = (*MODEL_UNAVAILABLE, TimeoutError)


class TurnHandler:
    def __init__(self, lane: SessionLane) -> None:
        if not isinstance(lane, SessionLane):
            raise TypeError("lane must be SessionLane")
        self.lane = lane

    async def attempt(self, item: RetryItem) -> Attempt:
        try:
            turn = decode_turn(item.payload)
        except BehaviorFusionError:
            return Attempt.DETERMINISTIC
        try:
            await self.lane.retry(turn)
        except _DETERMINISTIC:
            return Attempt.DETERMINISTIC
        except _UNAVAILABLE:
            return Attempt.UNAVAILABLE
        except (BehaviorFusionError, ValueError):
            return Attempt.INVALID  # 模型答得不合格、写入校验不过
        return Attempt.RECOVERED

    def recovered(self, item: RetryItem) -> bool:
        return True

    def abandon(self, item: RetryItem) -> None:
        try:
            turn = decode_turn(item.payload)
        except BehaviorFusionError:
            return
        self.lane.withdraw(turn)


class ChainHandler:
    def __init__(self, failed: FailedChains) -> None:
        if not isinstance(failed, FailedChains):
            raise TypeError("failed must be FailedChains")
        self.failed = failed

    async def attempt(self, item: RetryItem) -> Attempt:
        return Attempt.RESTORED if self.failed.restore(item) else Attempt.DETERMINISTIC

    def recovered(self, item: RetryItem) -> bool:
        return self.failed.recovered(item)

    def abandon(self, item: RetryItem) -> None:
        self.failed.withdraw(item)


__all__ = ["ChainHandler", "TurnHandler"]
