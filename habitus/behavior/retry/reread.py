"""时窗末用得着就当场重读（裁定 68 第 1 条）：预测层到时窗末要说时，说中的范围里有「没读懂」，就把那段里还在重试堆里的
会话轮重读一次——走的就是夜里重试那一个处理器。读出来了这一项就从堆里删掉（这一天它已经试过、成了）；模型此刻用不了、
答得不合格的留在堆里等当晚。
"""

from __future__ import annotations

from datetime import datetime

from habitus.behavior.retry.model import RetryKind
from habitus.behavior.retry.stage import Attempt, RetryHandler
from habitus.behavior.retry.store import RetryBacklogStore


class Rereader:
    def __init__(self, backlog: RetryBacklogStore, handler: RetryHandler) -> None:
        if not isinstance(backlog, RetryBacklogStore):
            raise TypeError("backlog must be RetryBacklogStore")
        if not isinstance(handler, RetryHandler):
            raise TypeError("handler must implement RetryHandler")
        self.backlog = backlog
        self.handler = handler

    async def reread(self, start: datetime, end: datetime) -> bool:
        """``start`` → ``end`` 里开口的会话轮，堆里有的逐条重读；有一条读出来了为真（这一拍不决定，下一拍按读出的再判）。"""

        recovered = False
        for item in self.backlog.snapshot().items:
            if item.kind is not RetryKind.TURN or not _opened_within(item.payload, start, end):
                continue
            outcome = await self.handler.attempt(item)
            if outcome is Attempt.RECOVERED:
                self.backlog.discard((item.item_id,))
                recovered = True
            elif outcome is Attempt.UNAVAILABLE:
                break  # 模型此刻用不了：别的也不必试了
        return recovered


def _opened_within(payload: object, start: datetime, end: datetime) -> bool:
    try:
        opened = datetime.fromisoformat(str(payload["instructed_at"]))  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return False
    return opened.utcoffset() is not None and start <= opened < end


__all__ = ["Rereader"]
