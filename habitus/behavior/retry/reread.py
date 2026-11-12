"""时窗末用得着就当场重读（裁定 68 第 1 条）：预测层到时窗末要说时，说中的范围里有「没读懂」，就把那段里还在重试堆里的
会话轮重读一次——走的就是重试那一个处理器。读出来了这一项就从堆里删掉（这一天它已经试过、成了）；模型此刻用不了、
锁被占、答得不合格的留在堆里，等模型队列就近再试、或那一天封口时丢弃。

与模型队列的就近重试不重复试同一项：两边先在堆上占这一项（``RetryBacklogStore.claim``），占不上的那一方跳过——队列正在试的、
队列写回了还在等归约的，这里都不再问一次模型。
"""

from __future__ import annotations

from datetime import datetime

from habitus.behavior.retry.model import RetryKind
from habitus.behavior.retry.stage import Attempt, RetryHandler, attempt
from habitus.behavior.retry.store import RetryBacklogStore


class Rereader:
    def __init__(self, backlog: RetryBacklogStore, handler: RetryHandler) -> None:
        self.backlog = backlog
        self.handler = handler

    async def reread(self, start: datetime, end: datetime) -> bool:
        """``start`` → ``end`` 里开口的会话轮，堆里有的逐条重读；有一条读出来了为真（这一拍不决定，下一拍按读出的再判）。"""

        recovered = False
        for item in self.backlog.snapshot().items:
            if (
                item.kind is not RetryKind.TURN
                or not item.cause.retried
                or not _opened_within(item.payload, start, end)
            ):
                continue  # 内容安全拦下的、我们自己处理不了的不白试（裁定 94 错误分类）
            if not self.backlog.claim(item.item_id):
                continue  # 模型队列正在试它、或写回了在等归约：不再问一次模型
            try:
                outcome = await attempt(self.handler, item)  # 一项抛了是 bug：只拦这一项，留在堆里等封口时报出
            finally:
                self.backlog.release(item.item_id)
            if outcome is Attempt.RECOVERED:
                self.backlog.discard((item.item_id,))
                recovered = True
            elif outcome in (Attempt.UNAVAILABLE, Attempt.BUSY):
                break  # 模型此刻用不了、或锁被另一进程拿着：别的也不必试了，留在堆里
        return recovered


def _opened_within(payload: object, start: datetime, end: datetime) -> bool:
    try:
        opened = datetime.fromisoformat(str(payload["instructed_at"]))  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return False
    return opened.utcoffset() is not None and start <= opened < end


__all__ = ["Rereader"]
