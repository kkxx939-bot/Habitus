"""模型队列的口径与值对象（``runtime.model_queue``）：各类活与先后、一项的结局、运维数、一趟的结果、给换代与健康面看的状态、
一轮里各类之间共用的那一份。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime

from habitus.foundation.observability import ObservationStatus
from habitus.model_client.breaker import CircuitBreaker
from habitus.series import EventSeries

RETRY, UNSEALED, SYNC, ADVICE = "retry", "unsealed", "sync", "advice"
#: 各类活，按一轮里的先后。
CATEGORIES = (RETRY, UNSEALED, ADVICE, SYNC)

#: 一项的结局：做成了；做完了但判据还说欠着（模型没答成）；抛了；还没准备好（锁被另一进程拿着）；模型此刻用不了。
DONE, UNFINISHED, FAILED, WAITING, UNREACHABLE = "done", "unfinished", "failed", "waiting", "unreachable"
ITEM_SPAN = "model_queue.item"

STATUS = {
    DONE: ObservationStatus.SUCCESS,
    WAITING: ObservationStatus.SUCCESS,
    UNFINISHED: ObservationStatus.DEGRADED,
    FAILED: ObservationStatus.FAILURE,
    UNREACHABLE: ObservationStatus.DEGRADED,
}


class ItemUnfinished(RuntimeError):
    """一项做完了，但从盘上看它还欠着（模型这一次没答成）：按重试账退避后再做。"""


@dataclass(frozen=True)
class QueueSettings:
    """模型队列的运维数（``config.scene.model_queue_*``，组合根换算过来）。"""

    poll_seconds: float = 60.0
    #: 欠账到这么多项、或最老一项等了这么久，健康面报降级。
    warning_items: int = 50
    warning_wait_seconds: float = 21_600.0
    kept_days: int = 366


@dataclass(frozen=True)
class ItemResult:
    """做了一项：哪一类、结局、原因、用时、调了几次模型、这一项自己的读数（只有数与固定枚举）。"""

    category: str
    outcome: str
    reason: str | None
    started_at: datetime
    duration_seconds: float
    model_calls: int = 0
    detail: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DrainReport:
    """``ModelQueue.drain`` 这一趟：做了哪些项、各类还欠着几项、对话熔断开着时到什么时候试探、队列租约是不是被另一个进程拿着。"""

    done: tuple[ItemResult, ...]
    owed: Mapping[str, int]
    backing_off_until: datetime | None = None
    held_elsewhere: bool = False

    @property
    def clear(self) -> bool:
        """没有欠账了。"""

        return not self.held_elsewhere and self.backing_off_until is None and not any(self.owed.values())


class ModelQueueState:
    """模型队列给别人看的那一面（换代与健康面读；只在进程里，重起后从盘上重新认）。"模型是不是整个用不了"不在这里记：读对话
    这一种能力的熔断（``breaker``，裁定 108；原来队列自己的整体退避并进了熔断）。"""

    def __init__(self, breaker: CircuitBreaker | None = None) -> None:
        #: 对话熔断（模型客户端的那一个，会话、选词、判官、记忆共用）；没接（离线回放）就当模型一直可用。
        self.breaker = breaker
        #: 各类最近一次从盘上认出来还欠着几项。
        self.owed: dict[str, int] = {}
        #: 各类最老的一项从什么时候起欠着（盘上看得出的按盘上的，看不出的按这个进程第一次看到它欠着）。
        self.since: dict[str, datetime] = {}
        #: 各类欠着的时候换代写成了几代（清完归零）。
        self.crossed: dict[str, int] = {}
        #: 认欠账时读盘出错的类 → 原因码（异常类型名）；那一类下一次认成了就清掉。
        self.errors: dict[str, str] = {}
        self.scanned_at: datetime | None = None
        #: 先验要问的题各有几份（健康面与换代报告读）：``priors_by_consequent`` / ``priors_by_antecedent`` 两种问法各要问几题。
        self.pending: dict[str, int] = {}
        #: 同一进程的换代正在做（``set_generating``）：队列在两项之间看到就先不取下一项，换代做完再继续。
        self.generating = False

    def set_generating(self, generating: bool) -> None:
        """换代开始 / 做完时立、撤这个标志（``runtime.generation.Generation.run``）。"""

        self.generating = generating

    def down(self) -> bool:
        """对话熔断开着（模型此刻用不了，还没有试探成功）：换代据它这一代不判关系、沿用上一代（裁定 96、108）。"""

        return self.breaker is not None and self.breaker.is_open()

    def retry_in_seconds(self) -> float | None:
        """熔断开着时离下一次试探还有多久；关着是 None。"""

        if self.breaker is None:
            return None
        return self.breaker.snapshot().retry_in_seconds

    def owed_now(self) -> dict[str, int]:
        """各类还欠着几项（每一类都给，没欠是 0）。"""

        return {category: self.owed.get(category, 0) for category in CATEGORIES}

    def oldest_wait_seconds(self, now: datetime) -> float:
        waits = [(now - since).total_seconds() for category, since in self.since.items() if self.owed.get(category)]
        return max(0.0, max(waits, default=0.0))

    def record(self, category: str, count: int, now: datetime, oldest: datetime | None = None) -> None:
        self.owed[category] = count
        if not count:
            self.since.pop(category, None)
            self.crossed.pop(category, None)
            return
        first = self.since.get(category, now)
        self.since[category] = first if oldest is None else min(first, oldest)

    def settled(self, category: str) -> None:
        """这一类做成了一项：欠着的少一项（下一轮从盘上重新认）。"""

        self.owed[category] = max(0, self.owed.get(category, 0) - 1)


@dataclass
class Item:
    category: str
    #: 重试账的键（带这一晚：下一晚重新给机会）；自己管重试的（出错重试）是 None。
    key: str | None
    run: Callable[[dict[str, object], list[str]], Awaitable[str]]
    day: date | None = None


@dataclass
class Round:
    """一轮里各类之间共用的：今天、已封口的活可以做的那一天、这一轮的时刻、读好的序列。"""

    now: datetime
    today: date
    night: date | None
    series: EventSeries | None = None
    #: 今天这一代做完了没有（这一轮第一次问到时读一次完成标记）。
    generated: bool | None = None


__all__ = [
    "ADVICE",
    "CATEGORIES",
    "DONE",
    "FAILED",
    "ITEM_SPAN",
    "RETRY",
    "STATUS",
    "SYNC",
    "UNFINISHED",
    "UNREACHABLE",
    "UNSEALED",
    "WAITING",
    "DrainReport",
    "Item",
    "ItemResult",
    "ItemUnfinished",
    "ModelQueueState",
    "QueueSettings",
    "Round",
]
