"""模型队列的口径与值对象（``runtime.model_queue``）：各类活与先后、一项的结局、运维数、一趟的结果、给换代与健康面看的状态、
一轮里各类之间共用的那一份。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from habitus.foundation.observability import ObservationStatus
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import Rhythm
from habitus.series import EventSeries

RETRY, VOCABULARY, UNSEALED, SYNC = "retry", "vocabulary", "unsealed", "sync"
CONCEPTS, MAPPING, ADVICE = "concepts", "mapping", "advice"
#: 各类活，按一轮里的先后。
CATEGORIES = (RETRY, VOCABULARY, UNSEALED, CONCEPTS, ADVICE, SYNC, MAPPING)

#: 一项的结局：做成了；做完了但判据还说欠着（模型没答成）；抛了；还没准备好（锁被另一进程拿着）；模型整个调不通。
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
    backoff_base_seconds: float = 60.0
    backoff_max_seconds: float = 3_600.0
    #: 欠账到这么多项、或最老一项等了这么久，健康面报降级。
    warning_items: int = 50
    warning_wait_seconds: float = 21_600.0
    #: 连着这么多项（不同的项）都是模型失败、中间没有一项做成，就整体退避。
    unreachable_after_failures: int = 3
    kept_days: int = 366
    #: 映射按槽：一条记录最后所见所在的槽（``slot_minutes`` 一槽）过后再留几槽就可以映射，不等那一天封口。
    mapping_margin_slots: int = 2
    slot_minutes: int = 15


@dataclass(frozen=True)
class VocabularyBranch:
    """一支的词表整理活：这一晚还欠不欠（纯读盘）、做这一晚的（持 sweep 锁，锁被占抛 ``BehaviorReductionBusyError``）。"""

    name: str
    due: Callable[[date], bool]
    run: Callable[[date], Awaitable[object]]
    #: 这一支有没有待生效的拆改（只给读数；没接是 None）。
    revision: Callable[[], bool] | None = None


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
    """``ModelQueue.drain`` 这一趟：做了哪些项、各类还欠着几项、整体退避到什么时候、队列租约是不是被另一个进程拿着。"""

    done: tuple[ItemResult, ...]
    owed: Mapping[str, int]
    backing_off_until: datetime | None = None
    held_elsewhere: bool = False

    @property
    def clear(self) -> bool:
        """没有欠账了。"""

        return not self.held_elsewhere and self.backing_off_until is None and not any(self.owed.values())


class ModelQueueState:
    """模型队列给别人看的那一面（换代与健康面读；只在进程里，重起后从盘上重新认）。"""

    def __init__(self) -> None:
        #: 各类最近一次从盘上认出来还欠着几项。
        self.owed: dict[str, int] = {}
        #: 各类最老的一项从什么时候起欠着（盘上看得出的按盘上的，看不出的按这个进程第一次看到它欠着）。
        self.since: dict[str, datetime] = {}
        #: 各类欠着的时候换代写成了几代（清完归零）。
        self.crossed: dict[str, int] = {}
        #: 认欠账时读盘出错的类 → 原因码（异常类型名）；那一类下一次认成了就清掉。
        self.errors: dict[str, str] = {}
        #: 已封口的日子的活这一轮因为什么没法看（词表迁移没做完、读序列时词表变了）；能看是 None。
        self.blocked: str | None = None
        #: 模型整个调不通：从什么时候起、连着几次没通、下一次最早什么时候再做。
        self.unreachable_since: datetime | None = None
        self.failures = 0
        self.retry_at: datetime | None = None
        #: 累计整体退避了几次、共多少秒。
        self.backoffs = 0
        self.backoff_seconds = 0.0
        self.scanned_at: datetime | None = None
        #: 模型失败后在退避里的项（只在进程里，重起后从头）：名字 → （连着失败了几次, 最早什么时候再做）。这些项仍算欠着。
        self.waits: dict[str, tuple[int, datetime]] = {}
        #: 连着模型失败、中间没有任何一项做成的那几项（不同的项）：到了配置的项数就整体退避；有一项做成就清空。
        self.streak: set[str] = set()
        #: 待生效的东西与先验要问的题各有几份（健康面与换代报告读）：``concepts`` 待生效的概念、``revisions`` 待生效的拆改、
        #: ``priors_by_consequent`` / ``priors_by_antecedent`` 两种问法各要问几题。
        self.pending: dict[str, int] = {}
        #: 同一进程的换代正在做（``set_generating``）：队列在两项之间看到就先不取下一项，换代做完再继续。
        self.generating = False

    def set_generating(self, generating: bool) -> None:
        """换代开始 / 做完时立、撤这个标志（``runtime.generation.Generation.run``）。"""

        self.generating = generating

    def waiting(self, name: str, now: datetime) -> bool:
        """这一项模型失败之后还在退避期内：这一轮不重问它。"""

        wait = self.waits.get(name)
        return wait is not None and now < wait[1]

    def waiting_names(self, now: datetime) -> frozenset[str]:
        return frozenset(name for name in self.waits if self.waiting(name, now))

    def down(self) -> bool:
        """正在整体退避（模型整个调不通，还没有一项重新做成）。"""

        return self.unreachable_since is not None

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
    #: 重试账的键（带这一晚：下一晚重新给机会）；自己管重试的（出错重试、映射）是 None。
    key: str | None
    run: Callable[[dict[str, object], list[str]], Awaitable[str]]
    day: date | None = None
    #: 这一项模型失败时，卡住的是它里面的哪几样（出错重试：卡在"模型用不了"上的那几项的名字）；不给就是这一项本身。
    stuck: Callable[[], Sequence[str]] | None = None

    @property
    def name(self) -> str:
        """这一项在进程里的名字（模型失败后的退避按它记）：重试账的键，没有键的是类别加日子。"""

        return (
            self.key if self.key is not None else f"{self.category}:{'' if self.day is None else self.day.isoformat()}"
        )


@dataclass
class Round:
    """一轮里各类之间共用的：今天、已封口的活可以做的那一天、这一轮的时刻、读好的序列与据盘上现算的材料。"""

    now: datetime
    today: date
    night: date | None
    series: EventSeries | None = None
    #: 含今天的序列（映射读它）。
    live: EventSeries | None = None
    #: 今天这一代做完了没有（这一轮第一次问到时读一次完成标记）。
    generated: bool | None = None
    _concepts: ConceptSet | None = None
    _rhythms: Mapping[str, Rhythm] | None = None

    def forget(self) -> None:
        """概念集变了（同步词表、写概念之后）：下一项重新从盘上取。"""

        self._concepts = self._rhythms = None


__all__ = [
    "ADVICE",
    "CATEGORIES",
    "CONCEPTS",
    "DONE",
    "FAILED",
    "ITEM_SPAN",
    "MAPPING",
    "RETRY",
    "STATUS",
    "SYNC",
    "UNFINISHED",
    "UNREACHABLE",
    "UNSEALED",
    "VOCABULARY",
    "WAITING",
    "DrainReport",
    "Item",
    "ItemResult",
    "ItemUnfinished",
    "ModelQueueState",
    "QueueSettings",
    "Round",
    "VocabularyBranch",
]
