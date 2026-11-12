"""换代的结果与口径（``runtime.generation``）：各步与它们的先后、哪几步必须成、一步的结局、各个退路的计数名、做一代的结果。

换代里没有模型调用（实时化阶段 2）：词表的每晚新增与定期拆改、语义树里调模型的那几段都归模型队列
（``runtime.model_queue``），步骤表里没有它们；换代只记下"换代时模型队列各类还欠着几项"（``GenerationResult.queue_owed``）
与"换代那一刻模型队列是不是在整体退避"（是就不判关系、沿用上一代）。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

from habitus.model_client import MODEL_UNUSABLE, ModelUnreachable
from habitus.runtime.generation_summary import UNREACHABLE, reason_code

MIGRATE, CALENDAR, RETRY, SETTLE = "migrate", "calendar", "retry", "settle"
SERIES, SCENE, OPPORTUNITIES, DAYS, MARK = "series", "scene", "opportunities", "days", "mark"
#: 换代的各步，按先后（实时化方案第 3 节：续做没做完的词表迁移排在最前）。单元历史与压实在机会表那一步里、跟在表后面
#: （各自一个 ``foresight.*`` 子 span）。
STEPS = (MIGRATE, CALENDAR, RETRY, SETTLE, SERIES, SCENE, OPPORTUNITIES, DAYS, MARK)
#: 必须成的几步：失败 = 今天没有这一代（改树是否做完由读序列那一步看词表状态挡下）。
MUST = frozenset({SERIES, OPPORTUNITIES, DAYS, MARK})
LABELS = {
    MIGRATE: "续做词表迁移",
    CALENDAR: "重读放假数据",
    RETRY: "重试收尾",
    SETTLE: "结算",
    SERIES: "读序列",
    SCENE: "语义树",
    OPPORTUNITIES: "机会表",
    DAYS: "日子预期",
    MARK: "完成标记",
}

#: 一步的结局：做成了、尽力的那一步没做成（走了退路）、这一代没轮到它（缺前提或这一代跳过）、必须成的那一步没成、
#: 还没准备好（锁被另一进程拿着：这一眼到此为止、不写完成标记、不算失败，下一眼从头做，用户 11-10 失败重试第 1 条）。
DONE, DEGRADED, SKIPPED, FAILED, HELD = "done", "degraded", "skipped", "failed", "held"
#: 健康面与报告上的原因（与 ``generation_summary.UNREACHABLE`` 同一句）。
MODEL_UNREACHABLE = UNREACHABLE
LOCK_BUSY = "锁被另一进程拿着"
MIGRATING = "词表迁移还没做完"
VOCABULARY_MOVED = "读序列时词表变了"
NO_RECORDS = "序列里还没有记录"
RETRY_UNFINISHED = "重试收尾没完成"
RETRY_REDUCING = "就近重试写回的判断还在等归约"
QUEUE_BACKING_OFF = "模型队列在整体退避"
NO_MATERIALS = "语义树没有交出材料"
#: ``Generation.ready`` 的两个"不换代"原因：worker 按它们清掉过时的失败、记下日期回退（健康面报）。
ALREADY_DONE = "今天这一代已经做完"
REWOUND = "日期回到了已换代的日子之前"

#: 各个退路的计数名（``fallbacks``；指标 ``generation_fallbacks_total{kind}``、span 属性 ``fallback.<名>``）。
FALLBACK_RELATIONS = "relations_previous"  # 关系判定退到上一代：机会表用的不是这一代关系表的 lane 数
FALLBACK_SETTLEMENT = "settlement_deferred"  # 结算留到下次
FALLBACK_HISTORY = "unit_history_skipped"  # 单元历史没做成（压实因此跳过）的 lane 数
FALLBACK_COMPACTION = "compaction_failed"  # 账本压实没做成的条数
FALLBACK_UNMAPPED = "unmapped_records"  # 换代时序列里还没映射的记录数
FALLBACK_CALENDAR = "calendar_stale"  # 放假数据重读失败，照用上一次的
FALLBACK_SCENE = "scene_step_failed"  # 语义树那一步里没做成的段数（同步词表、关系判定）
FALLBACKS = (
    FALLBACK_RELATIONS,
    FALLBACK_SETTLEMENT,
    FALLBACK_HISTORY,
    FALLBACK_COMPACTION,
    FALLBACK_UNMAPPED,
    FALLBACK_CALENDAR,
    FALLBACK_SCENE,
)

#: 线上任何一处调模型碰到它就是模型整个调不通：触点抛的 ``ModelUnreachable``，以及模型客户端原样抛出的"整个用不了"。
UNREACHABLE_ERRORS: tuple[type[BaseException], ...] = (ModelUnreachable, *MODEL_UNUSABLE)


@dataclass(frozen=True)
class Readiness:
    """现在能不能换代：能就是 ``day``（今天），不能是 None 与原因；定稿日不连续时 ``stuck`` 是卡住的那一天，当地日期回到了
    已换代的日子之前时 ``rewound`` 是（今天, 最晚完成标记的日子）（健康面报它们）。"""

    day: date | None
    reason: str
    stuck: date | None = None
    rewound: tuple[date, date] | None = None


@dataclass
class StepOutcome:
    step: str
    outcome: str
    reason: str | None
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    report: dict[str, object] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """做一代的结果：每一步的结局、各个退路的计数、这一代读到的输入是哪一代、有没有写成。"""

    day: date
    started_at: datetime
    steps: list[StepOutcome] = field(default_factory=list)
    fallbacks: Counter[str] = field(default_factory=Counter)
    #: 必须成的那一步没成：哪一步、为什么（写成了是 None）。
    failed: str | None = None
    reason: str | None = None
    #: 完成标记写成的时刻（没写成是 None）。
    finished_at: datetime | None = None
    #: 没东西可算（序列里还没有记录）：照样写完成标记。
    nothing: str | None = None
    series_cutoff: date | None = None
    vocabulary_version: int | None = None
    relation_nights: dict[str, date | None] = field(default_factory=dict)
    stamped_days: int | None = None
    #: 截止日之前还没归约的空判断条数（按槽留余量的读数：正常为 0）。
    unreduced_gaps: int | None = None
    #: 昨天「未观测」的时段（从这一代读到的序列里取）。
    unwatched: list[tuple[datetime, datetime]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    #: 从昨天定稿到这一代写出隔多久（秒；不知道定稿时刻是 None）。
    lag_seconds: float | None = None
    #: 还没准备好：哪一步、为什么（锁被占；不是失败，不退避）。
    held: str | None = None
    held_reason: str | None = None
    #: 换代那一刻模型队列各类还欠着几项（没接模型队列是 None）：进每代报告与 ``generation.run`` 的 span。
    queue_owed: dict[str, int] | None = None

    @property
    def complete(self) -> bool:
        return self.failed is None and self.finished_at is not None

    def step(self, name: str) -> StepOutcome | None:
        return next((item for item in self.steps if item.step == name), None)

    @property
    def degraded(self) -> tuple[StepOutcome, ...]:
        return tuple(item for item in self.steps if item.outcome == DEGRADED)


class MustFailed(Exception):
    """必须成的那一步没成，或这一眼还没准备好：这一代到此为止。"""


class NotReady(Exception):
    """锁被另一进程拿着（续做迁移的 sweep 锁、语义树一步的语义树写锁）：还没准备好，这一眼到此为止，下一眼从头做。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Skip(Exception):
    """这一步这一代不做（缺前提或按规矩跳过）：``reason`` 是为什么。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Degrade(Exception):
    """尽力的那一步没做成，但不是异常（锁被占、模型调不通的报告）：``reason`` 是为什么。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def retry_report(result: GenerationResult) -> object:
    step = result.step(RETRY)
    return None if step is None else step.report.get("retry")


def generation_attributes(result: GenerationResult) -> dict[str, object]:
    """``generation.run`` 的读数：写成没有、哪一步没成、读到的输入是哪一代、各个退路的计数、按槽留余量的读数。"""

    found: dict[str, object] = {
        "complete": result.complete,
        "failed_step": result.failed or "",
        "held_step": result.held or "",
        "held_reason": reason_code(result.held_reason),
        "reason": reason_code(result.reason),
        "series_cutoff": "" if result.series_cutoff is None else result.series_cutoff.isoformat(),
        "vocabulary_version": -1 if result.vocabulary_version is None else result.vocabulary_version,
        "stamped_days": -1 if result.stamped_days is None else result.stamped_days,
        "unreduced_gaps": -1 if result.unreduced_gaps is None else result.unreduced_gaps,
        "unwatched_spans": len(result.unwatched),
        "unwatched_minutes": round(sum((end - start).total_seconds() for start, end in result.unwatched) / 60, 1),
        **{f"fallback.{kind}": result.fallbacks.get(kind, 0) for kind in FALLBACKS},
        **{
            f"relation_night.{lane}": "" if night is None else night.isoformat()
            for lane, night in sorted(result.relation_nights.items())
        },
        **{f"outcome.{item.step}": item.outcome for item in result.steps},
        **{f"queue_owed.{category}": count for category, count in sorted((result.queue_owed or {}).items())},
    }
    if result.lag_seconds is not None:
        found["lag_seconds"] = result.lag_seconds
    return found


__all__ = [
    "ALREADY_DONE",
    "CALENDAR",
    "DAYS",
    "DEGRADED",
    "DONE",
    "FAILED",
    "FALLBACKS",
    "HELD",
    "MIGRATE",
    "LABELS",
    "LOCK_BUSY",
    "MARK",
    "MIGRATING",
    "MODEL_UNREACHABLE",
    "MUST",
    "NO_RECORDS",
    "OPPORTUNITIES",
    "QUEUE_BACKING_OFF",
    "RETRY",
    "RETRY_REDUCING",
    "RETRY_UNFINISHED",
    "REWOUND",
    "SCENE",
    "SERIES",
    "SETTLE",
    "SKIPPED",
    "STEPS",
    "UNREACHABLE_ERRORS",
    "Degrade",
    "GenerationResult",
    "MustFailed",
    "NotReady",
    "Readiness",
    "Skip",
    "StepOutcome",
    "generation_attributes",
    "retry_report",
]
