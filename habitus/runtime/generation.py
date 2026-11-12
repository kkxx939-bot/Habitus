"""换代（实时化方案第 3 节，阶段 1）：一天定稿之后，把"截至昨天的历史"算好、发布给预测层。取代夜线。

**触发**（``ready``）：今天的完成标记还没有，且

- 当地今天不早于最晚完成标记的日子：往西改时区、当地日期回到已换代的日子之前时不换代、不算失败、不退避（那一天的表不能早于
  盘上最晚一晚，做了也写不成），报出 ``rewound``，日期追上后自然消掉（用户 11-10 换时区，方案 19 第 6.1 节）；
- 定稿日从最早一天起**连续**到昨天（只看"最后定稿日 + 1"会越过还没定稿的日子，实时化缺陷 4）；不连续正常只持续一轮归约，
  一直不连续就是有判断封不了口——``ready`` 把卡住的那一天交给 worker 去报健康面；
- 预测拍水位过了 max(昨天日终, 最后一段「未观测」的结尾) 再 ``margin`` 那么久（用户 11-10「按槽留余量」）：没在看的空白先写成
  判断、再给归约留几槽把它落树，结算与读序列才看得到它（必须边"没在看的空白写进树 → 读那天的记录、结算"）。

**换代里没有模型调用**（实时化阶段 2）：出错重试、词表的每晚新增与定期拆改、迁移牵到的重映射、写概念、映射、候选条件
与先验的补问都归模型队列（``runtime.model_queue``），模型多慢、调不通都不影响这里写出今天这一代。换代不等队列：队列没做完的
按"不知道"算（没映射完的日子关系检验读不到、缺的先验权重按 1），下一代补上。

**先后**（各步一个子 span ``generation.<步>``，挂在 ``generation.run`` 下；每步记它读到的输入是哪一代）：

=====  ==================================================  ==========  ==================================================
 #      步                                                  必须成      没做成时
=====  ==================================================  ==========  ==================================================
 1      待生效的拆改交给迁移器、续做没做完的词表迁移        迁移是      没做完由结算（跳过）与读序列（这一代不成）各自挡下
 2      重读放假数据                                        否          照用上一次读到的日历
 3      重试收尾（through = 昨天；只丢弃、撤回，不试）      否          这一代跳过结算（守住"重试 → 结算"），下一代补结
 4      结算                                                否          下一代补结（幂等）
 5      读序列（前后各看一次词表状态）                      是          这一代不成，下一眼从头
 6      语义树（整步持语义树写锁）：同步词表（概念落盘）→    否          关系判定没做成、或换代那一刻模型队列在整体退避 →
        按缓存判关系 → 关系表 → 补日历取值                              这一代不写关系表，机会表退到上一代
 7      机会表（含各类的日历）                              是          今天没有这一代
 8      单元历史 → 压实（同上，跟在机会表后面）              否          表照用、压实跳过
 9      当天的日子预期（从零点起看）                        是          今天没有这一代
 10     完成标记 ``generation/<今天>.json``                  是          —
=====  ==================================================  ==========  ==================================================

只有必须成的那几步失败才算"今天没有这一代"（不写完成标记，下一眼从头再做：各步本来可重做）；其余失败大声报出（step 的结局、
健康面）、走退路、计数（``fallbacks``）、接着做。同一代不重写：完成标记存在就不再做。

``sealed_day`` 是"已封口的日子的活现在可以做了"的那一天：触发条件里除了"今天的完成标记还没有"之外的那几条都满足时是今天。
模型队列据它认已封口的日子的欠账（早于这一刻读序列，「未观测」的空白可能还没落树）；完成标记有没有不影响欠账。

**锁**（阶段 2 第二小步）：整代在预测 worker 那一槽的预测根租约之内做（``runtime.foresight_worker``：先看一眼换代、再做预测拍），
账本只有一个写者，各步不再各自拿它。语义树一步整步持语义树写锁（``runtime.scene_lock``）：模型队列的落盘等它做完，读到的语义树是
一份停住的样子；等一个租期还拿不到（另一个进程的队列正写着）是还没准备好，这一代到此为止、下一槽再看，不算失败、不退避。
换代在做时立一个标志（``generating``），同一进程的模型队列在两项之间看到就先不取下一项。

这里只有"做一代"本身；什么时候做、报告、指标与健康面在 ``runtime.generation_runner``。
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from zoneinfo import ZoneInfo

from habitus.behavior.retry import RetryStage
from habitus.behavior.tree import BehaviorTree
from habitus.foundation.tracing import current_span, set_attributes, span
from habitus.runtime import generation_steps as steps
from habitus.runtime.foresight_days import DayExpectationStage
from habitus.runtime.foresight_nightly import ForesightNightlyRun
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.generation_marks import GenerationMarkError, GenerationMarks
from habitus.runtime.generation_result import (
    ALREADY_DONE,
    CALENDAR,
    DAYS,
    DEGRADED,
    DONE,
    FAILED,
    HELD,
    LABELS,
    MARK,
    MIGRATE,
    MODEL_UNREACHABLE,
    MUST,
    OPPORTUNITIES,
    RETRY,
    REWOUND,
    SCENE,
    SERIES,
    SETTLE,
    SKIPPED,
    UNREACHABLE_ERRORS,
    Degrade,
    GenerationResult,
    MustFailed,
    NotReady,
    Readiness,
    Skip,
    StepOutcome,
    generation_attributes,
)
from habitus.runtime.generation_summary import flatten, reason_code
from habitus.runtime.generation_trace import GENERATION_SPAN, STEP_SPAN_PREFIX
from habitus.runtime.generation_watermark import TickWatermark
from habitus.runtime.scene_lock import SceneWriteLock
from habitus.runtime.scene_night import NightReport, SceneNightlyRun
from habitus.series import EventSeries
from habitus.series.reader import SeriesReader


class Generation:
    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        scene: SceneNightlyRun,
        zone: ZoneInfo,
        closed_days: Callable[[], Iterable[date]],
        vocabulary: Callable[[], tuple[int, bool]],
        marks: GenerationMarks,
        clock: Callable[[], datetime] | None = None,
        foresight: ForesightNightlyRun | None = None,
        settlement: SettlementStage | None = None,
        refresh_calendar: Callable[[], object] | None = None,
        series: SeriesReader | None = None,
        retry: RetryStage | None = None,
        days: DayExpectationStage | None = None,
        migration_jobs: Sequence[Callable[[], Awaitable[object]]] = (),
        models_down: Callable[[], bool] | None = None,
        queue_owed: Callable[[], Mapping[str, int]] | None = None,
        watermark: Callable[[], TickWatermark] | None = None,
        margin: timedelta = timedelta(0),
        scene_lock: SceneWriteLock | None = None,
        generating: Callable[[bool], None] | None = None,
        revisions: Callable[[], int] | None = None,
        closed_at: Callable[[date], datetime | None] | None = None,
        unreduced_gaps: Callable[[date], int] | None = None,
    ) -> None:
        if scene.behavior_tree is not behavior_tree:
            raise ValueError("the scene run must read the same behaviour tree")
        if series is not None and series.tree is not behavior_tree:
            raise ValueError("the series reader must read the same behaviour tree")
        if days is not None and foresight is None:
            raise ValueError("the day expectations are built from the opportunity table this generation writes")
        if margin < timedelta(0):
            raise ValueError("margin must not be negative")
        self.behavior_tree = behavior_tree
        #: 常驻的序列读者：按天缓存行为树文档，跨代复用（26 号清单 6.7）。
        self.series = series if series is not None else SeriesReader(behavior_tree, name="generation")
        self.scene = scene
        self.zone = zone
        self.closed_days = closed_days
        self.vocabulary = vocabulary  # 词表此刻的（版本, 有没有未完成的迁移）
        self.marks = marks
        self.retry = retry
        self.foresight = foresight
        self.settlement = settlement
        self.days = days
        self.refresh_calendar = refresh_calendar
        self.migration_jobs = tuple(migration_jobs)  # 各支续做没做完的词表迁移：排在最前（``steps.migrate_step``）
        #: 同一进程的模型队列：此刻是不是在整体退避（是就这一代不判关系、沿用上一代，裁定 96）、各类此刻还欠着几项。没接是 None。
        self.models_down, self.queue_owed = models_down, queue_owed
        #: 预测拍的水位（没接预测层就没有：触发不看它）与要过的余量。
        self.watermark, self.margin = watermark, margin
        #: 语义树写锁（语义树一步整步持有）、换代开始 / 做完时立撤的标志（模型队列看它）、各支待生效的拆改有几份（续做迁移
        #: 那一步先把它们交给迁移器）；没接都是 None。
        self.scene_lock, self.generating, self.revisions = scene_lock, generating, revisions
        #: 某一天什么时候定的稿（归约的定稿日账：量"从昨天定稿到今天这一代写出"）；截止日之前还没归约的空判断条数。
        self.closed_at, self.unreduced_gaps = closed_at, unreduced_gaps
        self.clock: Callable[[], datetime] = clock if clock is not None else lambda: datetime.now(UTC)

    def today(self) -> date:
        """主体所在地的"今天"（``zone``）。按进程默认时区算会差一天，每个数字看起来都还是对的。"""

        return self.clock().astimezone(self.zone).date()

    def ready(self) -> Readiness:
        """现在能不能做今天这一代（见模块文档串的触发条件）。"""

        today = self.today()
        try:
            if self.marks.exists(today):
                return Readiness(None, ALREADY_DONE)
        except GenerationMarkError:
            pass  # 今天的标记坏了：不当已做（重做时整份重写），健康面另报读不了
        return self._sealed(today)

    def sealed_day(self) -> date | None:
        """已封口的日子的活现在可以做了的那一天（今天）：触发条件里除"今天的完成标记还没有"之外的那几条都满足；否则 None。"""

        return self._sealed(self.today()).day

    def _sealed(self, today: date) -> Readiness:
        """触发条件里不看今天完成标记的那几条：日期没有回退、连续定稿到昨天、预测拍水位过了余量。"""

        try:
            latest = max(self.marks.days(), default=None)
        except GenerationMarkError:
            latest = None  # 标记目录读不了：健康面另报
        if latest is not None and today < latest:
            return Readiness(None, f"{REWOUND}（今天 {today}，最晚 {latest}）", rewound=(today, latest))
        yesterday = today - timedelta(days=1)
        # 只看昨天及以前的定稿日：今天不会定稿，也不能拿它顶昨天（codex R3）
        closed = frozenset(day for day in self.closed_days() if day <= yesterday)
        if not closed:
            return Readiness(None, "一天都还没定稿到昨天")
        # 从最早的定稿日往后找第一个还没定稿的日子；它之后还有已定稿的，就是定稿日不连续（卡在它）
        day = min(closed)
        while day in closed:
            day += timedelta(days=1)
        stuck = day if day < yesterday and any(item > day for item in closed) else None
        if day <= yesterday:
            return Readiness(None, f"{day} 还没定稿", stuck)
        if self.watermark is not None:
            mark = self.watermark()
            midnight = datetime.combine(today, clock_time(0), tzinfo=self.zone)
            floor = max(midnight, mark.unwatched_until or midnight) + self.margin
            if mark.ticked_at is None or mark.ticked_at < floor:
                return Readiness(None, f"预测拍水位还没过 {floor.isoformat()}")
        return Readiness(today, "")

    def clean_leftovers(self) -> int:
        """进程起来后第一眼调：上一次中途被杀剩下的重试项撤回、删掉（裁定 68 第 3 条）。"""

        return 0 if self.retry is None else self.retry.clean_leftovers()

    async def run(self, day: date) -> GenerationResult:
        """做第 ``day`` 代（``day`` 是今天，序列截止到它、含昨天）。必须成的那一步没成时结果的 ``failed`` 是它。"""

        result = GenerationResult(day, self.clock())
        if self.queue_owed is not None:
            result.queue_owed = dict(self.queue_owed())
        if self.generating is not None:
            self.generating(True)
        try:
            with span(GENERATION_SPAN, {"day": day.isoformat()}) as attributes:
                try:
                    await self._run(result)
                except MustFailed:
                    pass  # 必须成的那一步没成、或还没准备好：这一代到此为止

                set_attributes(attributes, generation_attributes(result))
        finally:
            if self.generating is not None:
                self.generating(False)
        return result

    async def _run(self, result: GenerationResult) -> None:
        day = result.day
        deferred: list[str] = []
        await self._step(result, MIGRATE, lambda report: steps.migrate_step(self, report))
        await self._step(result, CALENDAR, lambda report: steps.calendar_step(self, report, result))
        await self._step(result, RETRY, lambda report: steps.retry_step(self, report, day, deferred, result))
        await self._step(result, SETTLE, lambda report: steps.settle_step(self, report, deferred, result))
        holder: list[EventSeries] = []
        await self._step(result, SERIES, lambda report: steps.series_step(self, report, day, result, holder))
        series = holder[0]
        if result.nothing is None:
            scene: list[NightReport] = []
            await self._step(result, SCENE, lambda report: steps.scene_step(self, report, series, result, scene))
            await self._step(
                result, OPPORTUNITIES, lambda report: steps.opportunities_step(self, report, series, scene, result)
            )
            await self._step(result, DAYS, lambda report: steps.days_step(self, report))
        await self._step(result, MARK, lambda report: steps.mark_step(self, report, result))

    async def _step(
        self, result: GenerationResult, name: str, body: Callable[[dict[str, object]], Awaitable[None]]
    ) -> None:
        """跑一步：一个子 span、记结局；必须成的那一步没成抛 ``MustFailed``，尽力的那一步没成记 ``degraded``、接着做。"""

        report: dict[str, object] = {}
        started_at, started = self.clock(), time.monotonic()
        outcome, reason = DONE, None
        with span(f"{STEP_SPAN_PREFIX}{name}", {"day": result.day.isoformat()}) as attributes:
            try:
                await body(report)
            except Skip as skip:
                outcome, reason = SKIPPED, skip.reason
            except NotReady as held:
                outcome, reason = HELD, held.reason
            except Degrade as degrade:
                outcome, reason = (FAILED if name in MUST else DEGRADED), degrade.reason
            except UNREACHABLE_ERRORS as exc:
                outcome, reason = (FAILED if name in MUST else DEGRADED), f"{MODEL_UNREACHABLE}: {exc}"
            except Exception as exc:  # noqa: BLE001 - 一步抛了：必须成的这一代不成，尽力的报出、走退路、接着做
                outcome, reason = (FAILED if name in MUST else DEGRADED), f"{LABELS[name]}出错: {type(exc).__name__}"
            record = current_span()
            if record is not None and record.model_calls:
                calls: Counter[str] = Counter()
                for (touch, _provider, _model), count in record.model_calls.items():
                    calls[touch] += count
                report["model_calls"] = dict(calls)
            set_attributes(attributes, {"outcome": outcome, "reason": reason_code(reason), **flatten(report)})
        result.steps.append(
            StepOutcome(name, outcome, reason, started_at, self.clock(), max(0.0, time.monotonic() - started), report)
        )
        if outcome == HELD:
            result.held, result.held_reason = name, reason
            raise MustFailed
        if outcome == FAILED:
            result.failed, result.reason = name, reason
            raise MustFailed

    def scene_writes(self) -> AbstractAsyncContextManager[object]:
        """语义树一步持语义树写锁（没接就不持）。"""

        return self.scene_lock.held() if self.scene_lock is not None else nullcontext()


__all__ = ["Generation"]
