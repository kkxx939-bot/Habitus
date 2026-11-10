"""换代（实时化方案第 3 节，阶段 1）：一天定稿之后，把"截至昨天的历史"算好、发布给预测层。取代夜线。

**触发**（``ready``）：今天的完成标记还没有，且

- 当地今天不早于最晚完成标记的日子：往西改时区、当地日期回到已换代的日子之前时不换代、不算失败、不退避（那一天的表不能早于
  盘上最晚一晚，做了也写不成），报出 ``rewound``，日期追上后自然消掉（用户 11-10 换时区，方案 19 第 6.1 节）；
- 定稿日从最早一天起**连续**到昨天（只看"最后定稿日 + 1"会越过还没定稿的日子，实时化缺陷 4）；不连续正常只持续一轮归约，
  一直不连续就是有判断封不了口——``ready`` 把卡住的那一天交给 worker 去报健康面；
- 预测拍水位过了 max(昨天日终, 最后一段「未观测」的结尾) 再 ``margin`` 那么久（用户 11-10「按槽留余量」）：没在看的空白先写成
  判断、再给归约留几槽把它落树，结算与读序列才看得到它（必须边"没在看的空白写进树 → 读那天的记录、结算"）。

**先后**（各步一个子 span ``generation.<步>``，挂在 ``generation.run`` 下；每步记它读到的输入是哪一代）：

=====  ==================================================  ==========  ==================================================
 #      步                                                  必须成      没做成时
=====  ==================================================  ==========  ==================================================
 1      探模型                                              否          记"模型调不通"，后面调模型的步照样试、各自报
 2      重读放假数据                                        否          照用上一次读到的日历
 3      重试收尾（through = 昨天）                          否          这一代跳过结算（守住"重试 → 结算"），下一代补结
 4      结算（持预测根租约）                                否          下一代补结（幂等）
 5      词表：续迁移 → 每晚新增 → 定期拆改 → 迁移          迁移是      新增、拆改尽力；迁移没做完由读序列那一步挡下
 6      读序列（前后各看一次词表状态）                      是          这一代不成，下一眼从头
 7      语义树：同步词表 → 写概念 → 映射 → 先验 / 候选条件  否          关系判定没做成、或这一代里模型整个调不通 → 这一代不写
        → 关系判定 → 关系表 → 补日历取值                                关系表，机会表退到上一代
 8      机会表（持预测根租约，含各类的日历）                是          今天没有这一代
 9      单元历史 → 压实（同上，跟在机会表后面）              否          表照用、压实跳过
 10     当天的日子预期（持预测根租约，从零点起看）          是          今天没有这一代
 11     完成标记 ``generation/<今天>.json``（持预测根租约）  是          —
=====  ==================================================  ==========  ==================================================

只有必须成的那几步失败才算"今天没有这一代"（不写完成标记，下一眼从头再做：各步本来可重做）；其余失败大声报出（step 的结局、
健康面）、走退路、计数（``fallbacks``）、接着做。**模型调不通不停一天**：调模型的那几步各按尽力而为处理，换代照做、预测照常、
关系沿用上一次判定的。同一代不重写：完成标记存在就不再做。

这里只有"做一代"本身；什么时候做、租约、心跳、报告、指标与健康面在 ``runtime.generation_worker``。
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
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
from habitus.runtime.generation_lease import Lease
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
    PROBE,
    RETRY,
    REWOUND,
    SCENE,
    SERIES,
    SETTLE,
    SKIPPED,
    UNREACHABLE_ERRORS,
    VOCABULARY,
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
        probe: Callable[[], Awaitable[None]] | None = None,
        vocabulary_jobs: Sequence[Callable[[date], Awaitable[object]]] = (),
        migration_jobs: Sequence[Callable[[], Awaitable[object]]] = (),
        watermark: Callable[[], TickWatermark] | None = None,
        margin: timedelta = timedelta(0),
        ledger_lease: Lease | None = None,
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
        self.probe = probe
        #: 各支的每晚新增、这一天所在周期里还没做的定期拆改（持 sweep 锁、锁被占抛 ``BehaviorReductionBusyError``）。
        self.vocabulary_jobs = tuple(vocabulary_jobs)
        self.migration_jobs = tuple(migration_jobs)  # 各支续做没做完的词表迁移：排在最前（``steps.migrate_step``）
        #: 预测拍的水位（没接预测层就没有：触发不看它）。
        self.watermark = watermark
        self.margin = margin
        #: 预测根租约：只包写账本的那几步（结算、机会表 → 单元历史 → 压实、日子预期、完成标记）。
        self.ledger_lease = ledger_lease
        #: 某一天什么时候定的稿（归约的定稿日账）：量"从昨天定稿到今天这一代写出"。
        self.closed_at = closed_at
        #: 截止日之前还没归约的空判断条数（会话一支的判断存储与消费账本）。
        self.unreduced_gaps = unreduced_gaps
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
        """进程起来、第一次拿到换代租约时调：上一次中途被杀剩下的重试项撤回、删掉（裁定 68 第 3 条）。"""

        return 0 if self.retry is None else self.retry.clean_leftovers()

    async def run(self, day: date) -> GenerationResult:
        """做第 ``day`` 代（``day`` 是今天，序列截止到它、含昨天）。必须成的那一步没成时结果的 ``failed`` 是它。"""

        result = GenerationResult(day, self.clock())
        with span(GENERATION_SPAN, {"day": day.isoformat()}) as attributes:
            try:
                await self._run(result)
            except MustFailed:
                pass  # 必须成的那一步没成、或还没准备好：这一代到此为止

            set_attributes(attributes, generation_attributes(result))
        return result

    async def _run(self, result: GenerationResult) -> None:
        day = result.day
        deferred: list[str] = []
        await self._step(result, PROBE, lambda report: steps.probe_step(self, report, result))
        await self._step(result, MIGRATE, lambda report: steps.migrate_step(self, report))
        await self._step(result, CALENDAR, lambda report: steps.calendar_step(self, report, result))
        await self._step(result, RETRY, lambda report: steps.retry_step(self, report, day, deferred, result))
        await self._step(result, SETTLE, lambda report: steps.settle_step(self, report, deferred, result))
        await self._step(result, VOCABULARY, lambda report: steps.vocabulary_step(self, report, day, result))
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
                result.unreachable.append(name)
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

    def ledger(self) -> AbstractAsyncContextManager[object]:
        """写账本的那几步持预测根租约（没接就不持）。"""

        return self.ledger_lease.held() if self.ledger_lease is not None else nullcontext()


__all__ = ["Generation"]
