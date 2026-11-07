"""第 N 晚在组合根的串法：行为树只读一次，语义树与机会表用同一份序列（语义树新方案 ``13`` 第四节；预测树删了，裁定 59）。

夜里一条线（裁定 94 第 1 条）：前一天定稿之后接着跑，前一步完成即接下一步，不等钟点；结构（树、表、单元、日子预期）一天只算一次。

```
定稿     = 昨天在归约的定稿日里                              ← ``ready``：常驻节拍据它起跑（``runtime.nightly_worker``），不挂固定时刻
前置处理 = 探模型 → 重读放假数据 → RetryStage.run_once(through=昨天) ← 白天放一边的错误统一看一次：能重试的重试、落树、看完就删
                                                              （裁定 67、68、94 第 4 条）
结算     = SettlementStage.run_once()                       ← 预测层：已到最晚时限、那天定稿的预期对着行为树结；没接时跳过
词表     = 各支的每晚新增与定期拆改（``vocabulary``）         ← 结构之前改好词表，今晚的表就按它（裁定 94 第 8、9 条，不挂钟点）
结构     = 序列 → 语义树 → 机会表（同一拍重标单元历史）        ← 今晚 N = min(今天, 最后一个已封口日子的次日)
日子     = DayExpectationStage.run_once()                   ← 当天的日子预期，用这一晚含昨天的表建（``runtime.foresight_days``）
```

**只有整条线跑完，这一晚才算跑过**（``NightOutcome.complete``）。各段（``STEPS``）按先后记进度：一段成了、而且它前面的都成了，
才交给 ``on_step`` 记下。一段的结局只有三种（裁定 94 第 8 条）：

- **成了**：接着下一段；
- **还没准备好**（词表迁移中、读序列时词表变了、锁被另一进程拿着——含前置处理里重试那一支的归约与会话轮、今晚的表还不是今天这一晚的）：这一趟停在这里，这一晚不算
  跑完；下一次带着记下的进度（``resume``）来，从这一段接着跑（每一段重跑都幂等：结算只结一次、表与单元按这一晚覆盖、
  日子预期同一个身份只建一次）；
- **出错就停下这一晚**（``stopped``）：一段抛了异常是 bug（代码出错、数据卡住本不该有），拦住、不接着跑后面几段、健康面报出
  （``"<段>出错: <异常>"``），也记成这一晚跑过——不每隔几分钟整段重跑（结构一天只算一次），下一晚照常。**模型整个调不通**
  （线上任何一处调模型碰到 ``ModelUnreachable`` / ``MODEL_UNUSABLE``：前置处理的探针、重试与它的归约、每晚新增与定期拆改、语义树的
  各触点、日子预期的未封口归类）同样停下，
  健康面写"模型调不通"，不拿旧数据顶；停掉那天没有日子预期（那天按看不到）。前置处理里白天放一边的错误是逐项拦的（一项的 bug
  只拦这一项，``behavior.retry``），重读放假数据失败照用上一次读到的日历——这两处不停线。

**结算排在结构之前**：找原因只给已经结出来的落空找，所以同一晚里结算必须在它之前——排在后面的话，今晚结出的落空要到
明晚才找原因，第二天白天核"上次的原因今天又在"就晚了一天（年假第二天照样开口）。结算只读账本、行为树与上一晚的机会表、
不读这一晚的语义树，排在前面不缺材料。

语义树与预测层对同一条记录的读法一致，由序列保证（CI 钉住）。序列里还没有一条可用记录时语义树这一晚不跑（没有可量的东西；
原来这道闸是"预测树发布了没有"，清单 5.2），也没有日子预期可建：这一晚就算跑完。

词表迁移做到一半时不跑（第四轮评审 E3）：迁移先改树上的行、最后才写新版本，做到一半读出来的序列一半新编号一半旧编号；
读序列前后各看一次词表状态（版本、有没有未完成的迁移），不一致就停在结构这一段，下一眼再来。

截止日不能只取"今天"（第四轮评审 E1）：零点刚过时昨天那条链可能还在融合静默期、没归约，夜批把昨天当成完整的一天映射、
盖完成章，之后补发进来的记录就永远不映射了。所以只用到最后一个已封口的日子为止。"今天"按主体所在地的时区算
（``zone``），不看进程默认时区。节拍由 ``runtime.nightly_worker.NightlyWorker`` 管。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from habitus.behavior.reduction import BehaviorReductionBusyError
from habitus.behavior.retry import RetryReport, RetryStage
from habitus.behavior.tree import BehaviorTree
from habitus.model_client import MODEL_UNUSABLE, ModelUnreachable
from habitus.runtime.foresight_days import DayExpectationStage, DayReport
from habitus.runtime.foresight_nightly import ForesightNightlyRun, OpportunityReport
from habitus.runtime.foresight_settlement import SettlementReport, SettlementStage
from habitus.runtime.scene_night import NightReport, SceneNightlyRun
from habitus.series import EventSeries
from habitus.series.reader import SeriesReader

PREPARE, SETTLE, VOCABULARY, STRUCTURE, DAYS = "prepare", "settle", "vocabulary", "structure", "days"
#: 线的各段，按先后（裁定 94 第 1、8、9 条）：前置处理 → 结算 → 词表（每晚新增、定期拆改）→ 结构（序列、语义树、机会表、单元）→ 当天的日子预期。
STEPS = (PREPARE, SETTLE, VOCABULARY, STRUCTURE, DAYS)
_LABELS = {PREPARE: "前置处理", SETTLE: "结算", VOCABULARY: "词表", STRUCTURE: "结构", DAYS: "日子预期"}
#: 健康面上这一晚停下的原因（裁定 94 第 4 条）。
MODEL_UNREACHABLE = "模型调不通"
#: 结构这一段没东西可算（不是没跑完）：这一晚照样算跑完。
NOTHING_CLOSED, NO_RECORDS = "一天都还没封口", "序列里还没有记录"
_NOTHING_TO_BUILD = frozenset({NOTHING_CLOSED, NO_RECORDS})
#: 线上任何一处调模型碰到它就停下这一晚、报"模型调不通"（裁定 94 第 8 条）：触点抛的 ``ModelUnreachable``，以及词表整理活、
#: 日子预期的未封口归类这类直接调模型客户端、原样抛出的"整个用不了"。
_UNREACHABLE: tuple[type[BaseException], ...] = (ModelUnreachable, *MODEL_UNUSABLE)
#: 锁被另一进程拿着：还没准备好，下一眼再来。
LOCK_BUSY = "锁被另一进程拿着"


@dataclass(frozen=True)
class NightOutcome:
    night: date
    series: EventSeries
    scene: NightReport | None
    #: 这一趟停在哪儿没准备好（词表迁移中、锁被占……），或结构这一段没东西可算；都不是为 None。
    skipped: str | None = None
    #: 预测层的机会表（没接预测层、或语义树这一晚没跑到就是 None）。
    opportunities: OpportunityReport | None = None
    #: 这一晚的结算（没接预测层、或没跑到就是 None）。
    settled: SettlementReport | None = None
    #: 这一晚开头的重试（没接、或这一趟从断点接着跑、前置处理之前那一趟已做完是 None）。
    retried: RetryReport | None = None
    #: 这一晚开头重读放假数据抛了什么（没失败是 None）：照用上一次读到的日历，整晚照跑。
    calendar_failed: str | None = None
    #: 这一晚末尾建的当天日子预期（没接、或这一晚没跑到那一步是 None）。
    days: DayReport | None = None
    #: 这一晚停下的原因（模型整个调不通、某一段出错，裁定 94 第 4、8 条）；没停是 None。停下的这一晚不再接着跑，下一晚照常。
    stopped: str | None = None
    #: 这一趟结束时线上已经成了的段（含之前那一趟做完、这一趟接着跑时跳过的），按 ``STEPS`` 的先后。
    done: tuple[str, ...] = field(default=())

    @property
    def complete(self) -> bool:
        """整条线跑完了（只有这样这一晚才算跑过，裁定 94 第 1 条）。"""

        return self.done == STEPS


class _Line:
    """一趟里的进度：从第一段没成的那里接着跑；一段成了、前面的都成了才记下（交给 ``on_step``）。"""

    def __init__(self, resume: Iterable[str], on_step: Callable[[str], None] | None) -> None:
        resumed = set(resume)
        self.done: list[str] = []
        for step in STEPS:
            if step not in resumed:
                break
            self.done.append(step)
        self.on_step = on_step

    def pending(self, step: str) -> bool:
        return step not in self.done

    def finish(self, step: str) -> None:
        self.done.append(step)
        if self.on_step is not None:
            self.on_step(step)


@dataclass
class _Run:
    """一趟跑下来攒的结果。"""

    night: date
    series: EventSeries
    scene: NightReport | None = None
    skipped: str | None = None
    opportunities: OpportunityReport | None = None
    settled: SettlementReport | None = None
    retried: RetryReport | None = None
    calendar_failed: str | None = None
    days: DayReport | None = None
    stopped: str | None = None

    def outcome(self, done: Sequence[str]) -> NightOutcome:
        return NightOutcome(
            night=self.night,
            series=self.series,
            scene=self.scene,
            skipped=self.skipped,
            opportunities=self.opportunities,
            settled=self.settled,
            retried=self.retried,
            calendar_failed=self.calendar_failed,
            days=self.days,
            stopped=self.stopped,
            done=tuple(done),
        )


class _Stop(Exception):
    """这一晚停下（不是 bug 本身，是线据某一段的结局决定停）。"""


class Nightly:
    def __init__(
        self,
        *,
        behavior_tree: BehaviorTree,
        scene: SceneNightlyRun,
        zone: ZoneInfo,
        closed_days: Callable[[], Iterable[date]],
        vocabulary: Callable[[], tuple[int, bool]],
        clock: Callable[[], datetime] | None = None,
        foresight: ForesightNightlyRun | None = None,
        settlement: SettlementStage | None = None,
        refresh_calendar: Callable[[], object] | None = None,
        series: SeriesReader | None = None,
        retry: RetryStage | None = None,
        days: DayExpectationStage | None = None,
        probe: Callable[[], Awaitable[None]] | None = None,
        vocabulary_jobs: Sequence[Callable[[date], Awaitable[object]]] = (),
    ) -> None:
        if scene.behavior_tree is not behavior_tree:
            raise ValueError("the scene run must read the same behaviour tree")
        self.behavior_tree = behavior_tree
        if series is not None and series.tree is not behavior_tree:
            raise ValueError("the series reader must read the same behaviour tree")
        #: 常驻的序列读者：按天缓存行为树文档，跨晚复用（26 号清单 6.7）。
        self.series = series if series is not None else SeriesReader(behavior_tree)
        self.scene = scene
        self.zone = zone
        self.closed_days = closed_days
        #: 词表此刻的（版本, 有没有未完成的迁移）
        self.vocabulary = vocabulary
        self.retry = retry
        self.foresight = foresight
        self.settlement = settlement
        if days is not None and foresight is None:
            raise ValueError("the day expectations are built from the opportunity table this run writes")
        self.days = days
        #: 每晚开头重读一遍放假数据（读不了的年份由日历自己跳过并挂出问题；别的失败记进 ``calendar_failed``）。
        self.refresh_calendar = refresh_calendar
        #: 前置处理最先探一次模型：整个调不通就抛 ``ModelUnreachable``（裁定 94 第 4 条）；没接就不探。
        self.probe = probe
        #: 各支的词表整理活（归约的 ``nightly_vocabulary(这一晚)``：每晚新增、这一晚所在周期里还没做的定期拆改；持 sweep 锁、
        #: 锁被占抛 ``BehaviorReductionBusyError``）。
        self.vocabulary_jobs = tuple(vocabulary_jobs)
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._steps: dict[str, Callable[[_Run], Awaitable[bool]]] = {
            PREPARE: self._prepare,
            SETTLE: self._settle,
            VOCABULARY: self._vocabulary,
            STRUCTURE: self._structure,
            DAYS: self._days,
        }

    def tonight(self) -> date:
        """主体所在地的"今天"（``zone``）。按进程默认时区算会差一天，每个数字看起来都还是对的。"""

        return self._clock().astimezone(self.zone).date()

    def ready(self) -> date | None:
        """这条线现在能跑的那一晚：昨天已经定稿就是今天，否则 None（裁定 94 第 1 条：前一天定稿完成即接着跑）。"""

        today = self.tonight()
        return today if today - timedelta(days=1) in set(self.closed_days()) else None

    async def run_once(
        self, *, resume: Iterable[str] = (), on_step: Callable[[str], None] | None = None
    ) -> NightOutcome:
        """跑一趟。``resume`` 是这一晚之前那一趟记下的进度：从第一段没成的那里接着跑；``on_step`` 记下这一趟新成的段。"""

        line = _Line(resume, on_step)
        tonight = self.tonight()
        run = _Run(tonight, EventSeries(cutoff=tonight))
        for step in STEPS:
            if not line.pending(step):
                continue
            try:
                ready = await self._steps[step](run)
            except _Stop:
                break
            except _UNREACHABLE as exc:
                run.stopped = f"{MODEL_UNREACHABLE}: {exc}"
                break
            except Exception as exc:  # noqa: BLE001 - 一段抛了是 bug：这一晚停下、交给健康面，下一晚照常（裁定 94 第 8 条）
                run.stopped = f"{_LABELS[step]}出错: {type(exc).__name__}"
                break
            if not ready:
                break  # 还没准备好：这一晚没跑完，下一眼从这一段接着跑
            line.finish(step)
        return run.outcome(line.done)

    def clean_leftovers(self) -> int:
        """常驻节拍一起来就调：上一晚中途被杀剩下的重试项撤回、删掉（裁定 68 第 3 条）。"""

        return 0 if self.retry is None else self.retry.clean_leftovers()

    async def _prepare(self, run: _Run) -> bool:
        """前置处理（裁定 94 第 4 条）：先探模型（整个调不通就抛 ``ModelUnreachable``，不重试）、重读放假数据、再统一看白天
        放一边的错误；重试里（含它的归约）试到模型整个用不了，这一晚停下（报告留着，健康面照样看得见那一晚的计数）。重试那一支的
        锁被另一进程拿着是还没准备好（裁定 94 第 9 条）：停在这一段、不结算，下一眼重试从它停下的地方接着跑。"""

        if self.probe is not None:
            await self.probe()
        run.calendar_failed = await self._refresh_calendar()
        if self.retry is not None:
            # 这一趟处理前一天及更早的；零点之后新进、属于今天的等明天那一趟（裁定 68 第 3 条）
            run.retried = await self.retry.run_once(through=self.tonight() - timedelta(days=1))
            if run.retried.unreachable:
                run.stopped = f"{MODEL_UNREACHABLE}: the model is unavailable to the retry"
                raise _Stop
            if not run.retried.ready:
                run.skipped = LOCK_BUSY
                return False
        return True

    async def _settle(self, run: _Run) -> bool:
        if self.settlement is not None:
            run.settled = await asyncio.to_thread(self.settlement.run_once)
        return True

    async def _vocabulary(self, run: _Run) -> bool:
        """各支的每晚新增与定期拆改：锁被另一进程拿着就是还没准备好（已经做完的支再来一次只多问一遍待定池里还没聚组的；
        定期拆改这个周期做过就不再做）。"""

        try:
            for job in self.vocabulary_jobs:
                await job(run.night)
        except BehaviorReductionBusyError:
            run.skipped = LOCK_BUSY
            return False
        return True

    async def _structure(self, run: _Run) -> bool:
        """结构这一段：序列 → 语义树 → 机会表（同一拍重标单元历史）。"""

        closed = sorted(self.closed_days())
        if not closed:
            # 一天都还没封口（刚装上）：没有完整的一天可算
            run.skipped = NOTHING_CLOSED
            return True
        night = min(run.night, closed[-1] + timedelta(days=1))
        run.night, run.series = night, EventSeries(cutoff=night)
        before = self.vocabulary()
        if before[1]:
            run.skipped = "词表迁移还没做完"
            return False
        # 全量扫行为树与写机会表是同步 IO：下沉到线程，不卡白天那一拍与同一个循环上的在线请求（2026-10-18 四方评审）
        series = await asyncio.to_thread(self.series.read, cutoff=night)
        if self.vocabulary() != before:
            run.skipped = "读序列时词表变了"
            return False
        run.series = series
        if not series.records:
            # 只有观测空白、一条记录都没有（上游刚接入）：没有可量的东西
            run.skipped = NO_RECORDS
            return True
        run.scene = await self.scene.run(series=series, now=self._clock())
        run.opportunities = await self._opportunities(series, run.scene)
        if run.opportunities is not None and run.opportunities.failed:
            # 机会表或单元历史某条 lane 没做成（26 号清单 2.5.1）：同样是出错，停下这一晚
            run.stopped = f"机会表出错: {'; '.join(run.opportunities.failed)}"
            raise _Stop
        return True

    async def _opportunities(self, series: EventSeries, report: NightReport) -> OpportunityReport | None:
        if self.foresight is None or report.materials is None:
            return None
        return await asyncio.to_thread(
            self.foresight.run,
            night=series.cutoff,
            concepts=report.materials.concepts,
            timelines=report.materials.timelines,
            recurrences=report.materials.recurrences,
            coverage=report.materials.coverage,
        )

    async def _days(self, run: _Run) -> bool:
        """当天的日子预期：结构这一段没东西可算就没有可建的；这一晚的表还不是今天的（词表迁移中）是还没准备好。"""

        if self.days is None or run.skipped in _NOTHING_TO_BUILD:
            return True
        run.days = await self.days.run_once()
        if run.days.skipped is not None:
            run.skipped = run.days.skipped
            return False
        return True

    async def _refresh_calendar(self) -> str | None:
        """重读放假数据，返回它抛了什么。读不了的年份日历自己跳过；再有别的失败照用上一次读到的日历，不停线。"""

        if self.refresh_calendar is None:
            return None
        try:
            await asyncio.to_thread(self.refresh_calendar)
        except Exception as exc:  # noqa: BLE001 - 照用上一次读到的日历；不许拖着后面一夜不跑
            return type(exc).__name__
        return None


__all__ = ["LOCK_BUSY", "MODEL_UNREACHABLE", "STEPS", "NightOutcome", "Nightly"]
