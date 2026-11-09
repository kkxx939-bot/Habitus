"""第 N 晚在组合根的串法：行为树只读一次，语义树与机会表用同一份序列（语义树新方案 ``13`` 第四节；预测树删了，裁定 59）。

夜里一条线（裁定 94 第 1 条）：前一天定稿之后接着跑，前一步完成即接下一步，不等钟点；结构（树、表、单元、日子预期）一天只算一次。

```
定稿     = 昨天在归约的定稿日里                              ← ``ready``：夜线的 Job 执行器据它入队（``runtime.nightly_worker``），不挂固定时刻
前置处理 = 探模型 → 重读放假数据 → RetryStage.run_once(through=昨天) ← 白天放一边的错误统一看一次：能重试的重试、落树、看完就删
                                                              （裁定 67、68、94 第 4 条）
结算     = SettlementStage.run_once()                       ← 预测层：已到最晚时限、那天定稿的预期对着行为树结；没接时跳过
词表     = 各支的每晚新增与定期拆改（``vocabulary``）         ← 结构之前改好词表，今晚的表就按它（裁定 94 第 8、9 条，不挂钟点）
结构     = 序列 → 语义树 → 机会表（同一拍重标单元历史）        ← 今晚 N = min(这一晚, 最后一个已封口日子的次日)
日子     = DayExpectationStage.run_once()                   ← 当天的日子预期，用这一晚含昨天的表建（``runtime.foresight_days``）
```

**每一晚每一段是一个耐久 Job**（``runtime.nightly_jobs``）：这里只有"跑一段"（``run_step``），哪一段什么时候跑、跑完没有、
重试几次由执行器按 Job 账定。**只有 5 段都成了，这一晚才算跑完**。一段的结局只有三种（裁定 94 第 8 条）：

- **成了**（``DONE``）：执行器接着跑下一段；
- **还没准备好**（``HELD``：词表迁移中、读序列时词表变了、锁被另一进程拿着——含前置处理里重试那一支的归约与会话轮、今晚的表还不是
  今天这一晚的）：Job 保持排队，下一眼从这一段接着跑（每一段重跑都幂等：结算只结一次、表与单元按这一晚覆盖、日子预期同一个身份
  只建一次；重试那一支等锁的状态随 Job 落盘，``RetryCheckpoint``）；
- **停下**（``STOPPED``）：一段抛了异常是 bug（代码出错、数据卡住本不该有），拦住、不接着跑后面几段、健康面报出
  （``"<段>出错: <异常>"``）。**模型整个调不通**（线上任何一处调模型碰到 ``ModelUnreachable`` / ``MODEL_UNUSABLE``：前置处理的探针、
  重试与它的归约、每晚新增与定期拆改、语义树的各触点、日子预期的未封口归类）同样停下，健康面写"模型调不通"，不拿旧数据顶。
  停下的段由执行器有界退避重试，次数用完记失败；这一晚没跑完的那天没有日子预期（那天按看不到）。前置处理里白天放一边的错误是
  逐项拦的（一项的 bug 只拦这一项，``behavior.retry``），重读放假数据失败照用上一次读到的日历——这两处不停线。

段与段之间不经内存传东西：各段的输入本来就在盘上（重试堆、账本、词表、行为树、语义树与机会表的文件）；日子预期那一段要知道
"结构这一段没东西可算"，由执行器从 Job 账里结构那一段的报告取来交给它（``structure_skipped``）。

**结算排在结构之前**：找原因只给已经结出来的落空找，所以同一晚里结算必须在它之前——排在后面的话，今晚结出的落空要到
明晚才找原因，第二天白天核"上次的原因今天又在"就晚了一天（年假第二天照样开口）。结算只读账本、行为树与上一晚的机会表、
不读这一晚的语义树，排在前面不缺材料。

语义树与预测层对同一条记录的读法一致，由序列保证（CI 钉住）。序列里还没有一条可用记录时语义树这一晚不跑（没有可量的东西；
原来这道闸是"预测树发布了没有"，清单 5.2），也没有日子预期可建：这一晚就算跑完。

词表迁移做到一半时不跑（第四轮评审 E3）：迁移先改树上的行、最后才写新版本，做到一半读出来的序列一半新编号一半旧编号；
读序列前后各看一次词表状态（版本、有没有未完成的迁移），不一致就停在结构这一段，下一眼再来。

截止日不能只取"今天"（第四轮评审 E1）：零点刚过时昨天那条链可能还在融合静默期、没归约，夜批把昨天当成完整的一天映射、
盖完成章，之后补发进来的记录就永远不映射了。所以只用到最后一个已封口的日子为止。"今天"按主体所在地的时区算
（``zone``），不看进程默认时区。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from habitus.behavior.reduction import BehaviorReductionBusyError
from habitus.behavior.retry import RetryCheckpoint, RetryStage
from habitus.behavior.tree import BehaviorTree
from habitus.model_client import MODEL_UNUSABLE, ModelUnreachable
from habitus.runtime.foresight_days import DayExpectationStage
from habitus.runtime.foresight_nightly import ForesightNightlyRun, OpportunityReport
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.runtime.nightly_summary import (
    day_summary,
    opportunity_summary,
    pending_by_branch,
    retry_summary,
    scene_summary,
    settlement_summary,
    vocabulary_summary,
)
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


#: 一段的三种结局（裁定 94 第 8 条）。
DONE, HELD, STOPPED = "done", "held", "stopped"


@dataclass(frozen=True)
class StepResult:
    """跑一段的结局：成了、还没准备好（``reason`` 是为什么）、停下（``reason`` 是停下的原因）。``report`` 是这一段原有的报告
    （能落盘的摘要，进 Job 账与每晚报告），``retry_checkpoint`` 是前置处理还没准备好时重试那一支留下的、要随 Job 落盘的那一份。"""

    step: str
    outcome: str
    reason: str | None = None
    report: Mapping[str, object] = field(default_factory=dict)
    retry_checkpoint: RetryCheckpoint | None = None

    @property
    def done(self) -> bool:
        return self.outcome == DONE


@dataclass
class _Run:
    """一段跑下来攒的：结局之外的报告。"""

    night: date
    report: dict[str, object] = field(default_factory=dict)
    skipped: str | None = None
    stopped: str | None = None


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
            SETTLE: self._settle,
            VOCABULARY: self._vocabulary,
            STRUCTURE: self._structure,
        }

    def tonight(self) -> date:
        """主体所在地的"今天"（``zone``）。按进程默认时区算会差一天，每个数字看起来都还是对的。"""

        return self._clock().astimezone(self.zone).date()

    def ready(self) -> date | None:
        """这条线现在能跑的那一晚：昨天已经定稿就是今天，否则 None（裁定 94 第 1 条：前一天定稿完成即接着跑）。"""

        today = self.tonight()
        return today if today - timedelta(days=1) in set(self.closed_days()) else None

    async def run_step(
        self,
        step: str,
        night: date,
        *,
        structure_skipped: str | None = None,
        retry_checkpoint: RetryCheckpoint | None = None,
    ) -> StepResult:
        """跑这一晚的一段。``structure_skipped`` 是结构那一段报告里记的"没东西可算"（日子预期那一段据它跳过）；
        ``retry_checkpoint`` 是前置处理上一次没准备好时随 Job 落盘的那一份（换进程、重启之后接着等锁）。"""

        if step not in STEPS:
            raise ValueError(f"unknown nightly step: {step!r}")
        run = _Run(night)
        try:
            if step == PREPARE:
                ready = await self._prepare(run, retry_checkpoint)
            elif step == DAYS:
                ready = await self._days(run, structure_skipped)
            else:
                ready = await self._steps[step](run)
        except _Stop:
            return self._result(step, run, STOPPED, run.stopped)
        except _UNREACHABLE as exc:
            return self._result(step, run, STOPPED, f"{MODEL_UNREACHABLE}: {exc}")
        except Exception as exc:  # noqa: BLE001 - 一段抛了是 bug：这一晚停下、交给健康面（裁定 94 第 8 条）
            return self._result(step, run, STOPPED, f"{_LABELS[step]}出错: {type(exc).__name__}")
        if not ready:
            # 还没准备好：Job 保持排队，下一眼从这一段接着跑
            return self._result(step, run, HELD, run.skipped)
        return self._result(step, run, DONE, None)

    def _result(self, step: str, run: _Run, outcome: str, reason: str | None) -> StepResult:
        if run.skipped is not None:
            run.report.setdefault("skipped", run.skipped)
        checkpoint = self.retry.checkpoint() if step == PREPARE and outcome == HELD and self.retry is not None else None
        return StepResult(step, outcome, reason, dict(run.report), checkpoint)

    def clean_leftovers(self) -> int:
        """常驻节拍一起来就调：上一晚中途被杀剩下的重试项撤回、删掉（裁定 68 第 3 条）。"""

        return 0 if self.retry is None else self.retry.clean_leftovers()

    async def _prepare(self, run: _Run, checkpoint: RetryCheckpoint | None) -> bool:
        """前置处理（裁定 94 第 4 条）：先探模型（整个调不通就抛 ``ModelUnreachable``，不重试）、重读放假数据、再统一看白天
        放一边的错误；重试里（含它的归约）试到模型整个用不了，这一晚停下（报告留着，健康面照样看得见那一晚的计数）。重试那一支的
        锁被另一进程拿着是还没准备好（裁定 94 第 9 条）：停在这一段、不结算，下一眼重试从它停下的地方接着跑（等锁的状态随 Job 落盘，
        ``checkpoint`` 是上一次留下的那一份）。"""

        if self.probe is not None:
            await self.probe()
        run.report["probe"] = "ok" if self.probe is not None else "not_wired"
        calendar_failed = await self._refresh_calendar()
        run.report["calendar_failed"] = calendar_failed
        if self.retry is not None:
            # 这一趟处理前一天及更早的；零点之后新进、属于今天的等明天那一趟（裁定 68 第 3 条）
            through = run.night - timedelta(days=1)
            self.retry.restore(checkpoint)
            run.report["retry_pending"] = await asyncio.to_thread(pending_by_branch, self.retry, through)
            retried = await self.retry.run_once(through=through)
            run.report["retry"] = retry_summary(retried)
            if retried.unreachable:
                run.stopped = f"{MODEL_UNREACHABLE}: the model is unavailable to the retry"
                raise _Stop
            if not retried.ready:
                run.skipped = LOCK_BUSY
                return False
        return True

    async def _settle(self, run: _Run) -> bool:
        if self.settlement is not None:
            settlement = self.settlement
            report = await asyncio.to_thread(settlement.run_once)
            # 待结的预期 = 这一拍结了的 + 那天还没定稿、还挂着的（只给观测，从账本现读）
            still = await asyncio.to_thread(
                lambda: sum(len(settlement.ledger.unsettled_expectations(day)) for day in report.pending_days)
            )
            run.report.update(settlement_summary(report, due=report.settled + still))
        return True

    async def _vocabulary(self, run: _Run) -> bool:
        """各支的每晚新增与定期拆改：锁被另一进程拿着就是还没准备好（已经做完的支再来一次只多问一遍待定池里还没聚组的；
        定期拆改这个周期做过就不再做）。各支的报告进 Job 账（原来被丢掉）。"""

        branches: list[object] = []
        run.report["branches"] = branches
        try:
            for job in self.vocabulary_jobs:
                branches.append(vocabulary_summary(await job(run.night)))
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
        run.report["night"] = night.isoformat()
        before = self.vocabulary()
        if before[1]:
            run.skipped = "词表迁移还没做完"
            return False
        # 全量扫行为树与写机会表是同步 IO：下沉到线程，不卡白天那一拍与同一个循环上的在线请求（2026-10-18 四方评审）
        series = await asyncio.to_thread(self.series.read, cutoff=night)
        if self.vocabulary() != before:
            run.skipped = "读序列时词表变了"
            return False
        run.report["records"] = len(series.records)
        if not series.records:
            # 只有观测空白、一条记录都没有（上游刚接入）：没有可量的东西
            run.skipped = NO_RECORDS
            return True
        scene = await self.scene.run(series=series, now=self._clock())
        run.report["scene"] = scene_summary(scene)
        opportunities = await self._opportunities(series, scene)
        if opportunities is not None:
            run.report["opportunities"] = opportunity_summary(opportunities)
        if opportunities is not None and opportunities.failed:
            # 机会表或单元历史某条 lane 没做成（26 号清单 2.5.1）：同样是出错，停下这一晚
            run.stopped = f"机会表出错: {'; '.join(opportunities.failed)}"
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

    async def _days(self, run: _Run, structure_skipped: str | None) -> bool:
        """当天的日子预期：结构这一段没东西可算就没有可建的；这一晚的表还不是今天的（词表迁移中）是还没准备好。"""

        if self.days is None or structure_skipped in _NOTHING_TO_BUILD:
            run.report["built"] = False
            return True
        days = await self.days.run_once()
        run.report.update(day_summary(days))
        if days.skipped is not None:
            run.skipped = days.skipped
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


__all__ = [
    "DONE",
    "HELD",
    "LOCK_BUSY",
    "MODEL_UNREACHABLE",
    "NO_RECORDS",
    "NOTHING_CLOSED",
    "STEPS",
    "STOPPED",
    "Nightly",
    "StepResult",
]
