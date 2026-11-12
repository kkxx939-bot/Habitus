"""模型队列（实时化方案第 4 节；阶段 2 第一小步）：原来换代里调模型的活都在这里做，一项一项；换代里没有模型调用。

**欠账每一轮从盘上现算，不记"做到哪了"的账**：关机、杀进程之后不恢复任何队列状态，重起就接着做。一轮按固定先后看各类
（后一类要用前一类的结果）；前一类里有一项没做成，后面的类照做（尽力而为）：

==========  ==========================================  ====================================================
 类          欠着的判据（全部现读盘）                      一项是什么
==========  ==========================================  ====================================================
 retry       重试堆里属于还没封口的日子、能重试的项        就近试一遍（``RetryStage.run_once``；会话轮、链各归各的处理器）
 vocabulary  这一晚的每晚新增 / 定期拆改还没做            一支一项（``nightly_vocabulary``：长新类 → 到期的拆改 → 迁移，
             （``VocabularyJobs.due``）                    一次持 sweep 锁做完）
 unsealed    今天这一代还没做、有没归过类的未封口判断      白天归类一次（换代里的日子预期只读它的缓存）
 sync        同步词表的计划不空、或同步到的版本没记        同步词表 + 迁移牵到的重映射（``SyncStage.run``）
 concepts    有 lane 类清单指纹变了、或这一周还没写过      写概念（``SceneNightlyRun.author``）
 mapping     有记录没有命中记录、口径不是现在的、或输入变了  一条记录一项（``MappingStage.map_one``；情境留空，换代封口时算）
 advice      有 lane 要问候选条件、缺先验、提议簿要整理    一条 lane 一项（``RelationStage.ask``）
==========  ==========================================  ====================================================

retry 与 mapping 之外都是**已封口的日子**的活：只在"换代的触发条件里除今天的完成标记之外都满足"时才算欠着（``sealed_day``：
早于这一刻读序列，「未观测」的空白可能还没落树）。mapping 按条、不等封口（实时化阶段 2「映射按槽」）：一条记录最后所见所在的
槽过后再留 ``scene.mapping_margin_slots`` 槽（按预测拍水位算，没接预测层按时钟）、它那一天的那一代做完了，就可以映射；先后是
还没封口的日子里的新记录 → 口径或输入变了要重判的 → 历史回填（最新的先），没有一晚的预算。日完成章不在这里盖：换代的
语义树一步封口时统一重算情境、核条数、盖章；补问候选条件与先验排在今天这一代完成之后（实时化阶段 2「生效时点」），那时昨天已经封口，时间线读盘上盖了章的命中。完成标记有没有不影响映射的欠账——换代不等队列，换代先做完、队列后做是允许的；只有 unsealed
是专为今天这一代的日子预期备的，这一代做完之后归预测拍自己归类。

**一项没做成**分两种。

*不是模型失败的*（代码抛了、做完之后判据还说欠着——模型答了但不合格）：按重试账退避（``RetryLedger``；映射用它自己那一本），
队列接着做下一项——账的键带着这一晚，下一晚重新给机会。锁被另一进程拿着是还没准备好：这一类这一轮到此为止，不算失败。
认欠账时读盘抛了：这一类这一轮记失败、后面的类照做，这一趟做完再抛出来。

*模型调用失败*（到达队列的 ``UNREACHABLE_ERRORS`` 那一组，以及出错重试 / 归约报出的"模型用不了"，不管起因、不分错误
种类、不用探针）只有一条规则：

1. 这一项留着（仍算欠着），按进程内的退避等下一次（逐次加倍，封顶用整体退避的封顶值；进程重起后从头），永不放弃、不记
   落盘的重试账；退避期内不重问它，别的项与后面的类照做。
2. 连着 ``scene.model_queue_unreachable_after_failures`` 项（不同的项）都失败、中间没有任何一项做成——整体退避（逐次加倍、
   封顶可配；期间 ``drain`` 直接返回、不逐项试）。到点后接着做；头一项又失败的话连续数没清零，马上再次整体退避、时长加倍；
   有一项做成才清零、算恢复。整体退避中途返回时，排在后面的各类也从盘上认一遍"还欠几项"（只认、不做）。

同一进程的换代在那一刻问"队列是不是在整体退避"（``ModelQueueState.down``），是就不判关系、沿用上一代（裁定 96）。

**与换代不抢盘**（阶段 2 第二小步）：写语义树的那几类活（sync、concepts、mapping、advice）每一项在语义树写锁之内做，调模型
的那几段先放开锁（``runtime.scene_lock``）；同一进程的换代在做时（``ModelQueueState.generating``），两项之间先不取下一项，等它
做完。队列租约包着整趟，换代不拿它。

**做到没有欠账**：``drain`` 一轮一轮做，直到没有可做的项、或进入整体退避，返回这一趟做了哪些、还欠哪些。常驻 worker
（``runtime.model_queue_worker``）有活连着做、没活按间隔看；回放工具在每个槽的那一拍之前调它。

可查的痕迹：每一项一个 ``model_queue.item`` span（类别、结局、用时；模型调用是它下面的 ``model.call`` 子 span）、一条指标事件、
一行本地流水（``<scene_root>/model_queue/<日期>.jsonl``，带信号文字，只在本地）；各类还欠多少、最老一项等了多久、整体退避的
次数与时长记指标，健康面读 ``ModelQueueState``。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from datetime import UTC, date, datetime, timedelta

from habitus.behavior.reduction import BehaviorReductionBusyError
from habitus.behavior.retry import RetryStage
from habitus.foundation.lifecycle.backoff import BackoffPolicy
from habitus.foundation.observability import NullObserver, Observer
from habitus.foundation.tracing import current_span, open_trace, set_attributes, span
from habitus.infrastructure.lifecycle import RetryLedger
from habitus.runtime.generation_lease import Lease, LeaseBusyError
from habitus.runtime.generation_marks import GenerationMarks
from habitus.runtime.generation_result import LOCK_BUSY, MODEL_UNREACHABLE, UNREACHABLE_ERRORS
from habitus.runtime.generation_summary import flatten, reason_code
from habitus.runtime.generation_trace import TraceFiles
from habitus.runtime.generation_watermark import TickWatermark
from habitus.runtime.model_queue_journal import QueueJournal, QueueRecorder
from habitus.runtime.model_queue_state import (
    ADVICE,
    CATEGORIES,
    CONCEPTS,
    DONE,
    FAILED,
    ITEM_SPAN,
    MAPPING,
    RETRY,
    SYNC,
    UNFINISHED,
    UNREACHABLE,
    UNSEALED,
    VOCABULARY,
    WAITING,
    DrainReport,
    Item,
    ItemResult,
    ItemUnfinished,
    ModelQueueState,
    QueueSettings,
    Round,
    VocabularyBranch,
)
from habitus.runtime.model_queue_work import DayWork, SceneWork, owed
from habitus.runtime.scene_lock import SceneWriteLock
from habitus.runtime.scene_night import SceneNightlyRun
from habitus.runtime.unsealed import UnsealedFromJudgements
from habitus.series.reader import SeriesReader

logger = logging.getLogger("habitus.model_queue")

#: 一趟 ``drain`` 最多几轮（保护闸：正常两三轮就没有可做的了）。
_MAX_ROUNDS = 64
#: 换代在做时队列隔多久看一眼它做完没有（进程内的一个标志，不读盘）。
_GENERATING_POLL_SECONDS = 0.2
#: 写语义树的那几类活：一项在语义树写锁之内做。
_SCENE_CATEGORIES = frozenset({SYNC, CONCEPTS, MAPPING, ADVICE})
#: 不等日子封口就能做的类：出错重试、映射（按条，所在的槽过后留够余量就做）。
_UNSEALED_CATEGORIES = frozenset({RETRY, MAPPING})


class ModelQueue:
    def __init__(
        self,
        *,
        scene: SceneNightlyRun,
        series: SeriesReader,
        today: Callable[[], date],
        sealed_day: Callable[[], date | None],
        vocabulary: Callable[[], tuple[int, bool]],
        retries: RetryLedger,
        lease: Lease,
        state: ModelQueueState | None = None,
        scene_lock: SceneWriteLock | None = None,
        marks: GenerationMarks | None = None,
        retry: RetryStage | None = None,
        vocabulary_branches: Sequence[VocabularyBranch] = (),
        unsealed: UnsealedFromJudgements | None = None,
        settings: QueueSettings | None = None,
        traces: TraceFiles | None = None,
        journal: QueueJournal | None = None,
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
        watermark: Callable[[], TickWatermark] | None = None,
    ) -> None:
        if series.tree is not scene.behavior_tree:
            raise ValueError("the model queue must read the scene's behaviour tree")
        self.scene = scene
        self.today = today
        self.sealed_day = sealed_day
        #: 队列的重试账：一项没做成按它退避（映射另用语义树那一本，键带映射口径）。
        self.retries = retries
        #: 队列租约（语义树根派生）：两个进程挂同一个根时同一时刻只有一个在做。
        self.lease = lease
        self.state = state if state is not None else ModelQueueState()
        #: 语义树写锁：写语义树的那几类活每一项在它里面做，调模型的那几段除外（``runtime.scene_lock``）。没接就不持。
        self.scene_lock = scene_lock
        self.marks = marks
        self.settings = settings or QueueSettings()
        self.traces = traces
        self.journal = journal
        #: 指标、日志、本地流水与留存清理（都是旁路）。
        self.recorder = QueueRecorder(
            observer if observer is not None else NullObserver(), journal, marks, self.settings.kept_days
        )
        self.clock: Callable[[], datetime] = clock if clock is not None else lambda: datetime.now(UTC)
        #: 这一趟里认欠账时第一个读盘的错（做完这一趟再抛）。
        self._scan_error: Exception | None = None
        self._backoff = BackoffPolicy(self.settings.backoff_base_seconds, self.settings.backoff_max_seconds)
        #: 各类的欠账怎么从盘上认、一项怎么做（``runtime.model_queue_work``）。
        self.daily = DayWork(
            state=self.state,
            clock=self.clock,
            retry=retry,
            vocabulary_branches=vocabulary_branches,
            unsealed=unsealed,
            marks=marks,
        )
        self.sealed = SceneWork(
            state=self.state, scene=scene, reader=series, vocabulary=vocabulary,
            settings=self.settings, marks=marks, watermark=watermark,
        )  # fmt: skip

    # ── 做到没有欠账 ─────────────────────────────────────────────────────────

    async def drain(self, *, stop: Callable[[], bool] | None = None) -> DrainReport:
        """做到没有欠账、或进入整体退避为止（``stop`` 为真就在两项之间停下）。返回这一趟做了哪些项、各类还欠着几项、整体退避
        到什么时候。队列租约被另一个进程拿着时什么都不做（``held_elsewhere``）。有的类认欠账时读盘出错：后面的类照做，
        这一趟做完之后把第一个这样的错抛出来（不静默；常驻循环记成这一趟出错，原因码另在 ``state.errors`` 与健康面上）。"""

        done: list[ItemResult] = []
        self._scan_error = None
        try:
            async with self.lease.held(wait_seconds=0.0):
                for _ in range(_MAX_ROUNDS):
                    if not await self._round(done, stop or (lambda: False)):
                        break
        except LeaseBusyError:
            return DrainReport(tuple(done), self.state.owed_now(), self.state.retry_at, held_elsewhere=True)
        if self._scan_error is not None:
            raise self._scan_error
        state = self.state
        return DrainReport(tuple(done), state.owed_now(), state.retry_at if state.down() else None)

    def wait_seconds(self) -> float:
        """下一次最早隔多久再看：平时按间隔，整体退避时等到退避到点。"""

        retry_at = self.state.retry_at
        if not self.state.down() or retry_at is None or retry_at <= self.clock():
            return self.settings.poll_seconds
        return max(1.0, min(self.settings.poll_seconds, (retry_at - self.clock()).total_seconds()))

    async def _round(self, done: list[ItemResult], stop: Callable[[], bool]) -> bool:
        """一轮：从盘上认各类的欠账、按先后一项一项做。返回这一轮有没有做成的项（有就再来一轮：后面的欠账可能变了）。"""

        state = self.state
        now = self.clock()
        if state.down() and state.retry_at is not None and now < state.retry_at:
            return False  # 整体退避期内不逐项试；到点就接着做
        today = await asyncio.to_thread(self.today)
        current = Round(now=now, today=today, night=await asyncio.to_thread(self.sealed_day))
        state.blocked = None
        progressed = False
        for category in CATEGORIES:
            if category not in _UNSEALED_CATEGORIES and current.night is None:
                continue  # 已封口的日子的活现在还不能做：不看，原来的数留着（跨零点不清，清完才归零）
            for item in await self._scan(category, current):
                if stop() or not await self._generation_done(stop):
                    return False
                if item.key is not None and await asyncio.to_thread(self.retries.blocked, item.key):
                    continue  # 在退避或已放弃：这一晚不再排，账的键带着这一晚
                if state.waiting(item.name, now):
                    continue  # 上一次模型失败、还在进程内的退避期里：不重问它，仍算欠着
                result = await self._execute(item, current)
                done.append(result)
                if result.outcome == UNREACHABLE and self._model_failed(item):
                    return await self._halted(category, current)
                if result.outcome == WAITING:
                    break  # 锁被另一进程拿着：这一类这一轮到此为止
                if result.outcome == DONE:
                    progressed = True
                    state.settled(category)
                    state.streak.clear()  # 有一项做成：连着失败的数清零；在整体退避里的话这才算恢复
                    state.waits.pop(item.name, None)
                    if state.down():
                        self._recovered()
        state.scanned_at = now
        if state.down() and not any(state.owed.values()):
            self._recovered()  # 没有欠着的活了：没有什么可等的
        await asyncio.to_thread(self.recorder.round, state, current)
        return progressed

    async def _generation_done(self, stop: Callable[[], bool]) -> bool:
        """两项之间：同一进程的换代正在做就先不取下一项，等它做完（让它尽快拿到语义树写锁、读盘时盘不动）。返回假是等的
        时候被叫停了。"""

        while self.state.generating:
            if stop():
                return False
            await asyncio.sleep(_GENERATING_POLL_SECONDS)
        return True

    async def _scan(self, category: str, current: Round) -> list[Item]:
        """认这一类的欠账；读盘抛了（坏文件、缓存不是规范 JSON……）这一类这一轮记失败、报出原因码，后面的类照做。"""

        try:
            items = await owed(self.daily, self.sealed, category, current)
        except Exception as exc:  # noqa: BLE001 - 一类认不出欠账不挡后面的类；原因码进状态与健康面
            self.state.errors[category] = type(exc).__name__
            self._scan_error = self._scan_error or exc
            self.recorder.scan_failed(category, type(exc).__name__)
            return []
        self.state.errors.pop(category, None)
        return items

    def _model_failed(self, item: Item) -> bool:
        """一项模型调用失败（见模块文档串）：这一项留着、不记重试账，按进程内的退避等下一次。返回要不要整体退避——
        连着配置的项数（不同的项）都失败、中间没有一项做成。"""

        state, now = self.state, self.clock()
        names = tuple(item.stuck()) if item.stuck is not None else ()
        for name in names or (item.name,):
            count = state.waits.get(name, (0, now))[0] + 1
            state.waits[name] = (count, now + timedelta(seconds=self._backoff.delay(count)))
            state.streak.add(name)
        if len(state.streak) < self.settings.unreachable_after_failures:
            return False
        for name in state.streak:
            state.waits.pop(name, None)  # 整体退避到点就从它们做起，不再各等各的
        self._back_off(now)
        return True

    async def _halted(self, category: str, current: Round) -> bool:
        """整体退避、这一轮中途到此为止：排在后面的各类也从盘上认一遍"还欠几项"再记（只认、不做），换代报告与健康面
        读到的才是全的。总返回假。"""

        for later in CATEGORIES[CATEGORIES.index(category) + 1 :]:
            if later in _UNSEALED_CATEGORIES or current.night is not None:
                await self._scan(later, current)
        await asyncio.to_thread(self.recorder.round, self.state, current)
        return False

    # ── 模型整个调不通：整体退避 ─────────────────────────────────────────────

    def _back_off(self, now: datetime) -> bool:
        """整体退避一次（逐次加倍、封顶）；总返回假（调用处直接当"这一轮到此为止"用）。"""

        state = self.state
        state.failures += 1
        delay = self._backoff.delay(state.failures)
        if state.unreachable_since is None:
            state.unreachable_since = now
        state.retry_at = now + timedelta(seconds=delay)
        state.backoffs += 1
        state.backoff_seconds += delay
        self.recorder.backoff(state.failures, delay)
        return False

    def _recovered(self) -> None:
        state = self.state
        since = state.unreachable_since
        state.unreachable_since = state.retry_at = None
        state.failures = 0
        self.recorder.recovered(0.0 if since is None else max(0.0, (self.clock() - since).total_seconds()))

    # ── 做一项 ───────────────────────────────────────────────────────────────

    def _writes(self, item: Item) -> AbstractAsyncContextManager[object]:
        if self.scene_lock is None or item.category not in _SCENE_CATEGORIES:
            return nullcontext()
        return self.scene_lock.writing()

    async def _execute(self, item: Item, current: Round) -> ItemResult:
        """做一项：一个 span、记结局；没做成按重试账退避。取消（停机超时）原样抛：这一项下次从盘上重新认出来。"""

        started_at, started = self.clock(), time.monotonic()
        detail: dict[str, object] = {}
        signals: list[str] = []
        reason: str | None = None
        error: BaseException | None = None
        calls = 0
        trace = open_trace(self.traces.sink(current.today)) if self.traces is not None else nullcontext()
        with (
            trace,
            span(ITEM_SPAN, {"category": item.category, "day": (item.day or current.today).isoformat()}) as found,
        ):
            try:
                async with self._writes(item):
                    outcome = await item.run(detail, signals)
            except BehaviorReductionBusyError:
                outcome, reason = WAITING, LOCK_BUSY
            except UNREACHABLE_ERRORS as exc:
                outcome, reason = UNREACHABLE, f"{MODEL_UNREACHABLE}: {exc}"
            except Exception as exc:  # noqa: BLE001 - 一项抛了不挡后面的项：按重试账退避、报出
                outcome, reason, error = FAILED, type(exc).__name__, exc
            record = current_span()
            if record is not None:
                calls = sum(record.model_calls.values())
            set_attributes(found, {"outcome": outcome, "reason": reason_code(reason), **flatten(detail)})
        duration = max(0.0, time.monotonic() - started)
        if item.key is not None:
            if outcome == DONE:
                await asyncio.to_thread(self.retries.clear, item.key)
            elif outcome in (FAILED, UNFINISHED):
                failure = error if error is not None else ItemUnfinished(f"{item.category} is still owed")
                await asyncio.to_thread(self.retries.record_failure, item.key, failure)
        result = ItemResult(item.category, outcome, reason, started_at, duration, calls, detail)
        self.recorder.item(result, item, signals)
        return result


__all__ = [
    "ADVICE",
    "CATEGORIES",
    "CONCEPTS",
    "DONE",
    "FAILED",
    "ITEM_SPAN",
    "MAPPING",
    "RETRY",
    "SYNC",
    "UNFINISHED",
    "UNREACHABLE",
    "UNSEALED",
    "VOCABULARY",
    "WAITING",
    "DrainReport",
    "ItemResult",
    "ItemUnfinished",
    "ModelQueue",
    "ModelQueueState",
    "QueueJournal",
    "QueueSettings",
    "VocabularyBranch",
]
