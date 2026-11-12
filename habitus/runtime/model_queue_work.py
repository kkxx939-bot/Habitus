"""模型队列各类活的欠账（``runtime.model_queue``）：每一类从盘上现读"还欠不欠"，给出这一轮能做的项，顺带把还欠几项记到
``ModelQueueState``。``DayWork`` 是不读序列的三类（出错重试、词表、未封口判断的归类），``SceneWork`` 是语义树的四类
（同步词表、写概念、映射、候选条件与先验的补问），读同一份序列、材料从盘上现取。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, tzinfo
from datetime import time as clock_time

from habitus.behavior.retry import RetryStage
from habitus.runtime.generation_marks import GenerationMarkError, GenerationMarks
from habitus.runtime.generation_result import MIGRATING, VOCABULARY_MOVED
from habitus.runtime.generation_summary import retry_summary, vocabulary_summary
from habitus.runtime.generation_watermark import TickWatermark
from habitus.runtime.model_queue_advice import AdviceWork
from habitus.runtime.model_queue_state import (
    ADVICE,
    CONCEPTS,
    DONE,
    MAPPING,
    RETRY,
    SYNC,
    UNFINISHED,
    UNREACHABLE,
    UNSEALED,
    VOCABULARY,
    WAITING,
    Item,
    ModelQueueState,
    QueueSettings,
    Round,
    VocabularyBranch,
)
from habitus.runtime.scene_night import SceneNightlyRun
from habitus.runtime.scene_night_mapping import PendingRecord
from habitus.runtime.unsealed import UnsealedFromJudgements
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.occurrences.daily import mappable_from
from habitus.scene.occurrences.mapper import ConceptMapper
from habitus.scene.occurrences.model import UnresolvedReason
from habitus.series import EventSeries, SeriesRecord
from habitus.series.reader import SeriesReader


class DayWork:
    def __init__(
        self,
        *,
        state: ModelQueueState,
        clock: Callable[[], datetime],
        retry: RetryStage | None,
        vocabulary_branches: Sequence[VocabularyBranch],
        unsealed: UnsealedFromJudgements | None,
        marks: GenerationMarks | None,
    ) -> None:
        self.state = state
        self.clock = clock
        self.retry = retry
        self.vocabulary_branches = tuple(vocabulary_branches)
        self.bridge = unsealed  # 预测层那座未封口的桥
        self.marks = marks

    async def retries(self, current: Round) -> list[Item]:
        retry = self.retry
        if retry is None:
            self.state.record(RETRY, 0, current.now)
            return []
        due = await asyncio.to_thread(retry.due, through=current.today)
        self.state.record(RETRY, len(due), current.now, min((item.entered_at for item in due), default=None))
        # 上一次卡在"模型用不了"上的那几项在进程里退避着：这一遍不试它们（项留在堆里、仍算欠着），别的照试
        resting = self.state.waiting_names(current.now)
        waiting = frozenset(item.item_id for item in due if _retry_name(item.item_id) in resting)
        if len(waiting) == len(due):
            return []
        stuck: list[tuple[str, ...]] = [()]

        async def run(detail: dict[str, object], signals: list[str]) -> str:
            report = await retry.run_once(through=current.today, skip=waiting)
            detail.update(retry_summary(report))
            if report.unreachable:
                stuck[0] = tuple(_retry_name(identity) for identity in report.stuck_items)  # 卡在哪几项上
                return UNREACHABLE
            # 一项都没有定论（都被时窗末重读占着、或锁被占）不算进展：下一轮再来
            return DONE if report.ready and report.recovered + report.invalid + report.deterministic else WAITING

        return [Item(RETRY, None, run, stuck=lambda: stuck[0])]

    async def vocabulary(self, current: Round, night: date) -> list[Item]:
        due = [branch for branch in self.vocabulary_branches if await asyncio.to_thread(branch.due, night)]
        self.state.record(VOCABULARY, len(due), current.now)
        self.state.pending["revisions"] = sum(
            1 for branch in self.vocabulary_branches if branch.revision is not None and branch.revision()
        )

        def item(branch: VocabularyBranch) -> Item:
            async def run(detail: dict[str, object], signals: list[str]) -> str:
                summary = vocabulary_summary(await branch.run(night))
                noted = summary.pop("signals", None)
                if isinstance(noted, list):
                    signals.extend(str(signal) for signal in noted)
                detail.update(summary)
                current.series = current.live = None  # 树上的编号可能改了：后面的类重新读序列
                return UNFINISHED if await asyncio.to_thread(branch.due, night) else DONE

            return Item(VOCABULARY, f"vocabulary:{branch.name}:{night.isoformat()}", run, night)

        return [item(branch) for branch in due]

    async def unsealed(self, current: Round, night: date) -> list[Item]:
        unsealed = self.bridge
        count = 0
        if unsealed is not None and not self._generated(night):
            count = await asyncio.to_thread(unsealed.unclassified)
        self.state.record(UNSEALED, count, current.now)
        if unsealed is None or not count:
            return []

        async def run(detail: dict[str, object], signals: list[str]) -> str:
            await unsealed.prepare(until=self.clock())
            if unsealed.unavailable is not None:
                raise unsealed.unavailable  # 桥把"模型此刻用不了"吞了：交给队列按错误种类处理（整体退避，不磨这一项）
            left = await asyncio.to_thread(unsealed.unclassified)
            detail.update({"judgements": count, "unclassified": left})
            return UNFINISHED if left else DONE

        return [Item(UNSEALED, f"unsealed:{night.isoformat()}", run, night)]

    def _generated(self, night: date) -> bool:
        """今天这一代做完了没有（完成标记读不了当没做：多归一次类不要紧）。"""

        if self.marks is None:
            return False
        try:
            return self.marks.finished(night) is True
        except GenerationMarkError:
            return False


class SceneWork:
    def __init__(
        self,
        *,
        state: ModelQueueState,
        scene: SceneNightlyRun,
        reader: SeriesReader,
        vocabulary: Callable[[], tuple[int, bool]],
        settings: QueueSettings | None = None,
        marks: GenerationMarks | None = None,
        watermark: Callable[[], TickWatermark] | None = None,
    ) -> None:
        self.state = state
        self.scene = scene
        self.settings = settings or QueueSettings()
        #: 完成标记（一条还没封口的记录要它那一天的那一代做完才映射；没接就不等）与预测拍水位（"过了几槽"按它算；没接按时钟）。
        self.marks, self.watermark = marks, watermark
        #: 队列自己的序列读者（按天缓存行为树文档；读者名与换代的不同，各记各的脏日水位）。
        self.reader = reader
        self.vocabulary = vocabulary  # 词表此刻的（版本, 有没有未完成的迁移）
        #: 候选条件与先验"没有要补问的"时的输入摘要（只在进程里）：输入没变就不重算时间线。
        self.asking = AdviceWork(state=state, scene=scene, vocabulary=vocabulary)

    async def series(self, current: Round, night: date) -> EventSeries | None:
        """这一轮的序列（截止到 ``night``），读一次；与换代读序列同一个挡法：词表迁移中、或读的时候词表变了就不读。"""

        if current.series is not None or self.state.blocked is not None:
            return current.series
        before = self.vocabulary()
        if before[1]:
            self.state.blocked = MIGRATING
            return None
        series = await asyncio.to_thread(self.reader.read, cutoff=night)
        if self.vocabulary() != before:
            self.state.blocked = VOCABULARY_MOVED
            return None
        current.series = series
        return series

    async def _materials(
        self, current: Round, series: EventSeries, *, rhythms: bool
    ) -> tuple[ConceptSet, ConceptMapper, Mapping[str, Rhythm]]:
        """盘上现有的概念集与据它建的映射器；``rhythms`` 时连节律一起算（映射的材料）。一轮里概念集没变就不重取。"""

        if current._concepts is None:
            current._concepts = await asyncio.to_thread(self.scene.concepts.read_all)
        if rhythms and current._rhythms is None:
            current._concepts, _mapper, current._rhythms = await self.scene.materials(series)
        return current._concepts, self.scene.mapper_for(current._concepts), current._rhythms or {}

    def generated(self, current: Round) -> bool:
        """今天这一代做完了没有（没接完成标记就不等）。"昨天及以前"的词表整理活、写概念、候选条件与先验的补问、迁移牵到的
        重判，都在今天这一代完成之后才算欠着（实时化阶段 2「生效时点」）。"""

        if current.generated is None:
            current.generated = self._generated(current.today)
        return current.generated

    def settled(self, *categories: str) -> bool:
        """排在前面的这几类这一轮都没有欠着的：后一类要用前一类的结果（长新类 → 写概念 → 先验 → 候选条件），前面的没做完
        后面的先不做——补做的与准时做的才是同一份材料。"""

        return not any(self.state.owed.get(category) for category in categories)

    async def sync(self, current: Round, night: date, series: EventSeries) -> list[Item]:
        """迁移牵到的已映射日子重判、记下同步到的词表版本。概念不在这里写：新类的基础概念由换代的同步词表生成，所以要等
        换代把这一版的概念写完（计划里没有要写的概念）才轮到它；排在补问之后（重判会撤那几天的章，补问要对着今天这一代的
        时间线问）。"""

        syncing = self.scene.syncing
        plan = await syncing.plan(series, current.now)
        owed = not plan.write and syncing.owed(plan)
        self.state.record(SYNC, int(owed), current.now)
        if not owed:
            return []

        async def run(detail: dict[str, object], signals: list[str]) -> str:
            synced = await syncing.run(series, current.now, signals, remap=True)
            current.forget()
            detail.update(
                {"version": synced.version, "concepts": len(synced.concepts), "remapped_days": synced.remapped}
            )
            return UNFINISHED if syncing.owed(await syncing.plan(series, current.now)) else DONE

        return [Item(SYNC, f"sync:{night.isoformat()}", run, night)]

    async def concepts(self, current: Round, night: date, series: EventSeries) -> list[Item]:
        scene = self.scene
        authoring = scene.authoring

        def lanes() -> tuple[str, ...]:
            return () if authoring is None else authoring.due_lanes(scene.concepts, scene.catalog.classes(), night)

        due = await asyncio.to_thread(lanes)
        self.state.record(CONCEPTS, len(due), current.now)
        self.state.pending["concepts"] = len(await asyncio.to_thread(scene.concepts.pending))
        if not due:
            return []

        async def run(detail: dict[str, object], signals: list[str]) -> str:
            written = await scene.author(series, signals)  # 写出的待生效，下一次换代并入
            left = await asyncio.to_thread(lanes)
            self.state.pending["concepts"] = len(await asyncio.to_thread(scene.concepts.pending))
            detail.update({"lanes": len(due), "lanes_left": len(left), "written": written})
            return UNFINISHED if left else DONE

        return [Item(CONCEPTS, f"concepts:{night.isoformat()}", run, night)]

    async def live(self, current: Round) -> EventSeries | None:
        """含今天的序列（截止到明天）：映射读它——当天的记录不等封口。与 ``series`` 同一个挡法（词表迁移中、读的时候词表
        变了就不读）。"""

        if current.live is not None or self.state.blocked is not None:
            return current.live
        before = self.vocabulary()
        if before[1]:
            self.state.blocked = MIGRATING
            return None
        live = await asyncio.to_thread(self.reader.read, cutoff=current.today + timedelta(days=1))
        if self.vocabulary() != before:
            self.state.blocked = VOCABULARY_MOVED
            return None
        current.live = live
        return live

    def _ready(self, current: Round) -> Callable[[SeriesRecord], bool]:
        """还没封口的日子里的一条记录现在能不能映射：它那一天的那一代做完了（历史的那部分材料冻住了），且它最后所见所在的
        槽过后留够了余量——"过了几槽"按预测拍水位算（与换代的余量同一种写法），没接预测层时按时钟。"""

        passed = current.now if self.watermark is None else self.watermark().ticked_at
        settings = self.settings
        generated: dict[date, bool] = {}

        def ready(entry: SeriesRecord) -> bool:
            if passed is None:
                return False
            if entry.day not in generated:
                generated[entry.day] = self._generated(entry.day)
            return generated[entry.day] and passed >= self._ready_at(entry, settings)

        return ready

    @staticmethod
    def _ready_at(entry: SeriesRecord, settings: QueueSettings) -> datetime:
        return mappable_from(
            entry.last_observed_at, slot_minutes=settings.slot_minutes, margin_slots=settings.mapping_margin_slots
        )

    def _generated(self, day: date) -> bool:
        if self.marks is None:
            return True
        try:
            return self.marks.finished(day) is True
        except GenerationMarkError:
            return False

    async def mapping(self, current: Round) -> list[Item]:
        """映射按条：还没封口的日子里的新记录（所在的槽过后留够了余量）→ 口径或输入变了要重判的 → 历史回填（最新的先）。
        情境不在这里算（换代封口时统一算）。"""

        scene = self.scene
        stage = scene.mapping
        live = await self.live(current)
        if live is None:
            return []  # 词表迁移没做完、或读序列时词表变了：这一轮看不了，原来的数留着
        concepts = await asyncio.to_thread(scene.concepts.read_all)
        mapper = scene.mapper_for(concepts)
        pending = await asyncio.to_thread(
            stage.plan, live, concepts, mapper, sealed_before=current.night, ready=self._ready(current)
        )
        if not (self.generated(current) and self.settled(VOCABULARY, CONCEPTS, ADVICE, SYNC)):
            # 已盖章的日子的重判排在今天这一代与补问之后：重判会撤那几天的章，补问要对着今天这一代的时间线问
            pending = tuple(item for item in pending if not (item.rejudged and item.sealed))
        oldest = min(
            (
                _day_end(item.entry.day, scene.timezone) if item.sealed else self._ready_at(item.entry, self.settings)
                for item in pending
            ),
            default=None,
        )
        self.state.record(MAPPING, len(pending), current.now, oldest)

        def item(record: PendingRecord) -> Item:
            async def run(detail: dict[str, object], signals: list[str]) -> str:
                mapped = await stage.map_one(record)
                reasons = list(mapped.unresolved.values())
                failed = reasons.count(UnresolvedReason.MODEL_FAILED)
                detail.update(
                    {
                        "records": 1,
                        "rejudged": int(record.rejudged),
                        "sealed": record.sealed,
                        "unresolved": len(reasons),
                        "model_failed": failed,
                        "inconsistent": reasons.count(UnresolvedReason.INCONSISTENT),
                    }
                )
                return UNFINISHED if failed else DONE

            return Item(MAPPING, record.key, run, record.entry.day)

        return [item(record) for record in pending]

    async def advice(self, current: Round, night: date, series: EventSeries) -> list[Item]:
        concepts, _mapper, _rhythms = await self._materials(current, series, rhythms=False)
        return await self.asking.items(current, night, series, concepts)


def _day_end(day: date, zone: tzinfo) -> datetime:
    """那一天在当地结束的时刻：它的活从这时起才可能欠着。"""

    return datetime.combine(day + timedelta(days=1), clock_time(0), tzinfo=zone)


def _retry_name(identity: str) -> str:
    """出错重试里一项在进程里的名字（模型失败后的退避按它记）。"""

    return f"retry:{identity}"


async def owed(daily: DayWork, sealed: SceneWork, category: str, current: Round) -> list[Item]:
    """这一类此刻欠着的、这一轮能做的项；顺带把"还欠几项"记到状态上。已封口的日子的活在还不能做时不看（队列的一轮里跳过）。"""

    night = current.night
    if category == RETRY:
        return await daily.retries(current)
    if category == MAPPING:
        return await sealed.mapping(current)  # 按条：当天的记录不等封口
    assert night is not None
    if category == UNSEALED:
        return await daily.unsealed(current, night)
    # 其余几类都是今天这一代完成之后的活（先后固定：长新类 → 到期的拆改 → 写概念 → 先验 → 候选条件 → 迁移牵到的重判）；
    # 这一代还没做完就不看，原来的数留着
    if not sealed.generated(current):
        return []
    if category == VOCABULARY:
        return await daily.vocabulary(current, night)
    blocking = {CONCEPTS: (VOCABULARY,), ADVICE: (VOCABULARY, CONCEPTS), SYNC: (VOCABULARY, CONCEPTS, ADVICE)}
    if not sealed.settled(*blocking[category]):
        return []  # 排在前面的类还欠着：先不做，原来的数留着
    series = await sealed.series(current, night)
    if series is None:
        return []  # 词表迁移没做完、或读序列时词表变了：这一轮看不了，原来的数留着
    if category == SYNC:
        return await sealed.sync(current, night, series)
    if category == CONCEPTS:
        return await sealed.concepts(current, night, series)
    return await sealed.advice(current, night, series)


__all__ = ["DayWork", "SceneWork", "owed"]
