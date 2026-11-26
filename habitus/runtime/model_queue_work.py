"""模型队列各类活的欠账（``runtime.model_queue``）：每一类从盘上现读"还欠不欠"，给出这一轮能做的项，顺带把还欠几项记到
``ModelQueueState``。``DayWork`` 是不读序列的三类（出错重试、词表、未封口判断的归类），``SceneWork`` 是语义树的两类
（同步词表、先验的补问），读同一份序列、概念集从盘上现取。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date, datetime

from habitus.behavior.retry import RetryStage
from habitus.runtime.generation_marks import GenerationMarkError, GenerationMarks
from habitus.runtime.generation_summary import retry_summary
from habitus.runtime.model_queue_advice import AdviceWork
from habitus.runtime.model_queue_state import (
    ADVICE,
    DONE,
    RETRY,
    SYNC,
    UNFINISHED,
    UNREACHABLE,
    UNSEALED,
    WAITING,
    Item,
    ModelQueueState,
    Round,
)
from habitus.runtime.scene_night import SceneNightlyRun
from habitus.runtime.unsealed import UnsealedFromJudgements
from habitus.series import EventSeries
from habitus.series.reader import SeriesReader


class DayWork:
    def __init__(
        self,
        *,
        state: ModelQueueState,
        clock: Callable[[], datetime],
        retry: RetryStage | None,
        unsealed: UnsealedFromJudgements | None,
        marks: GenerationMarks | None,
    ) -> None:
        self.state = state
        self.clock = clock
        self.retry = retry
        self.bridge = unsealed  # 预测层那座未封口的桥
        self.marks = marks

    async def retries(self, current: Round) -> list[Item]:
        retry = self.retry
        if retry is None:
            self.state.record(RETRY, 0, current.now)
            return []
        due = await asyncio.to_thread(retry.due, through=current.today)
        self.state.record(RETRY, len(due), current.now, min((item.entered_at for item in due), default=None))
        if not due:
            return []

        async def run(detail: dict[str, object], signals: list[str]) -> str:
            report = await retry.run_once(through=current.today)
            detail.update(retry_summary(report))
            if report.unreachable:
                return UNREACHABLE  # 模型此刻用不了：项留在堆里、仍算欠着，下一轮再试（熔断开着时等试探）
            # 一项都没有定论（都被时窗末重读占着、或锁被占）不算进展：下一轮再来
            return DONE if report.ready and report.recovered + report.invalid + report.deterministic else WAITING

        return [Item(RETRY, None, run)]

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
                raise unsealed.unavailable  # 桥把"模型此刻用不了"吞了：交给队列（这一项留着，熔断开着时等试探）
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
        vocabulary: Callable[[], int],
        marks: GenerationMarks | None = None,
    ) -> None:
        self.state = state
        self.scene = scene
        #: 完成标记（"今天这一代做完了没有"；没接就不等）。
        self.marks = marks
        #: 队列自己的序列读者（按天缓存行为树文档；读者名与换代的不同，各记各的脏日水位）。
        self.reader = reader
        #: 先验的补问（"没有要补问的"时的输入摘要只在进程里）。
        self.asking = AdviceWork(state=state, scene=scene, vocabulary=vocabulary)

    async def series(self, current: Round, night: date) -> EventSeries:
        """这一轮的序列（截止到 ``night``），读一次。"""

        if current.series is None:
            current.series = await asyncio.to_thread(self.reader.read, cutoff=night)
        return current.series

    def generated(self, current: Round) -> bool:
        """今天这一代做完了没有（没接完成标记就不等）。"昨天及以前"的先验补问、记同步到的版本，
        都在今天这一代完成之后才算欠着。"""

        if current.generated is None:
            current.generated = self._generated(current.today)
        return current.generated

    def settled(self, *categories: str) -> bool:
        """排在前面的这几类这一轮都没有欠着的：后一类要用前一类的结果（先验 → 记同步到的版本），前面的没做完
        后面的先不做——补做的与准时做的才是同一份材料。"""

        return not any(self.state.owed.get(category) for category in categories)

    async def sync(self, current: Round, night: date, series: EventSeries) -> list[Item]:
        """记下同步到的词表版本（这一代之后在用状态的修订号还没记、却没有要写的概念时）。概念不在这里写：在用的词的行为概念
        由换代的同步词表生成，所以要等换代把这一版的概念写完（计划里没有要写的概念）才轮到它；排在补问之后。"""

        syncing = self.scene.syncing
        plan = await syncing.plan(series, current.now)
        owed = not plan.write and syncing.owed(plan)
        self.state.record(SYNC, int(owed), current.now)
        if not owed:
            return []

        async def run(detail: dict[str, object], signals: list[str]) -> str:
            synced = await syncing.run(series, current.now, signals)
            detail.update({"version": synced.version, "concepts": len(synced.concepts)})
            return UNFINISHED if syncing.owed(await syncing.plan(series, current.now)) else DONE

        return [Item(SYNC, f"sync:{night.isoformat()}", run, night)]

    def _generated(self, day: date) -> bool:
        if self.marks is None:
            return True
        try:
            return self.marks.finished(day) is True
        except GenerationMarkError:
            return False

    async def advice(self, current: Round, night: date, series: EventSeries) -> list[Item]:
        """先验的补问。"""

        return await self.asking.items(current, night, series)


async def owed(daily: DayWork, sealed: SceneWork, category: str, current: Round) -> list[Item]:
    """这一类此刻欠着的、这一轮能做的项；顺带把"还欠几项"记到状态上。已封口的日子的活在还不能做时不看（队列的一轮里跳过）。"""

    night = current.night
    if category == RETRY:
        return await daily.retries(current)
    assert night is not None
    if category == UNSEALED:
        return await daily.unsealed(current, night)
    # 其余几类都是今天这一代完成之后的活（先后固定：先验 → 记同步到的版本）；
    # 这一代还没做完就不看，原来的数留着
    if not sealed.generated(current):
        return []
    blocking: dict[str, tuple[str, ...]] = {ADVICE: (), SYNC: (ADVICE,)}
    if not sealed.settled(*blocking[category]):
        return []  # 排在前面的类还欠着：先不做，原来的数留着
    series = await sealed.series(current, night)
    if category == SYNC:
        return await sealed.sync(current, night, series)
    return await sealed.advice(current, night, series)


__all__ = ["DayWork", "SceneWork", "owed"]
