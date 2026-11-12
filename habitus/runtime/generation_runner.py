"""换代的那一眼（实时化方案第 3、9、10 节；阶段 2 第二小步）：不是常驻 worker——预测 worker 每个槽边界、在预测根租约之内
先调它看一眼，再做预测拍（``runtime.foresight_worker``）。没有自己的循环、没有自己的租约："同一个根同一天只换一次"由预测根租约
加完成标记保证。

一眼（``look``）：

1. 进程起来后第一眼：上一次中途被杀剩下的重试项撤回（裁定 68 第 3 条，``clean_leftovers``；等锁的状态只在进程里）；
2. 看能不能换代（``Generation.ready``）：连续定稿到昨天、预测拍水位过了余量、今天的完成标记还没有。定稿日不连续的那一天连着
   ``generation_stuck_ticks`` 眼都没好，``stuck`` 记下它（健康面报出）；
3. 能就做一代（``Generation.run``）。必须成的那一步没成：按退避（``scene.generation_retry_*``）在之后的槽再试；还没准备好（锁被
   占）不算失败、不退避，下一槽再看。

``slot`` 是预测 worker 调的口：一眼抛了只记 ``last_error`` 与观测，不挡这一槽的预测拍。

可查的痕迹：这一代一个 ``generation.run`` span、每步一个子 span（落本地 trace 文件，一天一份）；每步一条指标事件（耗时、结局）、
每代一条（写成没有、多久、从昨天定稿到写出隔多久）、每个退路一条计数；开始 / 结束各一行 INFO 日志；每代一份报告（没写成的
也写），写成时算一次关系表的双向变化。健康面读这里的 ``last_result``、``stuck``、``rewound``、``retry_at``、``last_error`` 与最新
的完成标记。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from habitus.foundation.lifecycle.backoff import BackoffPolicy
from habitus.foundation.observability import NullObserver, ObservationStatus, Observer, record_observation
from habitus.foundation.tracing import open_trace
from habitus.runtime.generation import Generation
from habitus.runtime.generation_report import GenerationReports
from habitus.runtime.generation_result import ALREADY_DONE, DEGRADED, FAILED, FALLBACKS, HELD, GenerationResult
from habitus.runtime.generation_trace import TraceFiles
from habitus.runtime.scene_rhythms import prune_rhythms

logger = logging.getLogger("habitus.generation")

#: 一步的结局在观测上的状态：退路记降级、没成记失败。
_OUTCOME_STATUS = {
    "done": ObservationStatus.SUCCESS,
    "skipped": ObservationStatus.SUCCESS,
    DEGRADED: ObservationStatus.DEGRADED,
    FAILED: ObservationStatus.FAILURE,
    HELD: ObservationStatus.SUCCESS,
}


@dataclass(frozen=True)
class GenerationSettings:
    """换代的运维数（``config.scene.generation_*``，组合根换算过来）。"""

    kept_days: int = 366
    stuck_ticks: int = 3
    #: 必须成的那一步没成之后的退避（用户 11-10 失败重试第 2 条：5 分钟起、逐次加倍、封顶 1 小时，同一天一直重试、不放弃）。
    retry_base_seconds: float = 300.0
    retry_max_seconds: float = 3_600.0


class GenerationRunner:
    def __init__(
        self,
        generation: Generation,
        *,
        settings: GenerationSettings | None = None,
        reports: GenerationReports | None = None,
        traces: TraceFiles | None = None,
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.generation = generation
        self.settings = settings or GenerationSettings()
        self.reports = reports
        self.traces = traces
        self.observer: Observer = observer if observer is not None else NullObserver()
        self.worker_id = worker_id or f"generation-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self._clock = clock or (lambda: datetime.now(UTC))
        #: 进程起来后清过上一次剩下的重试项没有（第一眼清一次）。
        self._cleaned = False
        #: 最近一次做一代的结果（健康面读：没写成的那一代停在哪一步、写成了走了哪些退路）。
        self.last_result: GenerationResult | None = None
        #: 最近一次出的错：一眼抛了、或这一代没写成、走了退路（健康面报；之后一眼顺利就清掉）。
        self.last_error: BaseException | None = None
        #: ``last_error`` 是不是一眼抛出来的（不是这一代的结果）：下一眼顺利就清掉。
        self._look_failed = False
        #: 定稿日不连续卡住的那一天，与连着几眼都是它。
        self._stuck: tuple[date, int] | None = None
        #: 最近一眼当地日期回到了已换代的日子之前：（今天, 最晚完成标记的日子）；日期追上后下一眼清掉（健康面报）。
        self.rewound: tuple[date, date] | None = None
        self._retries = _Retries(BackoffPolicy(self.settings.retry_base_seconds, self.settings.retry_max_seconds))

    @property
    def retry_at(self) -> datetime | None:
        """今天这一代没成、正在退避时下一次最早什么时候再做；没在退避是 None。"""

        return self._retries.at

    @property
    def stuck(self) -> date | None:
        """定稿日不连续、连着 ``stuck_ticks`` 眼都卡在同一天时是那一天，否则 None（正常只持续一轮归约）。"""

        if self._stuck is None or self._stuck[1] < self.settings.stuck_ticks:
            return None
        return self._stuck[0]

    @property
    def look_error(self) -> BaseException | None:
        """看那一眼抛出来的错（不是某一代的结果）；下一眼顺利就没有了。"""
        return self.last_error if self._look_failed else None

    def finished(self, day: date) -> bool | None:
        """这一天的换代做完了没有（完成标记的三值，``GenerationMarks.finished``）。"""

        return self.generation.marks.finished(day)

    async def slot(self) -> None:
        """预测 worker 每槽调的口（已在预测根租约之内）：看一眼；抛了只记下、报出，不挡这一槽的预测拍。取消原样抛。"""

        started = time.monotonic()
        try:
            result = await self.look()
        except Exception as exc:  # noqa: BLE001 - 换代这一眼出错不挡预测拍；下一槽从头看
            self.last_error, self._look_failed = exc, True
            self._observe("look", ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started=started)
            return
        if self._look_failed:
            self._look_failed = False
            if result is None:
                self.last_error = None  # 上一眼抛的错过去了（这一眼做了一代的话 ``_after`` 已按结果重写）

    async def look(self) -> GenerationResult | None:
        """看一眼：能换代就做今天这一代并返回结果；不能换代、还在退避是 None。调用方持预测根租约。"""

        if not self._cleaned:
            self._cleaned = True
            await asyncio.to_thread(self.generation.clean_leftovers)
        readiness = await asyncio.to_thread(self.generation.ready)
        self._track_stuck(readiness.stuck)
        self.rewound = readiness.rewound
        if readiness.reason == ALREADY_DONE and self.last_result is not None and not self.last_result.complete:
            # 今天这一代已有完成标记（另一个进程写成的、或日期已越过没写成的那一天）：上一次没写成的那一代过时了，
            # 健康面不再挂着它
            self.last_result = None
            self.last_error = None
        if readiness.day is None or self._retries.waiting(readiness.day, self._clock()):
            return None
        result = await self._generate(readiness.day)
        self._after(result)
        return result

    def _observe(
        self,
        operation: str,
        status: ObservationStatus,
        attributes: dict[str, str | int | float | bool],
        *,
        started: float | None = None,
    ) -> None:
        record_observation(self.observer, "generation", operation, status, started=started, attributes=attributes)

    def _track_stuck(self, day: date | None) -> None:
        if day is None:
            self._stuck = None
        elif self._stuck is not None and self._stuck[0] == day:
            self._stuck = (day, self._stuck[1] + 1)
        else:
            self._stuck = (day, 1)

    async def _generate(self, day: date) -> GenerationResult:
        logger.info("generation started day=%s worker=%s", day.isoformat(), self.worker_id)
        trace = open_trace(self.traces.sink(day)) if self.traces is not None else nullcontext()
        with trace:
            return await self.generation.run(day)

    def _after(self, result: GenerationResult) -> None:
        """做完一代：报告、指标、日志、留存清理、健康面的 ``last_error``。都是旁路：出错只记观测。"""

        self.last_result = result
        self._retries.count(result, self._clock())
        relations: dict[str, object] | None = None
        if self.reports is not None and result.complete:
            relations = self._relation_changes(result.day)
        if self.reports is not None:
            try:
                self.reports.write(result, now=self._clock(), relations=relations)
            except Exception as exc:  # noqa: BLE001 - 报告是旁路：写不出不挡换代
                self._observe("report", ObservationStatus.FAILURE, {"error_type": type(exc).__name__})
        self._record(result)
        if result.complete:
            self._prune(result.day)
        self._refresh_health(result)

    def _relation_changes(self, day: date) -> dict[str, object] | None:
        assert self.reports is not None
        try:
            section, changes = self.reports.relation_changes(day)
        except Exception as exc:  # noqa: BLE001 - 报告是旁路：算不出关系变化不挡换代
            self._observe("report", ObservationStatus.FAILURE, {"error_type": type(exc).__name__})
            return None
        grouped = Counter(
            (
                lane,
                change.kind,
                "none" if change.before is None else change.before.value,
                "none" if change.after is None else change.after.value,
                change.key.segment.value,
                bool(change.key.condition),
            )
            for lane, change in changes
        )
        for (lane, kind, before, after, segment, conditioned), count in sorted(grouped.items()):
            record_observation(
                self.observer,
                "scene",
                "relation_change",
                ObservationStatus.SUCCESS,
                attributes={
                    "lane": lane,
                    "kind": kind,
                    "status_from": before,
                    "status_to": after,
                    "segment": segment,
                    "conditioned": conditioned,
                    "count": count,
                },
            )
        return section

    def _record(self, result: GenerationResult) -> None:
        day = result.day.isoformat()
        for item in result.steps:
            record_observation(
                self.observer,
                "generation",
                "step",
                _OUTCOME_STATUS.get(item.outcome, ObservationStatus.FAILURE),
                attributes={
                    "step": item.step,
                    "outcome": item.outcome,
                    "day": day,
                    "duration_seconds": item.duration_seconds,
                },
            )
            logger.info(
                "generation step day=%s step=%s outcome=%s reason=%s duration=%.1fs",
                day,
                item.step,
                item.outcome,
                item.reason or "-",
                item.duration_seconds,
            )
        # 退路一个计数一条事件（零也记：验收要看得见"这一代没走退路"）
        for kind in FALLBACKS:
            record_observation(
                self.observer,
                "generation",
                "fallback",
                ObservationStatus.SUCCESS,
                attributes={"kind": kind, "count": result.fallbacks.get(kind, 0), "day": day},
            )
        duration = sum(item.duration_seconds for item in result.steps)
        attributes: dict[str, str | int | float | bool] = {
            "day": day,
            "outcome": "complete" if result.complete else "held" if result.held is not None else "failed",
            "held_step": result.held or "",
            "failed_step": result.failed or "",
            "duration_seconds": duration,
            "unreduced_gaps": -1 if result.unreduced_gaps is None else result.unreduced_gaps,
        }
        if result.lag_seconds is not None:
            attributes["lag_seconds"] = result.lag_seconds
        record_observation(
            self.observer,
            "generation",
            "run",
            ObservationStatus.SUCCESS if result.complete else ObservationStatus.FAILURE,
            attributes=attributes,
        )
        logger.info(
            "generation %s day=%s failed_step=%s reason=%s duration=%.1fs",
            "finished" if result.complete else "stopped",
            day,
            result.failed or "-",
            result.reason or "-",
            duration,
        )

    def _prune(self, day: date) -> None:
        kept = self.settings.kept_days
        try:
            self.generation.marks.prune(day, kept)
            if self.reports is not None:
                self.reports.prune(day, kept)
            if self.traces is not None:
                self.traces.prune(day, kept)
            prune_rhythms(self.generation.scene, day, kept)
        except Exception as exc:  # noqa: BLE001 - 留存清理是旁路：清不掉下一代再清
            self._observe("prune", ObservationStatus.FAILURE, {"error_type": type(exc).__name__})

    def _refresh_health(self, result: GenerationResult) -> None:
        """``last_error``：这一代没写成、或走了退路就挂上（健康面另读 ``last_result`` 给出原因）。"""

        problems: list[str] = []
        if not result.complete and result.held is None:
            problems.append(f"generation {result.day} not written: {result.failed} {result.reason}")
        problems.extend(f"generation {item.step} degraded: {item.reason}" for item in result.degraded)
        problems.extend(f"generation {item}" for item in result.problems if item.startswith("错误·看不到"))
        self.last_error = RuntimeError("; ".join(problems)) if problems else None


class _Retries:
    """必须成的那一步没成之后的退避（用户 11-10 失败重试第 2 条，F3）：哪一天、连着没成了几次、下一次最早什么时候再做。
    只在进程里，重启后从头；换了一天重新数；写成了清零；还没准备好（锁被占）不算失败、不退避。"""

    def __init__(self, backoff: BackoffPolicy) -> None:
        self.backoff = backoff
        self.state: tuple[date, int, datetime] | None = None

    @property
    def at(self) -> datetime | None:
        return None if self.state is None else self.state[2]

    def waiting(self, day: date, now: datetime) -> bool:
        if self.state is not None and self.state[0] != day:
            self.state = None
        return self.state is not None and now < self.state[2]

    def count(self, result: GenerationResult, now: datetime) -> None:
        if result.complete:
            self.state = None
        elif result.held is None:
            count = 1 if self.state is None or self.state[0] != result.day else self.state[1] + 1
            self.state = (result.day, count, now + timedelta(seconds=self.backoff.delay(count)))


__all__ = ["GenerationRunner", "GenerationSettings"]
