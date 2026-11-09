"""夜线的 Job 执行器（裁定 59，清单 5.1；裁定 94 第 1、8、9 条）：前一天定稿完成，就给这一晚入队 5 个耐久 Job（前置处理、结算、
词表、结构、日子预期，``runtime.nightly_jobs``），按先后执行；一天一晚，不挂固定时刻。

每隔一会儿（``scene.nightly_poll_seconds``）看一眼：先拿全局租约（``SQLiteLockStore``，两个进程挂同一个 storage root 时同一晚
同一段只跑一次），拿不到就是别人在跑、下一眼再看；拿到了就在独立线程里续租（续不上就取消这一段，账上留着"在跑"，租约过期后由
下一次认领递增 generation 接管），然后：

1. 上一晚过了那天还没跑完：没成的段记失败（"这一晚过去了，没跑完"），下一晚照常——旧的一晚不在新的日子上按新日子的时钟补算；
2. 昨天定稿了：这一晚没有账就建账（5 段都排队），清掉留存之外的账、报告与 trace；
3. 按先后跑没成的段：成了接着跑下一段；还没准备好 → 保持排队，下一眼从这一段接着跑；停下（模型调不通、出错）→ 按
   ``next_attempt_at`` 有界退避，到点再试，次数用完记失败、后面的段不再跑。

每段一个 span（``nightly.<段>``，挂在 ``nightly.night`` 下，落本地 trace 文件）、一条指标事件（耗时、结局）、开始 / 结束 / 停下各一行
INFO 日志；每次结算后重写这一晚的报告文件（``runtime.nightly_report``），结构那一段成了时算一次关系表的双向变化。

进程起来后第一次拿到租约时，上一晚中途被杀剩下的重试项撤回（裁定 68 第 3 条，``clean_leftovers``）——除非前置处理那个 Job 正
等着锁、账上留着重试那一支的状态（那时由它接着跑）。正确性全在 ``runtime.nightly.Nightly``（哪一晚、闸、哪一段算成）与 Job 账；
这里只管节奏、租约、记账、留下可查的痕迹。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, date, datetime
from typing import cast

from habitus.foundation.observability import ObservationStatus, Observer
from habitus.foundation.tracing import current_span, open_trace, set_attributes, span
from habitus.infrastructure.store.contracts import LockStore, LockToken
from habitus.runtime.nightly import PREPARE, STRUCTURE, Nightly, StepResult
from habitus.runtime.nightly_book import NightBook, require
from habitus.runtime.nightly_jobs import (
    FAILED,
    NightClaim,
    NightClaimLostError,
    NightJob,
    NightJobs,
    NightLedger,
    claim,
    claimed,
    decode_checkpoint,
    expired,
    renewed,
)
from habitus.runtime.nightly_report import NightReports
from habitus.runtime.nightly_settings import NightlySettings
from habitus.runtime.nightly_summary import LOCAL_ONLY_KEYS, kept_errors, reason_code
from habitus.runtime.nightly_trace import NIGHT_SPAN, STEP_SPAN_PREFIX, TraceFiles
from habitus.runtime.resident import ResidentWorker

logger = logging.getLogger("habitus.nightly")


class NightlyLeaseLostError(RuntimeError):
    """续不上全局租约：这一段已取消，账上留着"在跑"，租约过期后由下一次认领接管。"""


class NightlyWorker(ResidentWorker):
    _task_name = "habitus-nightly"
    _observation_category = "nightly"

    def __init__(
        self,
        nightly: Nightly,
        *,
        jobs: NightJobs,
        lock_store: LockStore,
        lease_key: str,
        shutdown_timeout_seconds: float,
        settings: NightlySettings | None = None,
        reports: NightReports | None = None,
        traces: TraceFiles | None = None,
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
        worker_id: str | None = None,
    ) -> None:
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        if not lease_key:
            raise ValueError("lease_key must be non-empty text")
        self.nightly = nightly
        self.jobs = jobs
        self.lock_store = lock_store
        self.lease_key = lease_key
        self.settings = settings or NightlySettings()
        self.reports = reports
        self.traces = traces
        self.worker_id = worker_id or f"nightly-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self._clock = clock or (lambda: datetime.now(UTC))
        self.book = NightBook(
            jobs, settings=self.settings, reports=reports, traces=traces, observer=self.observer, clock=self._clock
        )
        #: 进程起来后清过上一晚剩下的重试项没有（第一次拿到租约时清一次）。
        self._cleaned = False
        self._active: NightClaim | None = None

    def finished(self, night: date) -> bool | None:
        """这一晚的线整条跑完了没有（结算认停掉的那天、白天那一拍建不建预期都读它，``NightJobs.finished``）。"""

        return self.jobs.finished(night)

    async def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            try:
                await self.look()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动；账上的状态下一眼接着用
                self.last_error = exc
                self._observe("look", ObservationStatus.FAILURE, {"error_type": type(exc).__name__})
            await self._wait(self.settings.poll_seconds)

    async def look(self) -> NightLedger | None:
        """看一眼：拿得到租约就把这一晚能跑的段跑完（或跑到停下、没准备好），返回这一晚的账；拿不到租约是 None。"""

        try:
            token = await asyncio.to_thread(
                self.lock_store.acquire, self.lease_key, int(self.settings.lease_ttl_seconds)
            )
        except TimeoutError:
            return None
        execution = asyncio.create_task(self._drain(token), name=f"habitus-nightly-drain:{self.worker_id}")
        stop = threading.Event()
        loop = asyncio.get_running_loop()
        heartbeat = asyncio.create_task(asyncio.to_thread(self._heartbeat, token, execution, stop, loop))
        try:
            done, _pending = await asyncio.wait({execution, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                raise NightlyLeaseLostError("nightly lease could not be renewed") from heartbeat.exception()
            result = await execution
            self._refresh_health(result)
            return result
        finally:
            stop.set()
            await asyncio.gather(heartbeat, return_exceptions=True)
            if not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
            try:
                await asyncio.to_thread(self.lock_store.release, token)
            except Exception:  # noqa: BLE001 - 租约放不掉照样活着：过期之后别人能拿
                logger.warning("nightly lease release failed")

    def _heartbeat(
        self,
        token: LockToken,
        execution: asyncio.Task[NightLedger | None],
        stop: threading.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """在独立线程续租（同步文件、SQLite 操作卡住事件循环时也不丢租约），顺带续在跑那一段的认领。

        一次续不上（SQLite busy 之类的一时失败）不取消：租约到期之前别人拿不走它，下一拍再续；只有下一拍之前租约就要到期、
        还没续上，才取消这一段（原来一次 busy 就把几小时的结构段打回排队，验证轮"要人定"第 3 条）。"""

        ttl = int(self.settings.lease_ttl_seconds)
        beat = self.settings.heartbeat_seconds
        expires = time.monotonic() + ttl  # 刚拿到租约
        while not stop.wait(timeout=beat):
            attempted = time.monotonic()
            try:
                self.lock_store.renew(token, ttl)
                expires = attempted + ttl
                active = self._active
                if active is not None:
                    self._renew_claim(token, active)
            except NightClaimLostError:
                continue  # 这一段刚结算完：认领已经不在账上，下一拍没有要续的
            except Exception:
                if time.monotonic() + beat < expires:
                    logger.warning("nightly lease renewal failed; retrying before it expires")
                    continue
                loop.call_soon_threadsafe(execution.cancel)
                raise

    def _renew_claim(self, token: LockToken, active: NightClaim) -> None:
        ttl = self.settings.lease_ttl_seconds
        self.jobs.update(
            active.night,
            lambda ledger: renewed(require(ledger), active, now=self._clock(), ttl_seconds=ttl),
            fence=self._fence(token),
        )

    def _fence(self, token: LockToken) -> Callable[[], AbstractContextManager[object]]:
        return lambda: self.lock_store.fenced([token], int(self.settings.lease_ttl_seconds))

    async def _drain(self, token: LockToken) -> NightLedger | None:
        fence = self._fence(token)
        now = self._clock()
        latest = await asyncio.to_thread(self.jobs.latest)
        if latest is not None and latest.night < self.nightly.tonight() and not latest.closed:
            latest = await asyncio.to_thread(
                self.jobs.update, latest.night, lambda ledger: expired(require(ledger), now=now), fence=fence
            )
            await asyncio.to_thread(self.book.write_report, latest, None)
            logger.info("nightly night expired night=%s", latest.night.isoformat())
        night = await asyncio.to_thread(self.nightly.ready)
        if night is None:
            await self._clean_once(latest if latest is not None and latest.night == self.nightly.tonight() else None)
            return latest
        ledger = await asyncio.to_thread(self.book.ledger_of, night, fence)
        await self._clean_once(ledger)
        job = ledger.next_job()
        if job is None or job.state == FAILED or not job.runnable(self._clock()) or self._stop_requested.is_set():
            return ledger  # 跑完了、失败了、停下的还没到点重试、或正在停机：这一眼没有要跑的
        trace = open_trace(self.traces.sink(night)) if self.traces is not None else nullcontext()
        with trace, span(NIGHT_SPAN, {"night": night.isoformat(), "worker_id": self.worker_id}) as attributes:
            started = time.monotonic()
            ledger = await self._run_steps(ledger, fence)
            set_attributes(attributes, _night_attributes(ledger, time.monotonic() - started))
        return ledger

    async def _clean_once(self, ledger: NightLedger | None) -> None:
        """进程起来后第一次拿到租约：上一晚中途被杀剩下的重试项撤回——前置处理正等着锁（账上留着重试那一支的状态）时不撤。"""

        if self._cleaned:
            return
        self._cleaned = True
        waiting = ledger is not None and ledger.job(PREPARE).retry_checkpoint is not None
        if not waiting:
            await asyncio.to_thread(self.nightly.clean_leftovers)

    async def _run_steps(self, ledger: NightLedger, fence: Callable[[], AbstractContextManager[object]]) -> NightLedger:
        while not self._stop_requested.is_set():
            job = ledger.next_job()
            if job is None or job.state == FAILED or not job.runnable(self._clock()):
                break
            ledger, result = await self._run_job(ledger, job, fence)
            if result is None or not result.done:
                break
        return ledger

    async def _run_job(
        self, ledger: NightLedger, job: NightJob, fence: Callable[[], AbstractContextManager[object]]
    ) -> tuple[NightLedger, StepResult | None]:
        ttl = self.settings.lease_ttl_seconds
        token = claim(ledger, job.step, worker_id=self.worker_id, now=self._clock())
        ledger = await asyncio.to_thread(
            self.jobs.update,
            ledger.night,
            lambda current: claimed(require(current), token, now=self._clock(), ttl_seconds=ttl),
            fence=fence,
        )
        self._active = token
        logger.info(
            "nightly step started night=%s step=%s attempt=%d", ledger.night.isoformat(), job.step, token.attempt
        )
        started = time.monotonic()
        try:
            with span(f"{STEP_SPAN_PREFIX}{job.step}", _step_attributes(token)) as attributes:
                result = await self._step(ledger, job.step)
                record = current_span()
                calls = Counter[str]()
                if record is not None:
                    for (touch, _provider, _model), count in record.model_calls.items():
                        calls[touch] += count
                result = StepResult(
                    result.step,
                    result.outcome,
                    result.reason,
                    {**result.report, "model_calls": dict(calls)},
                    result.retry_checkpoint,
                )
                set_attributes(attributes, _result_attributes(result))
        except BaseException:
            # 被取消（停机超时、租约丢了）或执行器自己出错（不是段里的 bug，那由 ``run_step`` 记成停下）：放回排队，下一次接着跑
            self._active = None
            await asyncio.shield(asyncio.to_thread(self.book.release, ledger.night, token, fence))
            raise
        self._active = None
        duration = time.monotonic() - started
        ledger = await asyncio.to_thread(self.book.settle, ledger.night, token, result, duration, fence)
        self.book.after_step(ledger, result, duration)
        return ledger, result

    async def _step(self, ledger: NightLedger, step: str) -> StepResult:
        checkpoint_raw = ledger.job(PREPARE).retry_checkpoint
        structure = ledger.job(STRUCTURE).report.get("skipped")
        return await self.nightly.run_step(
            step,
            ledger.night,
            structure_skipped=structure if isinstance(structure, str) else None,
            retry_checkpoint=None if checkpoint_raw is None else decode_checkpoint(checkpoint_raw),
        )

    def _refresh_health(self, ledger: NightLedger | None) -> None:
        """常驻循环的 ``last_error``：这一晚有段停下或失败就挂上（健康面另读 Job 账给出原因）。"""

        if ledger is None:
            return
        problems = [
            f"nightly {job.step} {job.state}: {job.reason}" for job in ledger.jobs if job.state in ("stopped", FAILED)
        ]
        prepared = ledger.job(PREPARE).report
        if prepared.get("calendar_failed"):
            problems.append(f"nightly calendar refresh failed: {prepared['calendar_failed']}")
        kept = kept_errors(prepared.get("retry"))
        if kept is not None:
            problems.append(f"nightly errors kept unseen: {kept}")
        self.last_error = RuntimeError("; ".join(problems)) if problems else None


def _step_attributes(token: NightClaim) -> dict[str, str | int | float | bool]:
    return {
        "night": token.night.isoformat(),
        "step": token.step,
        "attempt": token.attempt,
        "claim_generation": token.claim_generation,
        "worker_id": token.worker_id,
    }


def _result_attributes(result: StepResult) -> dict[str, object]:
    return {"outcome": result.outcome, "reason": reason_code(result.reason), **flatten(result.report)}


def _night_attributes(ledger: NightLedger, duration: float) -> dict[str, object]:
    return {
        "duration_seconds": duration,
        "complete": ledger.complete,
        **{f"outcome.{job.step}": job.state for job in ledger.jobs},
        "stopped_reason": reason_code(
            next((job.reason for job in ledger.jobs if job.state in ("stopped", FAILED)), None)
        ),
    }


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    """报告摊成 span 属性：嵌套键用点连，列表记条数（短的字符串列表另记一份逗号连起来的），None 不记；``LOCAL_ONLY_KEYS``
    里的键（信号文字）整段跳过。"""

    found: dict[str, object] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in LOCAL_ONLY_KEYS:
                continue  # 带用户派生文字（概念名、关系两端、迁移原因）：只进本地每晚报告，不进 span（codex R3）
            found.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list | tuple):
        found[f"{prefix}.count"] = len(value)
        if value and all(isinstance(item, str) for item in value):
            found[prefix] = ",".join(cast(list[str], list(value)))[:1_000]
        elif value and all(isinstance(item, dict) for item in value):
            for index, item in enumerate(value):
                found.update(flatten(item, f"{prefix}.{index}"))
    elif value is not None:
        found[prefix] = value
    return found


__all__ = ["NightlyLeaseLostError", "NightlySettings", "NightlyWorker", "flatten"]
