"""预测层的常驻 worker：每个槽边界先看一眼要不要换代、要就先做这一代，再跑一拍 ``ForesightTick``（建预期 → 到了时窗的决定
说不说）。两件事在同一次预测根租约之内串行（实时化阶段 2 第二小步：换代并进这个 worker，账本只有一个写者）：换代那一槽的预测拍
顺延到换代做完、同一槽里做，从当天零点起补建；换代没成（必须成的那一步失败、这一眼抛了）也不挡这一拍——今天这一代还没有，
不建预期、递送照常，换代按它的退避在之后的槽再试。

只开语义树、没开预测层时没有预测拍（``tick`` 是 None）：这个 worker 照样每槽持预测根租约看一眼换代。

拍点对齐槽边界（实时化方案第 5 节：全系统一个时间刻度）：从当地零点起每 ``interval_seconds`` 一个边界，等到下一个边界再跑；
一拍拖过了几个边界，下一拍从此刻之后的那个边界接着走（不补跑错过的边界——那一段由下一拍按水位往回看补上）。等用墙钟、分小段
等：Mac 合盖时单调时钟不走，醒来后照墙钟重新算离边界还有多久（裁定 68 第 4 条）。边界按绝对时间推进：夏令时那天相邻两拍
的真实间隔仍是一个槽（拨回那天重复的那一小时照样每槽一拍，拨快那天不出现空当）；槽号按墙钟（``temporal.slots``）。

正确性全在换代与那一拍里，这里只管节奏与"循环别死"：一拍失败记 ``last_error`` 与观测事件，下一拍照跑；停得下来、能被叫醒。
停机时当前这一槽等有界的时间、超时取消：换代做到一半被取消就没有完成标记，下次启动后的那一槽从头做。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from datetime import time as clock_time
from zoneinfo import ZoneInfo

from habitus.foundation.observability import ObservationStatus, Observer
from habitus.runtime.foresight_tick import LEDGER_BUSY, ForesightTick, TickReport
from habitus.runtime.generation_lease import Lease, LeaseBusyError
from habitus.runtime.generation_runner import GenerationRunner
from habitus.runtime.resident import ResidentWorker

#: 等下一个槽边界时，最多一次等这么久就照墙钟重看一次（合盖醒来后不按单调时钟把剩下的等完）。
_WAIT_SLICE_SECONDS = 30.0
#: 没开预测层：这一槽只看了一眼换代，没有预测拍。
NO_TICK = "没开预测层"


class ForesightWorker(ResidentWorker):
    _task_name = "habitus-foresight-tick"
    _observation_category = "foresight"

    def __init__(
        self,
        tick: ForesightTick | None,
        *,
        generation: GenerationRunner | None = None,
        lease: Lease | None = None,
        interval_seconds: float,
        shutdown_timeout_seconds: float,
        zone: ZoneInfo,
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number")
        if tick is None and generation is None:
            raise ValueError("the slot worker needs a foresight tick or a generation to drive")
        self.tick = tick
        #: 换代的那一眼（``GenerationRunner.slot``）：每槽在预测根租约之内先做它。没接语义树就没有。
        self.generation = generation
        #: 没有预测拍时这个 worker 自己持的预测根租约（有预测拍时租约在那一拍手里，``ForesightTick.lease``）。
        self.lease = lease
        self.interval_seconds = float(interval_seconds)
        self.zone = zone
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    async def run_once(self) -> TickReport:
        """手动跑一槽（运维、测试与回放用）：与常驻循环的一槽同一个入口，只是不带槽边界。"""

        if self.running:
            raise RuntimeError("manual run_once cannot race the resident worker loop")
        return await self.slot()

    async def slot(self, scheduled: datetime | None = None) -> TickReport:
        """一槽：持预测根租约 → 看一眼换代（要就做今天这一代）→ 预测拍。租约只拿一次。"""

        if self.generation is None:
            assert self.tick is not None
            return await self.tick.run_once(scheduled=scheduled)
        before = self.generation.slot
        if self.tick is not None:
            return await self.tick.run_once(scheduled=scheduled, before=before)
        now = self._clock().astimezone(self.zone)
        try:
            if self.lease is None:
                await before()
            else:
                async with self.lease.held():
                    await before()
        except LeaseBusyError:
            return TickReport(now, 0, 0, skipped=LEDGER_BUSY)
        return TickReport(now, 0, 0, skipped=NO_TICK)

    def next_boundary(self, after: datetime, *, inclusive: bool) -> datetime:
        """``after`` 之后（``inclusive`` 时含它本身）的第一个槽边界：从当地零点起每 ``interval_seconds`` 一个，按绝对时间算。
        同一时区的两个带时区时刻相减、比大小、加 ``timedelta`` 都是墙钟算术（25 小时那天 01:45 BST 之后会算出 02:00 GMT，真实等
        75 分钟；重复那一小时里 01:45 BST 比 01:00 GMT "大"），所以都在 UTC 上算，返回的边界也是 UTC。"""

        midnight = datetime.combine(after.astimezone(self.zone).date(), clock_time(0), tzinfo=self.zone).astimezone(UTC)
        elapsed = (after.astimezone(UTC) - midnight).total_seconds()
        count = (
            math.ceil(elapsed / self.interval_seconds) if inclusive else math.floor(elapsed / self.interval_seconds) + 1
        )
        return midnight + timedelta(seconds=count * self.interval_seconds)

    async def _run_loop(self) -> None:
        boundary = self.next_boundary(self._clock(), inclusive=True)
        while not self._stop_requested.is_set():
            await self._until(boundary)
            if self._stop_requested.is_set():
                return
            started = time.monotonic()
            try:
                report = await self.slot(boundary)
            except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动
                self.last_error = exc
                self._observe(
                    "tick_loop", ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started=started
                )
            else:
                self._succeeded()
                self._observe(
                    "tick_loop",
                    ObservationStatus.SUCCESS,
                    {
                        "formed": report.formed,
                        "recorded": report.recorded,
                        "spoken": report.spoken,
                        "skipped": report.skipped or "",
                    },
                    started=started,
                )
            # 下一拍：这一个边界之后、且不早于此刻的那个边界（拖过了几个边界不补跑）
            boundary = max(
                self.next_boundary(boundary, inclusive=False), self.next_boundary(self._clock(), inclusive=True)
            )

    async def _until(self, boundary: datetime) -> None:
        """照墙钟等到 ``boundary``（分小段等，停机或醒来都会重看）。"""

        while not self._stop_requested.is_set():
            remaining = (boundary - self._clock()).total_seconds()
            if remaining <= 0:
                return
            await self._wait(min(remaining, _WAIT_SLICE_SECONDS))


__all__ = ["NO_TICK", "ForesightWorker"]
