"""预测层的常驻 worker：每个槽边界按支依次（会话在前、物理在后，裁定 42 两条独立管线；worker 不翻倍）做这一支的一槽——
先看一眼这一支要不要换代、要就先做这一代，再跑这一支的一拍 ``ForesightTick``（建预期 → 到了时窗的决定说不说）。同一支的两件事在
这一支的同一次预测根租约之内串行（实时化阶段 2 第二小步：换代并进这个 worker，账本只有一个写者）：换代那一槽的预测拍顺延到换代
做完、同一槽里做，从当天零点起补建；换代没成（必须成的那一步失败、这一眼抛了）也不挡这一拍——今天这一代还没有，不建预期、递送
照常，换代按它的退避在之后的槽再试。

两支各自独立：一支的预测根租约被占、这一拍抛了、换代没成，都只算这一支的，另一支照常做完这一槽；两支不同时持对方的锁。

只开语义树、没开预测层时没有预测拍（``SlotBranch.tick`` 是 None）：这一支那一槽自己持预测根租约看一眼换代。

拍点对齐槽边界（实时化方案第 5 节：全系统一个时间刻度）：从当地零点起每 ``interval_seconds`` 一个边界，等到下一个边界再跑；
一拍拖过了几个边界，下一拍从此刻之后的那个边界接着走（不补跑错过的边界——那一段由下一拍按水位往回看补上）。等用墙钟、分小段
等：Mac 合盖时单调时钟不走，醒来后照墙钟重新算离边界还有多久（裁定 68 第 4 条）。边界按绝对时间推进：夏令时那天相邻两拍
的真实间隔仍是一个槽（拨回那天重复的那一小时照样每槽一拍，拨快那天不出现空当）；槽号按墙钟（``temporal.slots``）。

正确性全在换代与那一拍里，这里只管节奏与"循环别死"：一支一拍失败记 ``last_error``（带是哪一支）与观测事件，另一支与下一拍照跑；
停得下来、能被叫醒。停机时当前这一槽等有界的时间、超时取消：换代做到一半被取消就没有完成标记，下次启动后的那一槽从头做。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SlotBranch:
    """预测 worker 替一支做的那一槽：这一支换代的那一眼（没接语义树是 None）、这一支的预测拍（没开预测层是 None）、没有预测拍时
    那一槽自己持的这一支预测根租约（有预测拍时租约在那一拍手里，``ForesightTick.lease``）。"""

    lane: str
    generation: GenerationRunner | None = None
    tick: ForesightTick | None = None
    lease: Lease | None = None

    def __post_init__(self) -> None:
        if not self.lane:
            raise ValueError("a slot branch needs its lane")
        if self.tick is None and self.generation is None:
            raise ValueError("the slot worker needs a foresight tick or a generation to drive")
        if self.tick is not None and self.tick.lane != self.lane:
            raise ValueError("a slot branch drives its own lane's foresight tick")


class SlotFailed(RuntimeError):
    """这一槽里有一支抛了（别的支照常做完了）：``failures`` 是支 → 异常，``reports`` 是做成了的那几支。"""

    def __init__(self, failures: dict[str, BaseException], reports: dict[str, TickReport]) -> None:
        lanes = ", ".join(f"{lane}: {type(exc).__name__}" for lane, exc in failures.items())
        super().__init__(f"slot failed on {lanes}")
        self.failures = failures
        self.reports = reports


class ForesightWorker(ResidentWorker):
    _task_name = "habitus-foresight-tick"
    _observation_category = "foresight"

    def __init__(
        self,
        branches: Sequence[SlotBranch],
        *,
        interval_seconds: float,
        shutdown_timeout_seconds: float,
        zone: ZoneInfo,
        observer: Observer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number")
        lanes = [branch.lane for branch in branches]
        if not lanes or len(lanes) != len(set(lanes)):
            raise ValueError("the slot worker drives one or more branches, each once")
        #: 按做的先后（会话在前）。
        self.branches = tuple(branches)
        self.interval_seconds = float(interval_seconds)
        self.zone = zone
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    async def run_once(self) -> dict[str, TickReport]:
        """手动跑一槽（运维、测试与回放用）：与常驻循环的一槽同一个入口，只是不带槽边界。有一支抛了：别的支照常做完，再抛
        ``SlotFailed``（带着做成了的那几支）。"""

        if self.running:
            raise RuntimeError("manual run_once cannot race the resident worker loop")
        reports, failures = await self.slot()
        if failures:
            raise SlotFailed(failures, reports)
        return reports

    async def slot(self, scheduled: datetime | None = None) -> tuple[dict[str, TickReport], dict[str, BaseException]]:
        """一槽：按支依次做（``branch_slot``），一支抛了记下、接着做下一支。返回（做成了的支 → 这一拍，抛了的支 → 异常）。
        取消（停机超时）原样抛。"""

        reports: dict[str, TickReport] = {}
        failures: dict[str, BaseException] = {}
        for branch in self.branches:
            try:
                reports[branch.lane] = await self.branch_slot(branch, scheduled)
            except Exception as exc:  # noqa: BLE001 - 一支抛了不挡另一支这一槽（裁定 42 两条独立管线）
                failures[branch.lane] = exc
        return reports, failures

    async def branch_slot(self, branch: SlotBranch, scheduled: datetime | None = None) -> TickReport:
        """一支的一槽：持这一支的预测根租约 → 看一眼这一支的换代（要就做今天这一代）→ 这一支的预测拍。租约只拿一次。"""

        if branch.generation is None:
            assert branch.tick is not None
            return await branch.tick.run_once(scheduled=scheduled)
        before = branch.generation.slot
        if branch.tick is not None:
            return await branch.tick.run_once(scheduled=scheduled, before=before)
        now = self._clock().astimezone(self.zone)
        try:
            if branch.lease is None:
                await before()
            else:
                async with branch.lease.held():
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
            # 每一支各自 try（``slot``）：常驻循环活过基础设施抖动，一支抛了不挡另一支
            reports, failures = await self.slot(boundary)
            for lane, failure in failures.items():
                self._observe(
                    "tick_loop",
                    ObservationStatus.FAILURE,
                    {"error_type": type(failure).__name__, "lane": lane},
                    started=started,
                )
            for lane, report in reports.items():
                self._observe(
                    "tick_loop",
                    ObservationStatus.SUCCESS,
                    {
                        "formed": report.formed,
                        "recorded": report.recorded,
                        "spoken": report.spoken,
                        "skipped": report.skipped or "",
                        "lane": lane,
                    },
                    started=started,
                )
            if failures:
                self.last_error = SlotFailed(failures, reports)
            else:
                self._succeeded()
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


__all__ = ["NO_TICK", "ForesightWorker", "SlotBranch", "SlotFailed"]
