"""预测层的常驻 worker：白天每隔一拍跑一次 ``ForesightTick``（建预期 → 到了时窗的决定说不说）。

正确性全在那一拍里，这里只管节奏与"循环别死"：一拍失败记 ``last_error`` 与观测事件，下一拍照跑；停得下来、能被叫醒。
"""

from __future__ import annotations

import time

from habitus.foundation.observability import ObservationStatus, Observer
from habitus.runtime.foresight_tick import ForesightTick, TickReport
from habitus.runtime.resident import ResidentWorker


class ForesightWorker(ResidentWorker):
    _task_name = "habitus-foresight-tick"
    _observation_category = "foresight"

    def __init__(
        self,
        tick: ForesightTick,
        *,
        interval_seconds: float,
        shutdown_timeout_seconds: float,
        observer: Observer | None = None,
    ) -> None:
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        if not isinstance(tick, ForesightTick):
            raise TypeError("tick must be a ForesightTick")
        if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, int | float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number")
        self.tick = tick
        self.interval_seconds = float(interval_seconds)

    async def run_once(self) -> TickReport:
        """手动跑一拍（运维与测试用）。"""

        if self.running:
            raise RuntimeError("manual run_once cannot race the resident worker loop")
        return await self.tick.run_once()

    async def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            started = time.monotonic()
            try:
                report = await self.tick.run_once()
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
            # 按固定节拍：等的是"一拍的长度减去这一拍花掉的"，拍点不随每拍的耗时往后漂
            await self._wait(max(0.0, self.interval_seconds - (time.monotonic() - started)))


__all__ = ["ForesightWorker"]
