"""模型队列的常驻 worker（实时化方案第 4 节）：有活连着做、没活按间隔看（``scene.model_queue_poll_seconds``）；模型整个调不通时
等到退避到点再看。正确性全在 ``ModelQueue`` 里（欠账每一轮从盘上现算），这里只管节奏与"循环别死、停得下来"。

停的时候在两项之间停下；当前这一项等有界的时间（``scene.model_queue_shutdown_timeout_seconds``），超时取消——取消掉的那一项
下次从盘上重新认出来。它在归约 worker 之后、预测 worker（每槽先看一眼换代）之前启动，停机时在预测 worker 之后、归约 worker 之前停。
"""

from __future__ import annotations

import asyncio

from habitus.foundation.observability import ObservationStatus, Observer
from habitus.runtime.model_queue import DrainReport, ModelQueue
from habitus.runtime.resident import ResidentWorker


class ModelQueueWorker(ResidentWorker):
    _task_name = "habitus-model-queue"
    _observation_category = "model_queue"

    def __init__(self, queue: ModelQueue, *, shutdown_timeout_seconds: float, observer: Observer | None = None) -> None:
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        self.queue = queue
        #: 最近一趟的结果（健康面与运维看）。
        self.last_drain: DrainReport | None = None

    async def drain(self) -> DrainReport:
        """手动做到没有欠账或进入整体退避为止（回放、运维与测试用；常驻循环在跑时不能并着调）。"""

        if self.running:
            raise RuntimeError("manual drain cannot race the resident worker loop")
        self.last_drain = await self.queue.drain()
        return self.last_drain

    async def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            try:
                self.last_drain = await self.queue.drain(stop=self._stop_requested.is_set)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动；下一眼从盘上重新认
                self.last_error = exc
                self._observe("drain", ObservationStatus.FAILURE, {"error_type": type(exc).__name__})
            else:
                self._succeeded()
            await self._wait(self.queue.wait_seconds())


__all__ = ["ModelQueueWorker"]
