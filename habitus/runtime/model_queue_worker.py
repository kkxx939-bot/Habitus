"""模型队列的常驻 worker：两支各一个队列（裁定 42：两条独立管线；worker 不翻倍），每一轮按支依次做（会话在前）。
有活连着做、没活按间隔看（``scene.model_queue_poll_seconds``）；某一支的模型整个调不通时，这一支等到它的退避到点再看，另一支照常。
正确性全在 ``ModelQueue`` 里（欠账每一轮从盘上现算），这里只管节奏与"循环别死、停得下来"。

一支这一趟抛了只记这一支的（``last_error`` 带是哪一支），另一支照做。停的时候在两项之间停下；当前这一项等有界的时间
（``scene.model_queue_shutdown_timeout_seconds``），超时取消——取消掉的那一项下次从盘上重新认出来。它在归约 worker 之后、预测
worker（每槽先看一眼换代）之前启动，停机时在预测 worker 之后、归约 worker 之前停。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from habitus.foundation.observability import ObservationStatus, Observer
from habitus.runtime.model_queue import DrainReport, ModelQueue
from habitus.runtime.resident import ResidentWorker


class DrainFailed(RuntimeError):
    """这一趟有一支抛了（别的支照常做完了）：``failures`` 是支 → 异常，``reports`` 是做完了的那几支。"""

    def __init__(self, failures: dict[str, BaseException], reports: dict[str, DrainReport]) -> None:
        lanes = ", ".join(f"{lane}: {type(exc).__name__}" for lane, exc in failures.items())
        super().__init__(f"model queue drain failed on {lanes}")
        self.failures = failures
        self.reports = reports


class ModelQueueWorker(ResidentWorker):
    _task_name = "habitus-model-queue"
    _observation_category = "model_queue"

    def __init__(
        self, queues: Sequence[ModelQueue], *, shutdown_timeout_seconds: float, observer: Observer | None = None
    ) -> None:
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        lanes = [queue.lane for queue in queues]
        if not lanes or len(lanes) != len(set(lanes)):
            raise ValueError("the model queue worker drives one or more branches' queues, each once")
        #: 按做的先后（会话在前）。
        self.queues = tuple(queues)
        #: 各支最近一趟的结果（健康面与运维看）。
        self.last_drain: dict[str, DrainReport] = {}

    async def drain(self) -> dict[str, DrainReport]:
        """手动把各支做到没有欠账或进入整体退避为止（运维与测试用；常驻循环在跑时不能并着调）。有一支抛了：别的支照常
        做完，再抛 ``DrainFailed``。"""

        if self.running:
            raise RuntimeError("manual drain cannot race the resident worker loop")
        failures = await self._drain_all(lambda: False)
        if failures:
            raise DrainFailed(failures, dict(self.last_drain))
        return dict(self.last_drain)

    async def _drain_all(self, stop: Callable[[], bool]) -> dict[str, BaseException]:
        failures: dict[str, BaseException] = {}
        for queue in self.queues:
            if stop():
                break
            try:
                self.last_drain[queue.lane] = await queue.drain(stop=stop)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 一支抛了不挡另一支；下一眼从盘上重新认
                failures[queue.lane] = exc
        return failures

    def wait_seconds(self) -> float:
        """下一次最早隔多久再看：各支里最早要再看的那一个（在退避的那一支按它的退避，别的按间隔）。"""

        return min(queue.wait_seconds() for queue in self.queues)

    async def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            failures = await self._drain_all(self._stop_requested.is_set)
            for lane, exc in failures.items():
                self._observe("drain", ObservationStatus.FAILURE, {"error_type": type(exc).__name__, "lane": lane})
            if failures:
                self.last_error = DrainFailed(failures, dict(self.last_drain))
            else:
                self._succeeded()
            await self._wait(self.wait_seconds())


__all__ = ["DrainFailed", "ModelQueueWorker"]
