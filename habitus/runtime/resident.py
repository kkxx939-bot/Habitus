"""常驻 Worker 的循环骨架：启停、唤醒、可观测事件。

行为管线的两个 Worker 与预测夜批共用这一份：三者的正确性都在各自的 runner 里，Worker 只负责
"循环别死、停得下来、能被叫醒"。这段骨架不含任何领域判断，因此可以放在组合根共享——复制一份
的代价是三处会各自漂移（停止超时的语义、观测事件的形状），而它们本该完全一致。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress

from habitus.foundation.observability import (
    NullObserver,
    ObservationEvent,
    ObservationStatus,
    Observer,
)


class ResidentWorker:
    """一个可启停、可唤醒的后台循环。子类只实现 ``_run_loop``。"""

    _task_name = "habitus-worker"
    _observation_category = "runtime"

    def __init__(self, *, shutdown_timeout_seconds: float, observer: Observer | None) -> None:
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self.observer: Observer = observer if observer is not None else NullObserver()
        self._stop_requested = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self.last_error: BaseException | None = None

    @property
    def running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._stop_requested.clear()
        self._wake_event.clear()
        self.last_error = None
        self._loop_task = asyncio.create_task(self._run_loop(), name=self._task_name)
        await asyncio.sleep(0)

    async def stop(self) -> None:
        task = self._loop_task
        if task is None or task.done():
            return
        self._stop_requested.set()
        self._wake_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self.shutdown_timeout_seconds)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def wake(self) -> None:
        self._wake_event.set()

    async def _wait(self, timeout: float) -> None:
        self._wake_event.clear()
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)

    def _succeeded(self) -> None:
        """一拍成功就清掉上一次的错误。

        不清零的话，一次磁盘抖动会让健康面永久 DEGRADED 直到进程重启——预测夜批一天一拍，
        后面三十天全部成功也翻不了案。告警变噪音之后就没人看了，那正是健康面要防的事。
        """

        self.last_error = None

    def _observe(
        self,
        operation: str,
        status: ObservationStatus,
        attributes: dict[str, str | int | float | bool],
        *,
        started: float | None = None,
    ) -> None:
        try:
            self.observer.record(
                ObservationEvent(
                    category=self._observation_category,
                    operation=operation,
                    status=status,
                    duration_seconds=(
                        0.0 if started is None else max(0.0, time.monotonic() - started)
                    ),
                    attributes=attributes,
                )
            )
        except Exception:  # noqa: BLE001 - 观测失败不许影响业务循环
            pass

    async def _run_loop(self) -> None:  # pragma: no cover - 子类实现
        raise NotImplementedError


__all__ = ["ResidentWorker"]
