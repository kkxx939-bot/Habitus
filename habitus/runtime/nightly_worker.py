"""整晚那一趟的常驻节拍（裁定 59，清单 5.1）：每 ``scene.nightly_interval_seconds`` 跑一次 ``Nightly``。

原来挂在预测树的重建 worker 上（预测树一拍、接上语义树后整晚代替重建）；预测树删了，整晚那一趟自己有一个 worker。
正确性全在 ``runtime.nightly.Nightly``：哪一晚、闸、先后顺序都在那里。这里只管节奏、"循环别死"、留下可查的痕迹——
一夜失败只设 ``last_error``、记一笔观测，下一拍照常来（派生层不做级联）。整晚跑完了、但其中的结算抛了，也设 ``last_error``：
结算失败被整晚吞掉、照跑后面几步，不留在 worker 上的话健康面一直报好。
"""

from __future__ import annotations

import time
from typing import Protocol

from habitus.foundation.observability import ObservationStatus, Observer
from habitus.runtime.nightly import NightOutcome
from habitus.runtime.resident import ResidentWorker


class NightRunner(Protocol):
    """ "第 N 晚"整趟；``runtime.nightly.Nightly`` 就是这个形状。"""

    async def run_once(self) -> NightOutcome: ...


class NightlyWorker(ResidentWorker):
    _task_name = "habitus-nightly"
    _observation_category = "nightly"

    def __init__(
        self,
        nightly: NightRunner,
        *,
        interval_seconds: float,
        shutdown_timeout_seconds: float,
        observer: Observer | None = None,
    ) -> None:
        if not hasattr(nightly, "run_once") or not callable(nightly.run_once):
            raise TypeError("nightly must expose an async run_once()")
        if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, int | float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number")
        super().__init__(shutdown_timeout_seconds=shutdown_timeout_seconds, observer=observer)
        self.nightly = nightly
        self.interval_seconds = float(interval_seconds)
        #: 最近一趟的结果（还没跑过、或最近一趟抛了是上一次成功的那份 / None）。
        self.last_outcome: NightOutcome | None = None

    async def run_once(self) -> NightOutcome:
        """手动跑一趟（运维与测试用）；不许和常驻循环抢。"""

        if self.running:
            raise RuntimeError("manual run_once cannot race the resident worker loop")
        outcome = await self.nightly.run_once()
        self.last_outcome = outcome
        return outcome

    async def _run_loop(self) -> None:
        while not self._stop_requested.is_set():
            started = time.monotonic()
            try:
                outcome = await self.nightly.run_once()
            except Exception as exc:  # noqa: BLE001 - 常驻循环必须活过基础设施抖动
                self.last_error = exc
                self._observe("night", ObservationStatus.FAILURE, {"error_type": type(exc).__name__}, started=started)
            else:
                self.last_outcome = outcome
                self._succeeded()
                if outcome.settlement_failed is not None:
                    self.last_error = RuntimeError(f"nightly settlement failed: {outcome.settlement_failed}")
                elif outcome.opportunities is not None and outcome.opportunities.failed:
                    # 机会表或单元历史某条 lane 没做成：照常往下，但健康面要看得见（26 号清单 2.5.1）
                    self.last_error = RuntimeError(
                        f"nightly opportunities failed: {'; '.join(outcome.opportunities.failed)}"
                    )
                self._observe("night", ObservationStatus.SUCCESS, _attributes(outcome), started=started)
            await self._wait(self.interval_seconds)


def _attributes(outcome: NightOutcome) -> dict[str, str | int | float | bool]:
    """这一趟的读数：哪一晚、为什么没跑、序列几条、语义树 / 机会表 / 结算跑没跑。"""

    return {
        "night": outcome.night.isoformat(),
        "skipped": outcome.skipped or "",
        "records": len(outcome.series.records),
        "scene": outcome.scene is not None,
        "opportunities": outcome.opportunities is not None,
        "settled": outcome.settled is not None,
        "settlement_failed": outcome.settlement_failed or "",
        "opportunities_failed": len(outcome.opportunities.failed) if outcome.opportunities is not None else 0,
    }


__all__ = ["NightRunner", "NightlyWorker"]
