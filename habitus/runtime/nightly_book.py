"""夜线 Job 执行器的记账一侧：建账、结算、放回、写每晚报告（含关系表双向变化）、记每段的指标事件与 INFO 日志。

执行器（``runtime.nightly_worker``）管租约、心跳与先后；这里只在它拿着租约时被调，写账都在它交来的 ``fence``（``LockStore.fenced``）里。
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import date, datetime

from habitus.foundation.observability import ObservationStatus, Observer, record_observation
from habitus.runtime.nightly import STEPS, STRUCTURE, StepResult
from habitus.runtime.nightly_jobs import (
    FAILED,
    STATES,
    NightClaim,
    NightClaimLostError,
    NightJobs,
    NightLedger,
    released,
    settled,
)
from habitus.runtime.nightly_report import NightReports
from habitus.runtime.nightly_settings import NightlySettings
from habitus.runtime.nightly_trace import TraceFiles

logger = logging.getLogger("habitus.nightly")
Fence = Callable[[], AbstractContextManager[object]]

#: 一段的结局在观测上的状态：停下记降级、失败记失败，不再记成成功。
_OUTCOME_STATUS = {
    "done": ObservationStatus.SUCCESS,
    "held": ObservationStatus.SUCCESS,
    "stopped": ObservationStatus.DEGRADED,
    FAILED: ObservationStatus.FAILURE,
}


class NightBook:
    def __init__(
        self,
        jobs: NightJobs,
        *,
        settings: NightlySettings,
        reports: NightReports | None,
        traces: TraceFiles | None,
        observer: Observer,
        clock: Callable[[], datetime],
    ) -> None:
        self.jobs = jobs
        self.settings = settings
        self.reports = reports
        self.traces = traces
        self.observer = observer
        self._clock = clock

    def observe(
        self, operation: str, status: ObservationStatus, attributes: dict[str, str | int | float | bool]
    ) -> None:
        record_observation(self.observer, "nightly", operation, status, attributes=attributes)

    def ledger_of(self, night: date, fence: Callable[[], AbstractContextManager[object]]) -> NightLedger:
        ledger = self.jobs.read(night)
        if ledger is not None:
            return ledger
        ledger = self.jobs.update(
            night, lambda existing: existing or NightLedger.fresh(night, self._clock()), fence=fence
        )
        logger.info("nightly night queued night=%s steps=%s", night.isoformat(), ",".join(STEPS))
        kept = self.settings.kept_nights
        self.jobs.prune(night)
        if self.reports is not None:
            self.reports.prune(night, kept)
        if self.traces is not None:
            self.traces.prune(night, kept)
        return ledger

    def settle(
        self,
        night: date,
        token: NightClaim,
        result: StepResult,
        duration: float,
        fence: Callable[[], AbstractContextManager[object]],
    ) -> NightLedger:
        settings = self.settings
        ledger = self.jobs.update(
            night,
            lambda current: settled(
                require(current),
                token,
                result,
                now=self._clock(),
                duration_seconds=duration,
                backoff=settings.backoff,
                max_attempts=settings.max_attempts,
            ),
            fence=fence,
        )
        relations = None
        if result.step == STRUCTURE and result.done and self.reports is not None:
            relations = self.relation_changes(ledger)
        self.write_report(ledger, relations)
        return ledger

    def release(self, night: date, token: NightClaim, fence: Callable[[], AbstractContextManager[object]]) -> None:
        try:
            self.jobs.update(night, lambda current: released(require(current), token), fence=fence)
        except Exception:  # noqa: BLE001 - 放不回去（租约已丢）：账上留着"在跑"，租约过期后由下一次认领接管
            logger.warning("nightly step could not be released night=%s step=%s", night.isoformat(), token.step)

    def relation_changes(self, ledger: NightLedger) -> dict[str, object] | None:
        assert self.reports is not None
        raw = ledger.job(STRUCTURE).report.get("night")
        if not isinstance(raw, str):
            return {}
        try:
            section, changes = self.reports.relation_changes(date.fromisoformat(raw))
        except Exception as exc:  # noqa: BLE001 - 报告是旁路：算不出关系变化不挡线
            self.observe("report", ObservationStatus.FAILURE, {"error_type": type(exc).__name__})
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

    def write_report(self, ledger: NightLedger, relations: dict[str, object] | None) -> None:
        if self.reports is None:
            return
        try:
            self.reports.write(ledger, now=self._clock(), relations=relations)
        except Exception as exc:  # noqa: BLE001 - 报告是旁路：写不出不挡线
            self.observe("report", ObservationStatus.FAILURE, {"error_type": type(exc).__name__})

    def after_step(self, ledger: NightLedger, result: StepResult, duration: float) -> None:
        job = ledger.job(result.step)
        outcome = job.state if job.state == FAILED else result.outcome
        status = _OUTCOME_STATUS.get(outcome, ObservationStatus.FAILURE)
        record_observation(
            self.observer,
            "nightly",
            "step",
            status,
            attributes={
                "step": result.step,
                "outcome": outcome,
                "night": ledger.night.isoformat(),
                "duration_seconds": duration,
            },
        )
        self.observe_jobs()
        logger.info(
            "nightly step %s night=%s step=%s outcome=%s reason=%s duration=%.1fs",
            "stopped" if outcome in ("stopped", FAILED) else "finished",
            ledger.night.isoformat(),
            result.step,
            outcome,
            result.reason or "-",
            duration,
        )

    def observe_jobs(self) -> None:
        """各段各状态的 Job 数（留存里的全部账），给 ``nightly_jobs{step,state}``。"""

        counts: Counter[str] = Counter({f"{step}:{state}": 0 for step in STEPS for state in STATES})
        try:
            for night in self.jobs.nights():
                current = self.jobs.read(night)
                if current is not None:
                    counts.update(f"{job.step}:{job.state}" for job in current.jobs)
        except Exception:  # noqa: BLE001 - 观测是旁路
            return
        record_observation(self.observer, "nightly", "jobs", ObservationStatus.SUCCESS, attributes=dict(counts))


def require(ledger: NightLedger | None) -> NightLedger:
    if ledger is None:
        raise NightClaimLostError("the nightly job ledger is gone")
    return ledger


__all__ = ["Fence", "NightBook", "require"]
