"""词表整理活的编排：持 sweep 锁（与发布互斥）、先续做没做完的迁移、改过的日子刷新语义面。

整理活本身（每晚新增、定期拆改、迁移）在 ``kinds_jobs`` / ``kinds_migration``；锁、检查点门与语义刷新由 runner 交进来，
与 sweep 共用同一把锁、同一份待刷新集合。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, datetime

from habitus.behavior.reduction.checkpoint import CheckpointStore
from habitus.behavior.reduction.kinds_jobs import JobsOutcome, VocabularyJobs
from habitus.behavior.reduction.refresh_pending import RefreshPendingLog
from habitus.infrastructure.store.contracts.path_lock import LeaseGuard


@dataclass(frozen=True)
class VocabularyJobReport:
    """一次词表整理活（每晚新增 / 定期拆改）的可观测结果。"""

    days: tuple[date, ...]
    model_calls: int
    signals: tuple[str, ...]
    #: 只给观测：见 ``JobsOutcome`` 的同名字段。
    pool: int = 0
    created: tuple[str, ...] = ()
    revision_due: bool = False
    split: int = 0
    merged: int = 0
    moves: int = 0
    #: 这一次持 sweep 锁多久（秒）：整段持锁、包括调模型的那几次，期间归约不能发布（实时化阶段 2 的读数）。
    lock_seconds: float = 0.0


class VocabularyRunner:
    """``BehaviorReductionRunner`` 的词表一摊：三种整理活共用一条持锁路径。"""

    def __init__(
        self,
        *,
        jobs: VocabularyJobs,
        checkpoints: CheckpointStore,
        refresh_pending: RefreshPendingLog,
        hold_lock: Callable[[ExitStack], LeaseGuard],
        refresh_semantics: Callable[[set[date], LeaseGuard], Awaitable[tuple[str, ...]]],
        now: Callable[[], datetime],
    ) -> None:
        self.jobs = jobs
        self.checkpoints = checkpoints
        self.refresh_pending = refresh_pending
        self._hold_lock = hold_lock
        self._refresh_semantics = refresh_semantics
        self._now = now

    async def nightly(self, night: date) -> VocabularyJobReport:
        async def job(now: datetime, *, checkpoint: Callable[[], object]) -> JobsOutcome:
            return await self.jobs.nightly(now, night=night, checkpoint=checkpoint)

        return await self.run(job)

    async def resume(self) -> VocabularyJobReport:
        """换代最前那一步（实时化方案第 3 节）：续做没做完的迁移，再把待生效的拆改交给迁移器（``VocabularyJobs.apply_revision``）。
        同一条持锁路径，不做整理活本身；两样都没有时是空操作（只拿一下 sweep 锁）；锁被占抛 ``BehaviorReductionBusyError``。"""

        async def pending(_now: datetime, *, checkpoint: Callable[[], object]) -> JobsOutcome:
            return self.jobs.apply_revision(checkpoint)

        return await self.run(pending)

    def revision_pending(self) -> bool:
        """有没有待生效的拆改（换代据它决定最前那一步要不要拿 sweep 锁）。"""

        return self.jobs.store.read_revision() is not None

    async def grow(self) -> VocabularyJobReport:
        return await self.run(self.jobs.grow)

    async def revise(self) -> VocabularyJobReport:
        return await self.run(self.jobs.revise)

    async def run(self, job: Callable[..., Awaitable[JobsOutcome]]) -> VocabularyJobReport:
        """词表整理活持 sweep 锁（与发布互斥）；先续做没做完的迁移，改过的日子刷新语义面。"""

        with ExitStack() as held:
            guard = self._hold_lock(held)
            locked = time.monotonic()
            self.checkpoints.require_none("vocabulary job")
            outcome = self.resume_migration(guard) + await job(self._now(), checkpoint=guard.checkpoint)
            self.refresh_pending.touch(outcome.days, guard)
            notes = await self._refresh_semantics(set(outcome.days), guard)
            return VocabularyJobReport(
                days=tuple(sorted(outcome.days)),
                model_calls=outcome.model_calls,
                signals=(*outcome.signals, *notes),
                pool=outcome.pool,
                created=outcome.created,
                revision_due=outcome.revision_due,
                split=outcome.split,
                merged=outcome.merged,
                moves=outcome.moves,
                lock_seconds=max(0.0, time.monotonic() - locked),
            )

    def resume_migration(self, guard: LeaseGuard) -> JobsOutcome:
        resumed = self.jobs.migrator.resume(checkpoint=guard.checkpoint)
        if resumed is None:
            return JobsOutcome()
        return JobsOutcome(resumed.days, 0, (f"kind_migration_resumed v{resumed.version}", *resumed.signals))
