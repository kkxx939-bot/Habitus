"""词表的三件整理活，在归约 sweep 锁下执行（改树与发布互斥）：

- 每晚新增（到点才跑，``nightly_due``）；
- 定期拆改（到点才跑，``revision_due``；所有用户每周都跑，放行靠证据规则与锚点自检，裁定 17）；

每晚新增认作"已有某类"的待定条目交白天归类重判一次（内容从树上取，``_content_of``），重判过的在池里打标记。
三者都只从词表要"下一版"这份事实，再交 ``KindMigrator`` 落到树上；返回受影响的日子，由 runner 去刷新语义面。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, tzinfo

from habitus.behavior.kinds.changes import VersionRecord
from habitus.behavior.kinds.classify import OccurrenceContent
from habitus.behavior.kinds.ids import is_marker, lane_of_token
from habitus.behavior.kinds.nightly import NightlyGrower
from habitus.behavior.kinds.revision import Member, Reviser
from habitus.behavior.kinds.schedule import JobState, nightly_due, revision_due, revision_period
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.model import BehaviorKind
from habitus.behavior.reduction.kinds_migration import KindMigrator
from habitus.behavior.reduction.kinds_step import KindStamping, content_from_fields, is_countable
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI


@dataclass(frozen=True)
class JobsOutcome:
    days: frozenset[date] = frozenset()
    model_calls: int = 0
    signals: tuple[str, ...] = ()

    def __add__(self, other: JobsOutcome) -> JobsOutcome:
        return JobsOutcome(
            self.days | other.days, self.model_calls + other.model_calls, (*self.signals, *other.signals)
        )


@dataclass(frozen=True)
class _Occurrence:
    uri: str
    token: str
    day: date
    content: OccurrenceContent = field(compare=False)


class VocabularyJobs:
    def __init__(
        self,
        *,
        tree: BehaviorTree,
        stamping: KindStamping,
        migrator: KindMigrator,
        grower: NightlyGrower,
        reviser: Reviser,
        zone: tzinfo,
    ) -> None:
        """``zone`` 是用户的本地时区（组合根从 ``config.locale`` 注入）：每晚几点、每周几都按它算，不读进程时区。"""

        if not isinstance(zone, tzinfo):
            raise TypeError("zone must be a tzinfo")
        self.zone = zone
        self.tree = tree
        self.stamping = stamping
        self.store: BehaviorKindStore = stamping.store
        self.migrator = migrator
        self.grower = grower
        self.reviser = reviser
        self.config = grower.config

    async def run_due(self, now: datetime, *, checkpoint: Callable[[], object]) -> JobsOutcome:
        local = now.astimezone(self.zone)
        state = self.store.read_jobs()
        outcome = JobsOutcome()
        if nightly_due(local, state.nightly_at, hour=self.config.nightly_hour):
            outcome += await self.grow(now, checkpoint=checkpoint)
        if revision_due(local, state.revision_at, weekday=self.config.revision_weekday, hour=self.config.revision_hour):
            outcome += await self.revise(now, checkpoint=checkpoint)
        return outcome

    async def grow(self, now: datetime, *, checkpoint: Callable[[], object]) -> JobsOutcome:
        vocabulary = self.store.read()
        result = await self.grower.grow(
            self.store.read_pending().pool, vocabulary, now=now, content_of=self._content_of, checkpoint=checkpoint
        )
        outcome = JobsOutcome(model_calls=result.model_calls, signals=result.signals)
        if result.rechecked:
            # 先打"已重判"标记再迁移：崩在两步之间时迁移计划还没写，下一晚这些条目不会被再问一次
            snapshot = self.store.read_pending()
            self.store.replace_pending(snapshot.pool.rechecked(result.rechecked), expected_revision=snapshot.revision)
        state = self.store.read_jobs().after_nightly(now)
        return outcome + self._migrate(result.record, state, checkpoint)

    def _content_of(self, occurrence: str) -> OccurrenceContent | None:
        """待定条目在树上的内容（给重判用，与白天归类同一形状）；树上已没有就给 None。"""

        address = BehaviorURI.parse(occurrence).to_address()
        if not self.tree.exists(address):
            return None
        return content_from_fields(self.tree.read(address).fields)

    async def revise(self, now: datetime, *, checkpoint: Callable[[], object]) -> JobsOutcome:
        vocabulary = self.store.read()
        members = [
            Member(item.uri, item.token, lane_of_token(item.token), item.day, item.content)
            for item in self._occurrences(checkpoint)
            if not is_marker(item.token)
        ]
        period = revision_period(
            now.astimezone(self.zone), weekday=self.config.revision_weekday, hour=self.config.revision_hour
        )
        result = await self.reviser.revise(
            vocabulary,
            members,
            self.store.records(),
            self.store.read_jobs(),
            now=now,
            period=period,
            checkpoint=checkpoint,
        )
        outcome = JobsOutcome(model_calls=result.model_calls, signals=result.signals)
        return outcome + self._migrate(result.record, result.state, checkpoint)

    def _migrate(
        self, record: VersionRecord | None, state: JobState | None, checkpoint: Callable[[], object]
    ) -> JobsOutcome:
        """顺序：写迁移计划 → 写定时状态 → 执行迁移。丢锁时计划在、状态也已存，下一轮续做完迁移，不会重跑这次整理。"""

        if record is not None:
            self.migrator.plan(record)
        if state is not None:
            self.store.replace_jobs(state)
        if record is None:
            return JobsOutcome()
        migrated = self.migrator.resume(checkpoint=checkpoint)
        assert migrated is not None
        notes = (
            f"kind_version v{migrated.version}: {record.note} (moved {migrated.applied}, skipped {migrated.skipped})",
        )
        return JobsOutcome(migrated.days, 0, (*migrated.signals, *notes))

    def _occurrences(self, checkpoint: Callable[[], object]) -> Iterator[_Occurrence]:
        """树上全部可计的 occurrence（撞车消歧的重复记录跳过，与预测树同一口径）。"""

        every = self.config.checkpoint_every
        for index, document in enumerate(self.tree.iter_documents(BehaviorKind.OCCURRENCE)):
            if index % every == 0:
                checkpoint()
            if not is_countable(document.fields):
                continue
            uri = str(BehaviorURI.from_address(document.address))
            token = str(document.fields.get("kind_token") or "")
            yield _Occurrence(uri, token, document.address.occurred_on, content_from_fields(document.fields))


__all__ = ["JobsOutcome", "VocabularyJobs"]
