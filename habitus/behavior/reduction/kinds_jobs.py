"""词表的两件整理活，在归约 sweep 锁下执行（改树与发布互斥）。都在夜里那条线上、结构那一段之前，不挂任何钟点（裁定 94 第 8、9 条）：
线经 ``BehaviorReductionRunner.nightly_vocabulary`` 调 ``nightly``——

- 每晚新增（``grow``）：每晚一次；
- 定期拆改（``revise``）：这一晚所在的拆改周期里还没拆改过才跑（``revision_due``；所有用户每周都跑，放行靠证据规则与锚点自检，
  裁定 17）。

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
from habitus.behavior.kinds.schedule import JobState, revision_due, revision_period
from habitus.behavior.kinds.store import BehaviorKindJournalFullError, BehaviorKindStore
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
        """``zone`` 是用户的本地时区（组合根从 ``config.locale`` 注入）：上一次拆改是哪一晚按它算，不读进程时区。"""

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

    async def nightly(self, now: datetime, *, night: date, checkpoint: Callable[[], object]) -> JobsOutcome:
        """线上的这一晚：先每晚新增，再看这一晚所在的拆改周期里拆改过没有，没有就拆改。"""

        outcome = await self.grow(now, checkpoint=checkpoint)
        last = self.store.read_jobs().revision_at
        if revision_due(night, None if last is None else last.astimezone(self.zone).date(), weekday=self.weekday):
            outcome += await self.revise(now, checkpoint=checkpoint, night=night)
        return outcome

    @property
    def weekday(self) -> int:
        return self.config.revision_weekday

    async def grow(self, now: datetime, *, checkpoint: Callable[[], object]) -> JobsOutcome:
        vocabulary = self.store.read()
        result = await self.grower.grow(
            self.store.read_pending().pool, vocabulary, now=now, content_of=self._content_of, checkpoint=checkpoint
        )
        outcome = JobsOutcome(model_calls=result.model_calls, signals=result.signals)
        # 要迁进新类的成员不打"聚组过"：它们靠这一版迁移出池；崩在打标记与写迁移计划之间时，它们得留着"新"，下一晚
        # 再交模型一次把类建出来，否则这个类永远建不成（L10.4）
        moving = frozenset() if result.record is None else frozenset(move.occurrence for move in result.record.moves)
        grouped = result.grouped - moving
        if result.rechecked or grouped or result.missing:
            # 先打"已重判""聚组过"标记、把树上没有了的拿出池，再迁移：崩在两步之间时迁移计划还没写，下一晚这些条目
            # 不会被再问一次，也不会再当新条目整批交给模型
            snapshot = self.store.read_pending()
            pool = (
                snapshot.pool.rechecked(result.rechecked)
                .grouped(grouped, on=now.astimezone(self.zone).date())
                .without(result.missing)
            )
            self.store.replace_pending(pool, expected_revision=snapshot.revision)
        return outcome + self._migrate(result.record, None, checkpoint)

    def _content_of(self, occurrence: str) -> OccurrenceContent | None:
        """待定条目在树上的内容（给重判用，与白天归类同一形状）；树上已没有就给 None。"""

        address = BehaviorURI.parse(occurrence).to_address()
        if not self.tree.exists(address):
            return None
        return content_from_fields(self.tree.read(address).fields)

    async def revise(
        self, now: datetime, *, checkpoint: Callable[[], object], night: date | None = None
    ) -> JobsOutcome:
        """拆改一次。``night`` 是线的那一晚（拆改周期按它算）；运维立刻拆改时不给，按 ``now`` 的本地日算。"""

        vocabulary = self.store.read()
        members = [
            Member(item.uri, item.token, lane_of_token(item.token), item.day, item.content)
            for item in self._occurrences(checkpoint)
            if not is_marker(item.token)
        ]
        period = revision_period(night or now.astimezone(self.zone).date(), weekday=self.weekday)
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
        """顺序：写迁移计划 → 写拆改状态（只有拆改有） → 执行迁移。丢锁时计划在、状态也已存，下一轮续做完迁移，不会重跑这次整理。"""

        if record is not None:
            try:
                self.migrator.plan(record)
            except BehaviorKindJournalFullError as exc:
                # 变更日志放不下这一版：不落计划、不动树；拆改状态照记（不每轮重做这次整理、重问模型），留信号等人处理
                if state is not None:
                    self.store.replace_jobs(state)
                return JobsOutcome(signals=(f"kind_version_refused v{record.version}: {exc}",))
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
        """树上全部可计的 occurrence（撞车消歧的重复记录跳过，与事件序列同一口径）。"""

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
