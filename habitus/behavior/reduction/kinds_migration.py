"""把词表的"下一版"落到树上（设计 三-③、四-④）：写迁移计划 → 扫一遍重打 ``kind_token`` → 写变更日志 → 清待定池 → 删计划。

计划文件就是那一版本身；崩溃后按它续做，不再问模型。每条迁移只在树上的 token 仍是迁移前的样子时才改
（已经是目标 = 上次做过；既不是源也不是目标 = 别处改过，跳过并留信号）。写进变更日志的是**真正生效的**
那些迁移，日志与树一致。调用方必须持归约的 sweep 锁（与发布互斥）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date

from habitus.behavior.kinds.changes import Move, VersionRecord
from habitus.behavior.kinds.model import BehaviorKindError
from habitus.behavior.kinds.store import BehaviorKindJournalFullError, BehaviorKindStore
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.tree.writer import BehaviorDocumentWriter, BehaviorWriteConfig
from habitus.behavior.uri import BehaviorURI
from habitus.infrastructure.store.contracts.lock import LockStore


@dataclass(frozen=True)
class MigrationOutcome:
    version: int | None
    applied: int
    skipped: int
    days: frozenset[date]
    signals: tuple[str, ...]


class KindMigrator:
    def __init__(
        self,
        tree: BehaviorTree,
        lock_store: LockStore,
        store: BehaviorKindStore,
        *,
        write_config: BehaviorWriteConfig | None = None,
    ) -> None:
        self.tree = tree
        self.lock_store = lock_store
        self.store = store
        self.write_config = write_config

    def apply(self, record: VersionRecord, *, checkpoint: Callable[[], object]) -> MigrationOutcome:
        """写计划并立刻执行。需要在两步之间记别的状态时，分开调 ``plan`` 与 ``resume``。"""

        self.plan(record)
        return self._carry_out(record, checkpoint)

    def plan(self, record: VersionRecord) -> None:
        """写下迁移计划（之后崩在哪一步，``resume`` 都按它续做）。"""

        if self.store.read_migration() is not None:
            raise BehaviorKindError("an unfinished migration plan exists; resume it first")
        self.store.write_migration(record)

    def resume(self, *, checkpoint: Callable[[], object]) -> MigrationOutcome | None:
        """有没做完的计划就续做；没有就返回 ``None``。"""

        plan = self.store.read_migration()
        if plan is None:
            return None
        if self.store.read().version >= plan.version:
            # 日志已经写了，只差清池与删计划。
            self._finish(plan)
            # 改过的日子从计划里算回来：崩在"写完日志、删计划之前"时，这些天的概览仍要刷新。
            days = frozenset(BehaviorURI.parse(move.occurrence).to_address().occurred_on for move in plan.moves)
            return MigrationOutcome(plan.version, 0, 0, days, (f"kind_migration_finished v{plan.version}",))
        return self._carry_out(plan, checkpoint)

    def _carry_out(self, record: VersionRecord, checkpoint: Callable[[], object]) -> MigrationOutcome:
        if not self.store.has_room_for(record):
            return self._refuse(record)
        writer = BehaviorDocumentWriter(self.tree, self.lock_store, clock=lambda: record.at, config=self.write_config)
        applied: list[Move] = []
        days: set[date] = set()
        signals: list[str] = []
        for index, move in enumerate(record.moves):
            if index % self.store.config.checkpoint_every == 0:
                checkpoint()
            address = BehaviorURI.parse(move.occurrence).to_address()
            try:
                current = self.tree.read(address).fields.get("kind_token")
            except FileNotFoundError:
                signals.append(f"kind_migration_skipped {move.occurrence}: occurrence is gone")
                continue
            if current == move.source:
                writer.restamp_kind_token(address, move.target)
            elif current != move.target:
                signals.append(f"kind_migration_skipped {move.occurrence}: token is {current!r}, not {move.source!r}")
                continue
            applied.append(move)
            days.add(address.occurred_on)
        if record.operations or applied:
            self.store.append(replace(record, moves=tuple(applied)), expected_version=record.version - 1)
        else:
            signals.append(f"kind_migration_empty v{record.version}: nothing left to change")
        self._finish(record)
        version = record.version if record.operations or applied else None
        skipped = len(record.moves) - len(applied)
        return MigrationOutcome(version, len(applied), skipped, frozenset(days), tuple(signals))

    def _refuse(self, record: VersionRecord) -> MigrationOutcome:
        """计划写下之后日志才放不下这一版（上限改小过）：树上一条都还没动就撤下计划、留信号，不每轮在同一处失败；
        已经动过就只能停在这里等人处理（放宽上限或截断日志）。"""

        for move in record.moves:
            address = BehaviorURI.parse(move.occurrence).to_address()
            if self.tree.exists(address) and self.tree.read(address).fields.get("kind_token") == move.target:
                raise BehaviorKindJournalFullError(
                    f"version {record.version} is half applied but the change log has no room for it"
                )
        self.store.clear_migration()
        signal = f"kind_migration_refused v{record.version}: the change log has no room for it; the tree is unchanged"
        return MigrationOutcome(None, 0, len(record.moves), frozenset(), (signal,))

    def _finish(self, record: VersionRecord) -> None:
        # 迁走的条目（转正、改归别的类）移出待定池；版本里的迁移不会把条目迁进「待定」。
        moved = {move.occurrence for move in record.moves}
        snapshot = self.store.read_pending()
        remaining = snapshot.pool.without(moved)
        if len(remaining.entries) != len(snapshot.pool.entries):
            self.store.replace_pending(remaining, expected_revision=snapshot.revision)
        self.store.clear_migration()


__all__ = ["KindMigrator", "MigrationOutcome"]
