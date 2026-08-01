"""Conversation 解析到统一记忆事务的领域编排入口。"""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from memory.document import MemoryDocument
from memory.editor.extraction import MemoryExtractionLoop, MemoryExtractionResult
from memory.editor.identity import MemoryFinalIdentityMap, MemoryIdentityPlanner
from memory.editor.link_plan import MemoryRelationPlan, MemoryRelationReadSet
from memory.editor.mutation import MemoryMutationPlan, MemoryMutationReadSet
from memory.editor.relation_reader import MemoryRelationReadSetLoader
from memory.editor.relationship import MemoryRelationshipEditor
from memory.editor.transaction import (
    MemoryCommitPlan,
    MemoryCommitResult,
    MemoryCommitTransaction,
)
from pre.conversation import ConversationSegment


@dataclass(frozen=True)
class MemoryEditorPlan:
    """保留一次编辑从模型候选到统一提交计划的全部稳定边界。"""

    extraction: MemoryExtractionResult | None
    mutations: MemoryMutationPlan
    identities: MemoryFinalIdentityMap
    commit: MemoryCommitPlan
    deferred: bool = False

    def __post_init__(self) -> None:
        if self.extraction is not None and not isinstance(self.extraction, MemoryExtractionResult):
            raise TypeError("extraction must be a MemoryExtractionResult or None")
        if not isinstance(self.mutations, MemoryMutationPlan):
            raise TypeError("mutations must be a MemoryMutationPlan")
        if not isinstance(self.identities, MemoryFinalIdentityMap):
            raise TypeError("identities must be a MemoryFinalIdentityMap")
        if not isinstance(self.commit, MemoryCommitPlan):
            raise TypeError("commit must be a MemoryCommitPlan")
        if not isinstance(self.deferred, bool):
            raise TypeError("deferred must be boolean")
        if self.deferred:
            if self.extraction is not None:
                raise ValueError("deferred editor plan cannot contain an extraction result")
            if (
                self.mutations.mutations
                or self.identities.entries
                or self.commit.read_set.snapshots
                or self.commit.changed_uris
            ):
                raise ValueError("deferred editor plan must be an empty deterministic plan")
        elif self.extraction is None:
            raise ValueError("analyzed editor plan requires an extraction result")

    @classmethod
    def deferred_until_turn_boundary(cls) -> MemoryEditorPlan:
        """构造不调用模型且不改变 L2 的显式延迟计划。"""

        empty: SnapshotBatch[MemoryDocument] = SnapshotBatch((), 0)
        mutations = MemoryMutationPlan(
            MemoryMutationReadSet(empty, empty),
            (),
        )
        identities = MemoryFinalIdentityMap(())
        relations = MemoryRelationPlan(
            MemoryRelationReadSet((), empty),
            (),
            (),
            (),
            (),
        )
        return cls(
            extraction=None,
            mutations=mutations,
            identities=identities,
            commit=MemoryCommitPlan.build(mutations, identities, relations),
            deferred=True,
        )


class MemoryEditor:
    """严格串联提取、字段规划、最终身份、关系规划和统一提交。"""

    def __init__(
        self,
        *,
        extraction_loop: MemoryExtractionLoop,
        identity_planner: MemoryIdentityPlanner,
        transaction: MemoryCommitTransaction,
        relation_reader: MemoryRelationReadSetLoader | None = None,
        relationship_editor: MemoryRelationshipEditor | None = None,
    ) -> None:
        if not isinstance(extraction_loop, MemoryExtractionLoop):
            raise TypeError("extraction_loop must be a MemoryExtractionLoop")
        if not isinstance(identity_planner, MemoryIdentityPlanner):
            raise TypeError("identity_planner must be a MemoryIdentityPlanner")
        if not isinstance(transaction, MemoryCommitTransaction):
            raise TypeError("transaction must be a MemoryCommitTransaction")
        if relation_reader is not None and not isinstance(
            relation_reader,
            MemoryRelationReadSetLoader,
        ):
            raise TypeError("relation_reader must be a MemoryRelationReadSetLoader")
        if relationship_editor is not None and not isinstance(
            relationship_editor,
            MemoryRelationshipEditor,
        ):
            raise TypeError("relationship_editor must be a MemoryRelationshipEditor")
        self.extraction_loop = extraction_loop
        self.identity_planner = identity_planner
        self.transaction = transaction
        self.relation_reader = relation_reader or MemoryRelationReadSetLoader(extraction_loop.mutation_reader.reader)
        self.relationship_editor = relationship_editor or MemoryRelationshipEditor()

    async def plan(self, segment: ConversationSegment) -> MemoryEditorPlan:
        """对不可变完整 Segment 形成不含落盘副作用的统一计划。"""

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        extraction = await self.extraction_loop.extract(segment)
        mutations = extraction.mutations
        identities = self.identity_planner.plan(extraction)
        snapshots = self._merge_snapshots(
            extraction.old_memories,
            mutations.read_set.target_memories,
        )
        relation_operations = self.relationship_editor.resolve(
            extraction.candidates,
            identities,
        )
        relation_read_set = self.relation_reader.load(
            snapshots,
            identities,
            relation_operations,
        )
        relations = self.relationship_editor.plan(
            identities,
            relation_operations,
            relation_read_set,
        )
        commit = MemoryCommitPlan.build(mutations, identities, relations)
        return MemoryEditorPlan(
            extraction=extraction,
            mutations=mutations,
            identities=identities,
            commit=commit,
        )

    async def edit(
        self,
        segment: ConversationSegment,
        *,
        transaction_id: str | None = None,
        retain_transaction_journal: bool = False,
    ) -> MemoryCommitResult:
        """先完成全部语义规划，再由唯一事务发布。"""

        plan = await self.plan(segment)
        return self.commit(
            plan,
            transaction_id=transaction_id,
            retain_transaction_journal=retain_transaction_journal,
        )

    def commit(
        self,
        plan: MemoryEditorPlan,
        *,
        transaction_id: str | None = None,
        retain_transaction_journal: bool = False,
    ) -> MemoryCommitResult:
        """发布已经完成语义提取和关系规划的统一计划。"""

        if not isinstance(plan, MemoryEditorPlan):
            raise TypeError("plan must be MemoryEditorPlan")
        return self.transaction.commit(
            plan.commit,
            transaction_id=transaction_id,
            retain_journal=retain_transaction_journal,
        )

    @staticmethod
    def _merge_snapshots(
        *batches: SnapshotBatch[MemoryDocument],
    ) -> SnapshotBatch[MemoryDocument]:
        snapshots: dict[str, VersionedSnapshot[MemoryDocument]] = {}
        for batch in batches:
            if not isinstance(batch, SnapshotBatch):
                raise TypeError("snapshot batch is invalid")
            for snapshot in batch.snapshots:
                previous = snapshots.get(snapshot.identity)
                if previous is not None and previous != snapshot:
                    raise ValueError(f"memory changed before relation planning: {snapshot.identity}")
                snapshots[snapshot.identity] = snapshot
        ordered = tuple(snapshots[key] for key in sorted(snapshots))
        return SnapshotBatch(
            snapshots=ordered,
            total_bytes=sum(snapshot.size_bytes for snapshot in ordered),
        )


__all__ = [
    "MemoryEditor",
    "MemoryEditorPlan",
]
