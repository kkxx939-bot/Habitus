"""串联最终身份解析与无副作用的关系变更规划。"""

from __future__ import annotations

from habitus.memory.editor.candidate import MemoryCandidateBatch
from habitus.memory.editor.identity import MemoryFinalIdentityMap
from habitus.memory.editor.link import MemoryRelationResolver, MemoryResolvedRelation
from habitus.memory.editor.link_plan import (
    MemoryRelationPlan,
    MemoryRelationPlanner,
    MemoryRelationReadSet,
)


class MemoryRelationshipEditor:
    """只计算最终 Links/Backlinks，不取得锁也不写入记忆树。"""

    def __init__(
        self,
        *,
        resolver: MemoryRelationResolver | None = None,
        planner: MemoryRelationPlanner | None = None,
    ) -> None:
        if resolver is not None and not isinstance(resolver, MemoryRelationResolver):
            raise TypeError("resolver must be a MemoryRelationResolver")
        if planner is not None and not isinstance(planner, MemoryRelationPlanner):
            raise TypeError("planner must be a MemoryRelationPlanner")
        self.resolver = resolver or MemoryRelationResolver()
        self.planner = planner or MemoryRelationPlanner()

    def plan(
        self,
        identities: MemoryFinalIdentityMap,
        operations: tuple[MemoryResolvedRelation, ...],
        read_set: MemoryRelationReadSet,
    ) -> MemoryRelationPlan:
        """按结构迁移、REMOVE、ADD 顺序形成纯计划。"""

        return self.planner.plan(identities, operations, read_set)

    def resolve(
        self,
        batch: MemoryCandidateBatch,
        identities: MemoryFinalIdentityMap,
    ) -> tuple[MemoryResolvedRelation, ...]:
        """在节点最终身份确定后解析临时关系端点。"""

        return self.resolver.resolve(batch, identities)


__all__ = ["MemoryRelationshipEditor"]
