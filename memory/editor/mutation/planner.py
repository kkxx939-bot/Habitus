"""串联严格节点匹配与确定性字段合并的纯记忆变更规划器。"""

from __future__ import annotations

from memory.editor.candidate import (
    MemoryCandidateBatch,
    MemoryIdentityProposalType,
)
from memory.editor.mutation.matcher import MemoryNodeMatcher
from memory.editor.mutation.merge import MemoryFieldMerger
from memory.editor.mutation.model import (
    MemoryMutation,
    MemoryMutationAction,
    MemoryMutationPlan,
    MemoryMutationReadSet,
    MemoryNodeMatchStatus,
)
from memory.editor.page_id import MemoryPageIdError, MemoryPageIdMap
from memory.model import MemoryKind
from memory.schema import MemorySchemaRegistry


class MemoryMutationPlanningError(ValueError):
    """节点内容计划无法形成完整一致的最终 URI 绑定。"""


class MemoryMutationPlanner:
    """只计算节点内容计划，不读取存储、不加锁也不推进 revision。"""

    def __init__(self, registry: MemorySchemaRegistry | None = None) -> None:
        if registry is not None and not isinstance(registry, MemorySchemaRegistry):
            raise TypeError("registry must be a MemorySchemaRegistry")
        resolved_registry = registry or MemorySchemaRegistry.load_default()
        self.matcher = MemoryNodeMatcher()
        self.merger = MemoryFieldMerger(resolved_registry)

    def plan(
        self,
        batch: MemoryCandidateBatch,
        read_set: MemoryMutationReadSet,
        page_ids: MemoryPageIdMap,
    ) -> MemoryMutationPlan:
        """生成 create、update 或 noop 计划，并验证每个候选最终 URI。"""

        if not isinstance(batch, MemoryCandidateBatch):
            raise TypeError("batch must be a MemoryCandidateBatch")
        if not isinstance(read_set, MemoryMutationReadSet):
            raise TypeError("read_set must be a MemoryMutationReadSet")
        if not isinstance(page_ids, MemoryPageIdMap):
            raise TypeError("page_ids must be a MemoryPageIdMap")

        mutations: list[MemoryMutation] = []
        working_page_ids = page_ids.copy()
        preservation_targets = {
            proposal.target_page_id
            for proposal in batch.identity_proposals
            if proposal.proposal_type is MemoryIdentityProposalType.SAME_MEMORY
        }
        for match in self.matcher.match(batch, read_set, page_ids):
            merged = self.merger.merge(match)
            if match.status is MemoryNodeMatchStatus.NEW:
                action = MemoryMutationAction.CREATE
            elif merged.changed_fields:
                action = MemoryMutationAction.UPDATE
            else:
                action = MemoryMutationAction.NOOP
            confirms_intention = match.candidate.confirmed is True
            if match.candidate.kind is MemoryKind.INTENTION:
                if (
                    not confirms_intention
                    and (
                        action is MemoryMutationAction.CREATE
                        or match.candidate.page_id not in preservation_targets
                    )
                ):
                    raise MemoryMutationPlanningError(
                        "unconfirmed Intention candidate may only preserve an existing same_memory merge target"
                    )
            mutation = MemoryMutation(
                match=match,
                action=action,
                fields=merged.fields,
                changed_fields=merged.changed_fields,
                confirms_intention=confirms_intention,
            )
            try:
                working_page_ids.register_resolved(match.uri, match.candidate.page_id)
            except MemoryPageIdError as exc:
                raise MemoryMutationPlanningError("mutation plan produced an invalid final page binding") from exc
            mutations.append(mutation)

        return MemoryMutationPlan(
            read_set=read_set,
            mutations=tuple(mutations),
        )


__all__ = ["MemoryMutationPlanner", "MemoryMutationPlanningError"]
