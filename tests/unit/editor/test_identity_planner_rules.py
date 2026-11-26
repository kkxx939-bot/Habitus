"""确定性身份裁决与 YAML 字段策略的跨类型组合测试。"""

from __future__ import annotations

import pytest

from habitus.infrastructure.editor.snapshot import SnapshotBatch
from habitus.memory.document import MemoryLinkType
from habitus.memory.editor import (
    MemoryCandidate,
    MemoryCandidateBatch,
    MemoryIdentityPlanner,
    MemoryIdentityPlanningError,
    MemoryIdentityProposal,
    MemoryIdentityProposalBasis,
    MemoryIdentityProposalType,
    MemoryMutationPlanner,
    MemoryMutationReadSet,
    MemoryNodeDisposition,
    MemoryRelationAction,
    MemoryRelationCandidate,
)
from habitus.memory.editor.extraction import (
    MemoryExtractionResult,
    MemoryRetrievalAction,
    MemoryRetrievalDecision,
    MemoryRetrievalStatus,
)
from habitus.memory.editor.mutation import MemoryFieldMerger, MemoryNodeMatch, MemoryNodeMatchStatus
from habitus.memory.model import MemoryKind
from habitus.memory.uri import MemoryURI
from tests.helpers import document, memory_snapshot, page_ids_from, snapshot_batch


def candidate_batch(candidate: MemoryCandidate | None, *, proposal, relations=()):
    values = {
        "profile": (),
        "preferences": (),
        "entities": (),
        "tools": (),
        "identity_proposals": (proposal,),
        "relations": relations,
    }
    if candidate is not None:
        field = {
            MemoryKind.PROFILE: "profile",
            MemoryKind.PREFERENCE: "preferences",
            MemoryKind.ENTITY: "entities",
            MemoryKind.TOOL: "tools",
        }[candidate.kind]
        values[field] = (candidate,)
    return MemoryCandidateBatch(**values)


def extraction(old_documents, candidate, proposal, *, relations=()):
    old = snapshot_batch(*old_documents)
    pages = page_ids_from(old)
    batch = candidate_batch(candidate, proposal=proposal, relations=relations)
    if candidate is None:
        targets = SnapshotBatch((), 0)
    else:
        snapshot = old.get(str(MemoryURI.from_address(candidate.address)))
        assert snapshot is not None
        targets = SnapshotBatch((snapshot,), snapshot.size_bytes)
    mutations = MemoryMutationPlanner().plan(
        batch,
        MemoryMutationReadSet(old, targets),
        pages,
    )
    return MemoryExtractionResult(
        conversation_id="conversation-1",
        segment_id="segment-1",
        source_segment_digest="a" * 64,
        candidates=batch,
        mutations=mutations,
        old_memories=old,
        page_ids=pages,
        retrieval_decisions=(
            MemoryRetrievalDecision(
                MemoryRetrievalStatus.IRRELEVANT,
                MemoryRetrievalAction.FINISH,
                None,
                (),
                "上下文已经足够。",
            ),
        ),
        retrieval_observations=(),
    )


def test_same_memory_merge_requires_allowed_kind_and_preserves_live_target() -> None:
    source = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "旧答复长度", "content": "- 偏好简洁回答"},
    )
    target = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "答复长度", "content": "- 偏好简洁回答"},
    )
    old = snapshot_batch(source, target)
    pages = page_ids_from(old)
    source_page = pages.page_id_for(MemoryURI.from_address(source.address))
    target_page = pages.page_id_for(MemoryURI.from_address(target.address))
    proposal = MemoryIdentityProposal(
        MemoryIdentityProposalType.SAME_MEMORY,
        source_page,  # type: ignore[arg-type]
        target_page,
        MemoryIdentityProposalBasis.DUPLICATE_IDENTITY,
    )
    result = MemoryIdentityPlanner().plan(
        extraction(
            (source, target),
            MemoryCandidate(target_page, MemoryKind.PREFERENCE, target.fields),  # type: ignore[arg-type]
            proposal,
        )
    )
    assert result.entry(source_page).disposition is MemoryNodeDisposition.MERGE  # type: ignore[arg-type]
    assert result.resolve(source_page) == MemoryURI.from_address(target.address)  # type: ignore[arg-type]


def test_same_memory_rejects_cross_kind_and_tool_merges_after_candidate_validation() -> None:
    preference = document(MemoryKind.PREFERENCE)
    entity = document(MemoryKind.ENTITY)
    old = snapshot_batch(preference, entity)
    pages = page_ids_from(old)
    source_page = pages.page_id_for(MemoryURI.from_address(preference.address))
    target_page = pages.page_id_for(MemoryURI.from_address(entity.address))
    proposal = MemoryIdentityProposal(
        MemoryIdentityProposalType.SAME_MEMORY,
        source_page,  # type: ignore[arg-type]
        target_page,
        MemoryIdentityProposalBasis.DUPLICATE_IDENTITY,
    )
    with pytest.raises(MemoryIdentityPlanningError, match="same memory type"):
        MemoryIdentityPlanner().plan(
            extraction(
                (preference, entity),
                MemoryCandidate(target_page, MemoryKind.ENTITY, entity.fields),  # type: ignore[arg-type]
                proposal,
            )
        )

    first_tool = document(
        MemoryKind.TOOL,
        fields={"tool_name": "workspace.inspect", "verification": "检查退出状态。"},
    )
    second_tool = document(
        MemoryKind.TOOL,
        fields={"tool_name": "workspace.read", "verification": "检查退出状态。"},
    )
    tools = snapshot_batch(first_tool, second_tool)
    tool_pages = page_ids_from(tools)
    source_id = tool_pages.page_id_for(MemoryURI.from_address(first_tool.address))
    target_id = tool_pages.page_id_for(MemoryURI.from_address(second_tool.address))
    tool_proposal = MemoryIdentityProposal(
        MemoryIdentityProposalType.SAME_MEMORY,
        source_id,  # type: ignore[arg-type]
        target_id,
        MemoryIdentityProposalBasis.DUPLICATE_IDENTITY,
    )
    with pytest.raises(MemoryIdentityPlanningError, match="not allowed"):
        MemoryIdentityPlanner().plan(
            extraction(
                (first_tool, second_tool),
                MemoryCandidate(target_id, MemoryKind.TOOL, second_tool.fields),  # type: ignore[arg-type]
                tool_proposal,
            )
        )


def test_retired_identity_source_cannot_also_participate_in_relation_candidate() -> None:
    source = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "旧名称", "content": "- 偏好简洁"},
    )
    target = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "新名称", "content": "- 偏好简洁"},
    )
    old = snapshot_batch(source, target)
    pages = page_ids_from(old)
    source_page = pages.page_id_for(MemoryURI.from_address(source.address))
    target_page = pages.page_id_for(MemoryURI.from_address(target.address))
    proposal = MemoryIdentityProposal(
        MemoryIdentityProposalType.SAME_MEMORY,
        source_page,  # type: ignore[arg-type]
        target_page,
        MemoryIdentityProposalBasis.DUPLICATE_IDENTITY,
    )
    relation = MemoryRelationCandidate(
        MemoryRelationAction.ADD,
        source_page,  # type: ignore[arg-type]
        target_page,  # type: ignore[arg-type]
        MemoryLinkType.RELATED_TO,
    )
    value = extraction(
        (source, target),
        MemoryCandidate(target_page, MemoryKind.PREFERENCE, target.fields),  # type: ignore[arg-type]
        proposal,
        relations=(relation,),
    )
    with pytest.raises(MemoryIdentityPlanningError, match="cannot participate"):
        MemoryIdentityPlanner().plan(value)


@pytest.mark.parametrize(
    "basis",
    [MemoryIdentityProposalBasis.EXPLICIT_FORGET, MemoryIdentityProposalBasis.FULLY_INVALIDATED],
)
def test_validated_remove_memory_becomes_delete_without_creating_a_content_mutation(basis) -> None:
    source = document(MemoryKind.ENTITY)
    old = snapshot_batch(source)
    pages = page_ids_from(old)
    source_page = pages.page_id_for(MemoryURI.from_address(source.address))
    proposal = MemoryIdentityProposal(
        MemoryIdentityProposalType.REMOVE_MEMORY,
        source_page,  # type: ignore[arg-type]
        None,
        basis,
    )
    result = MemoryIdentityPlanner().plan(extraction((source,), None, proposal))
    assert result.entry(source_page).disposition is MemoryNodeDisposition.DELETE  # type: ignore[arg-type]


def test_replace_fields_remove_absent_optional_values_while_patch_fields_preserve_them() -> None:
    tool = document(
        MemoryKind.TOOL,
        fields={
            "tool_name": "workspace.inspect",
            "purpose": "查看工作区状态",
            "constraints": "只读",
            "verification": "检查退出状态。",
        },
    )
    tool_candidate = MemoryCandidate(
        1,
        MemoryKind.TOOL,
        {"tool_name": "workspace.inspect", "verification": "检查退出状态和输出。"},
    )
    tool_match = MemoryNodeMatch(
        tool_candidate,
        MemoryURI.from_address(tool.address),
        MemoryNodeMatchStatus.EXISTING,
        memory_snapshot(tool),
    )
    merged_tool = MemoryFieldMerger().merge(tool_match)
    assert merged_tool.fields == {
        "tool_name": "workspace.inspect",
        "constraints": "只读",
        "verification": "检查退出状态和输出。",
    }
    assert set(merged_tool.changed_fields) == {"purpose", "verification"}

    entity = document(
        MemoryKind.ENTITY,
        fields={
            "category": "项目",
            "name": "Habitus",
            "summary": "一个记忆系统。",
            "details": "## 状态\n- 正在重构",
        },
    )
    entity_candidate = MemoryCandidate(
        1,
        MemoryKind.ENTITY,
        {"category": "项目", "name": "Habitus", "summary": "一个长期记忆系统。"},
    )
    entity_match = MemoryNodeMatch(
        entity_candidate,
        MemoryURI.from_address(entity.address),
        MemoryNodeMatchStatus.EXISTING,
        memory_snapshot(entity),
    )
    merged_entity = MemoryFieldMerger().merge(entity_match)
    assert merged_entity.fields["details"] == entity.fields["details"]
    assert merged_entity.changed_fields == ("summary",)
