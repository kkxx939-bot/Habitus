"""确定性身份裁决与 YAML 字段策略的跨类型组合测试。"""

from __future__ import annotations

import pytest

from infrastructure.editor.snapshot import SnapshotBatch
from memory.document import MemoryLinkType
from memory.editor import (
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
    MemoryPageIdMap,
    MemoryRelationAction,
    MemoryRelationCandidate,
)
from memory.editor.extraction import (
    MemoryCandidateReview,
    MemoryExtractionResult,
    MemoryRetrievalAction,
    MemoryRetrievalDecision,
    MemoryRetrievalStatus,
    MemoryReviewDecision,
)
from memory.editor.mutation import MemoryFieldMerger, MemoryNodeMatch, MemoryNodeMatchStatus
from memory.model import MemoryKind
from memory.uri import MemoryURI
from tests.helpers import document, memory_snapshot, snapshot_batch


def candidate_batch(candidate: MemoryCandidate | None, *, proposal, relations=()):
    values = {
        "profile": (),
        "preferences": (),
        "entities": (),
        "tools": (),
        "events": (),
        "intentions": (),
        "identity_proposals": (proposal,),
        "relations": relations,
    }
    if candidate is not None:
        field = {
            MemoryKind.PROFILE: "profile",
            MemoryKind.PREFERENCE: "preferences",
            MemoryKind.ENTITY: "entities",
            MemoryKind.TOOL: "tools",
            MemoryKind.EVENT: "events",
            MemoryKind.INTENTION: "intentions",
        }[candidate.kind]
        values[field] = (candidate,)
    return MemoryCandidateBatch(**values)


def extraction(old_documents, candidate, proposal, *, relations=()):
    old = snapshot_batch(*old_documents)
    pages = MemoryPageIdMap.from_snapshots(old)
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
                None,
                "上下文已经足够。",
            ),
        ),
        retrieval_observations=(),
        review=MemoryCandidateReview(MemoryReviewDecision.ACCEPT, ()),
        candidate_attempts=1,
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
    pages = MemoryPageIdMap.from_snapshots(old)
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


def test_same_memory_rejects_cross_kind_and_event_merges_even_after_model_review() -> None:
    preference = document(MemoryKind.PREFERENCE)
    entity = document(MemoryKind.ENTITY)
    old = snapshot_batch(preference, entity)
    pages = MemoryPageIdMap.from_snapshots(old)
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

    first_event = document(
        MemoryKind.EVENT,
        fields={"event_date": "2026-07-01", "event_name": "旧决定", "summary": "确认采用 A。"},
    )
    second_event = document(
        MemoryKind.EVENT,
        fields={"event_date": "2026-07-02", "event_name": "新决定", "summary": "确认采用 A。"},
    )
    events = snapshot_batch(first_event, second_event)
    event_pages = MemoryPageIdMap.from_snapshots(events)
    source_id = event_pages.page_id_for(MemoryURI.from_address(first_event.address))
    target_id = event_pages.page_id_for(MemoryURI.from_address(second_event.address))
    event_proposal = MemoryIdentityProposal(
        MemoryIdentityProposalType.SAME_MEMORY,
        source_id,  # type: ignore[arg-type]
        target_id,
        MemoryIdentityProposalBasis.DUPLICATE_IDENTITY,
    )
    with pytest.raises(MemoryIdentityPlanningError, match="not allowed"):
        MemoryIdentityPlanner().plan(
            extraction(
                (first_event, second_event),
                MemoryCandidate(target_id, MemoryKind.EVENT, second_event.fields),  # type: ignore[arg-type]
                event_proposal,
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
    pages = MemoryPageIdMap.from_snapshots(old)
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
def test_reviewed_remove_memory_becomes_delete_without_creating_a_content_mutation(basis) -> None:
    source = document(MemoryKind.ENTITY)
    old = snapshot_batch(source)
    pages = MemoryPageIdMap.from_snapshots(old)
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
    intention = document(
        MemoryKind.INTENTION,
        fields={
            "intent_name": "补齐测试",
            "status": "blocked",
            "next_step": "等待依赖",
            "blockers": "外部服务不可用",
            "target_time": "本月底",
        },
    )
    intention_candidate = MemoryCandidate(
        1,
        MemoryKind.INTENTION,
        {"intent_name": "补齐测试", "status": "open", "next_step": "继续实现"},
        confirmed=True,
    )
    intention_match = MemoryNodeMatch(
        intention_candidate,
        MemoryURI.from_address(intention.address),
        MemoryNodeMatchStatus.EXISTING,
        memory_snapshot(intention),
    )
    merged_intention = MemoryFieldMerger().merge(intention_match)
    assert merged_intention.fields == {
        "intent_name": "补齐测试",
        "status": "open",
        "next_step": "继续实现",
    }
    assert set(merged_intention.changed_fields) == {"status", "next_step", "blockers", "target_time"}

    entity = document(
        MemoryKind.ENTITY,
        fields={
            "category": "项目",
            "name": "m2bOS",
            "summary": "一个记忆系统。",
            "details": "## 状态\n- 正在重构",
        },
    )
    entity_candidate = MemoryCandidate(
        1,
        MemoryKind.ENTITY,
        {"category": "项目", "name": "m2bOS", "summary": "一个长期记忆系统。"},
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
