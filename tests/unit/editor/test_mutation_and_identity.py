"""精确节点匹配、YAML 字段合并和最终身份安全规划测试。"""

import pytest

from habitus.infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from habitus.memory.editor import (
    MemoryCandidate,
    MemoryCandidateBatch,
    MemoryFinalIdentity,
    MemoryFinalIdentityError,
    MemoryFinalIdentityMap,
    MemoryMutationAction,
    MemoryMutationPlanner,
    MemoryMutationPlanningError,
    MemoryMutationReadSet,
    MemoryNodeDisposition,
    MemoryPageIdMap,
)
from habitus.memory.model import MemoryKind
from habitus.memory.uri import MemoryURI
from tests.helpers import document, memory_fields, memory_snapshot


def target_batch(*snapshots: VersionedSnapshot) -> SnapshotBatch:
    ordered = tuple(sorted(snapshots, key=lambda item: item.identity))
    return SnapshotBatch(ordered, sum(item.size_bytes for item in ordered))


def test_existing_candidate_becomes_noop_or_update_without_llm_patch_operations() -> None:
    old = document(MemoryKind.PREFERENCE)
    old_snapshot = memory_snapshot(old)
    old_batch = target_batch(old_snapshot)
    page_ids = MemoryPageIdMap.from_snapshots(old_batch)
    planner = MemoryMutationPlanner()

    same = MemoryCandidate(1, MemoryKind.PREFERENCE, old.fields)
    noop = planner.plan(
        MemoryCandidateBatch(preferences=(same,)),
        MemoryMutationReadSet(old_batch, old_batch),
        page_ids,
    )
    assert noop.mutations[0].action is MemoryMutationAction.NOOP
    assert noop.changed_mutations == ()

    changed_fields = {**old.fields, "content": "- 偏好用一句话回答"}
    update = planner.plan(
        MemoryCandidateBatch(preferences=(MemoryCandidate(1, MemoryKind.PREFERENCE, changed_fields),)),
        MemoryMutationReadSet(old_batch, old_batch),
        page_ids,
    )
    assert update.mutations[0].action is MemoryMutationAction.UPDATE
    assert update.mutations[0].changed_fields == ("content",)
    assert update.mutations[0].fields["content"] == "- 偏好用一句话回答"


def test_new_candidate_requires_explicit_missing_target_and_uses_page_id_100_or_above() -> None:
    candidate = MemoryCandidate(100, MemoryKind.PREFERENCE, memory_fields(MemoryKind.PREFERENCE))
    uri = MemoryURI.from_address(candidate.address)
    missing = VersionedSnapshot.missing(str(uri))
    read_set = MemoryMutationReadSet(SnapshotBatch((), 0), target_batch(missing))
    plan = MemoryMutationPlanner().plan(MemoryCandidateBatch(preferences=(candidate,)), read_set, MemoryPageIdMap())
    assert plan.mutations[0].action is MemoryMutationAction.CREATE
    assert plan.mutations[0].uri == uri

    occupied = document(MemoryKind.PREFERENCE)
    occupied_snapshot = memory_snapshot(occupied)
    with pytest.raises(Exception, match="already exists"):
        MemoryMutationPlanner().plan(
            MemoryCandidateBatch(preferences=(candidate,)),
            MemoryMutationReadSet(SnapshotBatch((), 0), target_batch(occupied_snapshot)),
            MemoryPageIdMap(),
        )


def test_candidate_rewriting_an_old_add_only_event_keeps_it_as_audited_noop() -> None:
    """主链端到端扩 3 份 Job #12：复用旧 Event 的 page_id 改写 summary，旧的保留、记 NOOP，同批其余照常。"""

    old_event = document(MemoryKind.EVENT)
    old_preference = document(MemoryKind.PREFERENCE)
    batch = target_batch(memory_snapshot(old_event), memory_snapshot(old_preference))
    page_ids = MemoryPageIdMap.from_snapshots(batch)
    event_uri = MemoryURI.from_address(old_event.address)
    event_page_id = page_ids.page_id_for(event_uri)
    preference_page_id = page_ids.page_id_for(MemoryURI.from_address(old_preference.address))
    assert event_page_id is not None and preference_page_id is not None
    rewritten = {**old_event.fields, "summary": "不同事件内容", "details": "补上的细节。"}
    updated_preference = {**old_preference.fields, "content": "- 偏好用一句话回答"}

    plan = MemoryMutationPlanner().plan(
        MemoryCandidateBatch(
            preferences=(MemoryCandidate(preference_page_id, MemoryKind.PREFERENCE, updated_preference),),
            events=(MemoryCandidate(event_page_id, MemoryKind.EVENT, rewritten),),
        ),
        MemoryMutationReadSet(batch, batch),
        page_ids,
    )

    by_kind = {mutation.match.candidate.kind: mutation for mutation in plan.mutations}
    event = by_kind[MemoryKind.EVENT]
    assert event.action is MemoryMutationAction.NOOP
    assert dict(event.fields) == dict(old_event.fields)
    assert event.read_only_target is True
    assert by_kind[MemoryKind.PREFERENCE].action is MemoryMutationAction.UPDATE
    assert by_kind[MemoryKind.PREFERENCE].read_only_target is False
    assert plan.read_only_targets == (event_uri,)
    assert plan.changed_mutations == (by_kind[MemoryKind.PREFERENCE],)

    same = MemoryMutationPlanner().plan(
        MemoryCandidateBatch(events=(MemoryCandidate(event_page_id, MemoryKind.EVENT, old_event.fields),)),
        MemoryMutationReadSet(batch, target_batch(memory_snapshot(old_event))),
        page_ids,
    )
    assert same.mutations[0].action is MemoryMutationAction.NOOP
    assert same.read_only_targets == (event_uri,)


def test_new_event_page_id_reusing_an_old_event_uri_still_fails_planning() -> None:
    old_event = document(MemoryKind.EVENT)
    batch = target_batch(memory_snapshot(old_event))
    with pytest.raises(Exception, match="reuse its existing page_id"):
        MemoryMutationPlanner().plan(
            MemoryCandidateBatch(events=(MemoryCandidate(100, MemoryKind.EVENT, old_event.fields),)),
            MemoryMutationReadSet(batch, batch),
            MemoryPageIdMap.from_snapshots(batch),
        )


def test_unconfirmed_intention_cannot_create_or_update_without_merge_preservation() -> None:
    candidate = MemoryCandidate(
        100,
        MemoryKind.INTENTION,
        memory_fields(MemoryKind.INTENTION),
        confirmed=False,
    )
    missing = VersionedSnapshot.missing(str(MemoryURI.from_address(candidate.address)))
    with pytest.raises(MemoryMutationPlanningError, match="unconfirmed"):
        MemoryMutationPlanner().plan(
            MemoryCandidateBatch(intentions=(candidate,)),
            MemoryMutationReadSet(SnapshotBatch((), 0), target_batch(missing)),
            MemoryPageIdMap(),
        )


def test_final_identity_map_encodes_create_update_merge_delete_and_uri_remapping() -> None:
    profile_uri = MemoryURI.from_address(document(MemoryKind.PROFILE).address)
    preference_uri = MemoryURI.from_address(document(MemoryKind.PREFERENCE).address)
    entity_uri = MemoryURI.from_address(document(MemoryKind.ENTITY).address)
    entries = (
        MemoryFinalIdentity(1, MemoryNodeDisposition.MERGE, preference_uri, profile_uri),
        MemoryFinalIdentity(2, MemoryNodeDisposition.DELETE, entity_uri, None),
        MemoryFinalIdentity(100, MemoryNodeDisposition.CREATE, None, profile_uri),
    )
    identities = MemoryFinalIdentityMap(entries)
    assert identities.remap_uri(preference_uri) == profile_uri
    assert identities.remap_uri(entity_uri) is None
    assert identities.retired_uris == tuple(sorted((preference_uri, entity_uri), key=str))
    assert identities.merged_uri_map == {str(preference_uri): profile_uri}


def test_final_identity_map_rejects_merge_without_live_target_and_duplicate_live_identity() -> None:
    profile_uri = MemoryURI.from_address(document(MemoryKind.PROFILE).address)
    preference_uri = MemoryURI.from_address(document(MemoryKind.PREFERENCE).address)
    with pytest.raises(MemoryFinalIdentityError, match="separately planned live node"):
        MemoryFinalIdentityMap((MemoryFinalIdentity(1, MemoryNodeDisposition.MERGE, preference_uri, profile_uri),))
    with pytest.raises(MemoryFinalIdentityError, match="multiple node identities"):
        MemoryFinalIdentityMap(
            (
                MemoryFinalIdentity(100, MemoryNodeDisposition.CREATE, None, profile_uri),
                MemoryFinalIdentity(101, MemoryNodeDisposition.CREATE, None, profile_uri),
            )
        )
