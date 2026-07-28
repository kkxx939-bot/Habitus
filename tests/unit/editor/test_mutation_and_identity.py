"""精确节点匹配、YAML 字段合并和最终身份安全规划测试。"""


import pytest

from infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from memory.editor import (
    MemoryCandidate,
    MemoryCandidateBatch,
    MemoryFieldMergeError,
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
from memory.model import MemoryKind
from memory.uri import MemoryURI
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
    plan = MemoryMutationPlanner().plan(
        MemoryCandidateBatch(preferences=(candidate,)), read_set, MemoryPageIdMap()
    )
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


def test_add_only_event_cannot_be_silently_overwritten_at_same_identity() -> None:
    old = document(MemoryKind.EVENT)
    old_snapshot = memory_snapshot(old)
    batch = target_batch(old_snapshot)
    page_ids = MemoryPageIdMap.from_snapshots(batch)
    changed = {**old.fields, "summary": "不同事件内容"}
    with pytest.raises(MemoryFieldMergeError, match="add_only"):
        MemoryMutationPlanner().plan(
            MemoryCandidateBatch(events=(MemoryCandidate(1, MemoryKind.EVENT, changed),)),
            MemoryMutationReadSet(batch, batch),
            page_ids,
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
        MemoryFinalIdentityMap(
            (MemoryFinalIdentity(1, MemoryNodeDisposition.MERGE, preference_uri, profile_uri),)
        )
    with pytest.raises(MemoryFinalIdentityError, match="multiple node identities"):
        MemoryFinalIdentityMap(
            (
                MemoryFinalIdentity(100, MemoryNodeDisposition.CREATE, None, profile_uri),
                MemoryFinalIdentity(101, MemoryNodeDisposition.CREATE, None, profile_uri),
            )
        )

