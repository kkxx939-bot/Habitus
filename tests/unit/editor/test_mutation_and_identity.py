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
    MemoryMutationReadSet,
    MemoryNodeDisposition,
    MemoryPageIdMap,
)
from habitus.memory.model import MemoryKind
from habitus.memory.uri import MemoryURI
from tests.helpers import document, memory_fields, memory_snapshot, page_ids_from


def target_batch(*snapshots: VersionedSnapshot) -> SnapshotBatch:
    ordered = tuple(sorted(snapshots, key=lambda item: item.identity))
    return SnapshotBatch(ordered, sum(item.size_bytes for item in ordered))


def test_existing_candidate_becomes_noop_or_update_without_llm_patch_operations() -> None:
    old = document(MemoryKind.PREFERENCE)
    old_snapshot = memory_snapshot(old)
    old_batch = target_batch(old_snapshot)
    page_ids = page_ids_from(old_batch)
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


def test_new_page_id_reusing_an_old_memory_uri_still_fails_planning() -> None:
    old_entity = document(MemoryKind.ENTITY)
    batch = target_batch(memory_snapshot(old_entity))
    with pytest.raises(Exception, match="reuse its existing page_id"):
        MemoryMutationPlanner().plan(
            MemoryCandidateBatch(entities=(MemoryCandidate(100, MemoryKind.ENTITY, old_entity.fields),)),
            MemoryMutationReadSet(batch, batch),
            page_ids_from(batch),
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
