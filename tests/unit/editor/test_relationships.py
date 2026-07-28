"""最终身份确定后的关系解析、双向落点和结构迁移测试。"""

import pytest

from infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from memory.document import MemoryLinkType, MemoryStoredLink
from memory.editor import (
    MemoryCandidateBatch,
    MemoryFinalIdentity,
    MemoryFinalIdentityMap,
    MemoryNodeDisposition,
    MemoryRelationAction,
    MemoryRelationCandidate,
    MemoryRelationIntegrityError,
    MemoryRelationPlanner,
    MemoryRelationReadSet,
    MemoryRelationResolver,
    MemoryResolvedRelation,
)
from memory.model import MemoryKind
from memory.uri import MemoryURI
from tests.helpers import codec, document, memory_snapshot


def batch(*snapshots: VersionedSnapshot) -> SnapshotBatch:
    ordered = tuple(sorted(snapshots, key=lambda item: item.identity))
    return SnapshotBatch(ordered, sum(item.size_bytes for item in ordered))


def test_relation_candidate_is_resolved_only_after_final_uri_identity_exists() -> None:
    profile_uri = MemoryURI.from_address(document(MemoryKind.PROFILE).address)
    preference_uri = MemoryURI.from_address(document(MemoryKind.PREFERENCE).address)
    identities = MemoryFinalIdentityMap(
        (
            MemoryFinalIdentity(100, MemoryNodeDisposition.CREATE, None, profile_uri),
            MemoryFinalIdentity(101, MemoryNodeDisposition.CREATE, None, preference_uri),
        )
    )
    candidate = MemoryRelationCandidate(
        MemoryRelationAction.ADD, 100, 101, MemoryLinkType.BELONGS_TO
    )
    resolved = MemoryRelationResolver().resolve(
        MemoryCandidateBatch(relations=(candidate,)), identities
    )
    assert resolved == (
        MemoryResolvedRelation(
            MemoryRelationAction.ADD,
            profile_uri,
            preference_uri,
            MemoryLinkType.BELONGS_TO,
        ),
    )


def test_add_relation_updates_source_link_and_target_backlink_atomically_in_plan() -> None:
    profile_uri = MemoryURI.from_address(document(MemoryKind.PROFILE).address)
    preference_uri = MemoryURI.from_address(document(MemoryKind.PREFERENCE).address)
    identities = MemoryFinalIdentityMap(
        (
            MemoryFinalIdentity(100, MemoryNodeDisposition.CREATE, None, profile_uri),
            MemoryFinalIdentity(101, MemoryNodeDisposition.CREATE, None, preference_uri),
        )
    )
    operation = MemoryResolvedRelation(
        MemoryRelationAction.ADD, profile_uri, preference_uri, MemoryLinkType.BELONGS_TO
    )
    snapshots = batch(
        VersionedSnapshot.missing(str(profile_uri)),
        VersionedSnapshot.missing(str(preference_uri)),
    )
    read_set = MemoryRelationReadSet.build(snapshots, identities, (operation,))
    plan = MemoryRelationPlanner().plan(identities, (operation,), read_set)

    relation = operation.to_stored()
    assert plan.added == (relation,)
    assert plan.removed == ()
    assert plan.update_for(profile_uri).links == (relation,)
    assert plan.update_for(preference_uri).backlinks == (relation,)


def test_explicit_remove_keeps_both_nodes_and_removes_complete_link_backlink_pair() -> None:
    source = document(MemoryKind.PREFERENCE)
    target = document(MemoryKind.PROFILE)
    source_uri = MemoryURI.from_address(source.address)
    target_uri = MemoryURI.from_address(target.address)
    relation = MemoryStoredLink(source_uri, target_uri, MemoryLinkType.BELONGS_TO)
    source = codec().build(
        source.kind, source.fields, metadata=source.metadata, links=(relation,)
    )
    target = codec().build(
        target.kind, target.fields, metadata=target.metadata, backlinks=(relation,)
    )
    snapshots = batch(memory_snapshot(source), memory_snapshot(target))
    identities = MemoryFinalIdentityMap(
        (
            MemoryFinalIdentity(1, MemoryNodeDisposition.NOOP, source_uri, source_uri),
            MemoryFinalIdentity(2, MemoryNodeDisposition.NOOP, target_uri, target_uri),
        )
    )
    operation = MemoryResolvedRelation(
        MemoryRelationAction.REMOVE, source_uri, target_uri, MemoryLinkType.BELONGS_TO
    )
    read_set = MemoryRelationReadSet.build(snapshots, identities, (operation,))
    plan = MemoryRelationPlanner().plan(identities, (operation,), read_set)
    assert plan.removed == (relation,)
    assert plan.update_for(source_uri).links == ()
    assert plan.update_for(target_uri).backlinks == ()
    assert identities.retired_uris == ()


def test_structural_merge_migrates_old_relations_to_surviving_target() -> None:
    retired = document(MemoryKind.PREFERENCE)
    survivor = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "输出格式", "content": "- 使用 Markdown"},
    )
    neighbor = document(MemoryKind.PROFILE)
    retired_uri = MemoryURI.from_address(retired.address)
    survivor_uri = MemoryURI.from_address(survivor.address)
    neighbor_uri = MemoryURI.from_address(neighbor.address)
    relation = MemoryStoredLink(retired_uri, neighbor_uri, MemoryLinkType.BELONGS_TO)
    retired = codec().build(retired.kind, retired.fields, metadata=retired.metadata, links=(relation,))
    neighbor = codec().build(neighbor.kind, neighbor.fields, metadata=neighbor.metadata, backlinks=(relation,))
    snapshots = batch(memory_snapshot(retired), memory_snapshot(survivor), memory_snapshot(neighbor))
    identities = MemoryFinalIdentityMap(
        (
            MemoryFinalIdentity(1, MemoryNodeDisposition.MERGE, retired_uri, survivor_uri),
            MemoryFinalIdentity(2, MemoryNodeDisposition.NOOP, neighbor_uri, neighbor_uri),
            MemoryFinalIdentity(3, MemoryNodeDisposition.NOOP, survivor_uri, survivor_uri),
        )
    )
    read_set = MemoryRelationReadSet.build(snapshots, identities, ())
    plan = MemoryRelationPlanner().plan(identities, (), read_set)
    migrated = MemoryStoredLink(survivor_uri, neighbor_uri, MemoryLinkType.BELONGS_TO)
    assert plan.added == (migrated,)
    assert plan.removed == (relation,)
    assert plan.update_for(survivor_uri).links == (migrated,)
    assert plan.update_for(neighbor_uri).backlinks == (migrated,)


def test_structural_operation_fails_if_old_link_has_no_matching_backlink() -> None:
    source = document(MemoryKind.PREFERENCE)
    target = document(MemoryKind.PROFILE)
    source_uri = MemoryURI.from_address(source.address)
    target_uri = MemoryURI.from_address(target.address)
    relation = MemoryStoredLink(source_uri, target_uri, MemoryLinkType.BELONGS_TO)
    source = codec().build(source.kind, source.fields, metadata=source.metadata, links=(relation,))
    snapshots = batch(memory_snapshot(source), memory_snapshot(target))
    identities = MemoryFinalIdentityMap(
        (
            MemoryFinalIdentity(1, MemoryNodeDisposition.DELETE, source_uri, None),
            MemoryFinalIdentity(2, MemoryNodeDisposition.NOOP, target_uri, target_uri),
        )
    )
    with pytest.raises(MemoryRelationIntegrityError, match="backlink"):
        MemoryRelationReadSet.build(snapshots, identities, ())

