"""候选节点、最终身份、双向关系和统一事务提交的主链测试。"""

from pathlib import Path

import pytest

from infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.document import MemoryLinkType
from memory.editor import (
    MemoryCandidate,
    MemoryCandidateBatch,
    MemoryFinalIdentityMap,
    MemoryMutationPlanner,
    MemoryMutationReadSet,
    MemoryPageIdMap,
    MemoryRelationAction,
    MemoryRelationCandidate,
    MemoryRelationPlanner,
    MemoryRelationReadSet,
    MemoryRelationResolver,
)
from memory.editor.transaction import (
    MemoryCommitConflictError,
    MemoryCommitPlan,
    MemoryCommitStatus,
    MemoryCommitTransaction,
)
from memory.editor.transaction_log import MemoryTransactionJournal
from memory.model import MemoryKind
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI
from tests.helpers import BASE_TIME, codec, memory_fields


def missing_batch(*uris: MemoryURI) -> SnapshotBatch:
    snapshots = tuple(VersionedSnapshot.missing(str(uri)) for uri in sorted(uris, key=str))
    return SnapshotBatch(snapshots, 0)


def test_create_two_nodes_with_link_and_backlink_in_one_recoverable_transaction(tmp_path: Path) -> None:
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    profile = MemoryCandidate(100, MemoryKind.PROFILE, memory_fields(MemoryKind.PROFILE))
    preference = MemoryCandidate(101, MemoryKind.PREFERENCE, memory_fields(MemoryKind.PREFERENCE))
    candidates = MemoryCandidateBatch(
        profile=(profile,),
        preferences=(preference,),
        relations=(
            MemoryRelationCandidate(
                MemoryRelationAction.ADD,
                profile.page_id,
                preference.page_id,
                MemoryLinkType.BELONGS_TO,
            ),
        ),
    )
    profile_uri = MemoryURI.from_address(profile.address)
    preference_uri = MemoryURI.from_address(preference.address)
    targets = missing_batch(profile_uri, preference_uri)
    mutation = MemoryMutationPlanner().plan(
        candidates,
        MemoryMutationReadSet(SnapshotBatch((), 0), targets),
        MemoryPageIdMap(),
    )
    identities = MemoryFinalIdentityMap.from_mutation_plan(mutation, MemoryPageIdMap())
    relations = MemoryRelationResolver().resolve(candidates, identities)
    relation_read_set = MemoryRelationReadSet.build(targets, identities, relations)
    relation_plan = MemoryRelationPlanner().plan(identities, relations, relation_read_set)
    plan = MemoryCommitPlan.build(mutation, identities, relation_plan)
    transaction_journal = MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec)
    transaction = MemoryCommitTransaction(
        tree,
        MemorySnapshotReader(tree),
        PathLock(ProcessLocalLockStore()),
        transaction_journal,
        clock=lambda: BASE_TIME,
        transaction_id_factory=lambda: "a" * 32,
    )

    result = transaction.commit(plan)

    stored_profile = tree.read(profile.address)
    stored_preference = tree.read(preference.address)
    assert result.status is MemoryCommitStatus.UPDATED
    assert result.created_uris == tuple(sorted((profile_uri, preference_uri), key=str))
    assert stored_profile.links == result.added_relations
    assert stored_preference.backlinks == result.added_relations
    assert stored_profile.metadata.revision == stored_preference.metadata.revision == 1
    assert transaction_journal.try_read("a" * 32) is None

    with pytest.raises(MemoryCommitConflictError):
        transaction.commit(plan, transaction_id="b" * 32)


def test_empty_candidate_batch_is_a_noop_without_transaction_journal(tmp_path: Path) -> None:
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    empty = SnapshotBatch((), 0)
    candidates = MemoryCandidateBatch()
    mutation = MemoryMutationPlanner().plan(
        candidates,
        MemoryMutationReadSet(empty, empty),
        MemoryPageIdMap(),
    )
    identities = MemoryFinalIdentityMap.from_mutation_plan(mutation, MemoryPageIdMap())
    relation_read_set = MemoryRelationReadSet.build(empty, identities, ())
    relation_plan = MemoryRelationPlanner().plan(identities, (), relation_read_set)
    plan = MemoryCommitPlan.build(mutation, identities, relation_plan)
    journal = MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec)
    transaction = MemoryCommitTransaction(
        tree,
        MemorySnapshotReader(tree),
        PathLock(ProcessLocalLockStore()),
        journal,
    )

    result = transaction.commit(plan)

    assert result.status is MemoryCommitStatus.UNCHANGED
    assert result.transaction_id is None
    assert not journal.root.exists()

