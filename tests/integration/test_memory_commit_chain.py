"""候选节点、最终身份、双向关系和统一事务提交的主链测试。"""

from pathlib import Path
from threading import Event, Thread

import pytest

from habitus.infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.compaction import MemoryLifecycleOperationStore, MemoryLifecycleScanStore, MemoryRecoveryStore
from habitus.memory.document import MemoryLinkType
from habitus.memory.editor import (
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
from habitus.memory.editor.transaction import (
    MemoryCommitConflictError,
    MemoryCommitPlan,
    MemoryCommitStatus,
    MemoryCommitTransaction,
)
from habitus.memory.editor.transaction_log import MemoryTransactionJournal
from habitus.memory.indexing import MemoryIndexSourceReader, MemoryVectorIndexConfig
from habitus.memory.model import MemoryKind
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree, MemoryTreeConsistencyError
from habitus.memory.uri import MemoryURI
from tests.helpers import BASE_TIME, codec, memory_fields
from tests.lock_store import ProcessLocalLockStore


def missing_batch(*uris: MemoryURI) -> SnapshotBatch:
    snapshots = tuple(VersionedSnapshot.missing(str(uri)) for uri in sorted(uris, key=str))
    return SnapshotBatch(snapshots, 0)


def test_workflow_state_follows_the_injected_root_not_the_memory_root_parent(tmp_path: Path) -> None:
    """记忆根挪到别处、工作目录不动：事务日志与 L2 生命周期状态都落在注入的工作目录，不再从记忆根推。"""

    document_codec = codec()
    tree = MemoryTree(tmp_path / "elsewhere" / "memory", document_codec=document_codec)
    workflow_root = tmp_path / "data" / "workflow"
    path_lock = PathLock(ProcessLocalLockStore())
    journal = MemoryTransactionJournal(workflow_root / "transactions", document_codec)
    MemoryCommitTransaction(tree, MemorySnapshotReader(tree), path_lock, journal)
    recovery = MemoryRecoveryStore(tree, workflow_root)
    operations = MemoryLifecycleOperationStore(tree, path_lock, workflow_root)
    scan = MemoryLifecycleScanStore(tree, path_lock, workflow_root)
    for store in (recovery, operations, scan):
        store.initialize()

    resolved = workflow_root.resolve()
    assert tree.visibility_journal is journal
    assert recovery.root == resolved / "lifecycle" / "l2_recovery"
    assert operations.root == resolved / "lifecycle" / "l2_operations"
    assert scan.path == resolved / "lifecycle" / "l2_scan_state.json"
    assert not (tree.root.parent / "workflow").exists()


def test_workflow_state_is_refused_inside_the_memory_tree(tmp_path: Path) -> None:
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    inside = tree.root / "workflow"
    path_lock = PathLock(ProcessLocalLockStore())
    with pytest.raises(ValueError, match="outside the L2 memory tree"):
        MemoryCommitTransaction(
            tree,
            MemorySnapshotReader(tree),
            path_lock,
            MemoryTransactionJournal(inside / "transactions", document_codec),
        )
    with pytest.raises(ValueError, match="outside the L2 memory tree"):
        MemoryRecoveryStore(tree, inside)
    with pytest.raises(ValueError, match="outside the L2 memory tree"):
        MemoryLifecycleOperationStore(tree, path_lock, inside)
    with pytest.raises(ValueError, match="outside the L2 memory tree"):
        MemoryLifecycleScanStore(tree, path_lock, inside)


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


def test_multi_document_commit_is_visible_to_readers_as_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    reader = MemorySnapshotReader(tree)
    transaction = MemoryCommitTransaction(
        tree,
        reader,
        PathLock(ProcessLocalLockStore()),
        MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec),
        clock=lambda: BASE_TIME,
        transaction_id_factory=lambda: "c" * 32,
    )
    late_reader = MemorySnapshotReader(tree)
    index_sources = MemoryIndexSourceReader(
        tree,
        config=MemoryVectorIndexConfig(),
        path_lock=transaction.path_lock,
    )
    unlocked_index_sources = MemoryIndexSourceReader(
        tree,
        config=MemoryVectorIndexConfig(),
    )
    first_document_published = Event()
    release_writer = Event()
    original_write = tree.write
    writes = 0

    def blocking_write(value):
        nonlocal writes
        result = original_write(value)
        writes += 1
        if writes == 1:
            first_document_published.set()
            if not release_writer.wait(timeout=5):
                raise TimeoutError("reader did not release the transaction writer")
        return result

    monkeypatch.setattr(tree, "write", blocking_write)
    failures: list[BaseException] = []

    def commit() -> None:
        try:
            transaction.commit(plan)
        except BaseException as exc:
            failures.append(exc)

    worker = Thread(target=commit)
    worker.start()
    assert first_document_published.wait(timeout=5)
    with pytest.raises(RuntimeError, match="prepared transaction"):
        unlocked_index_sources.walk()
    with pytest.raises(MemoryTreeConsistencyError, match="prepared multi-document"):
        tree.exists(profile.address)
    with pytest.raises(MemoryTreeConsistencyError, match="prepared multi-document"):
        tree.list_addresses()
    # 另一个进程打开同一棵树：可见性日志由它自己的组装注入（同一个 workflow/transactions 目录），不从记忆根推。
    separately_opened_tree = MemoryTree(tree.root, document_codec=document_codec)
    assert separately_opened_tree.visibility_journal is None
    separately_opened_tree.bind_visibility_journal(
        MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec)
    )
    with pytest.raises(MemoryTreeConsistencyError, match="prepared multi-document"):
        separately_opened_tree.exists(profile.address)
    during = reader.read_many((profile_uri, preference_uri))
    observed = tuple(snapshot.exists for snapshot in during.snapshots)
    late_observed = tuple(
        snapshot.exists for snapshot in late_reader.read_many((profile_uri, preference_uri)).snapshots
    )
    source_started = Event()
    source_finished = Event()
    indexed_identities: list[str] = []

    def read_index_sources() -> None:
        source_started.set()
        indexed_identities.extend(source.identity for source in index_sources.walk())
        source_finished.set()

    source_worker = Thread(target=read_index_sources)
    source_worker.start()
    assert source_started.wait(timeout=5)
    assert not source_finished.wait(timeout=0.05)
    release_writer.set()
    worker.join(timeout=5)
    source_worker.join(timeout=5)

    assert not worker.is_alive()
    assert not source_worker.is_alive()
    assert failures == []
    assert observed in {(False, False), (True, True)}
    assert late_observed in {(False, False), (True, True)}
    assert str(profile_uri) in indexed_identities
    assert str(preference_uri) in indexed_identities
