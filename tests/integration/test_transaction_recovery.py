"""统一记忆事务的 UPDATE/MERGE/DELETE 与进程崩溃恢复测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from habitus.infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.memory.editor import (
    MemoryCandidate,
    MemoryCandidateBatch,
    MemoryFinalIdentity,
    MemoryFinalIdentityMap,
    MemoryMutationPlanner,
    MemoryMutationReadSet,
    MemoryNodeDisposition,
    MemoryPageIdMap,
    MemoryRelationPlanner,
    MemoryRelationReadSet,
)
from habitus.memory.editor.transaction import (
    MemoryCommitPlan,
    MemoryCommitRecoveryError,
    MemoryCommitTransaction,
)
from habitus.memory.editor.transaction_log import (
    MemoryTransactionJournal,
    MemoryTransactionJournalState,
)
from habitus.memory.model import MemoryKind
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from tests.helpers import BASE_TIME, codec, document, memory_fields


def transaction(tmp_path: Path):
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    reader = MemorySnapshotReader(tree)
    journal = MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec)
    value = MemoryCommitTransaction(
        tree,
        reader,
        PathLock(ProcessLocalLockStore()),
        journal,
        clock=lambda: BASE_TIME,
    )
    return tree, reader, journal, value


def empty_relation_plan(snapshots, identities):
    read_set = MemoryRelationReadSet.build(snapshots, identities, ())
    return MemoryRelationPlanner().plan(identities, (), read_set)


def two_create_plan():
    profile = MemoryCandidate(100, MemoryKind.PROFILE, memory_fields(MemoryKind.PROFILE))
    preference = MemoryCandidate(101, MemoryKind.PREFERENCE, memory_fields(MemoryKind.PREFERENCE))
    batch = MemoryCandidateBatch(profile=(profile,), preferences=(preference,))
    uris = tuple(sorted((MemoryURI.from_address(profile.address), MemoryURI.from_address(preference.address)), key=str))
    missing = SnapshotBatch(tuple(VersionedSnapshot.missing(str(uri)) for uri in uris), 0)
    mutation = MemoryMutationPlanner().plan(
        batch,
        MemoryMutationReadSet(SnapshotBatch((), 0), missing),
        MemoryPageIdMap(),
    )
    identities = MemoryFinalIdentityMap.from_mutation_plan(mutation, MemoryPageIdMap())
    return MemoryCommitPlan.build(mutation, identities, empty_relation_plan(missing, identities))


def test_update_advances_exactly_one_revision_and_delete_removes_entire_l2_node(tmp_path: Path) -> None:
    tree, reader, _journal, commit = transaction(tmp_path)
    old = document(MemoryKind.PREFERENCE)
    tree.write(old)
    uri = MemoryURI.from_address(old.address)
    old_batch = reader.read_many((uri,))
    pages = MemoryPageIdMap.from_snapshots(old_batch)
    page_id = pages.page_id_for(uri)
    changed = {**old.fields, "content": "- 现在偏好一句话回答"}
    mutation = MemoryMutationPlanner().plan(
        MemoryCandidateBatch(
            preferences=(MemoryCandidate(page_id, MemoryKind.PREFERENCE, changed),),  # type: ignore[arg-type]
        ),
        MemoryMutationReadSet(old_batch, old_batch),
        pages,
    )
    identities = MemoryFinalIdentityMap.from_mutation_plan(mutation, pages)
    update_plan = MemoryCommitPlan.build(
        mutation,
        identities,
        empty_relation_plan(old_batch, identities),
    )

    updated = commit.commit(update_plan)
    assert updated.updated_uris == (uri,)
    assert tree.read(old.address).metadata.revision == 2
    assert tree.read(old.address).fields["content"] == "- 现在偏好一句话回答"

    latest = reader.read_many((uri,))
    latest_pages = MemoryPageIdMap.from_snapshots(latest)
    source_page = latest_pages.page_id_for(uri)
    empty_mutation = MemoryMutationPlanner().plan(
        MemoryCandidateBatch(),
        MemoryMutationReadSet(latest, SnapshotBatch((), 0)),
        latest_pages,
    )
    delete_identities = MemoryFinalIdentityMap(
        (MemoryFinalIdentity(source_page, MemoryNodeDisposition.DELETE, uri, None),)  # type: ignore[arg-type]
    )
    delete_plan = MemoryCommitPlan.build(
        empty_mutation,
        delete_identities,
        empty_relation_plan(latest, delete_identities),
    )
    deleted = commit.commit(delete_plan)
    assert deleted.deleted_uris == (uri,)
    assert not tree.delete(old.address)


def test_merge_retires_source_but_keeps_explicitly_planned_target_identity(tmp_path: Path) -> None:
    tree, reader, _journal, commit = transaction(tmp_path)
    source = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "答复长度旧名称", "content": "- 偏好简洁回答"},
    )
    target = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "答复长度", "content": "- 偏好简洁回答"},
    )
    tree.write(source)
    tree.write(target)
    source_uri = MemoryURI.from_address(source.address)
    target_uri = MemoryURI.from_address(target.address)
    old = reader.read_many((source_uri, target_uri))
    pages = MemoryPageIdMap.from_snapshots(old)
    source_page = pages.page_id_for(source_uri)
    target_page = pages.page_id_for(target_uri)
    target_snapshot = old.get(str(target_uri))
    assert target_snapshot is not None
    target_batch = SnapshotBatch((target_snapshot,), target_snapshot.size_bytes)
    mutation = MemoryMutationPlanner().plan(
        MemoryCandidateBatch(
            preferences=(MemoryCandidate(target_page, MemoryKind.PREFERENCE, target.fields),),  # type: ignore[arg-type]
        ),
        MemoryMutationReadSet(old, target_batch),
        pages,
    )
    identities = MemoryFinalIdentityMap(
        tuple(sorted((
            MemoryFinalIdentity(source_page, MemoryNodeDisposition.MERGE, source_uri, target_uri),  # type: ignore[arg-type]
            MemoryFinalIdentity(target_page, MemoryNodeDisposition.NOOP, target_uri, target_uri),  # type: ignore[arg-type]
        ), key=lambda item: item.page_id))
    )
    plan = MemoryCommitPlan.build(
        mutation,
        identities,
        empty_relation_plan(old, identities),
    )

    result = commit.commit(plan)

    assert result.deleted_uris == (source_uri,)
    assert result.unchanged_uris == (target_uri,)
    assert tree.read(target.address) == target
    with pytest.raises(FileNotFoundError):
        tree.read(source.address)


def test_recovery_marks_fully_published_prepared_transaction_committed(tmp_path: Path) -> None:
    tree, _reader, journal, commit = transaction(tmp_path)
    plan = two_create_plan()
    writes = commit._build_writes(plan, timestamp=BASE_TIME)
    record = commit._journal_record(
        plan,
        writes,
        transaction_id="a" * 32,
        timestamp=BASE_TIME,
    )
    journal.prepare(record)
    for write in writes:
        tree.write(write.after)

    assert commit.recover_pending(discard_terminal=False) == ("a" * 32,)
    assert journal.read("a" * 32).state is MemoryTransactionJournalState.COMMITTED
    assert all(tree.read(write.after.address) == write.after for write in writes)


def test_recovery_rolls_back_partially_published_transaction_to_complete_before_state(tmp_path: Path) -> None:
    tree, reader, journal, commit = transaction(tmp_path)
    plan = two_create_plan()
    writes = commit._build_writes(plan, timestamp=BASE_TIME)
    record = commit._journal_record(
        plan,
        writes,
        transaction_id="b" * 32,
        timestamp=BASE_TIME,
    )
    journal.prepare(record)
    tree.write(writes[0].after)

    commit.recover_pending(discard_terminal=False)

    assert journal.read("b" * 32).state is MemoryTransactionJournalState.ROLLED_BACK
    assert all(not reader.read(write.uri).exists for write in writes)


def test_recovery_refuses_unknown_later_document_state_and_keeps_prepared_journal_for_manual_repair(
    tmp_path: Path,
) -> None:
    tree, _reader, journal, commit = transaction(tmp_path)
    plan = two_create_plan()
    writes = commit._build_writes(plan, timestamp=BASE_TIME)
    record = commit._journal_record(
        plan,
        writes,
        transaction_id="c" * 32,
        timestamp=BASE_TIME,
    )
    journal.prepare(record)
    tampered_fields = dict(writes[0].after.fields)
    if writes[0].after.kind is MemoryKind.PROFILE:
        tampered_fields["content"] = "- 不属于该事务的后续状态"
    else:
        tampered_fields["content"] = "- 不属于该事务的后续偏好"
    tree.write(
        tree.document_codec.build(
            writes[0].after.kind,
            tampered_fields,
            metadata=writes[0].after.metadata,
        )
    )

    with pytest.raises(MemoryCommitRecoveryError, match="unknown later"):
        commit.recover_pending(discard_terminal=False)
    assert journal.read("c" * 32).state is MemoryTransactionJournalState.PREPARED
