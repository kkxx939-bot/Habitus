"""MemoryEditorPlan、统一事务日志与耐久 Change Receipt 的一致性测试。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.document import MemoryLinkType
from memory.editor import (
    MemoryCandidate,
    MemoryCandidateBatch,
    MemoryEditorPlan,
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
from memory.editor.extraction import (
    MemoryExtractionResult,
    MemoryRetrievalAction,
    MemoryRetrievalDecision,
    MemoryRetrievalStatus,
)
from memory.editor.transaction import MemoryCommitPlan, MemoryCommitTransaction
from memory.editor.transaction_log import MemoryTransactionJournal
from memory.model import MemoryKind
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI
from memory.workflow.receipt import (
    MemoryChangeReceiptError,
    MemoryChangeReceiptState,
    MemoryChangeReceiptStore,
    MemoryChangeSource,
    MemoryNodeChangeAction,
)
from tests.helpers import BASE_TIME, codec, document, memory_fields, snapshot_batch


def editor_plan(*, preference_content: str = "- 偏好简洁回答", with_relation: bool = False):
    preference_fields = dict(memory_fields(MemoryKind.PREFERENCE))
    preference_fields["content"] = preference_content
    preference = MemoryCandidate(100, MemoryKind.PREFERENCE, preference_fields)
    if with_relation:
        profile = MemoryCandidate(101, MemoryKind.PROFILE, memory_fields(MemoryKind.PROFILE))
        batch = MemoryCandidateBatch(
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
    else:
        batch = MemoryCandidateBatch(preferences=(preference,))
    uris = tuple(sorted((MemoryURI.from_address(item.address) for item in batch.iter_candidates()), key=str))
    targets = SnapshotBatch(tuple(VersionedSnapshot.missing(str(uri)) for uri in uris), 0)
    old = SnapshotBatch((), 0)
    pages = MemoryPageIdMap()
    mutations = MemoryMutationPlanner().plan(
        batch,
        MemoryMutationReadSet(old, targets),
        pages,
    )
    identities = MemoryFinalIdentityMap.from_mutation_plan(mutations, pages)
    operations = MemoryRelationResolver().resolve(batch, identities)
    relation_read_set = MemoryRelationReadSet.build(targets, identities, operations)
    relations = MemoryRelationPlanner().plan(identities, operations, relation_read_set)
    commit = MemoryCommitPlan.build(mutations, identities, relations)
    extraction = MemoryExtractionResult(
        conversation_id="conversation-1",
        segment_id="000000000000-000000000001",
        source_segment_digest="d" * 64,
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
                "没有相关旧记忆。",
            ),
        ),
        retrieval_observations=(),
    )
    return MemoryEditorPlan(extraction, mutations, identities, commit)


def source(transaction_id: str = "a" * 32) -> MemoryChangeSource:
    return MemoryChangeSource(
        memory_sequence=1,
        transaction_id=transaction_id,
        conversation_id="conversation-1",
        started_on=date(2026, 7, 1),
        segment_id="000000000000-000000000001",
        source_segment_digest="d" * 64,
    )


def update_editor_plan(
    *,
    preference_content: str,
    initial_content: str = "- 初始偏好中等长度回答",
) -> MemoryEditorPlan:
    existing = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": initial_content},
    )
    old = snapshot_batch(existing)
    uri = MemoryURI.from_address(existing.address)
    pages = MemoryPageIdMap.from_snapshots(old)
    page_id = pages.page_id_for(uri)
    batch = MemoryCandidateBatch(
        preferences=(
            MemoryCandidate(
                page_id,  # type: ignore[arg-type]
                MemoryKind.PREFERENCE,
                {"topic": "回答风格", "content": preference_content},
            ),
        )
    )
    mutations = MemoryMutationPlanner().plan(
        batch,
        MemoryMutationReadSet(old, old),
        pages,
    )
    identities = MemoryFinalIdentityMap.from_mutation_plan(mutations, pages)
    operations = MemoryRelationResolver().resolve(batch, identities)
    relation_read_set = MemoryRelationReadSet.build(old, identities, operations)
    relations = MemoryRelationPlanner().plan(identities, operations, relation_read_set)
    commit = MemoryCommitPlan.build(mutations, identities, relations)
    extraction = MemoryExtractionResult(
        conversation_id="conversation-1",
        segment_id="000000000000-000000000001",
        source_segment_digest="d" * 64,
        candidates=batch,
        mutations=mutations,
        old_memories=old,
        page_ids=pages,
        retrieval_decisions=(
            MemoryRetrievalDecision(
                MemoryRetrievalStatus.SUFFICIENT,
                MemoryRetrievalAction.FINISH,
                None,
                None,
                "已读取同址旧偏好。",
            ),
        ),
        retrieval_observations=(),
    )
    return MemoryEditorPlan(extraction, mutations, identities, commit)


def test_prepared_receipt_is_finalized_only_from_actual_committed_journal(tmp_path: Path) -> None:
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    journal = MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec)
    transaction = MemoryCommitTransaction(
        tree,
        MemorySnapshotReader(tree),
        PathLock(ProcessLocalLockStore()),
        journal,
        clock=lambda: BASE_TIME,
    )
    receipts = MemoryChangeReceiptStore(tmp_path / "workflow", document_codec)
    plan = editor_plan(with_relation=True)
    change_source = source()

    prepared = receipts.prepare(change_source, plan, timestamp=BASE_TIME)
    result = transaction.commit(
        plan.commit,
        transaction_id=change_source.transaction_id,
        retain_journal=True,
    )
    committed = receipts.finalize(change_source, journal.read(change_source.transaction_id))

    assert prepared.state is MemoryChangeReceiptState.PREPARED
    assert committed.state is MemoryChangeReceiptState.COMMITTED
    assert committed.expected_created_uris == result.created_uris
    assert committed.added_relations == result.added_relations
    assert {item.action for item in committed.node_changes} == {MemoryNodeChangeAction.CREATE}
    assert all(item.before_revision is None and item.after_revision == 1 for item in committed.node_changes)
    assert all(item.before_digest is None and item.after_digest is not None for item in committed.node_changes)
    assert receipts.finalize(change_source, journal.read(change_source.transaction_id)) == committed


@pytest.mark.parametrize("with_relation", [False, True])
def test_prepare_is_idempotent_for_same_plan_but_rejects_semantically_different_replan(
    tmp_path: Path,
    with_relation: bool,
) -> None:
    receipts = MemoryChangeReceiptStore(tmp_path / "workflow", codec())
    change_source = source()
    first_plan = editor_plan(
        preference_content="- 偏好简洁回答",
        with_relation=with_relation,
    )
    second_plan = editor_plan(
        preference_content="- 偏好详细回答",
        with_relation=with_relation,
    )

    first = receipts.prepare(change_source, first_plan, timestamp=BASE_TIME)
    assert receipts.prepare(change_source, first_plan, timestamp=BASE_TIME) == first
    with pytest.raises(MemoryChangeReceiptError, match="conflicts with a new semantic plan"):
        receipts.prepare(change_source, second_plan, timestamp=BASE_TIME)


def test_prepare_rejects_update_replan_when_same_uri_has_a_different_final_body(
    tmp_path: Path,
) -> None:
    receipts = MemoryChangeReceiptStore(tmp_path / "workflow", codec())
    change_source = source()
    first_plan = update_editor_plan(preference_content="- 更新为简洁回答")
    second_plan = update_editor_plan(preference_content="- 更新为详细回答")

    receipts.prepare(change_source, first_plan, timestamp=BASE_TIME)
    with pytest.raises(MemoryChangeReceiptError, match="conflicts with a new semantic plan"):
        receipts.prepare(change_source, second_plan, timestamp=BASE_TIME)


def test_prepare_rejects_same_final_content_when_the_old_snapshot_differs(
    tmp_path: Path,
) -> None:
    receipts = MemoryChangeReceiptStore(tmp_path / "workflow", codec())
    change_source = source()
    first_plan = update_editor_plan(
        initial_content="- 旧偏好为中等长度回答",
        preference_content="- 更新为简洁回答",
    )
    second_plan = update_editor_plan(
        initial_content="- 旧偏好为非常详细回答",
        preference_content="- 更新为简洁回答",
    )

    receipts.prepare(change_source, first_plan, timestamp=BASE_TIME)
    with pytest.raises(MemoryChangeReceiptError, match="conflicts with a new semantic plan"):
        receipts.prepare(change_source, second_plan, timestamp=BASE_TIME)


def test_finalize_rejects_same_uri_and_action_when_committed_content_differs(
    tmp_path: Path,
) -> None:
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    journal = MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec)
    transaction = MemoryCommitTransaction(
        tree,
        MemorySnapshotReader(tree),
        PathLock(ProcessLocalLockStore()),
        journal,
        clock=lambda: BASE_TIME,
    )
    receipts = MemoryChangeReceiptStore(tmp_path / "workflow", document_codec)
    change_source = source()
    plan = editor_plan(preference_content="- 偏好简洁回答")
    receipts.prepare(change_source, plan, timestamp=BASE_TIME)
    transaction.commit(
        plan.commit,
        transaction_id=change_source.transaction_id,
        retain_journal=True,
    )
    committed_journal = journal.read(change_source.transaction_id)
    entry = committed_journal.entries[0]
    assert entry.after is not None
    tampered_after = document_codec.build(
        entry.after.kind,
        {"topic": entry.after.fields["topic"], "content": "- 被替换为详细回答"},
        metadata=entry.after.metadata,
        links=entry.after.links,
        backlinks=entry.after.backlinks,
    )
    tampered_journal = replace(
        committed_journal,
        entries=(replace(entry, after=tampered_after),),
    )

    with pytest.raises(MemoryChangeReceiptError, match="content differs"):
        receipts.finalize(change_source, tampered_journal)


def test_prepared_receipt_cannot_be_finalized_from_foreign_or_non_committed_transaction(
    tmp_path: Path,
) -> None:
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    journal = MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec)
    transaction = MemoryCommitTransaction(
        tree,
        MemorySnapshotReader(tree),
        PathLock(ProcessLocalLockStore()),
        journal,
        clock=lambda: BASE_TIME,
    )
    receipts = MemoryChangeReceiptStore(tmp_path / "workflow", document_codec)
    plan = editor_plan()
    receipts.prepare(source(), plan, timestamp=BASE_TIME)
    writes = transaction._build_writes(plan.commit, timestamp=BASE_TIME)
    prepared_journal = transaction._journal_record(
        plan.commit,
        writes,
        transaction_id="a" * 32,
        timestamp=BASE_TIME,
    )
    with pytest.raises(MemoryChangeReceiptError, match="COMMITTED"):
        receipts.finalize(source(), prepared_journal)
    with pytest.raises(MemoryChangeReceiptError, match="does not match"):
        receipts.projector.finalize(
            receipts.read(source()),
            source("b" * 32),
            prepared_journal,
        )
