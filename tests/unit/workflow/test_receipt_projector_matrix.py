"""Change Receipt 投影器的准备态、提交态和防篡改矩阵。"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.memory.editor.transaction import MemoryCommitTransaction
from habitus.memory.editor.transaction_log import (
    MemoryTransactionJournal,
    MemoryTransactionJournalEntry,
    MemoryTransactionJournalRecord,
    MemoryTransactionJournalState,
)
from habitus.memory.model import MemoryKind
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow.receipt import (
    MemoryChangeReceiptError,
    MemoryChangeReceiptProjector,
    MemoryChangeReceiptState,
    MemoryNodeChangeAction,
    MemoryPreparedNodeChange,
)
from tests.helpers import BASE_TIME, codec, document
from tests.integration.test_change_receipt_chain import editor_plan, source, update_editor_plan


def committed_triplet(
    tmp_path: Path,
    *,
    update: bool = False,
    with_relation: bool = False,
):
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    journal_store = MemoryTransactionJournal(
        tmp_path / "workflow" / "transactions",
        document_codec,
    )
    transaction = MemoryCommitTransaction(
        tree,
        MemorySnapshotReader(tree),
        PathLock(ProcessLocalLockStore()),
        journal_store,
        clock=lambda: BASE_TIME,
    )
    plan = update_editor_plan(preference_content="- 更新为简洁回答") if update else editor_plan(with_relation=with_relation)
    if update:
        old = plan.commit.logical_writes()[0].before.value
        assert old is not None
        tree.write(old)
    projector = MemoryChangeReceiptProjector(document_codec)
    change_source = source()
    prepared = projector.prepare(change_source, plan, timestamp=BASE_TIME)
    if update:
        timestamp = BASE_TIME + timedelta(seconds=1)
        writes = transaction._build_writes(plan.commit, timestamp=timestamp)
        journal = replace(
            transaction._journal_record(
                plan.commit,
                writes,
                transaction_id=change_source.transaction_id,
                timestamp=BASE_TIME,
            ),
            state=MemoryTransactionJournalState.COMMITTED,
            updated_at=timestamp,
        )
        return projector, prepared, journal
    transaction.commit(plan.commit, transaction_id=change_source.transaction_id, retain_journal=True)
    journal = journal_store.read(change_source.transaction_id)
    return projector, prepared, journal


def test_constructor_requires_memory_document_codec() -> None:
    with pytest.raises(TypeError, match="codec must be"):
        MemoryChangeReceiptProjector(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_value", "plan_value", "message"),
    [
        (object(), editor_plan(), "source must be"),
        (source(), object(), "plan must be"),
    ],
)
def test_prepare_rejects_invalid_inputs(source_value: object, plan_value: object, message: str) -> None:
    projector = MemoryChangeReceiptProjector(codec())
    with pytest.raises(TypeError, match=message):
        projector.prepare(source_value, plan_value, timestamp=BASE_TIME)  # type: ignore[arg-type]


@pytest.mark.parametrize("timestamp", [None, "2026-07-01", BASE_TIME.replace(tzinfo=None)])
def test_prepare_requires_timezone_aware_timestamp(timestamp: object) -> None:
    with pytest.raises(ValueError, match="prepared_at"):
        MemoryChangeReceiptProjector(codec()).prepare(
            source(),
            editor_plan(),
            timestamp=timestamp,  # type: ignore[arg-type]
        )


def test_prepare_projects_create_and_noop_sets_without_confusing_them() -> None:
    plan = editor_plan()
    receipt = MemoryChangeReceiptProjector(codec()).prepare(source(), plan, timestamp=BASE_TIME)
    assert receipt.state is MemoryChangeReceiptState.PREPARED
    assert receipt.expected_created_uris
    assert receipt.expected_updated_uris == ()
    assert receipt.expected_deleted_uris == ()
    assert all(change.action is MemoryNodeChangeAction.CREATE for change in receipt.prepared_node_changes)


def test_prepare_projects_update_with_bound_before_and_after_digests() -> None:
    receipt = MemoryChangeReceiptProjector(codec()).prepare(
        source(),
        update_editor_plan(preference_content="- 更新为简洁回答"),
        timestamp=BASE_TIME,
    )
    assert len(receipt.expected_updated_uris) == 1
    change = receipt.prepared_node_changes[0]
    assert change.action is MemoryNodeChangeAction.UPDATE
    assert change.before_digest is not None
    assert change.expected_after_digest is not None
    assert change.before_digest != change.expected_after_digest


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("current", object(), "current must be"),
        ("source", object(), "source must be"),
        ("journal", object(), "journal must be"),
    ],
)
def test_finalize_rejects_invalid_inputs(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    projector, prepared, journal = committed_triplet(tmp_path)
    arguments = {"current": prepared, "source": prepared.source, "journal": journal}
    arguments[field] = invalid
    with pytest.raises(TypeError, match=message):
        projector.finalize(**arguments)  # type: ignore[arg-type]


def test_finalize_create_uses_actual_revision_and_digest(tmp_path: Path) -> None:
    projector, prepared, journal = committed_triplet(tmp_path)
    committed = projector.finalize(prepared, prepared.source, journal)
    assert committed.state is MemoryChangeReceiptState.COMMITTED
    assert committed.committed_at == journal.updated_at
    assert committed.node_changes[0].action is MemoryNodeChangeAction.CREATE
    assert committed.node_changes[0].before_revision is None
    assert committed.node_changes[0].after_revision == 1
    assert committed.node_changes[0].after_digest is not None


def test_finalize_update_uses_actual_before_and_after_revisions(tmp_path: Path) -> None:
    projector, prepared, journal = committed_triplet(tmp_path, update=True)
    committed = projector.finalize(prepared, prepared.source, journal)
    change = committed.node_changes[0]
    assert change.action is MemoryNodeChangeAction.UPDATE
    assert change.before_revision == 1
    assert change.after_revision == 2
    assert change.before_digest is not None
    assert change.after_digest is not None
    assert change.before_digest != change.after_digest


def test_finalize_is_idempotent_only_for_identical_committed_receipt(tmp_path: Path) -> None:
    projector, prepared, journal = committed_triplet(tmp_path)
    committed = projector.finalize(prepared, prepared.source, journal)
    assert projector.finalize(committed, committed.source, journal) is committed
    conflicting = replace(committed, committed_at=committed.committed_at.replace(microsecond=1))
    with pytest.raises(MemoryChangeReceiptError, match="conflicts"):
        projector.finalize(conflicting, conflicting.source, journal)


def test_finalize_rejects_unprepared_extra_journal_entry(tmp_path: Path) -> None:
    projector, prepared, journal = committed_triplet(tmp_path)
    extra = document(MemoryKind.PROFILE)
    extra_uri = MemoryURI.from_address(extra.address)
    extra_entry = MemoryTransactionJournalEntry(extra_uri, None, extra)
    entries = tuple(sorted((*journal.entries, extra_entry), key=lambda item: str(item.uri)))
    locks = tuple(sorted((*journal.lock_identities, str(extra_uri))))
    tampered = replace(journal, entries=entries, lock_identities=locks)
    with pytest.raises(MemoryChangeReceiptError, match="unprepared node"):
        projector.finalize(prepared, prepared.source, tampered)


def test_finalize_rejects_relation_delta_different_from_prepared_plan(tmp_path: Path) -> None:
    projector, prepared, journal = committed_triplet(tmp_path, with_relation=True)
    assert prepared.added_relations
    tampered = replace(prepared, added_relations=())
    with pytest.raises(MemoryChangeReceiptError, match="added relations differ"):
        projector.finalize(tampered, tampered.source, journal)


def test_node_change_requires_journal_entry() -> None:
    with pytest.raises(TypeError, match="journal entry must be"):
        MemoryChangeReceiptProjector(codec())._node_change(object())  # type: ignore[arg-type]


def test_confirmation_cannot_change_without_prepared_intention_confirmation() -> None:
    before = document(MemoryKind.INTENTION)
    after = document(MemoryKind.INTENTION, revision=2, timestamp=BASE_TIME.replace(day=2))
    uri = MemoryURI.from_address(before.address)
    expected = MemoryPreparedNodeChange(
        MemoryNodeChangeAction.UPDATE,
        uri,
        "a" * 64,
        "b" * 64,
        confirms_intention=False,
    )
    with pytest.raises(MemoryChangeReceiptError, match="outside the prepared plan"):
        MemoryChangeReceiptProjector._verify_confirmation(before, after, expected)


def test_prepared_intention_confirmation_must_be_applied_at_commit_time() -> None:
    before = document(MemoryKind.INTENTION)
    after = replace(
        document(MemoryKind.INTENTION, revision=2),
        metadata=replace(document(MemoryKind.INTENTION, revision=2).metadata, updated_at=BASE_TIME.replace(day=2)),
    )
    expected = MemoryPreparedNodeChange(
        MemoryNodeChangeAction.UPDATE,
        MemoryURI.from_address(before.address),
        "a" * 64,
        "b" * 64,
        confirms_intention=True,
    )
    with pytest.raises(MemoryChangeReceiptError, match="did not apply"):
        MemoryChangeReceiptProjector._verify_confirmation(before, after, expected)


def test_forward_link_projection_rejects_non_document_source() -> None:
    with pytest.raises(TypeError, match="relation source"):
        MemoryChangeReceiptProjector._forward_links((object(),))  # type: ignore[arg-type]


def test_document_digest_is_none_for_missing_document_and_stable_for_same_document() -> None:
    projector = MemoryChangeReceiptProjector(codec())
    value = document()
    assert projector._document_digest(None) is None
    assert projector._document_digest(value) == projector._document_digest(value)


def test_same_change_intent_ignores_state_and_timestamps(tmp_path: Path) -> None:
    projector, prepared, journal = committed_triplet(tmp_path)
    committed = projector.finalize(prepared, prepared.source, journal)
    assert projector.same_change_intent(prepared, committed)


def test_same_change_intent_rejects_another_valid_editor_plan(tmp_path: Path) -> None:
    projector, prepared, _journal = committed_triplet(tmp_path)
    changed = projector.prepare(
        prepared.source,
        editor_plan(with_relation=True),
        timestamp=prepared.prepared_at,
    )
    assert not projector.same_change_intent(prepared, changed)


def test_same_change_intent_detects_source_change(tmp_path: Path) -> None:
    projector, prepared, _journal = committed_triplet(tmp_path)
    changed = replace(prepared, source=source("b" * 32))
    assert not projector.same_change_intent(prepared, changed)


def test_sorted_uris_returns_canonical_stable_order() -> None:
    values = {"memory://tools/z.md", "memory://preferences/a.md", "memory://profile.md"}
    result = MemoryChangeReceiptProjector._sorted_uris(values)
    assert tuple(str(uri) for uri in result) == tuple(sorted(values))


def test_finalize_rejects_non_committed_journal_beyond_store_boundary(tmp_path: Path) -> None:
    projector, prepared, journal = committed_triplet(tmp_path)
    rolled_back = MemoryTransactionJournalRecord(
        transaction_id=journal.transaction_id,
        state=MemoryTransactionJournalState.ROLLED_BACK,
        created_at=journal.created_at,
        updated_at=journal.updated_at,
        lock_identities=journal.lock_identities,
        entries=journal.entries,
    )
    with pytest.raises(MemoryChangeReceiptError, match="COMMITTED"):
        projector.finalize(prepared, prepared.source, rolled_back)
