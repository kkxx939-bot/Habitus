"""live 追加、不可变 History 封存、高水位和生命周期释放测试。"""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from infrastructure.store.contracts.path_lock import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import (
    ConversationAddress,
    ConversationAppendStatus,
    ConversationIngressError,
    ConversationIngressRequest,
    ConversationJournalConfig,
    ConversationLayoutError,
    ConversationMessageJournal,
    ConversationSealStatus,
    ConversationWriteConflictError,
)
from memory.conversation import messages as conversation_messages
from pre.conversation import ConversationBatch
from tests.helpers import closed_turn


def journal(tmp_path: Path) -> ConversationMessageJournal:
    return ConversationMessageJournal(tmp_path, PathLock(ProcessLocalLockStore()))


def address() -> ConversationAddress:
    return ConversationAddress("conversation-1", date(2026, 7, 1))


def test_append_is_contiguous_idempotent_and_rejects_conflicting_replay(tmp_path: Path) -> None:
    store = journal(tmp_path)
    batch = ConversationBatch("conversation-1", closed_turn())
    first = store.append(address(), batch)
    replay = store.append(address(), batch)
    assert first.status is ConversationAppendStatus.CREATED
    assert replay.status is ConversationAppendStatus.UNCHANGED
    assert replay.appended_count == 0

    conflict = ConversationBatch(
        "conversation-1",
        (replace(batch.messages[0], content="不同内容"), batch.messages[1]),
    )
    with pytest.raises(ConversationWriteConflictError, match="conflicts"):
        store.append(address(), conflict)

    gap = ConversationBatch("conversation-1", closed_turn(start_sequence=3))
    with pytest.raises(ConversationWriteConflictError, match="gap"):
        store.append(address(), gap)


def test_delivery_receipt_recovers_commit_lost_after_live_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = journal(tmp_path)
    batch = ConversationBatch("conversation-1", closed_turn())
    ingress = ConversationIngressRequest("a" * 64, "b" * 64)
    commit = store.ingress_receipts.commit

    def fail_commit(*_args: object, **_kwargs: object) -> object:
        raise ConversationIngressError("simulated receipt commit loss")

    monkeypatch.setattr(store.ingress_receipts, "commit", fail_commit)
    with pytest.raises(Exception, match="simulated receipt commit loss"):
        store.append(address(), batch, ingress=ingress)
    assert store.read_live(address()) == batch

    monkeypatch.setattr(store.ingress_receipts, "commit", commit)
    replay = store.append(address(), batch, ingress=ingress)
    assert replay.status is ConversationAppendStatus.UNCHANGED
    assert replay.next_sequence == 2
    receipt = store.ingress_receipts.read(address(), ingress.delivery_id)
    assert receipt is not None and receipt.state.value == "committed"


def test_delivery_id_survives_history_release_and_rejects_rebinding(tmp_path: Path) -> None:
    store = journal(tmp_path)
    batch = ConversationBatch("conversation-1", closed_turn())
    ingress = ConversationIngressRequest("c" * 64, "d" * 64)
    store.append(address(), batch, ingress=ingress)
    archived = store.seal(address(), through_sequence=1).segment
    store.release_history_prefix(address(), (archived,))

    replay = store.append(address(), batch, ingress=ingress)
    assert replay.status is ConversationAppendStatus.UNCHANGED
    assert replay.next_sequence == 2
    with pytest.raises(ConversationWriteConflictError, match="delivery_id"):
        store.append(
            address(),
            batch,
            ingress=ConversationIngressRequest(ingress.delivery_id, "e" * 64),
        )


def test_seal_publishes_history_trims_live_and_replay_is_unchanged(tmp_path: Path) -> None:
    store = journal(tmp_path)
    all_messages = (*closed_turn(start_sequence=0), *closed_turn(start_sequence=2))
    store.append(address(), ConversationBatch("conversation-1", all_messages))

    first = store.seal(address(), through_sequence=1)
    replay = store.seal(address(), through_sequence=1)
    assert first.status is ConversationSealStatus.CREATED
    assert replay.status is ConversationSealStatus.UNCHANGED
    assert first.segment.segment_id == "000000000000-000000000001"
    assert tuple(item.sequence for item in store.read_live(address()).messages) == (2, 3)
    assert store.list_history(address()) == (first.segment,)
    state = store.read_state(address())
    assert state.archived_through_sequence == 1
    assert state.next_sequence == 2


def test_outbox_callback_runs_before_history_publish_and_failure_leaves_live_intact(tmp_path: Path) -> None:
    store = journal(tmp_path)
    batch = ConversationBatch("conversation-1", closed_turn())
    store.append(address(), batch)
    observed = []

    def fail(segment) -> None:
        observed.append(segment.segment_id)
        assert not store.layout.history_path(address(), segment.segment_id).exists()
        raise RuntimeError("outbox failed")

    with pytest.raises(RuntimeError, match="outbox failed"):
        store.seal(address(), through_sequence=1, before_history_publish=fail)
    assert observed == ["000000000000-000000000001"]
    assert store.list_history(address()) == ()
    assert store.read_live(address()) == batch


def test_history_release_requires_oldest_prefix_and_preserves_high_watermark_after_delete(tmp_path: Path) -> None:
    store = journal(tmp_path)
    store.append(
        address(),
        ConversationBatch("conversation-1", (*closed_turn(start_sequence=0), *closed_turn(start_sequence=2))),
    )
    first = store.seal(address(), through_sequence=1).segment
    second = store.seal(address(), through_sequence=3).segment
    with pytest.raises(ConversationWriteConflictError, match="oldest"):
        store.release_history_prefix(address(), (second,))

    assert store.release_history_prefix(address(), (first,)) == (first.segment_id,)
    assert store.list_history(address()) == (second,)
    state = store.read_state(address())
    assert state.released_through_sequence == 1
    assert state.archived_through_sequence == 3

    next_batch = ConversationBatch("conversation-1", closed_turn(start_sequence=4))
    assert store.append(address(), next_batch).status is ConversationAppendStatus.EXTENDED


def test_purge_released_history_recovers_physical_file_left_by_interrupted_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = journal(tmp_path)
    store.append(address(), ConversationBatch("conversation-1", closed_turn()))
    archived = store.seal(address(), through_sequence=1).segment
    history_path = store.layout.history_path(address(), archived.segment_id)
    original_unlink = conversation_messages.durable_unlink
    monkeypatch.setattr(
        conversation_messages,
        "durable_unlink",
        lambda _path, *, artifact_root: False,
    )

    assert store.release_history_prefix(address(), (archived,)) == (archived.segment_id,)
    assert history_path.exists()
    assert store.list_history(address()) == ()

    monkeypatch.setattr(conversation_messages, "durable_unlink", original_unlink)
    assert store.purge_released_history(address(), max_items=1) == (archived.segment_id,)
    assert not history_path.exists()
    assert store.read_state(address()).released_through_sequence == archived.end_sequence


def test_address_enumeration_is_sorted_and_rejects_unknown_tree_entries(tmp_path: Path) -> None:
    store = journal(tmp_path)
    later = ConversationAddress("z", date(2026, 7, 2))
    earlier = ConversationAddress("a", date(2026, 7, 1))
    store.append(later, ConversationBatch("z", closed_turn()))
    store.append(earlier, ConversationBatch("a", closed_turn()))
    assert store.list_addresses() == (earlier, later)

    (store.layout.messages_root() / "unexpected.txt").write_text("bad")
    with pytest.raises(Exception, match="unsupported entry"):
        store.list_addresses()


def test_identity_directory_scan_uses_existing_journal_tree_bound(tmp_path: Path) -> None:
    config = ConversationJournalConfig(max_conversation_tree_entries=1)
    store = ConversationMessageJournal(
        tmp_path,
        PathLock(ProcessLocalLockStore()),
        config=config,
    )
    target = ConversationAddress("target", date(2026, 7, 1))
    parent = store.layout.messages_root() / "2026" / "07" / "01"
    (parent / target.identity).mkdir(parents=True)

    assert store.layout.conversation_directory(target) == parent / target.identity
    (parent / "second").mkdir()
    with pytest.raises(ConversationLayoutError, match="resolved safely"):
        store.layout.conversation_directory(target)


def test_case_aliased_conversations_share_one_identity_and_lock(tmp_path: Path) -> None:
    store = journal(tmp_path)
    upper = ConversationAddress("Chat", date(2026, 7, 1))
    try:
        lower = ConversationAddress("chat", date(2026, 7, 1))
    except ValueError:
        return
    store.append(upper, ConversationBatch("Chat", closed_turn()))
    upper_directory = store.layout.conversation_directory(upper)
    lower_directory = store.layout.conversation_directory(lower)
    if not lower_directory.exists() or not upper_directory.samefile(lower_directory):
        pytest.skip("filesystem is case-sensitive")

    assert upper == lower
    assert store.layout.lock_key(upper) == store.layout.lock_key(lower)


def test_legacy_case_preserving_conversation_directory_remains_visible(tmp_path: Path) -> None:
    store = journal(tmp_path)
    legacy = store.layout.messages_root() / "2026" / "07" / "01" / "Chat"
    legacy.mkdir(parents=True)
    canonical = ConversationAddress("chat", date(2026, 7, 1))

    assert store.layout.conversation_directory(canonical) == legacy
    store.append(
        ConversationAddress("Chat", date(2026, 7, 1)),
        ConversationBatch("Chat", closed_turn()),
    )
    assert store.read_live(canonical).conversation_id == "Chat"
    archived = store.seal(canonical, through_sequence=1).segment
    restored = store.read_segment(canonical, archived.segment_id)
    assert restored.conversation_id == "Chat"
    assert restored.digest == archived.digest
    assert store.list_addresses() == (canonical,)


def test_conversation_layout_rejects_multiple_physical_identity_aliases(tmp_path: Path) -> None:
    store = journal(tmp_path)
    parent = store.layout.messages_root() / "2026" / "07" / "01"
    upper = parent / "Chat"
    lower = parent / "chat"
    upper.mkdir(parents=True)
    try:
        lower.mkdir()
    except FileExistsError:
        return
    if upper.samefile(lower):
        return

    with pytest.raises(Exception, match="aliases"):
        store.layout.conversation_directory(ConversationAddress("chat", date(2026, 7, 1)))
