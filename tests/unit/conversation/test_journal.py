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
