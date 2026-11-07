"""live 追加、不可变 History 封存、高水位和生命周期释放测试。"""

import json
import re
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.memory.conversation import (
    ConversationAddress,
    ConversationAppendStatus,
    ConversationIngressError,
    ConversationIngressRequest,
    ConversationJournalConfig,
    ConversationJournalError,
    ConversationLayoutError,
    ConversationMessageJournal,
    ConversationSealStatus,
    ConversationWriteConflictError,
)
from habitus.memory.conversation import messages as conversation_messages
from habitus.pre.conversation import ConversationBatch
from tests.helpers import closed_turn
from tests.lock_store import ProcessLocalLockStore


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


@pytest.mark.parametrize(
    "separator",
    [" ", " ", "\u0085", "\u000b", "\u000c", "\u001c", "\u001d", "\u001e"],
)
def test_content_with_unicode_line_boundaries_reads_back_from_live_and_history(
    tmp_path: Path,
    separator: str,
) -> None:
    store = journal(tmp_path)
    turn = closed_turn()
    batch = ConversationBatch(
        "conversation-1",
        (replace(turn[0], content=f"第一行{separator}第二行"), turn[1]),
    )
    store.append(address(), batch)
    raw = store.layout.live_path(address()).read_bytes()
    assert raw.count(b"\n") == len(batch.messages)

    assert store.read_live(address()) == batch
    following = ConversationBatch("conversation-1", closed_turn(start_sequence=2))
    assert store.append(address(), following).status is ConversationAppendStatus.EXTENDED

    sealed = store.seal(address(), through_sequence=1).segment
    assert store.read_segment(address(), sealed.segment_id).messages == batch.messages
    assert store.read_live(address()) == following


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


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("delivery_id", 1, "receipt delivery_id must be lowercase SHA-256 text"),
        ("batch_digest", None, "receipt batch_digest must be lowercase SHA-256 text"),
        ("start_sequence", "0", "receipt start_sequence must be non-negative"),
        ("end_sequence", True, "receipt end_sequence must be non-negative"),
        ("next_sequence", 2.0, "committed receipt next_sequence is inconsistent"),
    ],
)
def test_ingress_receipt_read_rejects_wrongly_typed_fields(
    tmp_path: Path, field: str, bad: object, message: str
) -> None:
    """磁盘上类型错的投递回执字段在 ``from_dict`` 解码时就被拒（W16：类型检查在边界）。"""

    store = journal(tmp_path)
    ingress = ConversationIngressRequest("a" * 64, "b" * 64)
    store.append(address(), ConversationBatch("conversation-1", closed_turn()), ingress=ingress)
    path = store.layout.ingress_path(address(), ingress.delivery_id)
    value = json.loads(path.read_text(encoding="utf-8"))
    value[field] = bad
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ConversationIngressError, match="ingress receipt is corrupt") as caught:
        store.ingress_receipts.read(address(), ingress.delivery_id)
    assert isinstance(caught.value.__cause__, ConversationIngressError)
    assert message in str(caught.value.__cause__)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("conversation_id", 1, "journal state conversation_id must be non-empty text"),
        ("archived_through_sequence", "1", "journal state archived_through_sequence must be null or non-negative"),
        ("released_through_sequence", False, "journal state released_through_sequence must be null or non-negative"),
        ("latest_segment_id", 3, "journal state requires latest_segment_id"),
        ("latest_segment_digest", ["a"], "journal state latest_segment_digest must be lowercase SHA-256"),
    ],
)
def test_journal_state_read_rejects_wrongly_typed_fields(tmp_path: Path, field: str, bad: object, message: str) -> None:
    """磁盘上类型错的 ``state.json`` 字段在读回解码时就被拒。"""

    store = journal(tmp_path)
    store.append(address(), ConversationBatch("conversation-1", closed_turn()))
    store.seal(address(), through_sequence=1)
    path = store.layout.state_path(address())
    value = json.loads(path.read_text(encoding="utf-8"))
    value[field] = bad
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ConversationJournalError, match=re.escape(message)):
        store.read_state(address())


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
