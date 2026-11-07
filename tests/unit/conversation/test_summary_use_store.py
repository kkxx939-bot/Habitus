from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import timedelta

import pytest

from habitus.memory.conversation import (
    ConversationAddress,
    ConversationLayoutError,
    ConversationSummaryUseError,
    SQLiteConversationSummaryUseStore,
)
from habitus.memory.conversation.indexing import summary_reference
from tests.helpers import BASE_TIME, segment, segment_summary


def test_summary_actual_use_protects_recent_source_and_clears_retirement_candidate(tmp_path) -> None:
    address = ConversationAddress("conversation-1", BASE_TIME.date())
    summary = segment_summary()
    reference = summary_reference(address, summary)
    store = SQLiteConversationSummaryUseStore(tmp_path / "summary_use.sqlite3")

    candidate = store.mark_retire_candidate(reference, marked_at=BASE_TIME)
    assert candidate.useful_recall_count == 0
    assert candidate.retire_candidate_at == BASE_TIME

    used = store.record_use((reference,), used_at=BASE_TIME + timedelta(days=1))[0]
    assert used.useful_recall_count == 1
    assert used.retire_candidate_at is None
    assert store.recently_used(reference, now=BASE_TIME + timedelta(days=30), protection_days=90)
    assert not store.recently_used(reference, now=BASE_TIME + timedelta(days=100), protection_days=90)


def test_summary_use_store_is_monotonic_and_idempotently_deletes_state(tmp_path) -> None:
    address = ConversationAddress("conversation-1", BASE_TIME.date())
    reference = summary_reference(address, segment_summary())
    store = SQLiteConversationSummaryUseStore(tmp_path / "summary_use.sqlite3")
    first = store.record_use((reference,), used_at=BASE_TIME)[0]
    second = store.record_use((reference,), used_at=BASE_TIME - timedelta(days=1))[0]

    assert first.useful_recall_count == 1
    assert second.useful_recall_count == 2
    assert second.last_useful_recall_at == BASE_TIME
    assert store.delete_many((reference,)) == 1
    assert store.delete_many((reference,)) == 0


def test_summary_retirement_claim_fences_concurrent_use_by_version(tmp_path) -> None:
    address = ConversationAddress("conversation-1", BASE_TIME.date())
    reference = summary_reference(address, segment_summary())
    store = SQLiteConversationSummaryUseStore(tmp_path / "summary_use.sqlite3")
    candidate = store.mark_retire_candidate(reference, marked_at=BASE_TIME)
    claimed = store.claim_retirement(
        reference,
        expected_version=candidate.version,
        claimed_at=BASE_TIME + timedelta(days=1),
    )

    assert claimed.retiring_at == BASE_TIME + timedelta(days=1)
    with pytest.raises(ConversationSummaryUseError, match="retiring"):
        store.record_use((reference,), used_at=BASE_TIME + timedelta(days=2))


def test_summary_use_coverage_cleanup_paginates_beyond_batch_limit(tmp_path) -> None:
    address = ConversationAddress("conversation-1", BASE_TIME.date())
    store = SQLiteConversationSummaryUseStore(
        tmp_path / "summary_use.sqlite3",
        max_batch_size=2,
    )
    references = []
    for start in range(0, 10, 2):
        source = segment(segment_id=f"{start:012d}-{start + 1:012d}")
        reference = summary_reference(address, segment_summary(source))
        store.record_use((reference,), used_at=BASE_TIME)
        references.append(reference)

    assert store.delete_coverage(address, start_sequence=0, end_sequence=9) == 5
    assert store.read_many(tuple(references[:2])) == ()
    assert store.read_many(tuple(references[2:4])) == ()
    assert store.read_many((references[4],)) == ()


def test_summary_use_rows_are_keyed_by_the_case_folded_conversation(tmp_path) -> None:
    """W1 第 4 项 A1：只差大小写的会话号是同一个会话，用一种写法记下的使用，用另一种写法也清得掉。"""

    store = SQLiteConversationSummaryUseStore(tmp_path / "summary_use.sqlite3")
    written = ConversationAddress("Conv-ABC", BASE_TIME.date())
    source = segment(segment_id="000000000000-000000000001")
    store.record_use((summary_reference(written, segment_summary(source)),), used_at=BASE_TIME)

    cleaned = ConversationAddress("conv-abc", BASE_TIME.date())
    assert store.delete_coverage(cleaned, start_sequence=0, end_sequence=9) == 1


@pytest.mark.parametrize(
    ("column", "bad", "message"),
    [
        ("conversation_id", b"conversation-1", "conversation_id must be one safe non-empty path segment"),
        ("summary_id", b"7", "summary reference summary_id must be normalized non-empty text"),
        ("useful_recall_count", 1.5, "summary use state useful_recall_count must be non-negative"),
        ("version", "two", "summary use state version must be non-negative"),
    ],
)
def test_summary_use_store_rejects_wrongly_typed_sqlite_columns(
    tmp_path, column: str, bad: object, message: str
) -> None:
    """SQLite 表不是 STRICT（BLOB、非整数仍可落进列里），类型错的列在 ``_state`` 解码时就被拒（W16）。"""

    address = ConversationAddress("conversation-1", BASE_TIME.date())
    reference = summary_reference(address, segment_summary())
    path = tmp_path / "summary_use.sqlite3"
    store = SQLiteConversationSummaryUseStore(path)
    store.record_use((reference,), used_at=BASE_TIME)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(f"UPDATE conversation_summary_use SET {column} = ?", (bad,))

    with pytest.raises(ConversationSummaryUseError, match="failed to read") as caught:
        store.read_many((reference,))
    assert isinstance(caught.value.__cause__, ValueError)
    assert message in str(caught.value.__cause__)


def test_summary_use_coverage_cleanup_rejects_non_text_summary_id(tmp_path) -> None:
    """覆盖范围清理读回的 ``summary_id`` 不是文本时，在进入 ``segment_range`` 之前被拒。"""

    address = ConversationAddress("conversation-1", BASE_TIME.date())
    reference = summary_reference(address, segment_summary())
    path = tmp_path / "summary_use.sqlite3"
    store = SQLiteConversationSummaryUseStore(path)
    store.record_use((reference,), used_at=BASE_TIME)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE conversation_summary_use SET summary_id = ?", (b"1",))

    with pytest.raises(ConversationSummaryUseError, match="delete Summary use coverage") as caught:
        store.delete_coverage(address, start_sequence=0, end_sequence=1)
    assert isinstance(caught.value.__cause__, ConversationLayoutError)
    assert "segment_id must be text" in str(caught.value.__cause__)
