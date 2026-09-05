from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.memory.conversation import (
    ConversationAddress,
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
