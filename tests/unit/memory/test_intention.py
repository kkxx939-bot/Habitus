"""Intention 状态分区和只读时间复核测试。"""

from dataclasses import replace
from datetime import timedelta

import pytest

from habitus.memory.intention import (
    MemoryIntentionRecallScope,
    MemoryIntentionReviewConfig,
    MemoryIntentionReviewer,
    MemoryIntentionReviewLevel,
    allowed_memory_index_kinds,
    intention_matches_scope,
    memory_index_kind,
)
from habitus.memory.model import MemoryKind
from tests.helpers import BASE_TIME, document


@pytest.mark.parametrize("status", ["open", "waiting", "blocked"])
def test_active_intentions_share_active_partition_and_default_scope(status: str) -> None:
    assert memory_index_kind(MemoryKind.INTENTION, intention_status=status) == "intention"
    assert intention_matches_scope(status, MemoryIntentionRecallScope.ACTIVE)
    assert not intention_matches_scope(status, MemoryIntentionRecallScope.COMPLETED)


def test_completed_intention_uses_separate_partition_and_explicit_scope() -> None:
    assert memory_index_kind(MemoryKind.INTENTION, intention_status="completed") == "intention_completed"
    assert not intention_matches_scope("completed", MemoryIntentionRecallScope.ACTIVE)
    assert intention_matches_scope("completed", MemoryIntentionRecallScope.COMPLETED)
    assert allowed_memory_index_kinds(
        (MemoryKind.INTENTION,), MemoryIntentionRecallScope.ALL
    ) == ("intention", "intention_completed")


def test_kinds_filter_is_unique_and_completed_scope_selects_completed_partition() -> None:
    with pytest.raises(ValueError, match="unique"):
        allowed_memory_index_kinds(
            (MemoryKind.PROFILE, MemoryKind.PROFILE), MemoryIntentionRecallScope.ACTIVE
        )
    assert allowed_memory_index_kinds(
        (MemoryKind.INTENTION,), MemoryIntentionRecallScope.COMPLETED
    ) == ("intention_completed",)


def test_non_intention_kind_rejects_intention_status() -> None:
    with pytest.raises(ValueError, match="only Intention"):
        memory_index_kind(MemoryKind.PROFILE, intention_status="open")


@pytest.mark.parametrize(
    ("days", "level"),
    [
        (0, MemoryIntentionReviewLevel.CURRENT),
        (30, MemoryIntentionReviewLevel.FIRST_REVIEW),
        (60, MemoryIntentionReviewLevel.SECOND_REVIEW),
        (180, MemoryIntentionReviewLevel.STRONG_REVIEW),
        (365, MemoryIntentionReviewLevel.STRONG_REVIEW),
    ],
)
def test_reviewer_only_emits_escalating_reminder_without_changing_status(days: int, level: MemoryIntentionReviewLevel) -> None:
    intention = document(MemoryKind.INTENTION)
    review = MemoryIntentionReviewer().review(intention, now=BASE_TIME + timedelta(days=days))

    assert review is not None
    assert review.level is level
    assert review.unconfirmed_days == days
    assert intention.fields["status"] == "open"
    assert intention.metadata.revision == 1


def test_completed_intention_needs_no_time_review_but_remains_a_document() -> None:
    intention = document(
        MemoryKind.INTENTION,
        fields={"intent_name": "完成记忆系统重构", "status": "completed"},
    )
    assert MemoryIntentionReviewer().review(intention, now=BASE_TIME + timedelta(days=365)) is None
    assert intention.fields["status"] == "completed"


def test_reconfirmation_resets_review_clock() -> None:
    original = document(MemoryKind.INTENTION)
    reconfirmed = replace(
        original,
        metadata=original.metadata.next_revision(
            BASE_TIME + timedelta(days=170), refresh_confirmation=True
        ),
    )
    review = MemoryIntentionReviewer().review(reconfirmed, now=BASE_TIME + timedelta(days=181))
    assert review is not None
    assert review.level is MemoryIntentionReviewLevel.CURRENT
    assert review.unconfirmed_days == 11


def test_review_thresholds_must_be_strictly_increasing_and_time_cannot_move_backwards() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        MemoryIntentionReviewConfig(30, 30, 180)
    with pytest.raises(ValueError, match="precedes"):
        MemoryIntentionReviewer().review(document(MemoryKind.INTENTION), now=BASE_TIME - timedelta(days=1))
