"""连续前缀水位：乱序处理，只按连续前缀推进。"""

from datetime import date, timedelta

import pytest

from habitus.foundation.lifecycle import Watermark, covers, mark


def _next(value: int) -> int:
    return value + 1


def test_out_of_order_marks_advance_only_when_contiguous() -> None:
    watermark = Watermark(through=0)
    watermark = mark(watermark, 3, successor=_next)
    assert watermark == Watermark(through=0, above=frozenset({3}))
    watermark = mark(watermark, 1, successor=_next)
    assert watermark == Watermark(through=1, above=frozenset({3}))
    watermark = mark(watermark, 2, successor=_next)
    assert watermark == Watermark(through=3)


def test_covers_counts_prefix_and_scattered_marks() -> None:
    watermark = Watermark(through=5, above=frozenset({8}))
    assert covers(watermark, 5)
    assert covers(watermark, 1)
    assert covers(watermark, 8)
    assert not covers(watermark, 6)


def test_marking_a_covered_key_changes_nothing() -> None:
    watermark = Watermark(through=5)
    assert mark(watermark, 3, successor=_next) is watermark


def test_dates_work_as_keys() -> None:
    day = date(2026, 10, 18)
    watermark = mark(Watermark(through=day), day + timedelta(days=1), successor=lambda d: d + timedelta(days=1))
    assert watermark.through == date(2026, 10, 19)


def test_keys_above_must_be_later_than_through() -> None:
    with pytest.raises(ValueError, match="later than through"):
        Watermark(through=5, above=frozenset({4}))
