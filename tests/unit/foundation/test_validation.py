"""边界解码公共类型判定（W16）：bool 不能冒充数字，空白串不算文本，无时区的时间不算带偏移。"""

from datetime import UTC, date, datetime

import pytest

from habitus.foundation.validation import (
    is_aware_datetime,
    is_date_only,
    is_int,
    is_nonblank_str,
    is_number,
    is_str_list,
)


@pytest.mark.parametrize(("value", "expected"), [(0, True), (-3, True), (True, False), (1.0, False), ("1", False)])
def test_is_int_rejects_bool_and_non_integers(value: object, expected: bool) -> None:
    assert is_int(value) is expected


@pytest.mark.parametrize(("value", "expected"), [(1, True), (0.5, True), (False, False), ("0.5", False), (None, False)])
def test_is_number_rejects_bool_and_text(value: object, expected: bool) -> None:
    assert is_number(value) is expected


@pytest.mark.parametrize(("value", "expected"), [("a", True), (" a ", True), ("", False), (" \n", False), (1, False)])
def test_is_nonblank_str_requires_visible_content(value: object, expected: bool) -> None:
    assert is_nonblank_str(value) is expected


def test_is_aware_datetime_requires_an_offset() -> None:
    assert is_aware_datetime(datetime(2026, 1, 1, tzinfo=UTC))
    assert not is_aware_datetime(datetime(2026, 1, 1))
    assert not is_aware_datetime(date(2026, 1, 1))
    assert not is_aware_datetime("2026-01-01T00:00:00+00:00")


def test_is_date_only_excludes_datetime() -> None:
    assert is_date_only(date(2026, 1, 1))
    assert not is_date_only(datetime(2026, 1, 1, tzinfo=UTC))
    assert not is_date_only("2026-01-01")


@pytest.mark.parametrize(
    ("value", "expected"),
    [([], True), (["a", ""], True), (("a",), True), (["a", 1], False), ("ab", False), ({"a"}, False)],
)
def test_is_str_list_checks_every_item(value: object, expected: bool) -> None:
    assert is_str_list(value) is expected
