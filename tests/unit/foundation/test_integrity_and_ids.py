"""核心标识和规范摘要的不变量测试。"""

from datetime import UTC, date, datetime
from enum import Enum

import pytest

from habitus.foundation.ids import require_safe_path_segment
from habitus.foundation.integrity import (
    CanonicalSerializationError,
    canonical_digest,
    canonical_json,
    canonicalize,
    immutable_snapshot,
    text_digest,
)


@pytest.mark.parametrize("value", ["conversation-1", "中文主题", "name.md"])
def test_safe_path_segment_accepts_single_non_empty_segment(value: str) -> None:
    assert require_safe_path_segment(value, "id") == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "a/b",
        "a\\b",
        "a\x00b",
        "a:b",
        "a*",
        "a?",
        "Project.",
        "Project ",
        "CON",
        "con.txt",
        "LPT9",
        "COM¹",
        "COM².txt",
        "LPT³",
        "CONIN$",
        "CONOUT$",
        None,
    ],
)
def test_safe_path_segment_rejects_escape_and_empty_values(value: object) -> None:
    with pytest.raises(ValueError, match="safe non-empty path segment"):
        require_safe_path_segment(value, "id")


def test_canonical_json_normalizes_mapping_order_dates_enums_and_sets() -> None:
    class Choice(str, Enum):
        A = "a"

    value = {
        "z": {3, 1, 2},
        "time": datetime(2026, 7, 1, 16, tzinfo=UTC),
        "date": date(2026, 7, 1),
        "choice": Choice.A,
    }
    assert canonical_json(value) == (
        '{"choice":"a","date":"2026-07-01","time":"2026-07-01T16:00:00.000000Z","z":[1,2,3]}'
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_canonicalize_rejects_non_json_or_non_finite_values(value: object) -> None:
    with pytest.raises(CanonicalSerializationError):
        canonicalize(value)


def test_canonicalize_rejects_mapping_keys_that_collide_after_string_conversion() -> None:
    with pytest.raises(CanonicalSerializationError, match="collide"):
        canonicalize({1: "integer", "1": "string"})


def test_immutable_snapshot_detaches_and_freezes_nested_values() -> None:
    original = {"items": [{"name": "before"}]}
    snapshot = immutable_snapshot(original)
    original["items"][0]["name"] = "after"

    assert snapshot["items"][0]["name"] == "before"
    with pytest.raises(TypeError):
        snapshot["new"] = "value"


def test_digest_is_deterministic_but_text_digest_preserves_exact_bytes() -> None:
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})
    assert text_digest("a\n") != text_digest("a")
