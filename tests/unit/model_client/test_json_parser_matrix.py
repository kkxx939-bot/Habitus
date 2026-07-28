"""模型 JSON 提取、纯语法修复和语义保持边界矩阵。"""

from __future__ import annotations

import importlib
import json
import math

import pytest

import ModelClient.json_parser as json_parser
from ModelClient.json_parser import ParsedJSON, parse_json_response


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("{}", {}),
        ("[]", []),
        ('"text"', "text"),
        ("0", 0),
        ("-1", -1),
        ("1.25", 1.25),
        ("true", True),
        ("false", False),
        ("null", None),
        ('{"中文":"值"}', {"中文": "值"}),
        (' [ {"nested":[1,true,null]} ] ', [{"nested": [1, True, None]}]),
    ],
)
def test_strict_json_root_types_round_trip_without_repair(source: str, expected: object) -> None:
    parsed = parse_json_response(source)
    assert parsed == ParsedJSON(expected, "strict")
    assert parsed.repaired is False


@pytest.mark.parametrize(
    ("source", "expected_mode", "expected"),
    [
        ('```json\n{"a":1}\n```', "code_fence", {"a": 1}),
        ('```JSON\n[1,2]\n```', "code_fence", [1, 2]),
        ('```\n{"a":1}\n```', "code_fence", {"a": 1}),
        ('before {"a":{"b":[1,2]}} after', "extracted", {"a": {"b": [1, 2]}}),
        ('prefix [1,{"text":"} ] inside"},3] suffix', "extracted", [1, {"text": "} ] inside"}, 3]),
        ('prefix {"escaped":"quote \\\" and slash \\\\"} suffix', "extracted", {"escaped": 'quote " and slash \\'}),
    ],
)
def test_parser_extracts_fenced_or_balanced_json_without_changing_value(
    source: str,
    expected_mode: str,
    expected: object,
) -> None:
    parsed = parse_json_response(source)
    assert parsed.value == expected
    assert parsed.mode == expected_mode
    assert parsed.repaired is True


@pytest.mark.parametrize(
    "source",
    [
        '{"a":1,}',
        '[1,2,]',
        '{"a":[1,2,],}',
        'prefix {"a":1,} suffix',
        '```json\n{"a":1,}\n```',
    ],
)
def test_trailing_comma_repair_changes_only_punctuation(source: str) -> None:
    parsed = parse_json_response(source)
    if "[1,2" in source and "{" in source:
        expected: object = {"a": [1, 2]}
    elif "{" in source:
        expected = {"a": 1}
    else:
        expected = [1, 2]
    assert parsed.value == expected
    assert parsed.mode == "trailing_comma_repair"


@pytest.mark.parametrize(
    "source",
    [
        '{"a":1,}',
        "{'a': 1}",
        "prefix {\"a\":1,} suffix",
        "plain text",
    ],
)
def test_disabling_repair_rejects_every_non_json_candidate(source: str) -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_json_response(source, allow_repair=False)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('prefix {"a":1} suffix', {"a": 1}),
        ('prefix [1,2] suffix', [1, 2]),
        ('```json\n{"a":1}\n``` trailing {"b":2}', {"a": 1}),
    ],
)
def test_extraction_still_works_when_repair_is_disabled(source: str, expected: object) -> None:
    assert parse_json_response(source, allow_repair=False).value == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('prefix {"text":"[not structure]"} suffix', '{"text":"[not structure]"}'),
        ('x [{"a":1},2] y', '[{"a":1},2]'),
        ('x {"a":"escaped \\\" brace }","b":2} y', '{"a":"escaped \\\" brace }","b":2}'),
    ],
)
def test_balanced_extractor_respects_nested_structures_and_json_strings(source: str, expected: str) -> None:
    assert json_parser._extract_balanced_json(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "no structure",
        "prefix {unclosed",
        "prefix [1,2",
        "prefix } stray",
        "prefix {[}] suffix",
        "prefix [} suffix",
    ],
)
def test_balanced_extractor_rejects_missing_unclosed_or_mismatched_structures(source: str) -> None:
    assert json_parser._extract_balanced_json(source) is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -1,
        1.5,
        "text",
        [],
        [1, {"nested": None}],
        {},
        {"a": [1, 2]},
    ],
)
def test_json_value_guard_accepts_only_lossless_json_values(value: object) -> None:
    assert json_parser._is_json_value(value) is True


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        (1, 2),
        {1, 2},
        b"bytes",
        {1: "non-string-key"},
        [object()],
        {"nested": {1, 2}},
        object(),
    ],
)
def test_json_value_guard_rejects_non_json_python_values(value: object) -> None:
    assert json_parser._is_json_value(value) is False


@pytest.mark.parametrize("source", ["NaN", "Infinity", "-Infinity"])
def test_strict_loader_rejects_non_finite_json_constants(source: str) -> None:
    assert json_parser._loads(source) is json_parser._MISSING


def test_optional_repair_is_absent_without_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def missing(name: str):
        if name == "json_repair":
            raise ImportError("forced missing dependency")
        return original(name)

    monkeypatch.setattr(json_parser.importlib, "import_module", missing)
    assert json_parser._repair_with_optional_dependency("{'a': 1}") is json_parser._MISSING
    parsed = parse_json_response("{'a': 1}")
    assert parsed == ParsedJSON({"a": 1}, "python_literal_repair")


@pytest.mark.parametrize(
    "source",
    [
        "plain words",
        "identifier",
        "undefined",
        "None",
        "True",
        "False",
        "(1, 2)",
        "{1, 2}",
        "b'bytes'",
        "{'items': {1, 2}}",
        "{'key': (1, 2)}",
    ],
)
def test_non_json_semantics_are_also_rejected_without_optional_repair(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    original = importlib.import_module

    def missing(name: str):
        if name == "json_repair":
            raise ImportError("forced missing dependency")
        return original(name)

    monkeypatch.setattr(json_parser.importlib, "import_module", missing)
    with pytest.raises(ValueError):
        parse_json_response(source)


@pytest.mark.parametrize("error", [TypeError("bad type"), ValueError("bad value"), json.JSONDecodeError("bad", "x", 0)])
def test_optional_repair_normalizes_dependency_parse_failures(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    class BrokenRepair:
        @staticmethod
        def loads(_source: str) -> object:
            raise error

    monkeypatch.setattr(json_parser.importlib, "import_module", lambda _name: BrokenRepair)
    assert json_parser._repair_with_optional_dependency("broken") is json_parser._MISSING


@pytest.mark.parametrize(
    "source",
    [
        "plain words",
        "identifier",
        "undefined",
        "None",
        "True",
        "False",
        "(1, 2)",
        "{1, 2}",
        "b'bytes'",
        "{'items': {1, 2}}",
        "{'key': (1, 2)}",
    ],
)
def test_repair_must_not_convert_non_json_semantics_into_another_json_value(source: str) -> None:
    with pytest.raises(ValueError):
        parse_json_response(source)


@pytest.mark.parametrize("source", [None, 1, [], {}, b"bytes"])
def test_parser_rejects_non_text_input(source: object) -> None:
    with pytest.raises(ValueError, match="non-empty text"):
        parse_json_response(source)  # type: ignore[arg-type]


@pytest.mark.parametrize("source", ["", " ", "\n\t"])
def test_parser_rejects_empty_or_whitespace_text(source: str) -> None:
    with pytest.raises(ValueError, match="non-empty text"):
        parse_json_response(source)
