"""JSON Schema 核心校验器的关键字、类型和错误边界矩阵。"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

import pytest

import ModelClient.schema_validation as schema_validation
from ModelClient.schema_validation import JSONSchemaValidationError, validate_json_schema


@pytest.fixture
def fallback_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """强制执行无第三方依赖时必须保持一致的严格核心实现。"""

    original = schema_validation.importlib.import_module

    def missing_jsonschema(name: str):
        if name == "jsonschema":
            raise ImportError("forced fallback")
        return original(name)

    monkeypatch.setattr(schema_validation.importlib, "import_module", missing_jsonschema)


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ({"type": "null"}, None),
        ({"type": "boolean"}, True),
        ({"type": "boolean"}, False),
        ({"type": "integer"}, 0),
        ({"type": "integer"}, -7),
        ({"type": "number"}, 3),
        ({"type": "number"}, 3.5),
        ({"type": "string"}, ""),
        ({"type": "string"}, "中文"),
        ({"type": "array"}, []),
        ({"type": "array"}, [1, "x"]),
        ({"type": "object"}, {}),
        ({"type": "object"}, {"x": 1}),
        ({"type": ["null", "string"]}, None),
        ({"type": ["null", "string"]}, "value"),
    ],
)
def test_fallback_accepts_each_declared_json_type(
    fallback_validator: None,
    schema: Mapping[str, object],
    value: object,
) -> None:
    assert validate_json_schema(value, schema) is value


@pytest.mark.parametrize(
    ("expected_type", "value"),
    [
        ("null", False),
        ("boolean", 0),
        ("boolean", "false"),
        ("integer", True),
        ("integer", 1.0),
        ("number", False),
        ("number", "1"),
        ("string", 1),
        ("array", (1, 2)),
        ("array", {"0": 1}),
        ("object", []),
        ("object", "{}"),
    ],
)
def test_fallback_rejects_type_coercion(
    fallback_validator: None,
    expected_type: str,
    value: object,
) -> None:
    with pytest.raises(JSONSchemaValidationError, match="must have type"):
        validate_json_schema(value, {"type": expected_type})


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"type": 1},
        {"type": ["string", 1]},
        {"type": []},
        {"type": "date"},
    ],
)
def test_fallback_rejects_invalid_or_unsupported_type_declarations(
    fallback_validator: None,
    schema: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError):
        validate_json_schema("value", schema)


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ({"const": "fixed"}, "fixed"),
        ({"enum": ["open", "closed"]}, "open"),
        ({"allOf": [{"type": "integer"}, {"minimum": 1}]}, 2),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "x"),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, 1),
        ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, "x"),
        ({"not": {"type": "null"}}, "x"),
    ],
)
def test_fallback_accepts_valid_const_enum_and_combinators(
    fallback_validator: None,
    schema: Mapping[str, object],
    value: object,
) -> None:
    assert validate_json_schema(value, schema) is value


@pytest.mark.parametrize(
    ("schema", "value", "message"),
    [
        ({"const": "fixed"}, "other", "const"),
        ({"enum": ["open", "closed"]}, "unknown", "enum"),
        ({"enum": "open"}, "open", "enum"),
        ({"allOf": [{"type": "integer"}, {"minimum": 1}]}, 0, "minimum"),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, [], "anyOf"),
        ({"oneOf": [{"type": "number"}, {"type": "integer"}]}, 1, "oneOf"),
        ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, [], "oneOf"),
        ({"not": {"type": "null"}}, None, "forbidden"),
    ],
)
def test_fallback_rejects_invalid_const_enum_and_combinators(
    fallback_validator: None,
    schema: Mapping[str, object],
    value: object,
    message: str,
) -> None:
    with pytest.raises(JSONSchemaValidationError, match=message):
        validate_json_schema(value, schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"allOf": []},
        {"allOf": "not-array"},
        {"anyOf": []},
        {"anyOf": ["not-schema"]},
        {"oneOf": {}},
        {"not": []},
    ],
)
def test_fallback_rejects_malformed_combinator_schemas(
    fallback_validator: None,
    schema: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError):
        validate_json_schema("x", schema)


def test_fallback_resolves_local_refs_and_json_pointer_escaping(fallback_validator: None) -> None:
    schema = {
        "$defs": {
            "plain": {"type": "integer", "minimum": 1},
            "a/b": {"type": "string", "const": "slash"},
            "a~b": {"type": "string", "const": "tilde"},
        },
        "type": "object",
        "required": ["count", "slash", "tilde"],
        "properties": {
            "count": {"$ref": "#/$defs/plain"},
            "slash": {"$ref": "#/$defs/a~1b"},
            "tilde": {"$ref": "#/$defs/a~0b"},
        },
    }
    value = {"count": 2, "slash": "slash", "tilde": "tilde"}
    assert validate_json_schema(value, schema) is value


@pytest.mark.parametrize(
    "reference",
    [
        "https://example.test/schema.json",
        "plain-name",
        "#/missing",
        "#/$defs/non_object",
    ],
)
def test_fallback_rejects_external_unresolved_or_non_object_refs(
    fallback_validator: None,
    reference: str,
) -> None:
    schema = {"$defs": {"non_object": "value"}, "$ref": reference}
    with pytest.raises(ValueError):
        validate_json_schema("x", schema)


@pytest.mark.parametrize(
    "value",
    [
        {"name": "Habitus"},
        {"name": "Habitus", "count": 1},
        {"name": "Habitus", "labels": {"stable": True}},
    ],
)
def test_fallback_accepts_object_properties_and_typed_additional_properties(
    fallback_validator: None,
    value: dict[str, object],
) -> None:
    schema = {
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
        "additionalProperties": {"anyOf": [{"type": "integer"}, {"type": "object"}]},
        "minProperties": 1,
        "maxProperties": 3,
    }
    assert validate_json_schema(value, schema) is value


@pytest.mark.parametrize(
    ("schema", "value", "message"),
    [
        ({"type": "object", "required": ["name"]}, {}, "missing required"),
        ({"type": "object", "additionalProperties": False}, {"extra": 1}, "unknown field"),
        ({"type": "object", "additionalProperties": {"type": "string"}}, {"extra": 1}, "type string"),
        ({"type": "object", "minProperties": 2}, {"a": 1}, "minProperties"),
        ({"type": "object", "maxProperties": 1}, {"a": 1, "b": 2}, "maxProperties"),
        ({"type": "object"}, {1: "value"}, "keys must be strings"),
    ],
)
def test_fallback_rejects_object_contract_violations(
    fallback_validator: None,
    schema: Mapping[str, object],
    value: object,
    message: str,
) -> None:
    with pytest.raises(JSONSchemaValidationError, match=message):
        validate_json_schema(value, schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": []},
        {"type": "object", "properties": {"name": "string"}},
        {"type": "object", "required": "name"},
        {"type": "object", "required": [1]},
        {"type": "object", "minProperties": True},
        {"type": "object", "minProperties": None},
        {"type": "object", "maxProperties": -1},
        {"type": "object", "maxProperties": None},
    ],
)
def test_fallback_rejects_malformed_object_schemas(
    fallback_validator: None,
    schema: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError):
        validate_json_schema({"name": "x"}, schema)


@pytest.mark.parametrize(
    "value",
    [
        [],
        ["a"],
        ["a", "b", "c"],
    ],
)
def test_fallback_accepts_bounded_unique_typed_arrays(
    fallback_validator: None,
    value: list[object],
) -> None:
    schema = {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 3,
        "uniqueItems": True,
    }
    assert validate_json_schema(value, schema) is value


@pytest.mark.parametrize(
    ("schema", "value", "message"),
    [
        ({"type": "array", "items": {"type": "string"}}, ["a", 1], "type string"),
        ({"type": "array", "minItems": 2}, [1], "minItems"),
        ({"type": "array", "maxItems": 1}, [1, 2], "maxItems"),
        ({"type": "array", "uniqueItems": True}, [1, 1], "unique"),
        ({"type": "array", "uniqueItems": True}, [{"a": 1}, {"a": 1}], "unique"),
    ],
)
def test_fallback_rejects_array_contract_violations(
    fallback_validator: None,
    schema: Mapping[str, object],
    value: list[object],
    message: str,
) -> None:
    with pytest.raises(JSONSchemaValidationError, match=message):
        validate_json_schema(value, schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "items": []},
        {"type": "array", "minItems": True},
        {"type": "array", "minItems": None},
        {"type": "array", "maxItems": -1},
        {"type": "array", "maxItems": None},
    ],
)
def test_fallback_rejects_malformed_array_schemas(
    fallback_validator: None,
    schema: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError):
        validate_json_schema([], schema)


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ({"type": "string", "minLength": 0}, ""),
        ({"type": "string", "minLength": 2}, "中文"),
        ({"type": "string", "maxLength": 3}, "abc"),
        ({"type": "string", "pattern": r"^[a-z]+$"}, "memory"),
    ],
)
def test_fallback_accepts_string_boundaries(
    fallback_validator: None,
    schema: Mapping[str, object],
    value: str,
) -> None:
    assert validate_json_schema(value, schema) == value


@pytest.mark.parametrize(
    ("schema", "value", "message"),
    [
        ({"type": "string", "minLength": 2}, "a", "minLength"),
        ({"type": "string", "maxLength": 2}, "abc", "maxLength"),
        ({"type": "string", "pattern": r"^[a-z]+$"}, "123", "pattern"),
    ],
)
def test_fallback_rejects_string_contract_violations(
    fallback_validator: None,
    schema: Mapping[str, object],
    value: str,
    message: str,
) -> None:
    with pytest.raises(JSONSchemaValidationError, match=message):
        validate_json_schema(value, schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "pattern": 1},
        {"type": "string", "minLength": True},
        {"type": "string", "minLength": None},
        {"type": "string", "maxLength": -1},
        {"type": "string", "maxLength": None},
    ],
)
def test_fallback_rejects_malformed_string_schemas(
    fallback_validator: None,
    schema: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError):
        validate_json_schema("value", schema)


def test_fallback_propagates_invalid_regular_expression(fallback_validator: None) -> None:
    with pytest.raises(re.error):
        validate_json_schema("value", {"type": "string", "pattern": "["})


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ({"minimum": 1}, 1),
        ({"maximum": 1}, 1),
        ({"exclusiveMinimum": 1}, 2),
        ({"exclusiveMaximum": 2}, 1),
        ({"minimum": -1.5, "maximum": 1.5}, 0.5),
    ],
)
def test_fallback_accepts_numeric_boundaries(
    fallback_validator: None,
    schema: Mapping[str, object],
    value: int | float,
) -> None:
    assert validate_json_schema(value, schema) == value


@pytest.mark.parametrize(
    ("schema", "value", "message"),
    [
        ({"minimum": 1}, 0, "minimum"),
        ({"maximum": 1}, 2, "maximum"),
        ({"exclusiveMinimum": 1}, 1, "exclusiveMinimum"),
        ({"exclusiveMaximum": 1}, 1, "exclusiveMaximum"),
        ({"type": "number"}, math.inf, "finite"),
        ({"type": "number"}, -math.inf, "finite"),
        ({"type": "number"}, math.nan, "finite"),
    ],
)
def test_fallback_rejects_numeric_contract_violations(
    fallback_validator: None,
    schema: Mapping[str, object],
    value: int | float,
    message: str,
) -> None:
    with pytest.raises(JSONSchemaValidationError, match=message):
        validate_json_schema(value, schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"minimum": True},
        {"minimum": None},
        {"maximum": "1"},
        {"maximum": None},
        {"exclusiveMinimum": None},
        {"exclusiveMaximum": []},
        {"exclusiveMaximum": None},
    ],
)
def test_fallback_rejects_malformed_numeric_bounds(
    fallback_validator: None,
    schema: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError):
        validate_json_schema(1, schema)


def test_external_validator_and_fallback_agree_on_core_acceptance() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "values"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "values": {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
        },
    }
    value = {"name": "memory", "values": [1, 2]}
    assert validate_json_schema(value, schema) is value


@pytest.mark.parametrize("schema", [None, [], "object", 1, True])
def test_public_validator_rejects_non_mapping_schema(schema: object) -> None:
    with pytest.raises(ValueError):
        validate_json_schema({}, schema)  # type: ignore[arg-type]
