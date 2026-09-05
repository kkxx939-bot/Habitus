"""六类长期记忆 Schema 声明、字段、路径、渲染和注册表完整性矩阵。"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

import memory.schema.registry as registry_module
from memory.model import MemoryAddress, MemoryKind
from memory.schema import (
    MemoryFieldRole,
    MemoryFieldSchema,
    MemoryFieldType,
    MemoryMergeStrategy,
    MemoryOperationMode,
    MemorySchemaError,
    MemorySchemaRegistry,
    MemoryTypeSchema,
)
from memory.schema.model import _template_fields
from tests.helpers import memory_fields

REGISTRY = MemorySchemaRegistry.load_default()
SCHEMAS = tuple(REGISTRY.all())
FIELDS = tuple((schema.kind, field) for schema in SCHEMAS for field in schema.fields)


def _field(**overrides: object) -> MemoryFieldSchema:
    values: dict[str, object] = {
        "name": "content",
        "field_type": MemoryFieldType.STRING,
        "role": MemoryFieldRole.CONTENT,
        "required": True,
        "merge_strategy": MemoryMergeStrategy.PATCH,
        "description": "字段说明",
    }
    values.update(overrides)
    return MemoryFieldSchema(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["a", "content", "event_date", "field_1", "a1_b2"])
def test_field_schema_accepts_lowercase_snake_case_name(name: str) -> None:
    assert _field(name=name).name == name


@pytest.mark.parametrize("name", ["", "A", "Content", "1field", "_field", "field-name", "field name", "字段", None, 1])
def test_field_schema_rejects_non_snake_case_name(name: object) -> None:
    with pytest.raises((MemorySchemaError, TypeError)):
        _field(name=name)


@pytest.mark.parametrize("field_type", list(MemoryFieldType) + [item.value for item in MemoryFieldType])
def test_field_schema_normalizes_declared_field_types(field_type: object) -> None:
    assert isinstance(_field(field_type=field_type).field_type, MemoryFieldType)


@pytest.mark.parametrize("field_type", ["array", "object", "datetime", "", None, 1])
def test_field_schema_rejects_unknown_field_type(field_type: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        _field(field_type=field_type)


@pytest.mark.parametrize("role", list(MemoryFieldRole) + [item.value for item in MemoryFieldRole])
def test_field_schema_normalizes_declared_roles(role: object) -> None:
    overrides: dict[str, object] = {"role": role}
    if role in {MemoryFieldRole.ADDRESS, "address"}:
        overrides["merge_strategy"] = MemoryMergeStrategy.IMMUTABLE
    assert isinstance(_field(**overrides).role, MemoryFieldRole)


@pytest.mark.parametrize("role", ["identity", "metadata", "", None, 1])
def test_field_schema_rejects_unknown_role(role: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        _field(role=role)


@pytest.mark.parametrize("strategy", list(MemoryMergeStrategy) + [item.value for item in MemoryMergeStrategy])
def test_field_schema_normalizes_merge_strategies(strategy: object) -> None:
    assert isinstance(_field(merge_strategy=strategy).merge_strategy, MemoryMergeStrategy)


@pytest.mark.parametrize("strategy", ["append", "delete", "search_replace", "", None, 1])
def test_field_schema_rejects_unknown_merge_strategy(strategy: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        _field(merge_strategy=strategy)


@pytest.mark.parametrize("required", [True, False])
def test_field_schema_requires_explicit_boolean_required(required: bool) -> None:
    assert _field(required=required).required is required


@pytest.mark.parametrize("required", [0, 1, "true", None, []])
def test_field_schema_rejects_non_boolean_required(required: object) -> None:
    with pytest.raises(MemorySchemaError):
        _field(required=required)


@pytest.mark.parametrize("description", ["说明", " description ", "multi\nline"])
def test_field_schema_accepts_non_empty_description(description: str) -> None:
    assert _field(description=description).description == description


@pytest.mark.parametrize("description", ["", " ", "\n", None, 1])
def test_field_schema_rejects_empty_or_non_text_description(description: object) -> None:
    with pytest.raises(MemorySchemaError):
        _field(description=description)


@pytest.mark.parametrize("allowed", [(), ("open",), ("open", "closed"), ["open", "closed"]])
def test_string_field_accepts_unique_non_empty_allowed_values(allowed: object) -> None:
    assert _field(allowed_values=allowed).allowed_values == tuple(allowed)  # type: ignore[arg-type]


@pytest.mark.parametrize("allowed", ["open", ("",), (" ",), (1,), ("open", "open")])
def test_field_schema_rejects_invalid_allowed_values(allowed: object) -> None:
    with pytest.raises(MemorySchemaError):
        _field(allowed_values=allowed)


@pytest.mark.parametrize("field_type", [MemoryFieldType.INTEGER, MemoryFieldType.NUMBER, MemoryFieldType.BOOLEAN, MemoryFieldType.DATE])
def test_non_string_field_rejects_allowed_values(field_type: MemoryFieldType) -> None:
    with pytest.raises(MemorySchemaError, match="string fields"):
        _field(field_type=field_type, allowed_values=("value",))


@pytest.mark.parametrize(
    ("required", "strategy"),
    [
        (False, MemoryMergeStrategy.IMMUTABLE),
        (True, MemoryMergeStrategy.PATCH),
        (True, MemoryMergeStrategy.REPLACE),
        (False, MemoryMergeStrategy.PATCH),
    ],
)
def test_address_fields_must_be_required_and_immutable(required: bool, strategy: MemoryMergeStrategy) -> None:
    with pytest.raises(MemorySchemaError, match="required and immutable"):
        _field(role=MemoryFieldRole.ADDRESS, required=required, merge_strategy=strategy)


def test_default_schemas_follow_confirmed_tree_and_operation_modes() -> None:
    expected = {
        MemoryKind.PROFILE: ("profile.md", MemoryOperationMode.UPSERT),
        MemoryKind.PREFERENCE: ("preferences/{topic}.md", MemoryOperationMode.UPSERT),
        MemoryKind.ENTITY: ("entities/{category}/{name}.md", MemoryOperationMode.UPSERT),
        MemoryKind.TOOL: ("tools/{tool_name}.md", MemoryOperationMode.UPSERT),
        MemoryKind.EVENT: ("events/{event_date:%Y}/{event_date:%m}/{event_date:%d}/{event_name}.md", MemoryOperationMode.ADD_ONLY),
        MemoryKind.INTENTION: ("intentions/{intent_name}.md", MemoryOperationMode.UPSERT),
    }
    for schema in SCHEMAS:
        assert (schema.path_template, schema.operation_mode) == expected[schema.kind]


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda item: item.kind.value)
def test_default_schema_address_fields_exactly_match_path_placeholders(schema: MemoryTypeSchema) -> None:
    address_names = {field.name for field in schema.address_fields}
    assert address_names == set(_template_fields(schema.path_template, "memory path template"))
    assert all(field.required and field.merge_strategy is MemoryMergeStrategy.IMMUTABLE for field in schema.address_fields)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda item: item.kind.value)
def test_default_schema_content_fields_all_appear_in_markdown(schema: MemoryTypeSchema) -> None:
    assert schema.content_fields
    assert all("{" + field.name + "}" in schema.markdown_template for field in schema.content_fields)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
@pytest.mark.parametrize("forbidden", ["uri", "revision", "created_at", "updated_at", "owner", "tag", "confidence", "evidence", "page_id"])
def test_every_memory_payload_rejects_system_or_unknown_fields(kind: MemoryKind, forbidden: str) -> None:
    with pytest.raises(MemorySchemaError, match="unknown"):
        REGISTRY.validate(kind, {**memory_fields(kind), forbidden: "forbidden"})


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_every_memory_payload_rejects_non_mapping_root(kind: MemoryKind) -> None:
    for value in (None, [], "payload", 1):
        with pytest.raises(MemorySchemaError, match="object"):
            REGISTRY.validate(kind, value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "field"),
    [(schema.kind, field.name) for schema in SCHEMAS for field in schema.fields if field.required],
)
def test_every_required_memory_field_is_enforced(kind: MemoryKind, field: str) -> None:
    payload = memory_fields(kind)
    payload.pop(field)
    with pytest.raises(MemorySchemaError, match=field):
        REGISTRY.validate(kind, payload)


@pytest.mark.parametrize(
    ("kind", "field"),
    [(schema.kind, field.name) for schema in SCHEMAS for field in schema.fields if field.required],
)
def test_every_required_memory_field_rejects_null(kind: MemoryKind, field: str) -> None:
    with pytest.raises(MemorySchemaError, match=field):
        REGISTRY.validate(kind, {**memory_fields(kind), field: None})


STRING_FIELDS = tuple((kind, field.name, field.required) for kind, field in FIELDS if field.field_type is MemoryFieldType.STRING)
DATE_FIELDS = tuple((kind, field.name) for kind, field in FIELDS if field.field_type is MemoryFieldType.DATE)


@pytest.mark.parametrize(("kind", "field", "required"), STRING_FIELDS)
@pytest.mark.parametrize("invalid", [1, True, [], {}, date(2026, 7, 1)])
def test_every_string_memory_field_rejects_non_string_value(
    kind: MemoryKind,
    field: str,
    required: bool,
    invalid: object,
) -> None:
    payload = {**memory_fields(kind), field: invalid}
    with pytest.raises(MemorySchemaError, match="string"):
        REGISTRY.validate(kind, payload)


@pytest.mark.parametrize(("kind", "field", "required"), STRING_FIELDS)
def test_required_string_fields_reject_whitespace_but_optional_string_fields_may_be_empty(
    kind: MemoryKind,
    field: str,
    required: bool,
) -> None:
    payload = {**memory_fields(kind), field: " "}
    schema = REGISTRY.get(kind)
    remaining_non_empty = sum(
        1
        for content_field in schema.content_fields
        if content_field.name != field
        and content_field.name in payload
        and schema._is_non_empty(payload[content_field.name])
    )
    if required or remaining_non_empty < schema.min_non_empty_content_fields:
        with pytest.raises(MemorySchemaError):
            REGISTRY.validate(kind, payload)
    else:
        assert REGISTRY.validate(kind, payload)[field] == " "


@pytest.mark.parametrize(("kind", "field"), DATE_FIELDS)
@pytest.mark.parametrize("value", [date(2026, 7, 1), "2026-07-01"])
def test_date_memory_fields_accept_date_or_iso_string(kind: MemoryKind, field: str, value: object) -> None:
    assert REGISTRY.validate(kind, {**memory_fields(kind), field: value})[field] == date(2026, 7, 1)


@pytest.mark.parametrize(("kind", "field"), DATE_FIELDS)
@pytest.mark.parametrize("invalid", [datetime(2026, 7, 1, tzinfo=UTC), "2026/07/01", "", 1, True, None])
def test_date_memory_fields_reject_datetime_invalid_string_and_non_date(
    kind: MemoryKind,
    field: str,
    invalid: object,
) -> None:
    with pytest.raises(MemorySchemaError):
        REGISTRY.validate(kind, {**memory_fields(kind), field: invalid})


@pytest.mark.parametrize("status", ["open", "waiting", "blocked", "completed"])
def test_intention_accepts_only_declared_status_values(status: str) -> None:
    assert REGISTRY.validate(MemoryKind.INTENTION, {"intent_name": "事项", "status": status})["status"] == status


@pytest.mark.parametrize("status", ["cancelled", "done", "active", "", "OPEN", 1, None])
def test_intention_rejects_unknown_or_non_string_status(status: object) -> None:
    with pytest.raises(MemorySchemaError):
        REGISTRY.validate(MemoryKind.INTENTION, {"intent_name": "事项", "status": status})


@pytest.mark.parametrize("field", ["purpose", "when_to_use", "invocation", "constraints", "failure_recovery", "verification"])
def test_tool_accepts_each_knowledge_field_independently(field: str) -> None:
    payload = {"tool_name": "workspace.inspect", field: "稳定知识"}
    assert REGISTRY.validate(MemoryKind.TOOL, payload)[field] == "稳定知识"


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_name": "workspace.inspect"},
        {"tool_name": "workspace.inspect", "purpose": ""},
        {"tool_name": "workspace.inspect", "purpose": " ", "verification": None},
    ],
)
def test_tool_requires_at_least_one_non_empty_knowledge_field(payload: dict[str, object]) -> None:
    with pytest.raises(MemorySchemaError, match="non-empty content"):
        REGISTRY.validate(MemoryKind.TOOL, payload)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_every_memory_kind_builds_exact_confirmed_address(kind: MemoryKind) -> None:
    payload = memory_fields(kind)
    actual = REGISTRY.address_for(kind, payload)
    if kind is MemoryKind.PROFILE:
        expected = MemoryAddress.profile()
    elif kind is MemoryKind.PREFERENCE:
        expected = MemoryAddress.preference(str(payload["topic"]))
    elif kind is MemoryKind.ENTITY:
        expected = MemoryAddress.entity(str(payload["category"]), str(payload["name"]))
    elif kind is MemoryKind.TOOL:
        expected = MemoryAddress.tool(str(payload["tool_name"]))
    elif kind is MemoryKind.EVENT:
        expected = MemoryAddress.event(payload["event_date"], str(payload["event_name"]))
    else:
        expected = MemoryAddress.intention(str(payload["intent_name"]))
    assert actual == expected


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_every_memory_kind_renders_all_present_fields_and_no_system_metadata(kind: MemoryKind) -> None:
    payload = memory_fields(kind)
    markdown = REGISTRY.render_markdown(kind, payload)
    assert markdown.endswith("\n")
    for value in payload.values():
        rendered = value.isoformat() if isinstance(value, date) else str(value)
        assert rendered in markdown
    for forbidden in ("revision", "created_at", "updated_at", "HABITUS_MEMORY_FIELDS"):
        assert forbidden not in markdown


@pytest.mark.parametrize(
    ("kind", "optional_field", "section"),
    [
        (MemoryKind.ENTITY, "details", "## Details"),
        (MemoryKind.TOOL, "purpose", "## Purpose"),
        (MemoryKind.TOOL, "failure_recovery", "## Failure and Recovery"),
        (MemoryKind.INTENTION, "next_step", "## Next Step"),
        (MemoryKind.INTENTION, "blockers", "## Blockers or Constraints"),
        (MemoryKind.INTENTION, "target_time", "## Target Time"),
    ],
)
def test_omit_empty_sections_removes_absent_optional_semantics(
    kind: MemoryKind,
    optional_field: str,
    section: str,
) -> None:
    payload = memory_fields(kind)
    payload.pop(optional_field, None)
    if kind is MemoryKind.TOOL:
        payload["verification"] = "验证方式"
    markdown = REGISTRY.render_markdown(kind, payload)
    assert section not in markdown


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_registry_get_accepts_enum_or_exact_string(kind: MemoryKind) -> None:
    assert REGISTRY.get(kind) is REGISTRY.get(kind.value)


@pytest.mark.parametrize("kind", ["topic", "skill", "", None, 1])
def test_registry_rejects_unknown_memory_kind(kind: object) -> None:
    with pytest.raises(MemorySchemaError, match="unknown"):
        REGISTRY.get(kind)  # type: ignore[arg-type]


def test_registry_rejects_duplicate_or_incomplete_schema_set() -> None:
    with pytest.raises(MemorySchemaError, match="duplicate"):
        MemorySchemaRegistry((*SCHEMAS, SCHEMAS[0]))
    with pytest.raises(MemorySchemaError, match="incomplete"):
        MemorySchemaRegistry(SCHEMAS[:-1])


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda item: item.kind.value)
def test_type_schema_rejects_path_different_from_confirmed_tree(schema: MemoryTypeSchema) -> None:
    with pytest.raises(MemorySchemaError, match="confirmed memory tree"):
        replace(schema, path_template="other/{name}.md")


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda item: item.kind.value)
@pytest.mark.parametrize("description", ["", " ", None, 1])
def test_type_schema_rejects_empty_or_non_text_description(schema: MemoryTypeSchema, description: object) -> None:
    with pytest.raises(MemorySchemaError):
        replace(schema, description=description)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda item: item.kind.value)
@pytest.mark.parametrize("markdown", ["", None, 1])
def test_type_schema_rejects_empty_or_non_text_markdown(schema: MemoryTypeSchema, markdown: object) -> None:
    with pytest.raises(MemorySchemaError):
        replace(schema, markdown_template=markdown)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda item: item.kind.value)
def test_type_schema_rejects_duplicate_fields(schema: MemoryTypeSchema) -> None:
    with pytest.raises(MemorySchemaError, match="unique"):
        replace(schema, fields=(*schema.fields, schema.fields[0]))


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda item: item.kind.value)
def test_type_schema_requires_at_least_one_content_field(schema: MemoryTypeSchema) -> None:
    address_only = tuple(field for field in schema.fields if field.role is MemoryFieldRole.ADDRESS)
    with pytest.raises(MemorySchemaError, match="content field"):
        replace(schema, fields=address_only)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda item: item.kind.value)
def test_type_schema_rejects_minimum_above_declared_content_count(schema: MemoryTypeSchema) -> None:
    with pytest.raises(MemorySchemaError, match="exceeds"):
        replace(schema, min_non_empty_content_fields=len(schema.content_fields) + 1)


@pytest.mark.parametrize("minimum", [-1, True, 1.0, "1", None])
def test_type_schema_rejects_invalid_min_non_empty_content_fields(minimum: object) -> None:
    with pytest.raises(MemorySchemaError):
        replace(REGISTRY.get(MemoryKind.TOOL), min_non_empty_content_fields=minimum)


@pytest.mark.parametrize("value", [0, 1, "true", None, []])
def test_type_schema_requires_boolean_omit_empty_sections(value: object) -> None:
    with pytest.raises(MemorySchemaError):
        replace(REGISTRY.get(MemoryKind.TOOL), omit_empty_sections=value)


@pytest.mark.parametrize(
    "template",
    [
        "preferences/{unknown}.md",
        "preferences/{topic!r}.md",
        "preferences/{topic:>10}.md",
        "preferences/{Topic}.md",
        "preferences/{topic.md",
    ],
)
def test_type_schema_rejects_invalid_path_template_placeholders(template: str) -> None:
    schema = REGISTRY.get(MemoryKind.PREFERENCE)
    with pytest.raises(MemorySchemaError):
        replace(schema, path_template=template)


@pytest.mark.parametrize(
    "template",
    [
        "# Topic\n\n{unknown}\n",
        "# Topic\n\n{content!r}\n",
        "# Topic\n\n{content:>10}\n",
        "# Topic\n\n{Content}\n",
        "# Topic\n\n{content\n",
    ],
)
def test_type_schema_rejects_invalid_markdown_template_placeholders(template: str) -> None:
    schema = REGISTRY.get(MemoryKind.PREFERENCE)
    with pytest.raises(MemorySchemaError):
        replace(schema, markdown_template=template)


@pytest.mark.parametrize(
    "source",
    [
        "not: [valid",
        "- list-root",
        "scalar",
        "{}",
        "memory_type: profile\nunknown: true",
    ],
)
def test_schema_loader_rejects_invalid_yaml_root_missing_or_unknown_fields(source: str) -> None:
    with pytest.raises((MemorySchemaError, ValueError)):
        registry_module._load_schema(source, "invalid.yaml")


def test_schema_loader_rejects_non_list_fields_and_invalid_field_shape() -> None:
    base = """
memory_type: profile
description: description
path_template: profile.md
markdown_template: "{content}"
operation_mode: upsert
"""
    with pytest.raises(MemorySchemaError, match="fields must be a list"):
        registry_module._load_schema(base + "fields: {}\n", "invalid.yaml")
    with pytest.raises(MemorySchemaError):
        registry_module._load_schema(base + "fields:\n  - scalar\n", "invalid.yaml")


def test_schema_loader_rejects_unknown_or_missing_field_keys() -> None:
    valid_field = """
memory_type: profile
description: description
path_template: profile.md
markdown_template: "{content}"
operation_mode: upsert
fields:
  - name: content
    type: string
    role: content
    required: true
    merge: patch
    description: description
"""
    assert registry_module._load_schema(valid_field, "profile.yaml").kind is MemoryKind.PROFILE
    with pytest.raises(MemorySchemaError, match="unknown"):
        registry_module._load_schema(
            valid_field.replace(
                "    description: description\n",
                "    description: description\n    unknown: true\n",
            ),
            "profile.yaml",
        )
    with pytest.raises(MemorySchemaError, match="missing"):
        registry_module._load_schema(valid_field.replace("    merge: patch\n", ""), "profile.yaml")


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_schema_field_maps_and_role_views_are_complete(kind: MemoryKind) -> None:
    schema = REGISTRY.get(kind)
    assert set(schema.field_map) == {field.name for field in schema.fields}
    assert set(schema.address_fields).isdisjoint(schema.content_fields)
    assert len(schema.address_fields) + len(schema.content_fields) == len(schema.fields)


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        ("text", "text"),
        (1, "1"),
        (1.5, "1.5"),
        (True, "true"),
        (False, "false"),
        (date(2026, 7, 1), "2026-07-01"),
    ],
)
def test_memory_value_rendering_is_deterministic(value: object, rendered: str) -> None:
    assert MemoryTypeSchema._render_value(value) == rendered


def test_memory_value_rendering_and_non_empty_rules_cover_null_and_whitespace() -> None:
    assert MemoryTypeSchema._render_value(None) == ""
    assert MemoryTypeSchema._is_non_empty(None) is False
    assert MemoryTypeSchema._is_non_empty("") is False
    assert MemoryTypeSchema._is_non_empty(" ") is False
    assert MemoryTypeSchema._is_non_empty(0) is True
    assert MemoryTypeSchema._is_non_empty(False) is True
