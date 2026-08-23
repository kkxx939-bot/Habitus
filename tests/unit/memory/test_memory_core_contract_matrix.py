"""记忆地址、URI、文档元数据、关系与编解码的组合契约矩阵。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType

import pytest

from memory.document import (
    MemoryDocumentCodec,
    MemoryDocumentIntegrityError,
    MemoryDocumentMetadata,
    MemoryLinkType,
    MemoryStoredLink,
)
from memory.document.link import normalize_stored_links, parse_stored_links
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from memory.schema import MemorySchemaRegistry
from memory.uri import MemoryURI, MemoryURIError, MemoryURINodeType
from tests.helpers import BASE_TIME, document, memory_fields

REGISTRY = MemorySchemaRegistry.load_default()
CODEC = MemoryDocumentCodec(REGISTRY)
DYNAMIC_KINDS = (
    MemoryKind.PREFERENCE,
    MemoryKind.ENTITY,
    MemoryKind.TOOL,
    MemoryKind.EVENT,
    MemoryKind.INTENTION,
)
VALID_NAMES = (
    "a",
    "A",
    "回答风格",
    "视频 剪辑",
    "workspace.inspect",
    "name-with-dash",
    "name_with_underscore",
    "名字(版本2)",
    "C++",
    "100%完成",
    "mañana",
    "emoji-🎬",
    "~home",
    "a=b&c",
)
INVALID_NAMES: tuple[object, ...] = (
    "",
    ".",
    "..",
    "a/b",
    "a\\b",
    "\x00",
    "name\x00tail",
    " name",
    "name ",
    "\tname",
    "name\n",
    "profile.md",
    "NAME.MD",
    ".abstract",
    ".overview",
    None,
    1,
    True,
    [],
)


def _address(kind: MemoryKind, name: str) -> MemoryAddress:
    if kind is MemoryKind.PREFERENCE:
        return MemoryAddress.preference(name)
    if kind is MemoryKind.ENTITY:
        return MemoryAddress.entity("通用分类", name)
    if kind is MemoryKind.TOOL:
        return MemoryAddress.tool(name)
    if kind is MemoryKind.EVENT:
        return MemoryAddress.event(date(2026, 7, 28), name)
    return MemoryAddress.intention(name)


@pytest.mark.parametrize("kind", DYNAMIC_KINDS)
@pytest.mark.parametrize("name", VALID_NAMES)
def test_dynamic_memory_names_round_trip_without_semantic_rewriting(kind: MemoryKind, name: str) -> None:
    address = _address(kind, name)
    assert address.name == name
    assert MemoryURI.from_address(address).to_address() == address


@pytest.mark.parametrize("kind", DYNAMIC_KINDS)
@pytest.mark.parametrize("name", INVALID_NAMES)
def test_dynamic_memory_names_reject_escape_reserved_suffix_and_non_text(kind: MemoryKind, name: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _address(kind, name)  # type: ignore[arg-type]


@pytest.mark.parametrize("category", VALID_NAMES)
def test_entity_category_round_trips_as_an_independent_directory(category: str) -> None:
    address = MemoryAddress.entity(category, "实体")
    assert MemoryDirectory.for_address(address) == MemoryDirectory.entities(category)
    assert MemoryURI.from_address(address).to_address() == address


@pytest.mark.parametrize("category", INVALID_NAMES)
def test_entity_category_rejects_invalid_semantic_segment(category: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryAddress.entity(category, "实体")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "extra_field"),
    tuple((kind, "category") for kind in MemoryKind if kind is not MemoryKind.ENTITY)
    + tuple((kind, "event_date") for kind in MemoryKind if kind is not MemoryKind.EVENT),
)
def test_address_rejects_fields_not_owned_by_memory_kind(kind: MemoryKind, extra_field: str) -> None:
    kwargs: dict[str, object] = {"kind": kind}
    if kind is not MemoryKind.PROFILE:
        kwargs["name"] = "名称"
    if kind is MemoryKind.ENTITY:
        kwargs["category"] = "分类"
    if kind is MemoryKind.EVENT:
        kwargs["event_date"] = date(2026, 7, 28)
    kwargs[extra_field] = "额外" if extra_field == "category" else date(2026, 7, 27)
    with pytest.raises(ValueError):
        MemoryAddress(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("parts", "parent"),
    [
        ((), None),
        (("preferences",), ()),
        (("entities",), ()),
        (("entities", "项目"), ("entities",)),
        (("tools",), ()),
        (("events",), ()),
        (("events", "2026"), ("events",)),
        (("events", "2026", "07"), ("events", "2026")),
        (("events", "2026", "07", "28"), ("events", "2026", "07")),
        (("intentions",), ()),
    ],
)
def test_memory_directory_parent_and_lineage_are_lossless(
    parts: tuple[str, ...], parent: tuple[str, ...] | None
) -> None:
    directory = MemoryDirectory(parts)
    assert (None if directory.parent() is None else directory.parent().parts) == parent
    assert directory.lineage()[-1] == MemoryDirectory.root()
    assert len(directory.lineage()) == len(parts) + 1


@pytest.mark.parametrize("year", [1, 9, 99, 999, 2024, 2026, 9999])
@pytest.mark.parametrize("month", [1, 2, 6, 12])
@pytest.mark.parametrize("day", [1, 10, 28])
def test_event_directory_accepts_valid_calendar_components(year: int, month: int, day: int) -> None:
    directory = MemoryDirectory.events(year, month, day)
    assert directory.parts == ("events", f"{year:04d}", f"{month:02d}", f"{day:02d}")


@pytest.mark.parametrize(
    "parts",
    [
        ("unknown",),
        ("preferences", "nested"),
        ("tools", "nested"),
        ("intentions", "nested"),
        ("entities", "category", "nested"),
        ("events", "2026", "07", "28", "nested"),
        ("events", "26"),
        ("events", "abcd"),
        ("events", "0000"),
        ("events", "10000"),
        ("events", "2026", "00"),
        ("events", "2026", "13"),
        ("events", "2026", "7"),
        ("events", "2026", "02", "29"),
        ("events", "2024", "02", "30"),
        ("events", "2026", "04", "31"),
        ("events", "2026", "07", "1"),
    ],
)
def test_memory_directory_rejects_unknown_depth_format_and_invalid_calendar(parts: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        MemoryDirectory(parts)


@pytest.mark.parametrize(
    ("year", "month", "day"),
    [
        (None, 1, None),
        (None, None, 1),
        (2026, None, 1),
        (0, 1, 1),
        (10000, 1, 1),
        (2026, 0, 1),
        (2026, 13, 1),
        (2026, 2, 29),
        (2026, 1, 0),
        (2026, 1, 32),
    ],
)
def test_event_directory_factory_rejects_missing_parent_and_invalid_calendar(
    year: int | None,
    month: int | None,
    day: int | None,
) -> None:
    with pytest.raises(ValueError):
        MemoryDirectory.events(year, month, day)


@pytest.mark.parametrize(
    ("position", "value"),
    tuple((position, value) for position in ("year", "month", "day") for value in (True, False, 1.5, "2026", []))
    + (("year", None), ("month", None)),
)
def test_event_directory_factory_rejects_non_integer_components(position: str, value: object) -> None:
    kwargs: dict[str, object] = {"year": 2026, "month": 7, "day": 28}
    kwargs[position] = value
    with pytest.raises((TypeError, ValueError)):
        MemoryDirectory.events(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("level", [MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW])
@pytest.mark.parametrize(
    "directory",
    [
        MemoryDirectory.root(),
        MemoryDirectory.preferences(),
        MemoryDirectory.entities(),
        MemoryDirectory.entities("项目"),
        MemoryDirectory.tools(),
        MemoryDirectory.events(),
        MemoryDirectory.events(2026),
        MemoryDirectory.events(2026, 7),
        MemoryDirectory.events(2026, 7, 28),
        MemoryDirectory.intentions(),
    ],
)
def test_every_valid_directory_supports_l0_l1_uri_round_trip(directory: MemoryDirectory, level: MemoryLevel) -> None:
    uri = MemoryURI.from_layer(directory, level)
    assert uri.node_type is MemoryURINodeType.LAYER
    assert uri.to_layer() == (directory, level)
    assert uri.containing_directory == directory
    assert uri.parent == MemoryURI.from_directory(directory)


@pytest.mark.parametrize("level", [MemoryLevel.DETAIL, 2])
def test_l2_level_has_no_sidecar_uri(level: object) -> None:
    with pytest.raises(ValueError, match="L2"):
        MemoryURI.from_layer(MemoryDirectory.root(), level)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "memory:/profile.md",
        "memory:///profile.md",
        "MEMORY://profile.md",
        " memory://profile.md",
        "memory://profile.md ",
        "memory://profile.md/",
        "memory://preferences//x.md",
        "memory://preferences/%",
        "memory://preferences/%0.md",
        "memory://preferences/%GG.md",
        "memory://preferences/%FF.md",
        "memory://preferences/%2F.md",
        "memory://preferences/%5C.md",
        "memory://preferences/%00.md",
        "memory://preferences/.md",
        "memory://preferences/x.txt",
        "memory://profile.MD",
        "memory://profile.md/child",
        "memory://entities/category.md",
        "memory://entities/category/name",
        "memory://events/2026/02/30/event.md",
        "memory://events/26/07/28/event.md",
        "memory://events/2026/7/28/event.md",
        "memory://events/2026/07/8/event.md",
        "memory://events/2026/07/28/event.txt",
        "memory://topics/topic.md",
        "memory://skills/skill.md",
        "memory://conversations/id.md",
    ],
)
def test_uri_rejects_noncanonical_encoding_unknown_tree_and_invalid_node(uri: str) -> None:
    assert MemoryURI.is_valid(uri) is False
    with pytest.raises(MemoryURIError):
        MemoryURI(uri)


@pytest.mark.parametrize(
    ("raw", "canonical", "decoded"),
    [
        ("memory://preferences/a%20b.md", "memory://preferences/a%20b.md", "preferences/a b.md"),
        ("memory://preferences/a%23b.md", "memory://preferences/a%23b.md", "preferences/a#b.md"),
        ("memory://preferences/a%25b.md", "memory://preferences/a%25b.md", "preferences/a%b.md"),
        ("memory://preferences/a%2bb.md", "memory://preferences/a%2Bb.md", "preferences/a+b.md"),
        ("memory://preferences/a%40b.md", "memory://preferences/a%40b.md", "preferences/a@b.md"),
        ("memory://preferences/中文.md", "memory://preferences/中文.md", "preferences/中文.md"),
    ],
)
def test_uri_normalizes_reserved_percent_encoding_and_keeps_unicode(
    raw: str,
    canonical: str,
    decoded: str,
) -> None:
    uri = MemoryURI(raw)
    assert str(uri) == canonical
    assert uri.decoded_path == decoded
    assert MemoryURI.normalize(raw) == canonical


@pytest.mark.parametrize(
    "directory",
    [
        MemoryDirectory.root(),
        MemoryDirectory.preferences(),
        MemoryDirectory.entities(),
        MemoryDirectory.entities("项目"),
        MemoryDirectory.tools(),
        MemoryDirectory.events(),
        MemoryDirectory.events(2026, 7, 28),
        MemoryDirectory.intentions(),
    ],
)
def test_directory_uri_properties_and_wrong_conversions(directory: MemoryDirectory) -> None:
    uri = MemoryURI.from_directory(directory)
    assert uri.node_type is MemoryURINodeType.DIRECTORY
    assert uri.to_directory() == directory
    assert uri.containing_directory == directory
    assert uri.is_root is (directory == MemoryDirectory.root())
    with pytest.raises(MemoryURIError):
        uri.to_address()
    with pytest.raises(MemoryURIError):
        uri.to_layer()


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_document_uri_properties_and_wrong_conversions(kind: MemoryKind) -> None:
    address = REGISTRY.address_for(kind, memory_fields(kind))
    uri = MemoryURI.from_address(address)
    assert uri.node_type is MemoryURINodeType.DOCUMENT
    assert uri.to_address() == address
    assert uri.containing_directory == MemoryDirectory.for_address(address)
    with pytest.raises(MemoryURIError):
        uri.to_directory()
    with pytest.raises(MemoryURIError):
        uri.to_layer()


@pytest.mark.parametrize("value", [None, 1, True, [], {}])
def test_uri_parse_and_factory_reject_non_uri_types(value: object) -> None:
    with pytest.raises(TypeError):
        MemoryURI.parse(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MemoryURI.from_address(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MemoryURI.from_directory(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("part", ["", "a/b", "a\\b", None, 1, True])
def test_uri_build_and_join_reject_unsafe_or_non_text_parts(part: object) -> None:
    with pytest.raises((MemoryURIError, TypeError)):
        MemoryURI.build(part)  # type: ignore[arg-type]
    with pytest.raises((MemoryURIError, TypeError)):
        MemoryURI.root().join(part)  # type: ignore[arg-type]


def test_uri_is_immutable_hashable_and_compares_only_normalized_identity() -> None:
    uri = MemoryURI("memory://preferences/a%2bb.md")
    same = MemoryURI("memory://preferences/a%2Bb.md")
    assert uri == same
    assert uri == str(same)
    assert uri != "memory://preferences/other.md"
    assert uri != object()
    assert len({uri, same}) == 1
    assert repr(uri) == "MemoryURI('memory://preferences/a%2Bb.md')"
    with pytest.raises(AttributeError):
        uri._uri = "memory://profile.md"  # type: ignore[misc]


@pytest.mark.parametrize("revision", [1, 2, 10, 2**31 - 1])
@pytest.mark.parametrize("offset", [timezone.utc, timezone(timedelta(hours=8)), timezone(timedelta(hours=-5))])
def test_document_metadata_normalizes_timezones_and_preserves_revision(revision: int, offset: timezone) -> None:
    created = datetime(2026, 7, 28, 12, tzinfo=offset)
    metadata = MemoryDocumentMetadata(revision, created, created + timedelta(seconds=1), None)
    assert metadata.revision == revision
    assert metadata.created_at.tzinfo is timezone.utc
    assert metadata.updated_at.tzinfo is timezone.utc


@pytest.mark.parametrize("revision", [0, -1, True, False, 1.0, "1", None])
def test_document_metadata_rejects_invalid_revision(revision: object) -> None:
    with pytest.raises(ValueError):
        MemoryDocumentMetadata(revision, BASE_TIME, BASE_TIME, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    tuple(
        (field, value)
        for field in ("created_at", "updated_at")
        for value in (None, "2026-07-28", 1, date(2026, 7, 28), datetime(2026, 7, 28))
    )
    + tuple(("last_confirmed_at", value) for value in ("2026-07-28", 1, date(2026, 7, 28), datetime(2026, 7, 28))),
)
def test_document_metadata_rejects_missing_non_datetime_or_naive_timestamps(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "revision": 1,
        "created_at": BASE_TIME,
        "updated_at": BASE_TIME,
        "last_confirmed_at": BASE_TIME,
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        MemoryDocumentMetadata(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("confirmed", [True, False])
def test_initial_metadata_sets_confirmation_only_when_explicit(confirmed: bool) -> None:
    metadata = MemoryDocumentMetadata.initial(BASE_TIME, confirmed=confirmed)
    assert metadata.revision == 1
    assert metadata.created_at == metadata.updated_at == BASE_TIME
    assert (metadata.last_confirmed_at == BASE_TIME) is confirmed


@pytest.mark.parametrize("confirmed", [None, 0, 1, "true", []])
def test_initial_metadata_rejects_non_boolean_confirmation(confirmed: object) -> None:
    with pytest.raises(TypeError):
        MemoryDocumentMetadata.initial(BASE_TIME, confirmed=confirmed)  # type: ignore[arg-type]


@pytest.mark.parametrize("refresh", [True, False])
@pytest.mark.parametrize("seconds", [0, 1, 60, 86_400])
def test_next_revision_is_monotonic_and_refreshes_confirmation_only_on_request(refresh: bool, seconds: int) -> None:
    initial = MemoryDocumentMetadata.initial(BASE_TIME, confirmed=True)
    updated = initial.next_revision(BASE_TIME + timedelta(seconds=seconds), refresh_confirmation=refresh)
    assert updated.revision == 2
    assert updated.created_at == initial.created_at
    assert updated.updated_at == BASE_TIME + timedelta(seconds=seconds)
    assert updated.last_confirmed_at == (updated.updated_at if refresh else initial.last_confirmed_at)


@pytest.mark.parametrize("link_type", tuple(MemoryLinkType))
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (MemoryAddress.preference("回答风格"), MemoryAddress.entity("项目", "Habitus")),
        (MemoryAddress.tool("workspace.inspect"), MemoryAddress.event(date(2026, 7, 28), "检查项目")),
        (MemoryAddress.intention("完成重构"), MemoryAddress.profile()),
    ],
)
def test_every_relation_type_round_trips_and_respects_direction(
    link_type: MemoryLinkType,
    left: MemoryAddress,
    right: MemoryAddress,
) -> None:
    source = MemoryURI.from_address(left)
    target = MemoryURI.from_address(right)
    link = MemoryStoredLink(source, target, link_type)
    assert MemoryStoredLink.from_dict(link.to_dict()) == link
    assert link.identity == (str(link.from_uri), str(link.to_uri), link_type.value)
    if link_type.is_symmetric:
        assert str(link.from_uri) < str(link.to_uri)
        assert MemoryStoredLink(target, source, link_type) == link
    else:
        assert link.from_uri == source
        assert link.to_uri == target


@pytest.mark.parametrize("link_type", tuple(MemoryLinkType))
def test_relation_cannot_point_to_same_document(link_type: MemoryLinkType) -> None:
    uri = MemoryURI.from_address(MemoryAddress.profile())
    with pytest.raises(ValueError, match="same URI"):
        MemoryStoredLink(uri, uri, link_type)


@pytest.mark.parametrize(
    "endpoint",
    [
        MemoryURI.root(),
        MemoryURI.from_directory(MemoryDirectory.preferences()),
        MemoryURI.from_layer(MemoryDirectory.root(), MemoryLevel.ABSTRACT),
    ],
)
@pytest.mark.parametrize("position", ["from_uri", "to_uri"])
def test_relation_endpoints_must_be_l2_documents(endpoint: MemoryURI, position: str) -> None:
    kwargs = {
        "from_uri": MemoryURI.from_address(MemoryAddress.profile()),
        "to_uri": MemoryURI.from_address(MemoryAddress.preference("主题")),
        "link_type": MemoryLinkType.DERIVED_FROM,
    }
    kwargs[position] = endpoint
    with pytest.raises(ValueError, match="L2"):
        MemoryStoredLink(**kwargs)


@pytest.mark.parametrize("value", [None, 1, True, "memory://profile.md", object()])
@pytest.mark.parametrize("position", ["from_uri", "to_uri"])
def test_relation_endpoints_reject_non_uri_objects(value: object, position: str) -> None:
    kwargs = {
        "from_uri": MemoryURI.from_address(MemoryAddress.profile()),
        "to_uri": MemoryURI.from_address(MemoryAddress.preference("主题")),
        "link_type": MemoryLinkType.DERIVED_FROM,
    }
    kwargs[position] = value
    with pytest.raises(TypeError):
        MemoryStoredLink(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "link",
        {},
        {"from_uri": "memory://profile.md"},
        {
            "from_uri": "memory://profile.md",
            "to_uri": "memory://preferences/a.md",
            "link_type": "related_to",
            "extra": 1,
        },
        {"from_uri": 1, "to_uri": "memory://preferences/a.md", "link_type": "related_to"},
        {"from_uri": "memory://profile.md", "to_uri": 1, "link_type": "related_to"},
        {"from_uri": "memory://profile.md", "to_uri": "memory://preferences/a.md", "link_type": 1},
        {"from_uri": "memory://profile.md", "to_uri": "memory://preferences/a.md", "link_type": "unknown"},
    ],
)
def test_stored_link_parser_rejects_invalid_shapes_and_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryStoredLink.from_dict(value)


def test_relation_collection_is_sorted_immutable_and_rejects_duplicates() -> None:
    source = MemoryURI.from_address(MemoryAddress.profile())
    first = MemoryStoredLink(source, MemoryURI.from_address(MemoryAddress.preference("A")), MemoryLinkType.DERIVED_FROM)
    second = MemoryStoredLink(
        source, MemoryURI.from_address(MemoryAddress.preference("B")), MemoryLinkType.DERIVED_FROM
    )
    assert normalize_stored_links((second, first), label="links") == (first, second)
    assert parse_stored_links([second.to_dict(), first.to_dict()], label="links") == (first, second)
    with pytest.raises(ValueError, match="duplicate"):
        normalize_stored_links((first, first), label="links")
    with pytest.raises(TypeError, match="tuple"):
        normalize_stored_links([first], label="links")
    with pytest.raises(ValueError, match="array"):
        parse_stored_links((first.to_dict(),), label="links")


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_built_document_fields_are_immutable_snapshots(kind: MemoryKind) -> None:
    payload = memory_fields(kind)
    item = CODEC.build(
        kind,
        payload,
        metadata=MemoryDocumentMetadata.initial(BASE_TIME, confirmed=kind is MemoryKind.INTENTION),
    )
    assert isinstance(item.fields, MappingProxyType)
    payload.clear()
    assert item.fields
    with pytest.raises(TypeError):
        item.fields["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize("kind", tuple(MemoryKind))
@pytest.mark.parametrize("body", ["", " ", "body", 1, None])
def test_memory_document_rejects_empty_non_newline_or_non_text_body(kind: MemoryKind, body: object) -> None:
    source = document(kind)
    with pytest.raises((TypeError, ValueError)):
        replace(source, markdown_body=body)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_document_rejects_kind_address_mismatch(kind: MemoryKind) -> None:
    source = document(kind)
    other = next(candidate for candidate in MemoryKind if candidate is not kind)
    with pytest.raises(ValueError, match="address"):
        replace(source, kind=other)


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_codec_canonical_round_trip_for_every_kind_and_revision(kind: MemoryKind) -> None:
    source = document(kind, revision=3)
    encoded = CODEC.encode(source)
    restored = CODEC.decode(encoded, expected_address=source.address)
    assert restored == source
    assert CODEC.encode(restored) == encoded
    assert encoded.endswith("\n-->\n")


def _replace_metadata(raw: str, transform: object) -> str:
    marker = "\n<!-- HABITUS_MEMORY_FIELDS\n"
    body, metadata_source = raw.split(marker, 1)
    metadata = json.loads(metadata_source[: -len("\n-->\n")])
    transform(metadata)  # type: ignore[operator]
    return body + marker + json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n-->\n"


@pytest.mark.parametrize(
    "field",
    ["memory_type", "revision", "created_at", "updated_at", "last_confirmed_at", "fields", "links", "backlinks"],
)
def test_codec_rejects_missing_system_metadata_field(field: str) -> None:
    source = document(MemoryKind.INTENTION)
    raw = CODEC.encode(source)
    corrupted = _replace_metadata(raw, lambda metadata: metadata.pop(field))
    with pytest.raises(MemoryDocumentIntegrityError):
        CODEC.decode(corrupted, expected_address=source.address)


@pytest.mark.parametrize("field", ["uri", "owner", "confidence", "page_id", "evidence", "unknown"])
def test_codec_rejects_extra_system_metadata_field(field: str) -> None:
    source = document(MemoryKind.PREFERENCE)
    raw = CODEC.encode(source)
    corrupted = _replace_metadata(raw, lambda metadata: metadata.__setitem__(field, "forbidden"))
    with pytest.raises(MemoryDocumentIntegrityError):
        CODEC.decode(corrupted, expected_address=source.address)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("memory_type", 1),
        ("memory_type", "topic"),
        ("revision", 0),
        ("revision", True),
        ("created_at", 1),
        ("created_at", "not-time"),
        ("updated_at", None),
        ("updated_at", "2026-13-01T00:00:00Z"),
        ("last_confirmed_at", 1),
        ("fields", []),
        ("links", {}),
        ("backlinks", "links"),
    ],
)
def test_codec_rejects_invalid_system_metadata_types(field: str, value: object) -> None:
    source = document(MemoryKind.INTENTION)
    raw = CODEC.encode(source)
    corrupted = _replace_metadata(raw, lambda metadata: metadata.__setitem__(field, value))
    with pytest.raises(MemoryDocumentIntegrityError):
        CODEC.decode(corrupted, expected_address=source.address)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_codec_rejects_non_finite_json_constants(constant: str) -> None:
    source = document(MemoryKind.PREFERENCE)
    raw = CODEC.encode(source).replace('"revision": 1', f'"revision": {constant}')
    with pytest.raises(MemoryDocumentIntegrityError, match="strict JSON"):
        CODEC.decode(raw, expected_address=source.address)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), object(), [], {}, None])
def test_codec_rejects_non_serializable_schema_field_value(value: object) -> None:
    with pytest.raises(MemoryDocumentIntegrityError):
        CODEC._json_value(value)


def test_codec_rejects_duplicate_json_keys_even_when_values_match() -> None:
    source = document(MemoryKind.PREFERENCE)
    raw = CODEC.encode(source)
    corrupted = raw.replace('"revision": 1,', '"revision": 1,\n  "revision": 1,')
    with pytest.raises(MemoryDocumentIntegrityError, match="strict JSON"):
        CODEC.decode(corrupted, expected_address=source.address)


@pytest.mark.parametrize(
    "transform",
    [
        lambda raw: raw.replace("\n<!-- HABITUS_MEMORY_FIELDS\n", "", 1),
        lambda raw: raw + raw,
        lambda raw: raw[: -len("\n-->\n")],
        lambda raw: raw.replace("\n-->\n", "\n-- >\n"),
        lambda raw: raw.replace("偏好简洁直接", "偏好详细", 1),
        lambda raw: raw.replace('"topic": "回答风格"', '"topic": "其他主题"', 1),
    ],
)
def test_codec_rejects_missing_duplicate_nonterminal_tampered_or_path_divergent_document(transform: object) -> None:
    source = document(MemoryKind.PREFERENCE)
    raw = CODEC.encode(source)
    with pytest.raises(MemoryDocumentIntegrityError):
        CODEC.decode(transform(raw), expected_address=source.address)  # type: ignore[operator]


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_codec_rejects_noncanonical_whitespace_in_metadata(kind: MemoryKind) -> None:
    source = document(kind)
    raw = CODEC.encode(source)
    corrupted = raw.replace('\n  "backlinks"', '\n    "backlinks"', 1)
    with pytest.raises(MemoryDocumentIntegrityError, match="canonically"):
        CODEC.decode(corrupted, expected_address=source.address)


@pytest.mark.parametrize("value", [None, 1, True, [], {}, object()])
def test_codec_constructor_build_encode_and_decode_reject_wrong_contract_types(value: object) -> None:
    with pytest.raises(TypeError):
        MemoryDocumentCodec(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CODEC.encode(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CODEC.decode(value, expected_address=MemoryAddress.profile())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CODEC.decode(CODEC.encode(document()), expected_address=value)  # type: ignore[arg-type]
