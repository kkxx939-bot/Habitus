"""记忆地址、URI、文档元数据、关系与编解码的组合契约矩阵。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from types import MappingProxyType

import pytest

from habitus.infrastructure.tree.metadata import DocumentMetadata
from habitus.memory.document import (
    MemoryDocumentCodec,
    MemoryDocumentIntegrityError,
    MemoryLinkType,
    MemoryStoredLink,
)
from habitus.memory.document.link import normalize_stored_links, parse_stored_links
from habitus.memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.schema import MemorySchemaRegistry
from habitus.memory.uri import MemoryURI, MemoryURIError, MemoryURINodeType
from tests.helpers import BASE_TIME, document, memory_fields

REGISTRY = MemorySchemaRegistry.load_default()
CODEC = MemoryDocumentCodec(REGISTRY)
DYNAMIC_KINDS = (
    MemoryKind.PREFERENCE,
    MemoryKind.ENTITY,
    MemoryKind.TOOL,
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
    return MemoryAddress.tool(name)


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
    tuple((kind, "category") for kind in MemoryKind if kind is not MemoryKind.ENTITY),
)
def test_address_rejects_fields_not_owned_by_memory_kind(kind: MemoryKind, extra_field: str) -> None:
    kwargs: dict[str, object] = {"kind": kind}
    if kind is not MemoryKind.PROFILE:
        kwargs["name"] = "名称"
    if kind is MemoryKind.ENTITY:
        kwargs["category"] = "分类"
    kwargs[extra_field] = "额外"
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
    ],
)
def test_memory_directory_parent_and_lineage_are_lossless(
    parts: tuple[str, ...], parent: tuple[str, ...] | None
) -> None:
    directory = MemoryDirectory(parts)
    assert (None if directory.parent() is None else directory.parent().parts) == parent
    assert directory.lineage()[-1] == MemoryDirectory.root()
    assert len(directory.lineage()) == len(parts) + 1


@pytest.mark.parametrize(
    "parts",
    [
        ("unknown",),
        ("preferences", "nested"),
        ("tools", "nested"),
        ("entities", "category", "nested"),
        ("events",),
        ("intentions",),
    ],
)
def test_memory_directory_rejects_unknown_root_and_depth(parts: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        MemoryDirectory(parts)


@pytest.mark.parametrize("level", [MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW])
@pytest.mark.parametrize(
    "directory",
    [
        MemoryDirectory.root(),
        MemoryDirectory.preferences(),
        MemoryDirectory.entities(),
        MemoryDirectory.entities("项目"),
        MemoryDirectory.tools(),
    ],
)
def test_every_valid_directory_supports_l0_l1_uri_round_trip(directory: MemoryDirectory, level: MemoryLevel) -> None:
    uri = MemoryURI.from_layer(directory, level)
    assert uri.node_type is MemoryURINodeType.LAYER
    assert uri.to_layer() == (directory, level)
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
        "habitus:///profile.md",
        "MEMORY://profile.md",
        " habitus://profile.md",
        "habitus://profile.md ",
        "habitus://profile.md/",
        "habitus://preferences//x.md",
        "habitus://preferences/%",
        "habitus://preferences/%0.md",
        "habitus://preferences/%GG.md",
        "habitus://preferences/%FF.md",
        "habitus://preferences/%2F.md",
        "habitus://preferences/%5C.md",
        "habitus://preferences/%00.md",
        "habitus://preferences/.md",
        "habitus://preferences/x.txt",
        "habitus://profile.MD",
        "habitus://profile.md/child",
        "habitus://entities/category.md",
        "habitus://entities/category/name",
        "habitus://events/2026/07/28/event.md",
        "habitus://intentions/intent.md",
        "habitus://topics/topic.md",
        "habitus://skills/skill.md",
        "habitus://conversations/id.md",
    ],
)
def test_uri_rejects_noncanonical_encoding_unknown_tree_and_invalid_node(uri: str) -> None:
    assert MemoryURI.is_valid(uri) is False
    with pytest.raises(MemoryURIError):
        MemoryURI(uri)


@pytest.mark.parametrize(
    ("raw", "canonical", "decoded"),
    [
        ("habitus://preferences/a%20b.md", "habitus://preferences/a%20b.md", "preferences/a b.md"),
        ("habitus://preferences/a%23b.md", "habitus://preferences/a%23b.md", "preferences/a#b.md"),
        ("habitus://preferences/a%25b.md", "habitus://preferences/a%25b.md", "preferences/a%b.md"),
        ("habitus://preferences/a%2bb.md", "habitus://preferences/a%2Bb.md", "preferences/a+b.md"),
        ("habitus://preferences/a%40b.md", "habitus://preferences/a%40b.md", "preferences/a@b.md"),
        ("habitus://preferences/中文.md", "habitus://preferences/中文.md", "preferences/中文.md"),
    ],
)
def test_uri_normalizes_reserved_percent_encoding_and_keeps_unicode(
    raw: str,
    canonical: str,
    decoded: str,
) -> None:
    uri = MemoryURI(raw)
    assert str(uri) == canonical
    assert "/".join(uri.segments) == decoded
    assert MemoryURI.normalize(raw) == canonical


@pytest.mark.parametrize(
    "directory",
    [
        MemoryDirectory.root(),
        MemoryDirectory.preferences(),
        MemoryDirectory.entities(),
        MemoryDirectory.entities("项目"),
        MemoryDirectory.tools(),
    ],
)
def test_directory_uri_properties_and_wrong_conversions(directory: MemoryDirectory) -> None:
    uri = MemoryURI.from_directory(directory)
    assert uri.node_type is MemoryURINodeType.DIRECTORY
    assert uri.to_directory() == directory
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
    with pytest.raises(MemoryURIError):
        uri.to_directory()
    with pytest.raises(MemoryURIError):
        uri.to_layer()


@pytest.mark.parametrize("part", ["", "a/b", "a\\b", None, 1, True])
def test_uri_build_and_join_reject_unsafe_or_non_text_parts(part: object) -> None:
    with pytest.raises((MemoryURIError, TypeError)):
        MemoryURI.build(part)  # type: ignore[arg-type]
    with pytest.raises((MemoryURIError, TypeError)):
        MemoryURI.root().join(part)  # type: ignore[arg-type]


def test_uri_is_immutable_hashable_and_compares_only_normalized_identity() -> None:
    uri = MemoryURI("habitus://preferences/a%2bb.md")
    same = MemoryURI("habitus://preferences/a%2Bb.md")
    assert uri == same
    assert uri == str(same)
    assert uri != "habitus://preferences/other.md"
    assert uri != object()
    assert len({uri, same}) == 1
    assert repr(uri) == "MemoryURI('habitus://preferences/a%2Bb.md')"
    with pytest.raises(AttributeError):
        uri._uri = "habitus://profile.md"  # type: ignore[misc]


@pytest.mark.parametrize("revision", [1, 2, 10, 2**31 - 1])
@pytest.mark.parametrize("offset", [UTC, timezone(timedelta(hours=8)), timezone(timedelta(hours=-5))])
def test_document_metadata_normalizes_timezones_and_preserves_revision(revision: int, offset: timezone) -> None:
    created = datetime(2026, 7, 28, 12, tzinfo=offset)
    metadata = DocumentMetadata(revision, created, created + timedelta(seconds=1))
    assert metadata.revision == revision
    assert metadata.created_at.tzinfo is UTC
    assert metadata.updated_at.tzinfo is UTC


@pytest.mark.parametrize("revision", [0, -1])
def test_document_metadata_rejects_invalid_revision(revision: int) -> None:
    with pytest.raises(ValueError):
        DocumentMetadata(revision, BASE_TIME, BASE_TIME)


@pytest.mark.parametrize(
    ("field", "value"),
    tuple((field, datetime(2026, 7, 28)) for field in ("created_at", "updated_at")),
)
def test_document_metadata_rejects_naive_timestamps(field: str, value: datetime) -> None:
    kwargs: dict[str, object] = {
        "revision": 1,
        "created_at": BASE_TIME,
        "updated_at": BASE_TIME,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        DocumentMetadata(**kwargs)  # type: ignore[arg-type]


def test_initial_metadata_starts_at_revision_one() -> None:
    metadata = DocumentMetadata.initial(BASE_TIME)
    assert metadata.revision == 1
    assert metadata.created_at == metadata.updated_at == BASE_TIME


@pytest.mark.parametrize("seconds", [0, 1, 60, 86_400])
def test_next_revision_is_monotonic(seconds: int) -> None:
    initial = DocumentMetadata.initial(BASE_TIME)
    updated = initial.next_revision(BASE_TIME + timedelta(seconds=seconds))
    assert updated.revision == 2
    assert updated.created_at == initial.created_at
    assert updated.updated_at == BASE_TIME + timedelta(seconds=seconds)


@pytest.mark.parametrize("link_type", tuple(MemoryLinkType))
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (MemoryAddress.preference("回答风格"), MemoryAddress.entity("项目", "Habitus")),
        (MemoryAddress.tool("workspace.inspect"), MemoryAddress.entity("项目", "检查项目")),
        (MemoryAddress.preference("完成重构"), MemoryAddress.profile()),
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


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "link",
        {},
        {"from_uri": "habitus://profile.md"},
        {
            "from_uri": "habitus://profile.md",
            "to_uri": "habitus://preferences/a.md",
            "link_type": "related_to",
            "extra": 1,
        },
        {"from_uri": 1, "to_uri": "habitus://preferences/a.md", "link_type": "related_to"},
        {"from_uri": "habitus://profile.md", "to_uri": 1, "link_type": "related_to"},
        {"from_uri": "habitus://profile.md", "to_uri": "habitus://preferences/a.md", "link_type": 1},
        {"from_uri": "habitus://profile.md", "to_uri": "habitus://preferences/a.md", "link_type": "unknown"},
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
        metadata=DocumentMetadata.initial(BASE_TIME),
    )
    assert isinstance(item.fields, MappingProxyType)
    payload.clear()
    assert item.fields
    with pytest.raises(TypeError):
        item.fields["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize("kind", tuple(MemoryKind))
@pytest.mark.parametrize("body", ["", " ", "body"])
def test_memory_document_rejects_empty_non_newline_or_non_text_body(kind: MemoryKind, body: str) -> None:
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
    marker = "\n<!-- HABITUS_FIELDS\n"
    body, metadata_source = raw.split(marker, 1)
    metadata = json.loads(metadata_source[: -len("\n-->\n")])
    transform(metadata)  # type: ignore[operator]
    return body + marker + json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n-->\n"


@pytest.mark.parametrize(
    "field",
    ["type", "revision", "created_at", "updated_at", "fields", "links", "backlinks"],
)
def test_codec_rejects_missing_system_metadata_field(field: str) -> None:
    source = document(MemoryKind.PREFERENCE)
    raw = CODEC.encode(source)
    corrupted = _replace_metadata(raw, lambda metadata: metadata.pop(field))
    with pytest.raises(MemoryDocumentIntegrityError):
        CODEC.decode(corrupted, expected_address=source.address)


@pytest.mark.parametrize("field", ["uri", "owner", "confidence", "page_id", "evidence", "last_confirmed_at", "unknown"])
def test_codec_rejects_extra_system_metadata_field(field: str) -> None:
    source = document(MemoryKind.PREFERENCE)
    raw = CODEC.encode(source)
    corrupted = _replace_metadata(raw, lambda metadata: metadata.__setitem__(field, "forbidden"))
    with pytest.raises(MemoryDocumentIntegrityError):
        CODEC.decode(corrupted, expected_address=source.address)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", 1),
        ("type", "topic"),
        ("revision", 0),
        ("revision", True),
        ("created_at", 1),
        ("created_at", "not-time"),
        ("updated_at", None),
        ("updated_at", "2026-13-01T00:00:00Z"),
        ("fields", []),
        ("links", {}),
        ("backlinks", "links"),
    ],
)
def test_codec_rejects_invalid_system_metadata_types(field: str, value: object) -> None:
    source = document(MemoryKind.PREFERENCE)
    raw = CODEC.encode(source)
    corrupted = _replace_metadata(raw, lambda metadata: metadata.__setitem__(field, value))
    with pytest.raises(MemoryDocumentIntegrityError):
        CODEC.decode(corrupted, expected_address=source.address)


@pytest.mark.parametrize("value", ["1", True, 1.0, None, [1]])
def test_codec_decoder_rejects_non_integer_revision_before_building_metadata(value: object) -> None:
    """注释里 revision 类型错时在解码器就被拒（W16：类型检查在边界，不靠 ``DocumentMetadata``）。"""

    source = document(MemoryKind.PREFERENCE)
    corrupted = _replace_metadata(CODEC.encode(source), lambda metadata: metadata.__setitem__("revision", value))
    with pytest.raises(MemoryDocumentIntegrityError, match="do not satisfy their Schema") as raised:
        CODEC.decode(corrupted, expected_address=source.address)
    assert isinstance(raised.value.__cause__, ValueError)
    assert "document revision must be a positive integer" in str(raised.value.__cause__)


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
        lambda raw: raw.replace("\n<!-- HABITUS_FIELDS\n", "", 1),
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
