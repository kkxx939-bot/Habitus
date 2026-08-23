"""六类 Schema、规范 Markdown 文档和物理记忆树组合测试。"""

from dataclasses import replace
from datetime import date, timedelta

import pytest

from memory.document import (
    MemoryDocumentCodec,
    MemoryDocumentConfig,
    MemoryDocumentIntegrityError,
    MemoryDocumentLimitError,
    MemoryDocumentMetadata,
    MemoryLinkType,
    MemoryStoredLink,
)
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from memory.schema import MemorySchemaError, MemorySchemaRegistry
from memory.tree import MemoryTree, MemoryTreeConfig, MemoryTreeIntegrityError
from memory.uri import MemoryURI
from tests.helpers import BASE_TIME, document, memory_fields


def test_default_registry_contains_exactly_six_declared_memory_kinds() -> None:
    registry = MemorySchemaRegistry.load_default()
    assert tuple(schema.kind for schema in registry.all()) == tuple(MemoryKind)
    with pytest.raises(MemorySchemaError, match="unknown"):
        registry.get("topic")


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_each_schema_validates_fields_builds_address_and_renders_human_markdown(kind: MemoryKind) -> None:
    registry = MemorySchemaRegistry.load_default()
    fields = memory_fields(kind)

    normalized = registry.validate(kind, fields)
    address = registry.address_for(kind, normalized)
    markdown = registry.render_markdown(kind, normalized)

    assert address.kind is kind
    assert markdown.endswith("\n")
    assert "HABITUS_MEMORY_FIELDS" not in markdown


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_each_schema_rejects_unknown_business_or_system_fields(kind: MemoryKind) -> None:
    registry = MemorySchemaRegistry.load_default()
    valid = memory_fields(kind)
    for forbidden in ("uri", "revision", "owner", "confidence", "evidence"):
        with pytest.raises(MemorySchemaError, match="unknown"):
            registry.validate(kind, {**valid, forbidden: "forbidden"})


def test_profile_and_preference_require_content_but_do_not_share_address_fields() -> None:
    registry = MemorySchemaRegistry.load_default()
    with pytest.raises(MemorySchemaError):
        registry.validate(MemoryKind.PROFILE, {"topic": "回答风格", "content": "- 简洁"})
    with pytest.raises(MemorySchemaError):
        registry.validate(MemoryKind.PREFERENCE, {"content": "- 简洁"})


def test_entity_details_are_optional_and_absent_section_is_not_fabricated() -> None:
    registry = MemorySchemaRegistry.load_default()
    fields = {"category": "项目", "name": "Habitus", "summary": "Habitus 是记忆系统。"}
    normalized = registry.validate(MemoryKind.ENTITY, fields)
    markdown = registry.render_markdown(MemoryKind.ENTITY, normalized)

    assert "details" not in normalized
    assert "Habitus 是记忆系统" in markdown


def test_tool_requires_real_name_shape_and_at_least_one_non_empty_knowledge_field() -> None:
    registry = MemorySchemaRegistry.load_default()
    with pytest.raises(MemorySchemaError, match="non-empty content"):
        registry.validate(MemoryKind.TOOL, {"tool_name": "workspace.inspect"})
    with pytest.raises(MemorySchemaError, match="non-empty content"):
        registry.validate(MemoryKind.TOOL, {"tool_name": "workspace.inspect", "purpose": " "})
    assert registry.validate(
        MemoryKind.TOOL,
        {"tool_name": "workspace.inspect", "verification": "检查返回状态"},
    )["tool_name"] == "workspace.inspect"


def test_event_date_is_actual_date_type_and_intention_status_is_controlled() -> None:
    registry = MemorySchemaRegistry.load_default()
    event = registry.validate(
        MemoryKind.EVENT,
        {"event_date": "2026-07-01", "event_name": "确认方案", "summary": "用户确认方案。"},
    )
    assert event["event_date"] == date(2026, 7, 1)
    with pytest.raises(MemorySchemaError):
        registry.validate(
            MemoryKind.INTENTION,
            {"intent_name": "完成重构", "status": "cancelled"},
        )


def test_metadata_revision_and_confirmation_rules_are_monotonic() -> None:
    initial = MemoryDocumentMetadata.initial(BASE_TIME, confirmed=True)
    advanced = initial.next_revision(BASE_TIME + timedelta(days=1), refresh_confirmation=False)
    reconfirmed = advanced.next_revision(BASE_TIME + timedelta(days=2), refresh_confirmation=True)

    assert advanced.revision == 2
    assert advanced.created_at == initial.created_at
    assert advanced.last_confirmed_at == BASE_TIME
    assert reconfirmed.last_confirmed_at == BASE_TIME + timedelta(days=2)
    with pytest.raises(ValueError, match="backwards"):
        initial.next_revision(BASE_TIME - timedelta(seconds=1))


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_codec_round_trip_is_canonical_for_every_memory_kind(kind: MemoryKind) -> None:
    original = document(kind)
    codec = MemoryDocumentCodec(MemorySchemaRegistry.load_default())
    raw = codec.encode(original)
    restored = codec.decode(raw, expected_address=original.address)

    assert restored == original
    assert raw.count("HABITUS_MEMORY_FIELDS") == 1
    assert raw.endswith("-->\n")


def test_codec_rejects_body_tampering_path_mismatch_duplicate_marker_and_loose_json() -> None:
    codec = MemoryDocumentCodec(MemorySchemaRegistry.load_default())
    original = document(MemoryKind.PREFERENCE)
    raw = codec.encode(original)

    with pytest.raises(MemoryDocumentIntegrityError, match="body"):
        codec.decode(raw.replace("偏好简洁", "偏好冗长", 1), expected_address=original.address)
    with pytest.raises(MemoryDocumentIntegrityError, match="physical tree"):
        codec.decode(raw, expected_address=MemoryAddress.preference("另一个主题"))
    with pytest.raises(MemoryDocumentIntegrityError, match="one terminal"):
        codec.decode(raw + raw, expected_address=original.address)
    with pytest.raises(MemoryDocumentIntegrityError, match="strict JSON"):
        codec.decode(raw.replace('"revision": 1', '"revision": NaN'), expected_address=original.address)


def test_document_requires_intention_confirmation_and_relation_direction() -> None:
    with pytest.raises(ValueError, match="last_confirmed_at"):
        MemoryDocumentCodec(MemorySchemaRegistry.load_default()).build(
            MemoryKind.INTENTION,
            memory_fields(MemoryKind.INTENTION),
            metadata=MemoryDocumentMetadata.initial(BASE_TIME),
        )

    source = document(MemoryKind.PREFERENCE)
    target = document(MemoryKind.ENTITY)
    wrong_source = MemoryStoredLink(
        MemoryURI.from_address(target.address),
        MemoryURI.from_address(source.address),
        MemoryLinkType.BELONGS_TO,
    )
    with pytest.raises(ValueError, match="wrong source"):
        replace(source, links=(wrong_source,))


def test_symmetric_links_are_canonicalized_and_duplicate_links_are_rejected() -> None:
    left = MemoryURI.from_address(MemoryAddress.preference("A"))
    right = MemoryURI.from_address(MemoryAddress.entity("项目", "B"))
    link = MemoryStoredLink(right, left, MemoryLinkType.RELATED_TO)

    assert str(link.from_uri) < str(link.to_uri)
    assert MemoryStoredLink.from_dict(link.to_dict()) == link
    with pytest.raises(ValueError, match="same URI"):
        MemoryStoredLink(left, left, MemoryLinkType.RELATED_TO)


def test_document_limits_apply_to_body_encoded_file_and_both_relation_directions() -> None:
    config = MemoryDocumentConfig(max_markdown_body_chars=5, max_encoded_bytes=10, max_relations_per_document=1)
    with pytest.raises(MemoryDocumentLimitError, match="body"):
        config.validate_body("123456")
    with pytest.raises(MemoryDocumentLimitError, match="encoded"):
        config.validate_encoded(b"12345678901")
    with pytest.raises(MemoryDocumentLimitError, match="backlinks"):
        config.validate_relations(links=1, backlinks=2)


def test_tree_initialization_creates_only_confirmed_static_directories(tmp_path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    root = tree.initialize()
    assert sorted(path.name for path in root.iterdir()) == ["entities", "events", "intentions", "preferences", "tools"]
    assert not (root / "profile.md").exists()


def test_tree_writes_reads_lists_and_deletes_all_memory_kinds(tmp_path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    documents = tuple(document(kind) for kind in MemoryKind)
    for item in documents:
        assert tree.write(item) == item

    assert tuple(tree.read(item.address) for item in documents) == documents
    assert tree.list_addresses() == tuple(item.address for item in documents)
    for item in documents:
        assert tree.delete(item.address)
        assert not tree.delete(item.address)


def test_tree_layers_are_rebuildable_and_never_replace_l2(tmp_path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    item = document(MemoryKind.PREFERENCE)
    tree.write(item)
    directory = MemoryDirectory.preferences()

    tree.write_layers(directory, abstract="简短摘要", overview="完整目录概览")
    assert tree.read_layer(directory, MemoryLevel.ABSTRACT) == "简短摘要"
    assert tree.read_layer(directory, MemoryLevel.OVERVIEW) == "完整目录概览"
    assert tree.read(item.address) == item
    assert tree.delete_layers(directory)
    assert tree.read(item.address) == item


def test_tree_rejects_unknown_hidden_entries_symlinks_and_enumeration_overflow(tmp_path) -> None:
    tree = MemoryTree(tmp_path / "memory", tree_config=MemoryTreeConfig(max_children_per_directory=5))
    tree.initialize()
    (tree.root / "preferences" / ".unknown").write_text("x", encoding="utf-8")
    with pytest.raises(MemoryTreeIntegrityError, match="hidden"):
        tree.direct_addresses(MemoryDirectory.preferences())

    (tree.root / "preferences" / ".unknown").unlink()
    (tree.root / "preferences" / "bad.txt").write_text("x", encoding="utf-8")
    with pytest.raises(MemoryTreeIntegrityError, match="Markdown"):
        tree.direct_addresses(MemoryDirectory.preferences())


def test_tree_bounded_reads_and_directory_limits_fail_explicitly(tmp_path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    tree.write_layers(MemoryDirectory.preferences(), abstract="123456", overview="overview")
    with pytest.raises(MemoryTreeIntegrityError, match="read bound"):
        tree.read_layer_bounded(MemoryDirectory.preferences(), MemoryLevel.ABSTRACT, max_bytes=5)
    with pytest.raises(ValueError):
        tree.list_addresses(limit=0)


def test_tree_detects_document_tampering_on_read(tmp_path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    item = document(MemoryKind.PREFERENCE)
    tree.write(item)
    path = tree.path_for(item.address)
    path.write_text(path.read_text(encoding="utf-8").replace("偏好简洁", "偏好冗长", 1), encoding="utf-8")

    with pytest.raises(MemoryTreeIntegrityError, match="integrity validation"):
        tree.read(item.address)
