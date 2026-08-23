"""六类记忆地址、目录层级和 memory:// URI 的严格映射测试。"""

from datetime import date

import pytest

from memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from memory.uri import MemoryURI, MemoryURIError, MemoryURINodeType


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        (MemoryAddress.profile(), "memory://profile.md"),
        (MemoryAddress.preference("回答风格"), "memory://preferences/回答风格.md"),
        (MemoryAddress.entity("项目", "Habitus"), "memory://entities/项目/habitus.md"),
        (MemoryAddress.tool("workspace.inspect"), "memory://tools/workspace.inspect.md"),
        (MemoryAddress.event(date(2026, 7, 1), "确认记忆树"), "memory://events/2026/07/01/确认记忆树.md"),
        (MemoryAddress.intention("完成重构"), "memory://intentions/完成重构.md"),
    ],
)
def test_address_uri_round_trip_for_every_memory_kind(address: MemoryAddress, expected: str) -> None:
    uri = MemoryURI.from_address(address)
    assert str(uri) == expected
    assert uri.node_type is MemoryURINodeType.DOCUMENT
    assert uri.to_address() == address


@pytest.mark.parametrize("name", ["", "../escape", "name.md", ".abstract", " value ", "a/b"])
def test_dynamic_addresses_reject_empty_escape_suffix_and_reserved_names(name: str) -> None:
    with pytest.raises(ValueError):
        MemoryAddress.preference(name)


def test_address_fields_are_kind_specific() -> None:
    with pytest.raises(ValueError, match="profile"):
        MemoryAddress(MemoryKind.PROFILE, name="extra")
    with pytest.raises(ValueError, match="category"):
        MemoryAddress(MemoryKind.PREFERENCE, name="topic", category="wrong")
    with pytest.raises(ValueError, match="requires event_date"):
        MemoryAddress(MemoryKind.EVENT, name="event")


def test_directory_validation_and_lineage_follow_confirmed_tree_only() -> None:
    directory = MemoryDirectory.events(2026, 7, 1)
    assert directory.lineage() == (
        MemoryDirectory(("events", "2026", "07", "01")),
        MemoryDirectory(("events", "2026", "07")),
        MemoryDirectory(("events", "2026")),
        MemoryDirectory.events(),
        MemoryDirectory.root(),
    )
    with pytest.raises(ValueError, match="outside"):
        MemoryDirectory(("topics",))
    with pytest.raises(ValueError, match="calendar"):
        MemoryDirectory(("events", "2026", "02", "30"))


def test_directory_and_semantic_layer_uri_round_trip() -> None:
    directory = MemoryDirectory.entities("项目")
    directory_uri = MemoryURI.from_directory(directory)
    overview_uri = MemoryURI.from_layer(directory, MemoryLevel.OVERVIEW)

    assert directory_uri.to_directory() == directory
    assert overview_uri.to_layer() == (directory, MemoryLevel.OVERVIEW)
    assert overview_uri.parent == directory_uri


@pytest.mark.parametrize(
    "uri",
    [
        "preferences/topic.md",
        "viking://preferences/topic.md",
        "memoryos://preferences/topic.md",
        "memory://preferences/topic",
        "memory://preferences//topic.md",
        "memory://preferences/topic.md/",
        " memory://profile.md",
        "memory://../profile.md",
    ],
)
def test_uri_rejects_short_legacy_noncanonical_and_tree_escape_forms(uri: str) -> None:
    assert not MemoryURI.is_valid(uri)
    with pytest.raises(MemoryURIError):
        MemoryURI(uri)


def test_uri_preserves_unicode_and_canonicalizes_reserved_percent_encoding() -> None:
    uri = MemoryURI("memory://preferences/视频%20输出.md")
    assert str(uri) == "memory://preferences/视频%20输出.md"
    assert uri.decoded_path == "preferences/视频 输出.md"
    assert MemoryURI.parse(uri) is uri


def test_uri_join_requires_directory_and_prefix_matches_complete_segments() -> None:
    preferences = MemoryURI.from_directory(MemoryDirectory.preferences())
    assert preferences.join("回答风格.md") == "memory://preferences/回答风格.md"
    with pytest.raises(MemoryURIError, match="directory"):
        preferences.join("回答风格.md").join("child")
    assert MemoryURI("memory://entities/项目/Habitus.md").matches_prefix("memory://entities/项目")
    assert not MemoryURI("memory://entities/项目甲/Habitus.md").matches_prefix("memory://entities/项目")
