"""记忆树的公开浏览（方案 36 第 2 步 ``habitus memory``）：只给 ``memory://`` 地址，列目录、读文档与语义层。"""

from __future__ import annotations

from pathlib import Path

import pytest

from habitus.memory.browse import MemoryBrowseNotFoundError, MemoryBrowser, MemoryLayerText
from habitus.memory.document import MemoryDocument
from habitus.memory.model import MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from tests.helpers import codec, document


def _tree(tmp_path: Path) -> MemoryTree:
    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    tree.initialize()
    for kind in (MemoryKind.PROFILE, MemoryKind.PREFERENCE, MemoryKind.ENTITY, MemoryKind.EVENT):
        tree.write(document(kind))
    tree.write_layers(MemoryDirectory.preferences(), abstract="偏好：回答简洁。", overview="## 偏好\n- 回答风格：简洁")
    return tree


def test_the_root_lists_the_fixed_branches_and_the_profile(tmp_path: Path) -> None:
    listing = MemoryBrowser(_tree(tmp_path)).list()
    assert str(listing.uri) == "memory://"
    assert [str(item) for item in listing.directories] == sorted(
        f"memory://{name}" for name in ("entities", "events", "intentions", "preferences", "tools")
    )
    assert [str(item) for item in listing.documents] == ["memory://profile.md"]
    assert listing.abstract is None


def test_a_directory_lists_its_documents_with_its_abstract_and_nested_branches(tmp_path: Path) -> None:
    browser = MemoryBrowser(_tree(tmp_path))
    preferences = browser.list("memory://preferences")
    assert [str(item) for item in preferences.documents] == ["memory://preferences/回答风格.md"]
    assert preferences.abstract == "偏好：回答简洁。"
    assert [str(item) for item in browser.list("memory://entities").directories] == ["memory://entities/项目"]
    assert [str(item) for item in browser.list("memory://events/2026/07").directories] == ["memory://events/2026/07/01"]


def test_reading_gives_the_document_or_the_layer_text(tmp_path: Path) -> None:
    browser = MemoryBrowser(_tree(tmp_path))
    value = browser.read("memory://preferences/回答风格.md")
    assert isinstance(value, MemoryDocument)
    assert value.fields["content"] == "- 偏好简洁直接的回答"
    layer = browser.read(MemoryURI.from_layer(MemoryDirectory.preferences(), MemoryLevel.OVERVIEW))
    assert isinstance(layer, MemoryLayerText)
    assert layer.level is MemoryLevel.OVERVIEW and layer.text.startswith("## 偏好")


def test_missing_nodes_and_wrong_node_types_are_told_apart(tmp_path: Path) -> None:
    browser = MemoryBrowser(_tree(tmp_path))
    with pytest.raises(MemoryBrowseNotFoundError):
        browser.read("memory://preferences/没有这条.md")
    with pytest.raises(MemoryBrowseNotFoundError):
        browser.list("memory://entities/没有这类")
    with pytest.raises(MemoryBrowseNotFoundError):
        browser.read(MemoryURI.from_layer(MemoryDirectory.tools(), MemoryLevel.ABSTRACT))
    with pytest.raises(ValueError, match="directory URI"):
        browser.list("memory://profile.md")
    with pytest.raises(ValueError, match="document or semantic layer"):
        browser.read("memory://preferences")
    with pytest.raises(ValueError):
        browser.read("file:///etc/passwd")
