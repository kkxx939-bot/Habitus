"""L0/L1 刷新器的目录、并发快照和资源边界矩阵。"""

from __future__ import annotations

from pathlib import Path

import pytest

from infrastructure.store.contracts.path_lock import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from memory.semantic import (
    MemorySemanticConfig,
    MemorySemanticRefresher,
    MemorySemanticRefreshError,
    MemorySemanticRefreshStatus,
)
from memory.tree import MemoryTree
from tests.helpers import document


class Generator:
    def __init__(self, result: object = "# 概览\n\n稳定摘要。\n\n## 条目\n- 内容") -> None:
        self.result = result
        self.snapshots = []

    def generate(self, snapshot):
        self.snapshots.append(snapshot)
        return self.result


def make_refresher(
    tmp_path: Path,
    *,
    generator: Generator | None = None,
    config: MemorySemanticConfig | None = None,
) -> MemorySemanticRefresher:
    return MemorySemanticRefresher(
        MemoryTree(tmp_path / "memory"),
        generator or Generator(),
        PathLock(ProcessLocalLockStore()),
        config=config,
    )


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("tree", object(), "tree must be"),
        ("generator", object(), "generator must implement"),
        ("path_lock", object(), "path_lock must be"),
    ],
)
def test_constructor_rejects_invalid_collaborator(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    arguments = {
        "tree": MemoryTree(tmp_path / "memory"),
        "generator": Generator(),
        "path_lock": PathLock(ProcessLocalLockStore()),
    }
    arguments[field] = invalid
    with pytest.raises(TypeError, match=message):
        MemorySemanticRefresher(**arguments)  # type: ignore[arg-type]


def test_refresh_for_rejects_non_address(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="address must be"):
        make_refresher(tmp_path).refresh_for("memory://profile.md")  # type: ignore[arg-type]


@pytest.mark.parametrize("addresses", [[], [MemoryAddress.profile()], ("bad",), (MemoryAddress.profile(), "bad")])
def test_refresh_for_many_requires_address_tuple(tmp_path: Path, addresses: object) -> None:
    with pytest.raises(TypeError, match="addresses must contain"):
        make_refresher(tmp_path).refresh_for_many(addresses)  # type: ignore[arg-type]


def test_refresh_for_many_empty_tuple_is_a_noop(tmp_path: Path) -> None:
    assert make_refresher(tmp_path).refresh_for_many(()) == ()


def test_refresh_for_walks_leaf_to_root_and_refreshes_each_directory_once(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    entity = document(MemoryKind.ENTITY)
    instance.tree.write(entity)
    results = instance.refresh_for(entity.address)
    assert tuple(result.directory for result in results) == (
        MemoryDirectory.entities("项目"),
        MemoryDirectory.entities(),
        MemoryDirectory.root(),
    )
    assert all(result.status is MemorySemanticRefreshStatus.WRITTEN for result in results)


def test_refresh_for_many_deduplicates_shared_ancestors_and_orders_deepest_first(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    preference = document(MemoryKind.PREFERENCE)
    entity = document(MemoryKind.ENTITY)
    instance.tree.write(preference)
    instance.tree.write(entity)
    results = instance.refresh_for_many((preference.address, entity.address, preference.address))
    directories = tuple(result.directory for result in results)
    assert directories.count(MemoryDirectory.root()) == 1
    assert directories.index(MemoryDirectory.entities("项目")) < directories.index(MemoryDirectory.entities())
    assert directories[-1] == MemoryDirectory.root()


def test_refresh_missing_directory_does_not_call_generator(tmp_path: Path) -> None:
    generator = Generator()
    result = make_refresher(tmp_path, generator=generator).refresh_directory(MemoryDirectory.preferences())
    assert result.status is MemorySemanticRefreshStatus.MISSING
    assert generator.snapshots == []


def test_refresh_directory_rejects_non_directory(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="directory must be"):
        make_refresher(tmp_path).refresh_directory(("preferences",))  # type: ignore[arg-type]


@pytest.mark.parametrize("result", [None, 1, "", " \n\t"])
def test_generator_must_return_non_empty_text(tmp_path: Path, result: object) -> None:
    instance = make_refresher(tmp_path, generator=Generator(result))
    instance.tree.write(document(MemoryKind.PREFERENCE))
    with pytest.raises(MemorySemanticRefreshError, match="empty text"):
        instance.refresh_directory(MemoryDirectory.preferences())


def test_generator_output_cannot_exceed_configured_overview_bound(tmp_path: Path) -> None:
    config = MemorySemanticConfig(max_overview_chars=20, max_abstract_chars=10)
    instance = make_refresher(tmp_path, generator=Generator("x" * 21), config=config)
    instance.tree.write(document(MemoryKind.PREFERENCE))
    with pytest.raises(MemorySemanticRefreshError, match="overview exceeds"):
        instance.refresh_directory(MemoryDirectory.preferences())


@pytest.mark.parametrize(
    ("overview", "expected"),
    [
        ("# 标题\n\n第一句。\n## 条目\n第二句", "第一句。\n"),
        ("\n# 标题\n正文一\n正文二", "正文一 正文二\n"),
        ("# 标题\n## 条目\n- 项目", "- 项目\n"),
        ("正文。\n### 明细\n忽略", "正文。\n"),
    ],
)
def test_abstract_is_deterministically_derived_from_overview(
    tmp_path: Path,
    overview: str,
    expected: str,
) -> None:
    instance = make_refresher(tmp_path)
    assert instance._abstract_from_overview(overview) == expected


@pytest.mark.parametrize("overview", ["", " \n", "# 标题\n## 条目"])
def test_overview_without_semantic_content_cannot_form_abstract(tmp_path: Path, overview: str) -> None:
    with pytest.raises(MemorySemanticRefreshError, match="non-empty abstract"):
        make_refresher(tmp_path)._abstract_from_overview(overview)


@pytest.mark.parametrize(
    ("maximum", "text", "expected"),
    [
        (3, "abcdef", "abc"),
        (6, "abcdef", "abcdef"),
        (10, "第一句话呢。第二句非常非常长", "第一句话呢。"),
        (8, "abcdefghijk", "abcde..."),
    ],
)
def test_abstract_truncation_preserves_sentence_when_possible(
    tmp_path: Path,
    maximum: int,
    text: str,
    expected: str,
) -> None:
    config = MemorySemanticConfig(max_overview_chars=100, max_abstract_chars=maximum)
    assert make_refresher(tmp_path, config=config)._truncate_abstract(text) == expected


def test_snapshot_contains_direct_memory_and_child_abstract(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    entity = document(MemoryKind.ENTITY)
    instance.tree.write(entity)
    child = MemoryDirectory.entities("项目")
    instance.tree.write_layers(child, abstract="项目摘要", overview="项目完整概览")
    snapshot = instance._snapshot(MemoryDirectory.entities())
    assert tuple((entry.kind.value, entry.name, entry.content) for entry in snapshot.entries) == (
        ("directory", "项目", "项目摘要"),
    )


def test_child_overview_is_used_when_abstract_is_absent(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    entity = document(MemoryKind.ENTITY)
    instance.tree.write(entity)
    child = MemoryDirectory.entities("项目")
    overview_path = instance.tree.layer_path(child, MemoryLevel.OVERVIEW)
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    overview_path.write_text("# 标题\n\n项目概要。\n", encoding="utf-8")
    snapshot = instance._snapshot(MemoryDirectory.entities())
    assert snapshot.entries[0].content == "项目概要。\n"


def test_child_without_semantic_layers_is_not_in_parent_snapshot(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    instance.tree.write(document(MemoryKind.ENTITY))
    assert instance._snapshot(MemoryDirectory.entities()).entries == ()


@pytest.mark.parametrize("level", [MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW])
def test_oversized_child_semantic_layer_fails_closed(tmp_path: Path, level: MemoryLevel) -> None:
    config = MemorySemanticConfig(max_overview_chars=20, max_abstract_chars=10)
    instance = make_refresher(tmp_path, config=config)
    instance.tree.write(document(MemoryKind.ENTITY))
    child = MemoryDirectory.entities("项目")
    path = instance.tree.layer_path(child, level)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * 22, encoding="utf-8")
    message = "abstract exceeds" if level is MemoryLevel.ABSTRACT else "overview exceeds"
    with pytest.raises(MemorySemanticRefreshError, match=message):
        instance._snapshot(MemoryDirectory.entities())


def test_empty_directory_without_layers_is_unchanged(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    instance.tree.initialize()
    result = instance.refresh_directory(MemoryDirectory.preferences())
    assert result.status is MemorySemanticRefreshStatus.UNCHANGED


def test_rebuild_rejects_invalid_root_and_returns_empty_for_missing_subtree(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    with pytest.raises(TypeError, match="directory must be"):
        instance.rebuild("preferences")  # type: ignore[arg-type]
    assert instance.rebuild(MemoryDirectory.entities("不存在")) == ()


def test_rebuild_enforces_directory_capacity(tmp_path: Path) -> None:
    config = MemorySemanticConfig(max_rebuild_directories=1)
    instance = make_refresher(tmp_path, config=config)
    instance.tree.write(document(MemoryKind.ENTITY))
    with pytest.raises(MemorySemanticRefreshError, match="directory bound"):
        instance.rebuild()


def test_rebuild_refreshes_deepest_directories_before_root(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    instance.tree.write(document(MemoryKind.ENTITY))
    results = instance.rebuild()
    directories = tuple(result.directory for result in results)
    assert directories.index(MemoryDirectory.entities("项目")) < directories.index(MemoryDirectory.entities())
    assert directories.index(MemoryDirectory.entities()) < directories.index(MemoryDirectory.root())
    assert directories[-1] == MemoryDirectory.root()


def test_lock_key_is_stable_per_tree_and_distinct_per_directory(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    root_key = instance._lock_key(MemoryDirectory.root())
    preference_key = instance._lock_key(MemoryDirectory.preferences())
    assert root_key.endswith(":/")
    assert preference_key.endswith(":preferences")
    assert root_key != preference_key
