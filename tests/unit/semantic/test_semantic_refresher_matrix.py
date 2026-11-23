"""L0/L1 模板刷新器的目录顺序、模板内容和资源边界矩阵。"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.memory.model import MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.semantic import (
    MemoryOverview,
    MemoryOverviewFormatError,
    MemorySemanticConfig,
    MemorySemanticRefresher,
    MemorySemanticRefreshError,
    MemorySemanticRefreshStatus,
)
from habitus.memory.tree import MemoryTree
from tests.helpers import BASE_TIME, document, registry
from tests.lock_store import ProcessLocalLockStore


def make_refresher(
    tmp_path: Path,
    *,
    config: MemorySemanticConfig | None = None,
) -> MemorySemanticRefresher:
    return MemorySemanticRefresher(
        MemoryTree(tmp_path / "memory"),
        registry(),
        PathLock(ProcessLocalLockStore()),
        config=config,
    )


def preference(topic: str, *, minutes: int = 0):
    return document(
        MemoryKind.PREFERENCE,
        fields={"topic": topic, "content": f"- 关于{topic}的偏好"},
        timestamp=BASE_TIME + timedelta(minutes=minutes),
    )


def read_layer(instance: MemorySemanticRefresher, directory: MemoryDirectory, level: MemoryLevel) -> str:
    return instance.tree.layer_path(directory, level).read_text(encoding="utf-8")


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
    first = document(MemoryKind.PREFERENCE)
    entity = document(MemoryKind.ENTITY)
    instance.tree.write(first)
    instance.tree.write(entity)
    results = instance.refresh_for_many((first.address, entity.address, first.address))
    directories = tuple(result.directory for result in results)
    assert directories.count(MemoryDirectory.root()) == 1
    assert directories.index(MemoryDirectory.entities("项目")) < directories.index(MemoryDirectory.entities())
    assert directories[-1] == MemoryDirectory.root()


def test_refresh_missing_directory_reports_missing(tmp_path: Path) -> None:
    result = make_refresher(tmp_path).refresh_directory(MemoryDirectory.preferences())
    assert result.status is MemorySemanticRefreshStatus.MISSING


def test_overview_lists_cards_by_template_and_abstract_counts_items(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    card = preference("回答风格")
    instance.tree.write(card)
    instance.refresh_directory(MemoryDirectory.preferences())

    overview_text = read_layer(instance, MemoryDirectory.preferences(), MemoryLevel.OVERVIEW)
    abstract_text = read_layer(instance, MemoryDirectory.preferences(), MemoryLevel.ABSTRACT)
    overview = MemoryOverview.parse(MemoryDirectory.preferences(), overview_text)
    (line,) = overview.cards
    assert line.filename == instance.tree.path_for(card.address).name
    assert line.description == registry().get(MemoryKind.PREFERENCE).overview_line(card.fields)
    assert line.description
    assert (line.created_at, line.updated_at, line.revision) == (
        card.metadata.created_at,
        card.metadata.updated_at,
        card.metadata.revision,
    )
    assert overview.render() == overview_text
    assert abstract_text == "preferences：1 项\n"


def test_refresh_is_deterministic_and_unchanged_when_nothing_changed(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    instance.tree.write(preference("回答风格"))
    first = instance.refresh_directory(MemoryDirectory.preferences())
    second = instance.refresh_directory(MemoryDirectory.preferences())
    assert first.status is MemorySemanticRefreshStatus.WRITTEN
    assert second.status is MemorySemanticRefreshStatus.UNCHANGED


def test_parent_overview_lists_child_directories_with_item_counts(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    instance.tree.write(document(MemoryKind.ENTITY))
    instance.refresh_for(document(MemoryKind.ENTITY).address)
    overview = MemoryOverview.parse(
        MemoryDirectory.entities(),
        read_layer(instance, MemoryDirectory.entities(), MemoryLevel.OVERVIEW),
    )
    assert overview.cards == ()
    assert tuple((child.name, child.items) for child in overview.children) == (("项目", 1),)
    assert read_layer(instance, MemoryDirectory.root(), MemoryLevel.ABSTRACT) == "记忆根：3 项\n"


def test_cards_are_ordered_newest_first(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    older = preference("旧主题", minutes=0)
    newer = preference("新主题", minutes=5)
    instance.tree.write(older)
    instance.tree.write(newer)
    instance.refresh_directory(MemoryDirectory.preferences())
    overview = MemoryOverview.parse(
        MemoryDirectory.preferences(),
        read_layer(instance, MemoryDirectory.preferences(), MemoryLevel.OVERVIEW),
    )
    assert tuple(card.name for card in overview.cards) == ("新主题", "旧主题")


def test_direct_entry_limit_keeps_newest_cards_and_counts_omitted(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path, config=MemorySemanticConfig(max_direct_entries=2))
    for index in range(3):
        instance.tree.write(preference(f"主题{index}", minutes=index))
    instance.refresh_directory(MemoryDirectory.preferences())
    overview = MemoryOverview.parse(
        MemoryDirectory.preferences(),
        read_layer(instance, MemoryDirectory.preferences(), MemoryLevel.OVERVIEW),
    )
    assert tuple(card.name for card in overview.cards) == ("主题2", "主题1")
    assert overview.omitted_cards == 1
    assert read_layer(instance, MemoryDirectory.preferences(), MemoryLevel.ABSTRACT) == "preferences：3 项\n"


def test_overview_character_bound_drops_oldest_cards_first(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    for index in range(2):
        instance.tree.write(preference(f"主题{index}", minutes=index))
    full = instance._overview(MemoryDirectory.preferences())
    bound = len(MemoryOverview(full.directory, full.cards[:1], full.children, 1).render())
    limited = make_refresher(tmp_path, config=MemorySemanticConfig(max_overview_chars=bound, max_abstract_chars=10))
    overview = limited._overview(MemoryDirectory.preferences())
    assert tuple(card.name for card in overview.cards) == ("主题1",)
    assert overview.omitted_cards == 1


def test_overview_that_cannot_fit_even_without_cards_fails_closed(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path, config=MemorySemanticConfig(max_overview_chars=10, max_abstract_chars=10))
    instance.tree.write(preference("回答风格"))
    with pytest.raises(MemorySemanticRefreshError, match="character bound"):
        instance.refresh_directory(MemoryDirectory.preferences())


def test_emptied_directory_deletes_its_layers(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    card = preference("回答风格")
    instance.tree.write(card)
    instance.refresh_directory(MemoryDirectory.preferences())
    instance.tree.delete(card.address)
    result = instance.refresh_directory(MemoryDirectory.preferences())
    assert result.status is MemorySemanticRefreshStatus.DELETED
    assert not instance.tree.layer_exists(MemoryDirectory.preferences(), MemoryLevel.OVERVIEW)


def test_empty_directory_without_layers_is_unchanged(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    instance.tree.initialize()
    result = instance.refresh_directory(MemoryDirectory.preferences())
    assert result.status is MemorySemanticRefreshStatus.UNCHANGED


@pytest.mark.parametrize(
    "text",
    [
        "# 概览\n\n稳定摘要。\n",
        "# preferences\n<!-- 每行：文件名、创建时间、更新时间、版本、说明；由模板生成，可删除重建 -->\n\n- 坏行\n",
        "# entities\n<!-- 每行：文件名、创建时间、更新时间、版本、说明；由模板生成，可删除重建 -->\n",
    ],
)
def test_overview_parse_rejects_text_that_is_not_this_template(text: str) -> None:
    with pytest.raises(MemoryOverviewFormatError):
        MemoryOverview.parse(MemoryDirectory.preferences(), text)


def test_rebuild_returns_empty_for_missing_subtree(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
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


def test_rebuild_replaces_a_non_template_overview(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    instance.tree.write(preference("回答风格"))
    instance.tree.write_layers(MemoryDirectory.preferences(), abstract="旧简介", overview="# 旧的模型概览\n")
    instance.rebuild(MemoryDirectory.preferences())
    MemoryOverview.parse(
        MemoryDirectory.preferences(),
        read_layer(instance, MemoryDirectory.preferences(), MemoryLevel.OVERVIEW),
    )


def test_lock_key_is_stable_per_tree_and_distinct_per_directory(tmp_path: Path) -> None:
    instance = make_refresher(tmp_path)
    root_key = instance._lock_key(MemoryDirectory.root())
    preference_key = instance._lock_key(MemoryDirectory.preferences())
    assert root_key.endswith(":/")
    assert preference_key.endswith(":preferences")
    assert root_key != preference_key
