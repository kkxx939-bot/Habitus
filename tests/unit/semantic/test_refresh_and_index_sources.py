"""L0/L1 可重建语义层和 MemoryTree 索引源投影测试。"""

from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts.path_lock import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.memory.indexing import MemoryIndexSourceReader, MemoryVectorIndexConfig
from habitus.memory.model import MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.semantic import (
    MemorySemanticConfig,
    MemorySemanticRefresher,
    MemorySemanticRefreshError,
    MemorySemanticRefreshStatus,
)
from habitus.memory.tree import MemoryTree
from tests.helpers import document


class OverviewGenerator:
    def __init__(self, tree: MemoryTree | None = None, mutate: bool = False) -> None:
        self.calls = []
        self.tree = tree
        self.mutate = mutate

    def generate(self, snapshot):
        self.calls.append(snapshot)
        if self.mutate:
            self.tree.write(
                document(
                    MemoryKind.PREFERENCE,
                    fields={"topic": "新增偏好", "content": "- 生成期间发生变化"},
                )
            )
        return "# 目录概览\n\n该目录保存用户已经确认的长期信息。\n\n## 条目\n\n- 内容"


def refresher(tree: MemoryTree, generator: OverviewGenerator, **config: object) -> MemorySemanticRefresher:
    return MemorySemanticRefresher(
        tree,
        generator,
        PathLock(ProcessLocalLockStore()),
        config=MemorySemanticConfig(**config),
    )


def test_refresh_builds_l1_then_deterministically_derives_l0_and_is_idempotent(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    preference = document(MemoryKind.PREFERENCE)
    tree.write(preference)
    generator = OverviewGenerator()
    service = refresher(tree, generator)

    first = service.refresh_directory(MemoryDirectory.preferences())
    second = service.refresh_directory(MemoryDirectory.preferences())
    assert first.status is MemorySemanticRefreshStatus.WRITTEN
    assert second.status is MemorySemanticRefreshStatus.UNCHANGED
    assert tree.read_layer(MemoryDirectory.preferences(), MemoryLevel.OVERVIEW).startswith("# 目录概览")
    assert tree.read_layer(MemoryDirectory.preferences(), MemoryLevel.ABSTRACT) == (
        "该目录保存用户已经确认的长期信息。\n"
    )


def test_empty_directory_deletes_stale_layers_without_invoking_model(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.initialize()
    directory = MemoryDirectory.preferences()
    tree.write_layers(directory, abstract="旧摘要", overview="旧概览")
    generator = OverviewGenerator()
    result = refresher(tree, generator).refresh_directory(directory)
    assert result.status is MemorySemanticRefreshStatus.DELETED
    assert generator.calls == []
    assert not tree.layer_exists(directory, MemoryLevel.ABSTRACT)


def test_changed_snapshot_is_never_published_and_exhausts_bounded_retry(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    tree.write(document(MemoryKind.PREFERENCE))
    generator = OverviewGenerator(tree, mutate=True)
    with pytest.raises(MemorySemanticRefreshError, match="changed repeatedly"):
        refresher(tree, generator, stale_retries=0).refresh_directory(MemoryDirectory.preferences())
    assert not tree.layer_exists(MemoryDirectory.preferences(), MemoryLevel.OVERVIEW)


def test_index_source_walk_contains_l2_and_available_l0_l1_but_no_missing_layers(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory")
    preference = document(MemoryKind.PREFERENCE)
    tree.write(preference)
    tree.write_layers(MemoryDirectory.preferences(), abstract="偏好摘要", overview="偏好完整概览")
    sources = MemoryIndexSourceReader(tree, config=MemoryVectorIndexConfig()).walk()
    levels = {source.level for source in sources}
    assert levels == {MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW, MemoryLevel.DETAIL}
    detail = next(source for source in sources if source.level is MemoryLevel.DETAIL)
    assert detail.index_kind == "preference"
    assert detail.revision == preference.metadata.revision
    assert detail.scope_roots[0] == "memory://"

