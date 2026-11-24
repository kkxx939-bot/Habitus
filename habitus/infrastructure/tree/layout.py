"""一棵 habitus 树的顶层布局：树根在存储根下的位置，以及根下允许出现的条目。

只列名字，不认识名字背后的领域类型：四类卡片（``profile.md``、``preferences``、``entities``、``tools``）由记忆读写，
事件与空白（``occurrences``、``gaps``，下面再按支分 ``session`` / ``physical``）由行为读写。两边各自校验自己那部分的结构；
根目录只许出现这里列出的条目，多出来的就是旧盘面或手放的东西，读写时明确报错。
"""

from __future__ import annotations

from pathlib import Path

#: 树根是存储根下的这个子目录（不可配置）。任务、冷热、语义树、预测的工作目录都在树外。
TREE_DIRECTORY = "habitus"

PROFILE_FILE = "profile.md"
PREFERENCES_DIRECTORY = "preferences"
ENTITIES_DIRECTORY = "entities"
TOOLS_DIRECTORY = "tools"
OCCURRENCES_DIRECTORY = "occurrences"
GAPS_DIRECTORY = "gaps"

#: 目录 L0 / L1 的侧车文件名；只有卡片目录生成它们，事件与空白的目录不生成（裁定 112）。
ABSTRACT_FILENAME = ".abstract.md"
OVERVIEW_FILENAME = ".overview.md"
SIDECAR_FILENAMES = frozenset({ABSTRACT_FILENAME, OVERVIEW_FILENAME})

ROOT_FILES = frozenset({PROFILE_FILE})
ROOT_DIRECTORIES = frozenset(
    {PREFERENCES_DIRECTORY, ENTITIES_DIRECTORY, TOOLS_DIRECTORY, OCCURRENCES_DIRECTORY, GAPS_DIRECTORY}
)


def tree_root(storage_root: Path) -> Path:
    """存储根下的树根路径。"""

    return storage_root / TREE_DIRECTORY


__all__ = [
    "ABSTRACT_FILENAME",
    "ENTITIES_DIRECTORY",
    "GAPS_DIRECTORY",
    "OCCURRENCES_DIRECTORY",
    "OVERVIEW_FILENAME",
    "PREFERENCES_DIRECTORY",
    "PROFILE_FILE",
    "ROOT_DIRECTORIES",
    "ROOT_FILES",
    "SIDECAR_FILENAMES",
    "TOOLS_DIRECTORY",
    "TREE_DIRECTORY",
    "tree_root",
]
