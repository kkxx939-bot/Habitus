"""词表注入口：语义树从基础词表要的东西（裁定 20；词表方案 2026-11-14）。

scene 不 import 词表包（``behavior/kinds``）；组合根用词表的在用状态装出一个 ``ClassCatalog`` 交进来。这里只声明形状：
在用的词（编号、名字、定义、lane）与在用状态的修订号（下游记的"词表版本"）。不在用的词没有概念，不交。
词表没有拆分、合并、迁移，所以没有变更流：在用集合只在换代里变，换代把这一代数出来的集合显式交给语义树。

occurrence 上的 ``kind_token`` 怎么读（词编号 / 占位标记、哪条 lane）不在这里：那是事件序列的读法，
只写在 ``series.reader``，语义树与 ``temporal`` 共用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CatalogWord:
    """一个在用的词。"""

    id: str
    name: str
    definition: str
    lane: str

    def __post_init__(self) -> None:
        for label in ("id", "name", "definition", "lane"):
            value = getattr(self, label)
            if not value.strip():
                raise ValueError(f"catalog word {label} must be non-empty text")


@dataclass(frozen=True)
class CatalogWords:
    """某一版在用的词：``version`` 是在用状态的修订号。"""

    version: int
    words: tuple[CatalogWord, ...]

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("catalog version must be a non-negative integer")
        if len({word.id for word in self.words}) != len(self.words):
            raise ValueError("catalog words must not repeat")


@runtime_checkable
class ClassCatalog(Protocol):
    def current(self) -> CatalogWords:
        """盘上此刻的在用状态（只在换代里变）。"""
        ...


__all__ = ["CatalogWord", "CatalogWords", "ClassCatalog"]
