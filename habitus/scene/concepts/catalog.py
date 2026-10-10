"""词表注入口：语义树从基础词表要的几样东西（裁定 20）。

scene 不 import 词表包（``behavior/kinds``）；组合根用词表的只读口装出一个 ``ClassCatalog`` 交进来，
与节律口同一种做法。这里只声明形状：

- 全部类（含停用的）：编号、类名、判据、lane、在不在用；
- 当前版本号；
- 变更流：某个版本之后哪些类是从哪个类拆出来的、哪些被并进了哪个类，迁移改了哪些 occurrence 的编号
  （改名、改判据不用变更流：同步时按全部类逐个核对）。

occurrence 上的 ``kind_token`` 怎么读（类编号 /「待定」/「非事件」、哪条 lane）不在这里：那是事件序列的读法，
只写在 ``series.reader``，预测树与语义树共用。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CatalogClass:
    """词表里的一个行为类。"""

    id: str
    name: str
    criterion: str
    lane: str
    active: bool

    def __post_init__(self) -> None:
        for label in ("id", "name", "criterion", "lane"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"catalog class {label} must be non-empty text")
        if not isinstance(self.active, bool):
            raise ValueError("catalog class active must be a boolean")


@dataclass(frozen=True)
class MovedOccurrence:
    """迁移改写的一条 occurrence：从哪个编号改到哪个编号。"""

    uri: str
    source: str
    target: str


@dataclass(frozen=True)
class CatalogChanges:
    """``since`` 之后到 ``version`` 为止词表的变化。"""

    since: int
    version: int
    #: 接收条目的类 → 交出条目的类（源类没停用）：通常是新拆出来的类，也可能是把一部分条目交给了已有的类。
    split_from: Mapping[str, str] = field(default_factory=dict)
    #: 停用的类 → 它并进的类。
    merged_into: Mapping[str, str] = field(default_factory=dict)
    moved: tuple[MovedOccurrence, ...] = ()

    def __post_init__(self) -> None:
        if self.version < self.since:
            raise ValueError("catalog changes cannot end before they start")
        object.__setattr__(self, "split_from", MappingProxyType(dict(self.split_from)))
        object.__setattr__(self, "merged_into", MappingProxyType(dict(self.merged_into)))


@runtime_checkable
class ClassCatalog(Protocol):
    def classes(self) -> tuple[CatalogClass, ...]: ...

    def version(self) -> int: ...

    def changes_since(self, version: int) -> CatalogChanges: ...


__all__ = ["CatalogChanges", "CatalogClass", "ClassCatalog", "MovedOccurrence"]
