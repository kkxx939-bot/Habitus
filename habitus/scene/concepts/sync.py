"""同步词表（夜批 B1，裁定 20）：让基础概念跟上词表，并说出词表这一段时间里发生了什么。

纯规划，不落盘：输入是现有概念集、词表的全部类、上次同步之后的变更；输出是要写的概念（基础概念与改写了成员的汇总概念）
与迁移改了编号的 occurrence。落盘与重映射由夜批按这份计划做。

- **每个类恰好一个基础概念**：身份 = 类编号，显示名 = 类名，判据 = 类判据。没有的生成；类改名、改判据的跟着改
  （身份是编号，引用它的命中不动）；类停用的标成停用（不再命中）。这一条按全部类核对，
  不只看变更——所以第一次同步就是给全部类生成基础概念，中途丢了一次同步也会被补上。
- 拆分与合并只从变更里来：它们说的是"历史上的条目去了哪"，现状里看不出来；这里只拿它们改写汇总概念的成员。
- **汇总概念的成员跟着改写成现编号**：成员里有类拆出了新类，新类加进来；成员被并掉，换成并入的那个类。这样曲线（桥）
  与命中聚合、重言判断读的是同一份成员（第二轮评审 C2）。改写之后不足两个成员的，保持原样（并掉的那个类已经没有记录，
  不影响读数）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from habitus.scene.concepts.catalog import CatalogChanges, CatalogClass, MovedOccurrence
from habitus.scene.concepts.model import (
    ConceptDefinition,
    ConceptKind,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ConceptSource,
)


@dataclass(frozen=True)
class SyncPlan:
    """一次同步要做的事。"""

    version: int
    #: 新生成或要更新的概念（基础概念，以及成员改写了的汇总概念）。
    write: tuple[ConceptDefinition, ...] = ()
    #: 迁移改了编号的 occurrence：它们所在的日子要重映射。
    moved: tuple[MovedOccurrence, ...] = ()


def base_concept(item: CatalogClass, *, now: datetime) -> ConceptDefinition:
    """一个类的基础概念。"""

    return ConceptDefinition(
        name=item.id,
        definition=item.criterion,
        role=ConceptRole.BEHAVIOR,
        source=ConceptSource(ConceptOrigin.VOCABULARY),
        created_at=now,
        kind=ConceptKind.BASE,
        classes=(item.id,),
        lane=item.lane,
        title=item.name,
        retired=not item.active,
    )


def plan_sync(
    existing: ConceptSet, classes: Sequence[CatalogClass], changes: CatalogChanges, *, now: datetime
) -> SyncPlan:
    if not isinstance(existing, ConceptSet):
        raise TypeError("existing must be a ConceptSet")
    write = tuple(item for item in (_reconcile(existing, cls, now=now) for cls in classes) if item is not None)
    write += tuple(
        item
        for item in (_regroup(existing[identity], changes) for identity in existing.behaviors())
        if item is not None
    )
    return SyncPlan(version=changes.version, write=write, moved=changes.moved)


def _reconcile(existing: ConceptSet, item: CatalogClass, *, now: datetime) -> ConceptDefinition | None:
    """这个类的基础概念要不要写：没有就生成；名字、判据、停用与否变了就更新（保留出处与生成时刻）。"""

    identity = existing.base_for(item.id)
    if identity is None:
        if item.id in existing:
            raise ValueError(f"concept {item.id!r} is not the base concept of class {item.id!r}")
        return base_concept(item, now=now)
    current = existing[identity]
    if (current.title, current.definition, current.retired) == (item.name, item.criterion, not item.active):
        return None
    return replace(current, title=item.name, definition=item.criterion, retired=not item.active)


def _regroup(item: ConceptDefinition, changes: CatalogChanges) -> ConceptDefinition | None:
    """汇总概念的成员改写成现编号：拆出来的新类加进来，被并掉的换成并入的类；没变或改写后不足两个就不写。"""

    if item.kind is not ConceptKind.GROUP:
        return None
    members = list(item.classes)
    changed = True
    while changed:
        changed = False
        for new, source in changes.split_from.items():
            if source in members and new not in members:
                members.append(new)
                changed = True
        for old, target in changes.merged_into.items():
            if old in members:
                members = [target if member == old else member for member in members]
                changed = True
        members = list(dict.fromkeys(members))
    if tuple(members) == item.classes or len(members) < 2:
        return None
    return replace(item, classes=tuple(members))


__all__ = ["SyncPlan", "base_concept", "plan_sync"]
