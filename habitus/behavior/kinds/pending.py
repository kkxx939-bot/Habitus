"""待定池：白天归不进任何在用类的 occurrence，等每晚新增时交给模型聚组。

按 occurrence 地址幂等——同一条重复放入只留一份（崩溃重放、重复发布都不会多算复现次数）。

**每晚只交新条目**（裁定 60 第 2 条）：没聚组过的条目，加上按字符二元组相似召回的几条旧条目（``nightly.recall``），
一晚有上限、多的留到下一晚。旧条目不清，当索引留着：两件不同月份、措辞相近的同一件事靠召回凑成组。原来每晚整池进提示词，
一年几百条一次性的事每晚都要让模型再看一遍。树上已经没有的条目在重判时出池；池满时挤掉进池最早的旧条目（只是索引，树上不动），留信号、进容量快照。

TODO（生命周期，留给以后）：词表类的退休（``model.ClassStatus.RETIRED`` 目前只由拆改的合并产生）与语义树情境概念的退休
用同一套机制，见 ``~/Desktop/Habitus实验数据/事件融合与词表重做/待办.md``。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType

from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.model import BehaviorKindError


@dataclass(frozen=True)
class PendingEntry:
    """一条待定：哪条 occurrence、哪条 lane、哪天、模型提议的名字、给聚组看的内容。

    ``rechecked``：每晚新增认作"已有某类"时，已经交白天归类重判过一次、仍判「都不是」。只问一次——
    之后照常参与长新类，不再重判（反复重问会把它磨进一个最像的类，2026-10-09 54 天对照）。

    ``entered_on``：哪天进的池；``grouped_on``：哪一晚交模型聚组过（裁定 60 第 2 条）。没聚组过的是"新条目"，每晚只把
    新条目与按相似召回的几条旧条目交给模型；旧条目不清，当索引留着。模型输出出错那晚交过去的不算聚组过，第二晚仍是新的。
    """

    occurrence: str
    lane: Lane
    day: date
    proposed: str
    content: str
    rechecked: bool = False
    entered_on: date | None = None
    grouped_on: date | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane", Lane(self.lane))
        if isinstance(self.day, bool) or not isinstance(self.day, date):
            raise BehaviorKindError("pending day must be a date")
        for name in ("occurrence", "proposed", "content"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise BehaviorKindError(f"pending {name} must be non-empty text")
        if not isinstance(self.rechecked, bool):
            raise BehaviorKindError("pending rechecked must be a bool")
        for name in ("entered_on", "grouped_on"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, date)):
                raise BehaviorKindError(f"pending {name} must be a date or None")

    @property
    def fresh(self) -> bool:
        """还没交模型聚组过。"""

        return self.grouped_on is None


@dataclass(frozen=True)
class PendingPool:
    entries: Mapping[str, PendingEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, entry in self.entries.items():
            if not isinstance(entry, PendingEntry) or entry.occurrence != key:
                raise BehaviorKindError("pending entries must be keyed by their occurrence")
        object.__setattr__(self, "entries", MappingProxyType(dict(sorted(self.entries.items()))))

    def lane(self, lane: Lane) -> tuple[PendingEntry, ...]:
        resolved = Lane(lane)
        return tuple(entry for entry in self.entries.values() if entry.lane is resolved)

    def with_entries(self, entries: Iterable[PendingEntry]) -> PendingPool:
        merged = dict(self.entries)
        for entry in entries:
            merged.setdefault(entry.occurrence, entry)
        return PendingPool(merged)

    def rechecked(self, occurrences: Iterable[str]) -> PendingPool:
        """把这些条目标成"已重判"；不在池里的忽略（可能已被迁走）。"""

        marked = set(occurrences)
        return PendingPool(
            {key: replace(value, rechecked=True) if key in marked else value for key, value in self.entries.items()}
        )

    def grouped(self, occurrences: Iterable[str], *, on: date) -> PendingPool:
        """把这些条目记成 ``on`` 那一晚交模型聚组过；不在池里的忽略。"""

        marked = set(occurrences)
        return PendingPool(
            {key: replace(value, grouped_on=on) if key in marked else value for key, value in self.entries.items()}
        )

    def without(self, occurrences: Iterable[str]) -> PendingPool:
        drop = set(occurrences)
        return PendingPool({key: value for key, value in self.entries.items() if key not in drop})


__all__ = ["PendingEntry", "PendingPool"]
