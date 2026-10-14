"""待定池：白天归不进任何在用类的 occurrence，等每晚新增时整体交给模型聚组。

按 occurrence 地址幂等——同一条重复放入只留一份（崩溃重放、重复发布都不会多算复现次数）。

TODO（生命周期，用户 2026-10-12 裁定"整个都没有生命周期，记下"；各层一起设计、不单独做，见
``~/Desktop/Habitus实验数据/事件融合与词表重做/待办.md``）：池里从未转正的条目永远留着，每晚整池进提示词，
到 ``max_pending`` 才拒收。造数实验（人物 A 一年、真实模型 35 晚）池子只有 6–13 条，是因为粒度规则把 707 条一次性杂事
里的 57 条并进了类；规则改成"对象就是行为的留对象"之后，"订去杭州的机票""问孩子不肯写作业"这类一次性的事会老实留在
池里，一年几百条，每晚都要让模型再看一遍。要定的：待定条目多久没凑成类就出池（按 ``recurrence_min_days`` 的倍数）、
出池后要不要留痕给预测树占位、与词表类的退休（``model.ClassStatus.RETIRED`` 目前只由拆改的合并产生）、语义树概念的
``retired``（目前从不置真）用同一套机制。
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
    """

    occurrence: str
    lane: Lane
    day: date
    proposed: str
    content: str
    rechecked: bool = False

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

    def without(self, occurrences: Iterable[str]) -> PendingPool:
        drop = set(occurrences)
        return PendingPool({key: value for key, value in self.entries.items() if key not in drop})


__all__ = ["PendingEntry", "PendingPool"]
