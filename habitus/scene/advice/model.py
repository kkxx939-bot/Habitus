"""大模型给关系检验的建议的值：先验的三档与权重。纯值，不调模型。

大模型只提议，统计握否决权：先验只调第 1 道的权重（有界、归一到平均 1、冻结后不随结果重估）。
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class PriorLevel(str, Enum):
    """A 之后 B 更容易或更难发生（不论方向）这件事，按常识有多可能。"""

    LIKELY = "likely"
    UNSURE = "unsure"
    UNLIKELY = "unlikely"

    @property
    def weight(self) -> float:
        """三档权重 2 / 1 / 0.5（检验族里再归一到平均 1）。"""

        return {PriorLevel.LIKELY: 2.0, PriorLevel.UNSURE: 1.0, PriorLevel.UNLIKELY: 0.5}[self]

    @property
    def rank(self) -> int:
        return {PriorLevel.LIKELY: 2, PriorLevel.UNSURE: 1, PriorLevel.UNLIKELY: 0}[self]


def settle(answers: Sequence[PriorLevel]) -> PriorLevel:
    """乱序问几遍取中位；最高与最低差了两档（一遍说很可能、一遍说不太可能）的给中性——模型自己都拿不准。"""

    if not answers:
        raise ValueError("a prior needs at least one answer")
    ranks = [answer.rank for answer in answers]
    if max(ranks) - min(ranks) >= 2:
        return PriorLevel.UNSURE
    middle = round(statistics.median(ranks))
    return next(level for level in PriorLevel if level.rank == middle)


@dataclass(frozen=True)
class PriorTable:
    """一条 lane 的先验：（前因, 后果）→ 档。没问到的对不在里面（权重按 1）。"""

    lane: str
    levels: Mapping[tuple[str, str], PriorLevel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", MappingProxyType(dict(self.levels)))

    def weight(self, antecedent: str, consequent: str) -> float:
        level = self.levels.get((antecedent, consequent))
        return 1.0 if level is None else level.weight


__all__ = ["PriorLevel", "PriorTable", "settle"]
