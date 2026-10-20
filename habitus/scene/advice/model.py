"""大模型给关系检验的两样建议（语义树新方案 ``13`` ②）的值：先验的三档、权重、调节条件提议的去留。纯值，不调模型。

**大模型只提议，统计握否决权**：先验只调第 1 道的权重（有界、归一到平均 1、冻结后不随结果重估），候选条件要编译成三种
可算的模板、再在提出之后的新数据上过同一套把关。
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from types import MappingProxyType

from habitus.foundation.integrity import canonical_json


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


@dataclass(frozen=True)
class Proposal:
    """大模型提的一条调节条件：编译成功的记模板文字，失败的记原话与原因（已拒清单）。"""

    antecedent: str
    proposed_on: date
    why: str
    condition: str | None = None
    raw: Mapping[str, object] = field(default_factory=dict)
    refused: str | None = None
    #: 出处：提示词与 schema 的版本、实际作答的模型（第四轮评审 E15）。
    version: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        if (self.condition is None) == (self.refused is None):
            raise ValueError("a proposal is either compiled into a condition or refused, not both")
        object.__setattr__(self, "raw", MappingProxyType(dict(self.raw)))


@dataclass(frozen=True)
class ProposalBook:
    """一条 lane 的提议簿：收下的、拒掉的、每个前因上次问模型时概念集的指纹（概念集没变就不再问）。"""

    lane: str
    proposals: tuple[Proposal, ...] = ()
    asked: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "asked", MappingProxyType(dict(self.asked)))

    def accepted(self, antecedent: str) -> tuple[Proposal, ...]:
        return tuple(item for item in self.proposals if item.antecedent == antecedent and item.condition is not None)

    def refused(self, antecedent: str) -> tuple[Proposal, ...]:
        return tuple(item for item in self.proposals if item.antecedent == antecedent and item.refused is not None)

    def known(self, antecedent: str) -> frozenset[str]:
        """这个前因已经收下的条件文字（同一个条件不重复收）。"""

        return frozenset(item.condition for item in self.accepted(antecedent) if item.condition is not None)


def tidy_book(
    book: ProposalBook,
    *,
    alive: Callable[[str], bool],
    compiles: Callable[[str], bool],
    max_refused_per_antecedent: int,
) -> ProposalBook:
    """提议簿的退出（26 号清单 2.5.5）：前因已经不在的提议与问过的指纹清掉；收下的条件现在编译不了的清掉（以前每晚跳过、
    每晚报一句）；拒掉的按（前因, 拒因, 原话）去重、每个前因只留最近的 ``max_refused_per_antecedent`` 条——它们会全量进提示词。"""

    if max_refused_per_antecedent < 1:
        raise ValueError("max_refused_per_antecedent must be positive")
    kept: list[Proposal] = []
    seen_refusals: set[tuple[str, str, str]] = set()
    for item in sorted(book.proposals, key=lambda proposal: proposal.proposed_on, reverse=True):
        if not alive(item.antecedent):
            continue
        if item.condition is not None:
            if compiles(item.condition):
                kept.append(item)
            continue
        identity = (item.antecedent, str(item.refused), canonical_json(dict(item.raw)))
        if identity in seen_refusals:
            continue
        if sum(1 for other in kept if other.antecedent == item.antecedent and other.refused is not None) >= (
            max_refused_per_antecedent
        ):
            continue
        seen_refusals.add(identity)
        kept.append(item)
    order = {id(item): position for position, item in enumerate(book.proposals)}
    proposals = tuple(sorted(kept, key=lambda item: order[id(item)]))
    asked = {name: digest for name, digest in book.asked.items() if alive(name)}
    return ProposalBook(lane=book.lane, proposals=proposals, asked=asked)


__all__ = ["PriorLevel", "PriorTable", "Proposal", "ProposalBook", "settle", "tidy_book"]
