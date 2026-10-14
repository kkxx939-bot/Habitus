"""把模型提的差分解析成三种提议：合并、拆分、改判据。形状不合的整条丢掉、留信号，不重问。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from habitus.behavior.kinds.ids import ClassId
from habitus.behavior.kinds.model import BehaviorClass, BehaviorKindError, ClassOrigin, Exclusion
from habitus.behavior.kinds.revision.prompt import NEW


@dataclass(frozen=True)
class Draft:
    """一份还没分配编号的条目；``excludes`` 的去向可以是 ``NEW``（拆分时指向拆出的新类）。"""

    name: str
    criterion: str
    reminder: str
    excludes: tuple[tuple[str, ClassId | str | None], ...]
    examples: tuple[str, ...]

    def __post_init__(self) -> None:
        texts = (self.name, self.criterion, self.reminder, *(text for text, _ in self.excludes), *self.examples)
        if any(not text for text in texts):
            raise BehaviorKindError("draft text must not be empty")
        if len(set(self.examples)) != len(self.examples):
            raise BehaviorKindError("draft examples repeat")

    def to_class(self, class_id: ClassId, *, origin: ClassOrigin, new_id: ClassId | None = None) -> BehaviorClass:
        excludes = tuple(
            Exclusion(text, new_id if target == NEW else target if isinstance(target, ClassId) else None)
            for text, target in self.excludes
        )
        return BehaviorClass(
            id=class_id,
            name=self.name,
            criterion=self.criterion,
            reminder=self.reminder,
            excludes=excludes,
            examples=self.examples,
            origin=origin,
        )


@dataclass(frozen=True)
class MergeProposal:
    source: ClassId
    target: ClassId
    reason: str

    @property
    def touched(self) -> frozenset[ClassId]:
        return frozenset({self.source, self.target})


@dataclass(frozen=True)
class SplitProposal:
    """拆分不由模型提（v3）：``reviser`` 按分组 + 复现造出来——``new`` 是一次拆出的全部新类（可以几个），
    ``narrowed`` 是收窄后的原类条目。"""

    source: ClassId
    narrowed: Draft
    new: tuple[Draft, ...]
    reason: str
    #: 分组说"这一组就是词表里已有的某一类"：成员搬过去，不另起新类（第 35 晚重放：查看信息里 14 条其实是「询问知识性问题」，
    #: 另起同名新类撞名、整处拆分作废）
    into: tuple[ClassId, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.new, tuple) or not all(isinstance(item, Draft) for item in self.new):
            raise BehaviorKindError("split drafts must be a tuple of Draft")
        if not isinstance(self.into, tuple) or not all(isinstance(item, ClassId) for item in self.into):
            raise BehaviorKindError("split targets must be a tuple of ClassId")
        if not self.new and not self.into:
            raise BehaviorKindError("a split names at least one new class draft or an existing class to move into")

    @property
    def touched(self) -> frozenset[ClassId]:
        return frozenset({self.source, *self.into})


@dataclass(frozen=True)
class ReviseProposal:
    source: ClassId
    revised: Draft
    reason: str

    @property
    def touched(self) -> frozenset[ClassId]:
        return frozenset({self.source})


Proposal = MergeProposal | SplitProposal | ReviseProposal


def parse_proposals(
    raw: Sequence[Mapping[str, Any]], classes: Mapping[str, BehaviorClass], *, max_examples: int
) -> tuple[tuple[Proposal, ...], tuple[str, ...]]:
    proposals: list[Proposal] = []
    signals: list[str] = []
    for item in raw:
        try:
            proposals.append(_proposal(item, classes, max_examples=max_examples))
        except (BehaviorKindError, KeyError, TypeError) as exc:
            signals.append(f"kind_revision_proposal_dropped {item.get('kind')!r} {item.get('class')!r}: {exc}")
    return tuple(proposals), tuple(signals)


def _proposal(item: Mapping[str, Any], classes: Mapping[str, BehaviorClass], *, max_examples: int) -> Proposal:
    kind, source, reason = item["kind"], classes[item["class"]].id, str(item["reason"]).strip()
    if kind == "merge":
        if item["target"] is None or item["target"] == item["class"]:
            raise BehaviorKindError("merge needs a different target class")
        return MergeProposal(source, classes[item["target"]].id, reason)
    if kind != "revise":
        raise BehaviorKindError(f"unknown proposal kind {kind!r} (splits are decided by grouping, not proposed)")
    if item["draft"] is None:
        raise BehaviorKindError("revise needs a draft")
    draft = _draft(item["draft"], classes, max_examples=max_examples)
    if any(target == NEW for _, target in draft.excludes):
        raise BehaviorKindError("a revise may not point an exclusion at NEW")
    return ReviseProposal(source, draft, reason)


def _draft(raw: Mapping[str, Any], classes: Mapping[str, BehaviorClass], *, max_examples: int) -> Draft:
    excludes: list[tuple[str, ClassId | str | None]] = []
    for exclusion in raw["excludes"]:
        target = exclusion["goes_to"]
        excludes.append(
            (str(exclusion["text"]).strip(), NEW if target == NEW else classes[target].id if target else None)
        )
    return Draft(
        name=str(raw["name"]).strip(),
        criterion=str(raw["criterion"]).strip(),
        reminder=str(raw["reminder"]).strip(),
        excludes=tuple(excludes),
        examples=tuple(dict.fromkeys(str(text).strip() for text in raw["examples"]))[:max_examples],
    )


__all__ = ["Draft", "MergeProposal", "Proposal", "ReviseProposal", "SplitProposal", "parse_proposals"]
