"""命中记录：一条 occurrence 在概念层的读法，与行为树同构、叶名相同；封口时由代码直接生成，不经模型。

- ``kind_token`` 是封口时这条在行为树上的编号；``classified`` 说它是不是类编号（「待定」不是）。
  **行为概念的命中不存**：类编号本身就是那个行为概念的身份（``base_concept`` 从编号现读，裁定 20）；
- ``situation_hits`` 是那一刻成立的**日历情境**，由算法直接得出；``situations_checked`` 是**判过**的日历情境（含成立的）——
  没判过（那天日历答不出这个键）与判过不成立在盘上要分得开（2026-09-30 裁定八）。日历情境多了，这一天的摘要就变了，
  封口按现在的概念集整天重封，所以判过的总是封口那一刻概念集里日历答得出的全部；
- ``lane`` 是这条记录所在的 lane（编号前缀读出来）：时间线、叙事、结算都只在同一条 lane 里看；
- ``started_at`` / ``last_observed_at`` 来自行为树，落在这里让 ``occurrences/`` 对下游自足：窗锚在
  ``started_at``（2026-09-26 裁定，不锚"结束时刻"——树上没有结束时刻，``last_observed_at`` 是我们停止
  观测的那一刻，会随观测覆盖动）；
- ``hits_version`` 是生成它的命中记录口径（``daily.HITS_VERSION``），``sealed_at`` 是封口写下它的时刻。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property

from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.behavior.uri import BehaviorURI, BehaviorURIError, BehaviorURINodeType
from habitus.scene.concepts.model import ConceptError, concept_identity


class ConceptHitsError(ValueError):
    """命中记录与自己的约束矛盾。"""


@dataclass(frozen=True)
class ConceptHit:
    concept: str

    def __post_init__(self) -> None:
        try:
            concept_identity(self.concept)
        except ConceptError as exc:
            raise ConceptHitsError(str(exc)) from exc

    @property
    def identity(self) -> str:
        return concept_identity(self.concept)


@dataclass(frozen=True)
class ConceptHits:
    occurrence_uri: str
    kind_token: str
    last_observed_at: datetime
    situation_hits: tuple[ConceptHit, ...]
    hits_version: str
    sealed_at: datetime
    lane: str
    #: ``kind_token`` 是类编号（True）还是「待定」（False）；「非事件」不进序列、没有记录。
    classified: bool = True
    situations_checked: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            uri = BehaviorURI.parse(self.occurrence_uri)
        except (TypeError, BehaviorURIError) as exc:
            raise ConceptHitsError(f"occurrence_uri is not a behaviour URI: {exc}") from exc
        if uri.node_type is not BehaviorURINodeType.DOCUMENT or uri.to_address().kind is not BehaviorKind.OCCURRENCE:
            raise ConceptHitsError("occurrence_uri must identify an occurrence document")
        object.__setattr__(self, "occurrence_uri", str(uri))
        if not self.kind_token.strip() or self.kind_token != self.kind_token.strip():
            raise ConceptHitsError("kind_token must be non-empty text without surrounding whitespace")
        if not self.lane.strip() or self.lane != self.lane.strip():
            raise ConceptHitsError("lane must be non-empty text without surrounding whitespace")
        started_at = uri.to_address().started_at
        if self.last_observed_at.utcoffset() is None:
            raise ConceptHitsError("last_observed_at must be a timezone-aware datetime")
        if self.last_observed_at < started_at:
            # 上游不自洽不该在这里被静默抹平：树上的行为不可能在开始之前就被最后看见。
            raise ConceptHitsError("last_observed_at cannot precede the occurrence's start")
        object.__setattr__(self, "last_observed_at", self.last_observed_at.astimezone(started_at.tzinfo))
        situations = tuple(self.situation_hits)
        if len({hit.identity for hit in situations}) != len(situations):
            raise ConceptHitsError("situation_hits repeats a concept")
        object.__setattr__(self, "situation_hits", situations)
        if not self.hits_version.strip() or "\n" in self.hits_version:
            raise ConceptHitsError("hits_version must be a non-empty single line")
        if self.sealed_at.utcoffset() is None:
            raise ConceptHitsError("sealed_at must be a timezone-aware datetime")
        object.__setattr__(self, "sealed_at", self.sealed_at.astimezone(UTC))
        # 成立的必然判过：没列进 checked 的成立情境补进去。
        checked = _names(
            tuple(dict.fromkeys((*self.situations_checked, *(hit.concept for hit in situations)))), "situations_checked"
        )
        object.__setattr__(self, "situations_checked", checked)

    @cached_property
    def address(self) -> BehaviorAddress:
        """地址解析一次存着：结算对每条开着的承诺扫窗内全部记录、每次比较都取 ``started_at``，逐次重解析 URI
        是平方级的活（评审 B-10：第 120 夜解析 50,533 次）。"""

        return BehaviorURI.parse(self.occurrence_uri).to_address()

    @property
    def started_at(self) -> datetime:
        """窗的锚点。"""

        return self.address.started_at

    @property
    def base_concept(self) -> str | None:
        """这条记录算的行为概念的身份：类编号本身；「待定」没有。"""

        return concept_identity(self.kind_token) if self.classified else None


def _names(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    items = tuple(values)
    identities: list[str] = []
    for item in items:
        try:
            identities.append(concept_identity(item))
        except ConceptError as exc:
            raise ConceptHitsError(f"{label}: {exc}") from exc
    if len(set(identities)) != len(identities):
        raise ConceptHitsError(f"{label} repeats a concept")
    return items


__all__ = ["ConceptHit", "ConceptHits", "ConceptHitsError"]
