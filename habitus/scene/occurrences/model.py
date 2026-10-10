"""概念命中：一条 occurrence 在概念层的读法，与行为树同构、叶名相同。

- ``kind_token`` 是映射时这条在行为树上的编号；``classified`` 说它是不是类编号（「待定」不是）。
  **基础概念的命中不存**：类编号本身就是那个基础概念的身份，``graded_hits`` 从编号现读（裁定 20）；
  编号被词表迁移改写后，这条记录与树上不一致，映射按"输入变了"重判；
- ``hits`` 是映射出来的**细分概念**（可多个、可为空、每个可带档）；汇总概念不在这里，读侧按成员类聚合；
- ``situation_hits`` 是那一刻成立的**情境概念**，由算法直接得出、不经 LLM；
- ``unresolved`` 是候选里**没判成**的概念 → 为什么没判成（``UnresolvedReason``：材料缺、常态解不出、模型答看不到、
  模型这一次没答成）。它与 ``hits=()`` 必须分得开——没命中在关系检验里是负样本，没判不是；续跑按原因决定要不要重问；
- ``lane`` 是这条记录所在的 lane（编号前缀读出来，由映射时的词表口给）：时间线、叙事、结算都只在同一条 lane 里看；
- ``recent_digest`` 是映射时给的"近几天同类记录"的摘要：别的日子被词表迁移改了编号，这份材料就变了，要重判；
- ``started_at`` / ``last_observed_at`` 来自行为树，落在这里让 ``occurrences/`` 对下游自足：窗锚在
  ``started_at``（2026-09-26 裁定，不锚"结束时刻"——树上没有结束时刻，``last_observed_at`` 是我们停止
  观测的那一刻，会随观测覆盖动），``last_observed_at`` 只参与删失判定与档的复核；
- ``baseline_snapshot`` 是映射时给规则/模型看的这个人的常态值——常态一变，同一条重跑就是另一个答案，
  不记就不可重放；
- ``mapper`` / ``mapped_at`` 是出处，不是版本；``signals`` 是这次映射留下的痕（第几轮答对、JSON 是不是修的……）。

模型只校验自洽；"命中的概念确实是挂在这个类上的细分概念"要对着概念集才能判，那在映射器里。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from functools import cached_property
from types import MappingProxyType

from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.behavior.uri import BehaviorURI, BehaviorURIError, BehaviorURINodeType
from habitus.foundation.text import clean_line
from habitus.scene.concepts.model import ConceptError, concept_identity

MAX_BASELINE_ENTRIES = 32
MAX_BASELINE_CHARS = 120
MAX_SIGNALS = 64
MAX_SIGNAL_CHARS = 200


class ConceptHitsError(ValueError):
    """概念命中记录与自己的约束矛盾。"""


class UnresolvedReason(str, Enum):
    """一个候选为什么没判成。前四种是材料问题（材料换了才值得重问），后两种是模型那一侧的。"""

    BASELINE_MISSING = "baseline_missing"
    BASELINE_UNPARSEABLE = "baseline_unparseable"
    TIMELINE_MISSING = "timeline_missing"
    RECENT_MISSING = "recent_missing"
    MODEL_UNSEEN = "model_unseen"
    MODEL_FAILED = "model_failed"
    #: 同一条问两遍（第二遍候选倒过来排）答得不一样：两边都不算（语义树新方案 ``13`` ②；冒烟实测同输入两遍 7 条里 3 处不同）。
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True)
class ConceptHit:
    concept: str
    grade: str | None = None

    def __post_init__(self) -> None:
        try:
            concept_identity(self.concept)
        except ConceptError as exc:
            raise ConceptHitsError(str(exc)) from exc
        if self.grade is not None and (
            not isinstance(self.grade, str)
            or not self.grade.strip()
            or self.grade != self.grade.strip()
            or "\n" in self.grade
        ):
            raise ConceptHitsError("a grade must be non-empty single-line text without surrounding whitespace")

    @property
    def identity(self) -> str:
        return concept_identity(self.concept)


@dataclass(frozen=True)
class ConceptHits:
    occurrence_uri: str
    kind_token: str
    last_observed_at: datetime
    hits: tuple[ConceptHit, ...]
    situation_hits: tuple[ConceptHit, ...]
    unresolved: Mapping[str, UnresolvedReason]
    baseline_snapshot: Mapping[str, str]
    mapper: str
    mapped_at: datetime
    lane: str
    signals: tuple[str, ...] = ()
    #: ``kind_token`` 是类编号（True）还是「待定」（False）；「非事件」不映射、没有记录。
    classified: bool = True
    recent_digest: str | None = None
    #: 这条记录**判过**哪些情境概念（含成立的）。没判过（材料不全、情境概念是后来才加的）与判过不成立
    #: 在盘上要分得开，稳定性分层才知道哪一层该收哪条（2026-09-30 裁定八）。
    situations_checked: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            uri = BehaviorURI.parse(self.occurrence_uri)
        except (TypeError, BehaviorURIError) as exc:
            raise ConceptHitsError(f"occurrence_uri is not a behaviour URI: {exc}") from exc
        if uri.node_type is not BehaviorURINodeType.DOCUMENT or uri.to_address().kind is not BehaviorKind.OCCURRENCE:
            raise ConceptHitsError("occurrence_uri must identify an occurrence document")
        object.__setattr__(self, "occurrence_uri", str(uri))
        if (
            not isinstance(self.kind_token, str)
            or not self.kind_token.strip()
            or self.kind_token != self.kind_token.strip()
        ):
            raise ConceptHitsError("kind_token must be non-empty text without surrounding whitespace")
        if not isinstance(self.lane, str) or not self.lane.strip() or self.lane != self.lane.strip():
            raise ConceptHitsError("lane must be non-empty text without surrounding whitespace")
        if self.recent_digest is not None and (not isinstance(self.recent_digest, str) or not self.recent_digest):
            raise ConceptHitsError("recent_digest must be non-empty text or None")
        if not isinstance(self.classified, bool):
            raise ConceptHitsError("classified must be a boolean")
        if not self.classified and self.hits:
            raise ConceptHitsError("a pending occurrence has no class, so no refinement of a class can hit it")
        started_at = uri.to_address().started_at
        if not isinstance(self.last_observed_at, datetime) or self.last_observed_at.utcoffset() is None:
            raise ConceptHitsError("last_observed_at must be a timezone-aware datetime")
        if self.last_observed_at < started_at:
            # 上游不自洽不该在这里被静默抹平：树上的行为不可能在开始之前就被最后看见。
            raise ConceptHitsError("last_observed_at cannot precede the occurrence's start")
        object.__setattr__(self, "last_observed_at", self.last_observed_at.astimezone(started_at.tzinfo))
        hits = _hits(self.hits, "hits")
        situations = _hits(self.situation_hits, "situation_hits")
        unresolved = _reasons(self.unresolved)
        taken = {hit.identity for hit in hits}
        if self.classified and concept_identity(self.kind_token) in taken:
            raise ConceptHitsError("the base concept is read from kind_token; it is never stored as a hit")
        if taken & {hit.identity for hit in situations}:
            raise ConceptHitsError("a concept cannot appear both as a behaviour hit and a situation hit")
        if (taken | {hit.identity for hit in situations}) & {concept_identity(name) for name in unresolved}:
            raise ConceptHitsError("an unresolved concept cannot also be a hit")
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "situation_hits", situations)
        object.__setattr__(self, "unresolved", unresolved)
        object.__setattr__(self, "baseline_snapshot", _baseline(self.baseline_snapshot))
        if not isinstance(self.mapper, str) or not self.mapper.strip() or "\n" in self.mapper:
            raise ConceptHitsError("mapper must be a non-empty single line")
        if not isinstance(self.mapped_at, datetime) or self.mapped_at.utcoffset() is None:
            raise ConceptHitsError("mapped_at must be a timezone-aware datetime")
        object.__setattr__(self, "mapped_at", self.mapped_at.astimezone(UTC))
        object.__setattr__(self, "signals", _signals(self.signals))
        # 成立的必然判过：没列进 checked 的成立情境补进去（旧记录与只给命中的夹具都走这里）。
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
    def duration_minutes(self) -> float:
        """开始到最后所见有多少分钟。它是**观测到的**时长（``last_observed_at`` 会随覆盖动），
        常态时长与时长档比的都是这个量。"""

        return (self.last_observed_at - self.started_at).total_seconds() / 60.0

    @property
    def base_concept(self) -> str | None:
        """这条命中的基础概念身份：类编号本身；「待定」没有。"""

        return concept_identity(self.kind_token) if self.classified else None

    @property
    def graded_hits(self) -> Mapping[str, str | None]:
        """身份 → 档：基础概念（从编号现读，不带档）+ 细分概念。判前因命中用这个，不用丢了档的集合。"""

        table: dict[str, str | None] = {} if self.base_concept is None else {self.base_concept: None}
        table.update((hit.identity, hit.grade) for hit in self.hits)
        return MappingProxyType(table)

    @property
    def graded_situations(self) -> Mapping[str, str | None]:
        return MappingProxyType({hit.identity: hit.grade for hit in self.situation_hits})

    @property
    def unresolved_identities(self) -> frozenset[str]:
        return frozenset(concept_identity(name) for name in self.unresolved)

    def matches(self, concept: str, grade: str | None = None) -> bool:
        """这条记录是否命中 ``concept``（行为或情境）；给了 ``grade`` 就要求档也相同，不给则任何档都算。"""

        identity = concept_identity(concept)
        for table in (self.graded_hits, self.graded_situations):
            if identity in table:
                return grade is None or table[identity] == grade
        return False


def _hits(values: object, label: str) -> tuple[ConceptHit, ...]:
    if isinstance(values, str) or not isinstance(values, tuple | list):
        raise ConceptHitsError(f"{label} must be a sequence of ConceptHit")
    items = tuple(values)
    if any(not isinstance(item, ConceptHit) for item in items):
        raise ConceptHitsError(f"{label} must contain ConceptHit values")
    if len({item.identity for item in items}) != len(items):
        raise ConceptHitsError(f"{label} repeats a concept")
    return items


def _names(values: object, label: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, tuple | list):
        raise ConceptHitsError(f"{label} must be a sequence of concept names")
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


def _reasons(values: object) -> Mapping[str, UnresolvedReason]:
    if not isinstance(values, Mapping):
        raise ConceptHitsError("unresolved must map concept names to reasons")
    names = _names(tuple(values), "unresolved")
    try:
        # 按名字排好：盘上 JSON 按键排序，正文按这里的顺序渲染，两边不一致回读就对不上。
        return MappingProxyType({name: UnresolvedReason(values[name]) for name in sorted(names)})
    except ValueError as exc:
        raise ConceptHitsError(f"unresolved reason is unknown: {exc}") from exc


def _baseline(values: object) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ConceptHitsError("baseline_snapshot must be a mapping")
    if len(values) > MAX_BASELINE_ENTRIES:
        raise ConceptHitsError("baseline_snapshot carries too many entries")
    cleaned: dict[str, str] = {}
    for key, value in values.items():
        for item, label in ((key, "key"), (value, "value")):
            if not isinstance(item, str) or not item.strip() or item != item.strip() or "\n" in item:
                raise ConceptHitsError(f"baseline_snapshot {label} must be a non-empty single line")
            if len(item) > MAX_BASELINE_CHARS:
                raise ConceptHitsError(f"baseline_snapshot {label} exceeds {MAX_BASELINE_CHARS} characters")
        cleaned[key] = value
    return MappingProxyType(dict(sorted(cleaned.items())))


def _signals(values: object) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, tuple | list):
        raise ConceptHitsError("signals must be a sequence of text")
    items = tuple(values)
    if len(items) > MAX_SIGNALS:
        raise ConceptHitsError(f"a record carries at most {MAX_SIGNALS} signals")
    cleaned: list[str] = []
    for item in items:
        text = clean_line(item)
        if not text or len(text) > MAX_SIGNAL_CHARS:
            raise ConceptHitsError("each signal must be a non-empty line within the length budget")
        cleaned.append(text)
    return tuple(cleaned)


__all__ = ["ConceptHit", "ConceptHits", "ConceptHitsError", "UnresolvedReason"]
