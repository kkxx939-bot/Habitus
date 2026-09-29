"""情境命中：算法按概念上的情境说明算出"那一刻外面是什么样"。

映射器不算这个——它只判"这一条是不是那个行为"（``situation_hits`` 由调用方填，七d-7 那条边界）。
这里是纯计算：给定一天的日历、这条 occurrence 的字段、那一刻还立着的承诺、以及前几天的命中，
把概念集里**带情境说明**的那些逐条判一遍。

**材料由调用方备好**：账在组合根那边（本支不许触达 ``ledger``），所以"那一刻还立着哪些后件"以
``open_consequents`` 这个普通集合传进来。历史命中同理，以每天一份"签名"传进来。

**派生情境只看过去的日子**（``STREAK`` / ``YESTERDAY`` 从 ``day`` 的前一天往回数）：把今天算进连续里
就是拿今天解释今天，而这些情境正是要给今天的行为分层用的。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from habitus.scene.concepts.model import ConceptSet, concept_identity
from habitus.scene.concepts.situation import SituationBasis, SituationRule
from habitus.scene.occurrences.model import ConceptHit, ConceptHits

#: 一天的命中签名：(概念身份, 档)。祖先以 ``(身份, None)`` 记——祖先概念没有档。
DaySignature = frozenset[tuple[str, str | None]]


@dataclass(frozen=True)
class SituationInputs:
    """算一条 occurrence 的情境要的全部材料。"""

    day: date
    calendar_note: str | None = None
    subjects: tuple[str, ...] = ()
    place: str | None = None
    open_consequents: frozenset[str] = frozenset()
    history: Mapping[date, DaySignature] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.day, date):
            raise TypeError("day must be a date")


def day_signature(records: Iterable[ConceptHits], concepts: ConceptSet) -> DaySignature:
    """一天的命中记录 → 那天命中过的 (概念身份, 档)，**含祖先**。

    含祖先是因为派生情境多半盯着粗的那一层（「赶工中」盯的是「写代码」，而命中的是「修改代码」）；
    读侧沿 parent 链聚合是同一条规矩。
    """

    signature: set[tuple[str, str | None]] = set()
    for record in records:
        names = [hit.concept for hit in record.hits if hit.concept in concepts]
        for hit in record.hits:
            if hit.concept in concepts:
                signature.add((hit.identity, hit.grade))
        for ancestor in concepts.with_ancestors(names) - {concepts[name].identity for name in names}:
            signature.add((ancestor, None))
    return frozenset(signature)


def situation_hits(concepts: ConceptSet, inputs: SituationInputs) -> tuple[ConceptHit, ...]:
    """概念集里带情境说明、且此刻成立的那些情境概念，按身份排序。

    没有说明的情境概念一个都不出（「出差中」要等事实门），带说明但材料给不全的（要日历却没有日历）
    也不出——**不把"不知道"算成"成立"**。
    """

    if not isinstance(concepts, ConceptSet):
        raise TypeError("concepts must be a ConceptSet")
    if not isinstance(inputs, SituationInputs):
        raise TypeError("inputs must be SituationInputs")
    found = [
        concepts[identity].name
        for identity in sorted(concepts.situations())
        if concepts[identity].situation is not None and _holds(concepts[identity].situation, concepts, inputs)  # type: ignore[arg-type]
    ]
    return tuple(ConceptHit(name) for name in found)


def _holds(rule: SituationRule, concepts: ConceptSet, inputs: SituationInputs) -> bool:
    if rule.basis is SituationBasis.WEEKDAYS:
        return inputs.day.weekday() in rule.weekdays
    if rule.basis is SituationBasis.CALENDAR_NOTE:
        return inputs.calendar_note is not None and rule.value is not None and rule.value in inputs.calendar_note
    if rule.basis is SituationBasis.SUBJECT:
        return rule.value in inputs.subjects
    if rule.basis is SituationBasis.PLACE:
        return inputs.place is not None and inputs.place == rule.value
    watched = concept_identity(rule.concept) if rule.concept is not None else None
    if watched is None:  # pragma: no cover - SituationRule 已经要求这三族带概念
        return False
    if rule.basis is SituationBasis.OPEN_CLAIM:
        return watched in inputs.open_consequents
    if rule.basis is SituationBasis.YESTERDAY:
        return _hit_on(inputs, inputs.day - timedelta(days=1), watched, rule.grade)
    return all(
        _hit_on(inputs, inputs.day - timedelta(days=step), watched, rule.grade) for step in range(1, rule.days + 1)
    )


def _hit_on(inputs: SituationInputs, day: date, watched: str, grade: str | None) -> bool:
    """那天命中过吗。那天**没有签名**（还没映射过）一律算没命中——由调用方保证材料齐，
    这里不把"没读到"当成"成立"。"""

    signature = inputs.history.get(day)
    if not signature:
        return False
    return any(identity == watched and (grade is None or hit_grade == grade) for identity, hit_grade in signature)


def history_days(concepts: ConceptSet, day: date) -> tuple[date, ...]:
    """要备哪几天的签名：概念集里最长的那个连续天数决定往回读多少天（含昨天）。"""

    spans = [
        item.situation.days
        for identity in concepts.situations()
        for item in (concepts[identity],)
        if item.situation is not None and item.situation.basis.reads_history
    ]
    depth = max(spans) if spans else 0
    return tuple(day - timedelta(days=step) for step in range(1, depth + 1))


def signatures_of(
    records_by_day: Mapping[date, Sequence[ConceptHits]], concepts: ConceptSet
) -> Mapping[date, DaySignature]:
    return {day: day_signature(records, concepts) for day, records in records_by_day.items()}


__all__ = [
    "DaySignature",
    "SituationInputs",
    "day_signature",
    "history_days",
    "signatures_of",
    "situation_hits",
]
