"""证据包里的"已成立的关系"：此刻在场的前因（语义树新方案 ``13`` ⑤⑥；裁定 24：先接现在的预测层，反馈后做）。

一个候选（行为类 B）配上语义树里**已成立**、后果正是这个类、而且**前因此刻在它那一段跨度里**的关系。在场的判法与关系检验
**同一把尺**（裁定 27 第 1 条：同一条链从前因做完起量；E8）：

- 同一条链：前因今天做完的那一槽到此刻不超过转移窗口的槽数（今天已封口的行 + 还没封口的判断）；
- 当天剩下的：前因今天更早做完、已经过了转移窗口；
- 次日：前因昨天发生过；
- 第 2–7 天：前因 2–7 天前发生过。

同一条链与当天剩下的：前因做完之后后果**已经开始过**的，不再摆（那段窗口说的事已经发生了，再摆是让模型把它算第二遍）。

**只读、零模型、零统计**：关系是语义树每晚算好存下的，这里只判"前因在不在场"、把读数摆出来。说法用中性的"之后出现的比例"，
不用"带出来""让……更容易"这类因果措辞（关系只是相关）。读数带样本数；维持检验这一段攒的次数不够前向验证的门槛时，用全部数据的
读数（样本少不出数——零宽的区间看起来最"确定"，其实最不可信；E7）。

这一版还不摆的（各有原因，记着）：
- 长跨度关系（状态是否还在要重算标志；前期也成立不了）；
- 带条件的关系（条件要对着此刻这条前因判，等预测层能拿到此刻那条记录的结构字段）；
- 前因或后果是细分概念的（今天的记录还没映射，判不了"这一条是不是返工"；裁定 27 第 7 条：进证据前还要用户标样本量精确率）；
- 汇总概念当前因、而它的成员也有同一段的关系在场时，只摆成员那条（同一个效应不重复报）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import MappingProxyType

from habitus.foresight.model import Moment, UnsealedRow
from habitus.scene.concepts.model import ConceptKind, ConceptSet
from habitus.scene.relations import Relation, RelationTest, Segment, Status, Subset
from habitus.scene.views import DayIndexCache

#: 每一段在证据里的说法：跨度、对照是什么。同一条链与预测树的转移边量的几乎是同一件事（"做完 A 接着做什么"），标出来。
SPANS = {
    Segment.CHAIN: ("同一条链：做完之后 {window} 分钟内，与转移边重叠", "同一天别的事做完之后"),
    Segment.REST_OF_DAY: ("当天之后", "别的周同一个周几、同一钟点之后"),
    Segment.NEXT_DAY: ("次日", "别的周同一个周几的次日"),
    Segment.DAYS_2_7: ("之后第 2–7 天", "别的周同一个周几之后第 2–7 天"),
}


@dataclass(frozen=True)
class RelationNote:
    """一条此刻在场的已成立关系。``key`` 是关系身份（承诺上记它，预测层的反馈将来按它读回语义树）。
    ``antecedents`` 是读数用了几次独立的前因；``seen_at`` 是前因这一次做完的时刻（跨天两段是那天最后一次做完）。"""

    key: str
    antecedent: str
    segment: Segment
    span: str
    control: str
    treated_rate: float
    control_rate: float
    interval: tuple[float, float] | None
    upward: bool
    antecedents: int
    seen_at: datetime
    #: 前因这一次的出处：已封口的是那条记录的 URI；还没封口、靠现场归类读出来的是 ``unsealed:<开始时刻>``——晚上封口时它可能被
    #: 归成别的类，承诺上记下来，反馈读回语义树时才分得清（第四轮评审 E19）。
    source: str = ""
    #: 这条读数出自第几晚的关系表（夜批停了也看得出来是多久以前的；E10）。
    as_of: date | None = None


@dataclass(frozen=True)
class RelationTable:
    """预测层要从语义树读的那一份：已存的关系、它们用的概念集、是第几晚的（各 lane 里最晚的那晚；没有关系表是 None）。"""

    relations: tuple[Relation, ...]
    concepts: ConceptSet
    night: date | None


@dataclass(frozen=True)
class _Seen:
    started_at: datetime
    ended_at: datetime
    kind_token: str
    source: str


def present_relations(
    relations: Iterable[Relation],
    concepts: ConceptSet,
    moment: Moment,
    cache: DayIndexCache,
    unsealed: Sequence[UnsealedRow],
    *,
    slot_minutes: int,
    window_slots: int,
    minimum_antecedents: int,
    as_of: date | None = None,
) -> Mapping[str, tuple[RelationNote, ...]]:
    """候选的类编号 → 此刻在场的已成立关系（按前因名、跨度排）。``window_slots`` 是转移窗口的槽数（与预测树、关系检验同一个数）；
    ``minimum_antecedents`` 是前向验证的最少次数（维持检验这一段不到它就用全部数据的读数）。"""

    today = _today(moment, cache, unsealed)
    found: dict[str, list[tuple[Relation, frozenset[str], tuple[datetime, str]]]] = {}
    for relation in relations:
        key = relation.key
        if relation.status is not Status.ESTABLISHED or key.condition or key.segment not in SPANS:
            continue
        consequent = concepts.get(key.consequent)
        antecedent = concepts.get(key.antecedent)
        if consequent is None or antecedent is None or consequent.kind is not ConceptKind.BASE:
            continue
        if antecedent.kind not in (ConceptKind.BASE, ConceptKind.GROUP):
            continue
        seen = _seen(
            set(antecedent.classes), consequent.classes[0], key.segment, moment, cache, today, slot_minutes, window_slots
        )
        if seen is not None:
            found.setdefault(consequent.classes[0], []).append((relation, frozenset(antecedent.classes), seen))
    notes: dict[str, tuple[RelationNote, ...]] = {}
    for candidate, items in found.items():
        kept = [
            (relation, seen)
            for relation, classes, seen in items
            if not any(
                other is not relation and other.key.segment == relation.key.segment and others < classes
                for other, others, _ in items
            )
        ]
        made = [
            note
            for relation, (seen, source) in kept
            if (
                note := _note(relation, concepts, seen, source, slot_minutes * window_slots, minimum_antecedents, as_of)
            )
            is not None
        ]
        if made:
            notes[candidate] = tuple(sorted(made, key=lambda note: (note.antecedent, list(SPANS).index(note.segment))))
    return MappingProxyType(notes)


def _today(moment: Moment, cache: DayIndexCache, unsealed: Sequence[UnsealedRow]) -> list[_Seen]:
    """今天到此刻为止开始过的事（已封口的行 + 未封口的判断），按开始时刻排。"""

    now = moment.at
    rows = [
        _Seen(row.at, min(row.last_observed_at, now), row.kind_token, row.uri)
        for row in cache.day(moment.day).rows
        if row.at <= now
    ]
    rows += [
        _Seen(row.started_at, min(row.last_observed_at, now), row.kind_token, f"unsealed:{row.started_at.isoformat()}")
        for row in unsealed
        if row.kind_token is not None and row.started_at <= now
    ]
    return sorted(rows, key=lambda item: item.started_at)


def _seen(
    classes: set[str],
    consequent: str,
    segment: Segment,
    moment: Moment,
    cache: DayIndexCache,
    today: list[_Seen],
    slot_minutes: int,
    window_slots: int,
) -> tuple[datetime, str] | None:
    """前因在这一段里最近一次做完的（时刻, 出处）；不在场（或那段窗口里后果已经开始过）是 None。"""

    now = moment.at
    if segment in (Segment.CHAIN, Segment.REST_OF_DAY):
        here = _slot(now, slot_minutes)
        done = [item for item in today if item.kind_token in classes and item.ended_at <= now]
        if segment is Segment.CHAIN:
            done = [item for item in done if here - _slot(item.ended_at, slot_minutes) <= window_slots]
        else:
            done = [item for item in done if here - _slot(item.ended_at, slot_minutes) > window_slots]
        if not done:
            return None
        last = max(done, key=lambda item: item.ended_at)
        if any(item.kind_token == consequent and item.started_at >= last.ended_at for item in today):
            return None  # 前因做完之后后果已经开始过：这段窗口说的事已经发生了
        return last.ended_at, last.source
    days: list[date] = (
        [moment.day - timedelta(days=1)]
        if segment is Segment.NEXT_DAY
        else [moment.day - timedelta(days=offset) for offset in range(2, 8)]
    )
    rows = [row for day in days for row in cache.day(day).rows if row.kind_token in classes]
    if not rows:
        return None
    latest = max(rows, key=lambda row: row.last_observed_at)
    return latest.last_observed_at, latest.uri


def _slot(moment: datetime, slot_minutes: int) -> int:
    return (moment.hour * 60 + moment.minute) // slot_minutes


def _note(
    relation: Relation,
    concepts: ConceptSet,
    seen: datetime,
    source: str,
    window_minutes: int,
    minimum_antecedents: int,
    as_of: date | None,
) -> RelationNote | None:
    # 比例、区间、次数取同一份读数：维持检验这一段攒够了就用它（"现在还成立"判的就是它），不够就用当晚全部数据的检验
    reading: Subset | RelationTest | None = relation.maintenance
    if reading is None or reading.antecedents < minimum_antecedents:
        reading = relation.tonight
    if reading is None:
        return None
    span, control = SPANS[relation.key.segment]
    return RelationNote(
        key=relation.key.text,
        antecedent=concepts.label_of(relation.key.antecedent),
        segment=relation.key.segment,
        span=span.format(window=window_minutes),
        control=control,
        treated_rate=reading.effect.treated_rate,
        control_rate=reading.effect.control_rate,
        interval=reading.interval,
        upward=relation.upward,
        antecedents=reading.antecedents,
        seen_at=seen,
        source=source,
        as_of=as_of,
    )


__all__ = ["SPANS", "RelationNote", "RelationTable", "present_relations"]
