"""对照答案找原因（裁定 55，方案第五节「纠错」）：一次落空（到最晚时限没做、也没补做），他接下来做了什么、在不在我们的语义关联里。
纯函数，只读。

读的是语义树这一晚的材料（概念集、时间线、关系表）与这条预期本身；不写任何树、不改账——找到的原因由调用方作为同值改判挂在
这次落空上，白天那一拍据此核"上次的原因今天还在不在"。

**下一步 X** = 该做 B 的时窗开始之后他做的第一件事（到最晚时限为止；时窗开始前就结束了的不算——早上叫的车替不了晚上该记的跑步）。
按顺序推，停在第一个对上的：

1. 同一件事：X 与 B 同在一个汇总概念下（订家人机票 / 订酒店都在「出行预订」下）→ 粒度。
2. 线索没了：B 惯常的前因（这一晚关系表里抬高 B 的成立 / 候选关系的前因）这次没来、之后到最晚时限也没来；或这天的日型与依据
   那几次都不同。习惯由稳定的情境线索触发，线索没了习惯就断（Wood, Tam & Guerrero Witt 2005，只当旁证）。
3. 别的起因：X 的成立 / 候选前因这一次正好把 X 带了出来（同一把尺：链合并、``Controls.arrival``）。
4. 事件触发再看前因 A 之后的分支：A → X 成立 / 候选 = 本来就有分支——带条件的 A → B 或 A → X 这一次条件判得出就是「分支有条件」，
   判不出是「分支条件不够」。钟面与日子触发没有前因，不走这一步。
5. X 还没进任何关联：「还没进关联」，带 X 的类与编号——记下来攒次数，不是终点。"B 做了接着 X"走行为链路由语义树自己检验；
   "B 没来接着 X"由诊断按（账 × X）比落空后与命中后接着做 X 的比例。
6. 到最晚时限没有任何动作：「没看到下一步」，观测的边界。

单次落空不说"习惯在变"：连着落空、账判出"规律断了"才说（裁定 54）。"推迟"也不在这里：补做了，那条事实就是「后补」。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from habitus.foresight.ledger.model import Cause, Expectation
from habitus.foresight.model import Seen
from habitus.foresight.triggers.events import chains_in_segment
from habitus.foresight.triggers.marks import absolute_slot, timeline_of
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations import (
    SEGMENTS,
    Chain,
    ConditionError,
    Controls,
    Field,
    LaneTimeline,
    Relation,
    Segment,
    Status,
    Template,
    chains_of,
    evaluator,
    merge_for,
    parse_condition,
)
from habitus.series import SeriesRecord

#: 上次落空的原因此刻又成立时的递送原因（裁定 55「再预测」）：不说；证据里写 ``CITE + 那次落空的预期身份``，下一次的答案按它验这次调整。
AGAIN = "上次的原因今天又在"
CITE = "原因:"

#: 能当原因的关系状态：已经成立或正在验证的。已检不显著、失效、前向未复现说的是"不是它带出来的"。
_LIVE = frozenset({Status.ESTABLISHED, Status.CANDIDATE})


@dataclass(frozen=True)
class CauseMaterials:
    """语义树这一晚的材料。``notes`` 是当地日历对一天的说法；``window_slots`` 是转移窗口（与关系检验同一个数）。"""

    line: LaneTimeline
    concepts: ConceptSet
    relations: Sequence[Relation]
    window_slots: int
    notes: Callable[[date], str | None]


def find_cause(
    expectation: Expectation,
    materials: CauseMaterials,
    *,
    current: Callable[[str], Sequence[str]],
    basis_days: Iterable[date] = (),
) -> Cause:
    """这一次落空的原因。``current`` 是词表的编号映射（预期记的是形成时的编号）；``basis_days`` 是依据那几次各在哪天。"""

    wanted = {expectation.kind_token, *current(expectation.kind_token)}
    concepts = materials.concepts
    base = next((hit for token in sorted(wanted) if (hit := concepts.base_for(token)) is not None), None)
    following = next_action(expectation, materials.line, wanted, concepts)
    if base is not None and following is not None:
        same = _same_matter(base, following, concepts)
        if same is not None:
            return same
    missing = _cue_missing(expectation, base, materials, basis_days)
    if missing is not None:
        return missing
    by_key = {relation.key.text: relation for relation in materials.relations}
    controls = Controls(materials.line, materials.window_slots)
    # 事件触发的前因 A 自己带出了 X，那是 A 之后的分支（下一步），不是"别的起因"
    trigger = expectation.trigger.split("|")[1] if expectation.source == "事件" and expectation.trigger.count("|") >= 3 else None
    if following is None:
        return Cause("没看到下一步")
    record, own = following
    found = _own_cause(record, own, materials, controls, besides=trigger)
    if found is not None:
        return found
    if expectation.source == "事件" and base is not None:
        branch = _branch(expectation, base, following, materials, by_key)
        if branch is not None:
            return branch
    return Cause("还没进关联", classes=(record.kind_token,), uris=(record.uri,))


def next_action(
    expectation: Expectation, line: LaneTimeline, wanted: set[str], concepts: ConceptSet
) -> tuple[SeriesRecord, str] | None:
    """下一步 X：时窗开始之后他做的第一件别的事（成了类的；到最晚时限为止；时窗开始前就结束了的不算），带它的基础概念。
    不取整个说中的范围：说中的范围常是一整天，早上 08:55 叫的车替不了晚上 21:03 该记的跑步（2026-10-17 甲盘探针误归 5 次）。"""

    start, end = expectation.window[0], expectation.deadline
    found = []
    for record in line.records.values():
        if (
            not record.classified
            or record.kind_token in wanted
            or record.started_at > end
            or record.last_observed_at < start
        ):
            continue
        own = concepts.base_for(record.kind_token)
        if own is not None:
            found.append((record, own))
    return min(found, key=lambda pair: (pair[0].started_at, pair[0].uri)) if found else None


def _same_matter(base: str, following: tuple[SeriesRecord, str], concepts: ConceptSet) -> Cause | None:
    record, own = following
    shared = sorted(set(concepts.ancestors(base)) & set(concepts.ancestors(own)))
    if shared:
        return Cause("同一件事", classes=(record.kind_token,), uris=(record.uri,), relation=shared[0])
    return None


def _cue_missing(
    expectation: Expectation, base: str | None, materials: CauseMaterials, basis_days: Iterable[date]
) -> Cause | None:
    """B 惯常的前因这次没来：这一晚关系表里以 B 为后果、抬高它的成立 / 候选关系（不带条件、短跨度），前因在时窗开始那一刻
    不在它那一段里、时窗开始到最晚时限之间也没来。看的是答案那一晚的关系表，不是形成时的前因清单——形成时还只是候选的
    关系不进清单（甲盘探针：「预约保洁 → 开门码」三月还是候选，开门码的预期前因清单全空，那周没约保洁的落空一条也没归对）。
    几条都没来时取它不来 B 就掉得最多的那条。再看这天的日型是不是和依据那几次都不同。"""

    line = materials.line
    if base is not None:
        # 几条惯常前因都没来时，取它不来 B 就掉得最多的那条（甲盘探针：按编号、按超额比例排都把那周没约保洁的落空归给了
        # 「看今天的日程 → 开门码」）
        for relation in sorted(materials.relations, key=_explains_most):
            key = relation.key
            if (
                key.consequent != base
                or relation.status not in _LIVE
                or not relation.upward
                or key.condition
                or key.segment not in SEGMENTS
                or key.lane != line.lane
                or key.antecedent not in materials.concepts
                or not materials.concepts[key.antecedent].role.is_behavior
            ):
                continue
            if not cue_present(
                line, (key.antecedent,), key.segment, at=expectation.window[0], window_slots=materials.window_slots
            ) and not any(
                expectation.window[0] <= mark.started_at <= expectation.deadline
                for mark in line.marks.get(key.antecedent, [])
            ):
                return Cause(
                    "线索没了",
                    classes=tuple(materials.concepts.classes_of(key.antecedent)) or (key.antecedent,),
                    relation=key.text,
                    segment=key.segment.value,
                )
    today = materials.notes(expectation.day)
    earlier = {materials.notes(day) for day in basis_days}
    if today is not None and earlier and today not in earlier:
        return Cause("线索没了", condition=f"日型={today}")
    return None


def _explains_most(relation: Relation) -> tuple[float, bool, str]:
    """它不来时 B 掉得最多的先：前因在时 B 的发生率减前因不在时的（语义树这一晚的读数）。不按超额比例：天天都有的前因
    （「看今天的日程」一天一次）份额也高，可它不来时开门码照样很少（0.14 对 0.05）；没约保洁时开门码几乎不会有（0.71 对 0）。"""

    reading = relation.tonight
    drop = reading.effect.difference if reading is not None else 0.0
    return (-drop, relation.status is not Status.ESTABLISHED, relation.key.text)


def cue_present(line: LaneTimeline, keys: Iterable[str], segment: Segment, *, at: datetime, window_slots: int) -> bool:
    """这几个键（类编号或概念身份）的链此刻（``at``）正处在 ``segment`` 那一段里没有：与事件触发同一把尺。"""

    slot_minutes = 24 * 60 // line.slots_per_day
    return bool(
        chains_in_segment(
            line,
            keys,
            segment,
            today=at.date(),
            now_slot=absolute_slot(at, slot_minutes=slot_minutes),
            window=window_slots,
        )
    )


def _own_cause(
    record: SeriesRecord, own: str, materials: CauseMaterials, controls: Controls, *, besides: str | None
) -> Cause | None:
    """以 X 为后果的成立 / 候选关系里，它的前因这一次正好把这条 X 带了出来（``besides`` 是这条预期自己的前因，不算）。"""

    line, window = materials.line, materials.window_slots
    for relation in sorted(materials.relations, key=lambda item: (item.status is not Status.ESTABLISHED, item.key.text)):
        key = relation.key
        if (
            key.consequent != own
            or relation.status not in _LIVE
            or not relation.upward
            or key.condition
            or key.segment not in SEGMENTS
            or key.lane != line.lane
            or key.antecedent not in materials.concepts
            or key.antecedent == besides
        ):
            continue
        for chain in merge_for(key.segment, chains_of(line, key.antecedent, window)):
            if not 0 <= (record.day - chain.day).days <= 7 or record.uri in chain.uris:
                continue
            arrival = controls.arrival(key.segment, chain, own)
            if arrival.came and arrival.uri == record.uri:
                return Cause(
                    "别的起因",
                    classes=tuple(materials.concepts.classes_of(key.antecedent)) or (key.antecedent,),
                    uris=(chain.anchor.uri, record.uri),
                    relation=key.text,
                    segment=key.segment.value,
                )
    return None


def _branch(
    expectation: Expectation,
    base: str,
    following: tuple[SeriesRecord, str],
    materials: CauseMaterials,
    by_key: Mapping[str, Relation],
) -> Cause | None:
    """事件触发：前因 A 之后的分支。A → X 不在表里（样本不够、从没有过）或已检不显著 / 失效 / 前向未复现：X 不是 A 带出来的，
    交给下一步记「还没进关联」。"""

    parts = expectation.trigger.split("|")
    if len(parts) < 4 or parts[3] not in {segment.value for segment in SEGMENTS}:
        return None
    lane, antecedent, segment = parts[0], parts[1], Segment(parts[3])
    chain = _chain_at(materials.line, antecedent, segment, expectation.trigger_instance, materials.window_slots)
    record, own = following
    relation = by_key.get(f"{lane}|{antecedent}|{own}|{segment.value}")
    if relation is None or relation.status not in _LIVE:
        return None
    conditioned = _conditioned(materials, by_key, lane, antecedent, segment, (base, own), chain)
    if conditioned is not None:
        return conditioned
    return Cause("分支条件不够", classes=(record.kind_token,), uris=(record.uri,), relation=relation.key.text)


def _conditioned(
    materials: CauseMaterials,
    by_key: Mapping[str, Relation],
    lane: str,
    antecedent: str,
    segment: Segment,
    consequents: tuple[str, str],
    chain: Chain | None,
) -> Cause | None:
    """带条件的 A → B（条件不成立时 B 不来）或 A → X（条件成立时来的是 X）：这一次条件判得出就是它。"""

    if chain is None:
        return None
    base = consequents[0]
    notes = {chain.day: note} if (note := materials.notes(chain.day)) is not None else {}
    for relation in sorted(by_key.values(), key=lambda item: item.key.text):
        key = relation.key
        if (
            relation.status is not Status.ESTABLISHED
            or not key.condition
            or not relation.upward
            or key.lane != lane
            or key.antecedent != antecedent
            or key.segment is not segment
            or key.consequent not in consequents
        ):
            continue
        try:
            judge = evaluator(materials.line, parse_condition(key.condition), window=materials.window_slots, notes=notes)
        except (ConditionError, ValueError):
            continue
        holds = judge(chain)
        if holds is None:
            continue
        # A → B｜c：c 不成立时 B 不来；A → X｜c：c 成立时来的是 X
        keeps_away = key.consequent != base
        if holds is keeps_away:
            return Cause("分支有条件", relation=key.text, condition=key.condition, holds_when=keeps_away)
    return None


def _chain_at(line: LaneTimeline, concept: str, segment: Segment, instance: str, window: int) -> Chain | None:
    """这条预期的前因那一次（实例是链头的开始时刻）在时间线上的那条链。"""

    try:
        started = datetime.fromisoformat(instance)
    except ValueError:
        return None
    for chain in merge_for(segment, chains_of(line, concept, window)):
        if chain.anchor.started_at == started or any(
            mark.started_at == started for mark in line.marks.get(concept, []) if mark.uri in chain.uris
        ):
            return chain
    return None


def cause_holds(
    cause: Cause,
    expectation: Expectation,
    seen: Sequence[Seen],
    *,
    now: datetime,
    lane: str,
    slot_minutes: int,
    window_slots: int,
    day_note: Callable[[date], str | None],
) -> bool:
    """白天那一拍：上次那次落空的原因，此刻对这条预期又成立了没有（裁定 55「再预测」）。只用类编号与最近几天开始过的事，
    不读语义树。核得了的只有三种：线索没了（同一条惯常前因此刻又不在它那一段里 / 又是那个日型）、别的起因（起因那一类此刻又正处在
    它带出 X 的那一段里）、分支有条件里的周末与日历两种结构字段；其余几种（同一件事、分支条件不够、还没进关联、没看到下一步）说的是粒度、关联还没攒起来或观测边界，此刻核不出来。"""

    today = now.date()
    if cause.kind == "线索没了":
        if cause.classes and cause.segment:
            # 同一条惯常前因此刻又不在它那一段里
            line = timeline_of(seen, lane=lane, slot_minutes=slot_minutes, horizon=today)
            return not cue_present(line, cause.classes, Segment(cause.segment), at=now, window_slots=window_slots)
        note = cause.condition.removeprefix("日型=")
        return cause.condition.startswith("日型=") and day_note(today) == note
    if cause.kind == "别的起因":
        line = timeline_of(seen, lane=lane, slot_minutes=slot_minutes, horizon=today)
        return cue_present(line, cause.classes, Segment(cause.segment), at=now, window_slots=window_slots)
    if cause.kind == "分支有条件" and cause.condition:
        try:
            condition = parse_condition(cause.condition)
        except (ConditionError, ValueError):
            return False
        if condition.template is not Template.FIELD:
            return False
        # 条件说的是前因那一次那天（与语义树的判法同一份）：事件触发按前因的开始时刻，取不到按今天
        try:
            anchor = datetime.fromisoformat(expectation.trigger_instance).astimezone(now.tzinfo).date()
        except ValueError:
            anchor = today
        if condition.field is Field.WEEKEND:
            value = anchor.weekday() >= 5
        elif condition.field is Field.CALENDAR:
            value = day_note(anchor) == condition.value
        else:
            return False
        return value is cause.holds_when
    return False


__all__ = ["AGAIN", "CITE", "CauseMaterials", "cause_holds", "cue_present", "find_cause", "next_action"]
