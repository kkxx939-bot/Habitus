"""单元的历史（裁定 55 新版第 2 条）：每晚用当晚的语义树把每个行为历史上每一刻重标一遍在场的输入，单元白天拿它去数。
纯函数、形状与编解码；落盘在组合根（``runtime/foresight_history.py``）。

照 transformer 训练里"每一遍用最新的模型把全部数据重算一次"：旧数据不改标签、不回头改账，只是用今晚的关系表重新看历史上
那一刻有哪些输入在场。一条新关系（"约游泳 → 压低健身"）长出来那一晚，它在场的那几刻从「离上次够久」换进「离上次够久 + 约游泳」，
不再拖低平常那一组；下一次同样的组合出现，单元读低、不预测。

**一刻在场的输入**：这一刻每条预期自己那本账，加上当晚关系表里以这个行为为后果、状态在 ``statuses`` 里的每条关系——前因在
那条预期形成的那一刻正处在关系那一段跨度里（与建预期同一把尺 ``chains_in_segment``）；情境段（日历情境，裁定 59）看那一天
日历条件成不成立（时间线上那天的日历条件，与建预期同一个判法 ``calendar_holds``；日历答不出的那天不算在场）。抬高的记成它那本事件 / 情境账的身份，压低的记
``↓`` + 关系身份。时间线是语义树这一晚的（按概念身份挂记号；基础概念的身份就是类编号）。新关系"周三 → 例会材料"长出来
那一晚，过去的周三都换进这个输入。

**带条件的关系与巧合**（裁定 61 第 6 条）：同一族（不带条件的 + 各带条件版本）里那一刻只记给条件全成立、条件最多的那一条
（与建预期同一个取法 ``triggers.conditions.most_specific``，条件按语义树的 ``evaluator`` 在当晚的时间线上判）；组合成立那一晚，
历史上条件成立的各刻自动归到组合边。语义树已判成巧合（效应不够大）的输入不进签名，权重照数、只是不参与分组。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from habitus.foresight.edges import DOWN, PastMoment
from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.model import NATURAL_HITS, AccountFact, Expectation, account_key, parse_account
from habitus.foresight.opportunities.model import OpportunityTable, RelationOpportunities
from habitus.foresight.triggers.conditions import families, most_specific
from habitus.foresight.triggers.events import chains_in_segment
from habitus.foresight.units import connected
from habitus.foundation.lifecycle import rank_by
from habitus.scene.relations import LaneTimeline, Segment
from habitus.temporal.slots import absolute_slot

#: v2（裁定 61 第 7 条）：每刻的分量是原始强度，"规律断了"的打折读的时候算；签名按族取最具体的成立组合、不含巧合输入。
#: v3：每刻带着它由哪几条预期组成——单元的提醒反馈时钟按"到时窗末还没做"的机会推进（裁定 60 第 4 条），要认出哪几刻
#: 已经作为提醒反馈记过、不再数一遍。
HISTORY_SCHEMA_VERSION = 3
MOMENT_RECORD_SCHEMA = "foresight_unit_moment_v1"
#: 看得见的落空（强度大于 0 才算：观测空白里的中性不算知道）。
_MISSES = frozenset({"过期待决", "后补"})


@dataclass(frozen=True)
class MomentRecord:
    """单元历史的原料（26 号清单 4.9）：一条预期结完之后它在预测轴上的结果，一条一份落盘、不随账本压实而丢。

    ``came``：自然命中为真，看得见的落空为假，其余（被提醒才做、观测空白、作废）为 None——不算知道，但照样参与"说中的范围
    连成一片"（它能把前后两条连成同一刻）。``strength`` 是原始强度：规律断了的打折在读的时候对正在读的那串刻算
    （``edges.fold_sequence``），不存打折后的。"""

    expectation_id: str
    kind_token: str
    account: str
    formed_at: datetime
    observed_from: datetime
    deadline: datetime
    came: bool | None
    strength: float

    def __post_init__(self) -> None:
        for label in ("expectation_id", "kind_token", "account"):
            if not isinstance(getattr(self, label), str) or not getattr(self, label):
                raise ForesightError(f"moment record {label} must be non-empty text")
        for label in ("formed_at", "observed_from", "deadline"):
            value = getattr(self, label)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ForesightError(f"moment record {label} must be a timezone-aware datetime")
        if not 0.0 <= self.strength <= 1.0 or (self.came is None and self.strength != 0.0):
            raise ForesightError("a moment record carries a strength in [0, 1], and none when its outcome is unknown")


def record_of(item: Expectation, fact: AccountFact | None) -> MomentRecord:
    """一条结完的预期、它在预测轴上现在算数的那一笔（没有为 None），变成一份记录。"""

    came: bool | None = None
    strength = 0.0
    if fact is not None and fact.kind in NATURAL_HITS and fact.strength > 0.0:
        came, strength = True, fact.strength
    elif fact is not None and fact.kind in _MISSES and fact.strength > 0.0:
        came, strength = False, fact.strength
    return MomentRecord(
        item.expectation_id,
        item.kind_token,
        item.account,
        item.formed_at,
        item.observed_from,
        item.deadline,
        came,
        strength,
    )


def encode_record(item: MomentRecord) -> dict[str, Any]:
    return {
        "schema_version": MOMENT_RECORD_SCHEMA,
        "expectation_id": item.expectation_id,
        "kind_token": item.kind_token,
        "account": item.account,
        "formed_at": item.formed_at.isoformat(),
        "observed_from": item.observed_from.isoformat(),
        "deadline": item.deadline.isoformat(),
        "came": item.came,
        "strength": item.strength,
    }


def decode_record(raw: Mapping[str, Any]) -> MomentRecord:
    try:
        if raw["schema_version"] != MOMENT_RECORD_SCHEMA:
            raise ValueError(f"schema_version {raw['schema_version']!r}")
        came = raw["came"]
        if came is not None and not isinstance(came, bool):
            raise ValueError("came must be a boolean or null")
        return MomentRecord(
            str(raw["expectation_id"]),
            str(raw["kind_token"]),
            str(raw["account"]),
            datetime.fromisoformat(str(raw["formed_at"])),
            datetime.fromisoformat(str(raw["observed_from"])),
            datetime.fromisoformat(str(raw["deadline"])),
            came,
            float(raw["strength"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ForesightError(f"unit moment record is malformed: {exc}") from exc


@dataclass(frozen=True)
class Moment:
    """重标过的一刻与它由哪几条预期组成（退出时整刻一起删；提醒反馈时钟认出记过的那几刻）。"""

    past: PastMoment
    expectation_ids: tuple[str, ...]


@dataclass(frozen=True)
class UnitHistory:
    """一条 lane 第 ``night`` 晚重标过的历史：行为类 → 它历史上的每一刻（连同每刻由哪几条预期组成）。"""

    lane: str
    night: date
    moments: Mapping[str, tuple[Moment, ...]] = field(default_factory=dict)

    def of(self, kind_token: str) -> tuple[PastMoment, ...]:
        """这个行为历史上的每一刻（边与单元读数用）。"""

        return tuple(item.past for item in self.moments.get(kind_token, ()))

    def labelled(self, kind_token: str) -> tuple[Moment, ...]:
        """这个行为历史上的每一刻连同组成它的预期（提醒反馈时钟用）。"""

        return self.moments.get(kind_token, ())


def present_inputs(
    group: Sequence[MomentRecord], *, table: OpportunityTable, line: LaneTimeline, statuses: Iterable[str]
) -> frozenset[str]:
    """历史上这一刻（同一个行为、说中的范围连成一片的几条预期）按当晚的关系表在场的输入。"""

    if not group:
        raise ForesightError("a moment needs at least one expectation")
    kind = group[0].kind_token
    wanted = set(statuses)
    coincidental = {relation.key for relation in table.relations if relation.coincidental}
    relations = [relation for relation in table.by_consequent(kind) if relation.status in wanted]
    situations = [relation for relation in relations if relation.segment == "situation"]
    events = families(relation for relation in relations if relation.segment != "situation")
    found: set[str] = set()
    for item in group:
        _kind, source, trigger = parse_account(item.account)
        at = item.formed_at
        now_slot = absolute_slot(at, slot_minutes=table.slot_minutes)
        for relation in situations:
            conditions = line.calendar_conditions.get(at.date(), {})
            if relation.key not in coincidental and relation.calendar_holds(
                conditions, official=table.unseen_day_types == "official"
            ):
                found.add(_input_of(kind, "情境", relation))
        decided: set[str] = set()
        for base, members in events.items():
            chains = chains_in_segment(
                line,
                (members[0].antecedent,),
                Segment(members[0].segment),
                today=at.date(),
                now_slot=now_slot,
                window=table.window_slots,
            )
            if not chains:
                continue
            chosen = most_specific(members, chains[0], line, window=table.window_slots)
            if chosen is None:
                continue
            decided.add(base)
            if chosen.key not in coincidental:
                found.add(_input_of(kind, "事件", chosen))
        # 这条记录自己那本账：事件源的账属于某一族、而这一族这一刻已判出最具体的那条时，就由那一条代表（同一次结果只记
        # 一次，组合成立后旧刻自动归到组合边）；族不在表上（关系退回了）或日子源的照旧记自己
        if trigger not in coincidental and (source != "事件" or _base_of(trigger) not in decided):
            found.add(item.account)
    return frozenset(found)


def _base_of(trigger: str) -> str:
    """事件账的触发身份（关系键 ``lane|前因|后果|段[|条件]``）去掉条件之后的那一族（条件文本里不许有 ``|``）。"""

    return "|".join(trigger.split("|")[:4])


def _input_of(kind: str, source: str, relation: RelationOpportunities) -> str:
    """抬高的记成它那本账的身份，压低的记 ``↓`` + 关系身份。"""

    return account_key(kind, source, relation.key) if relation.upward else f"{DOWN}{relation.key}"


def labelled_moments(
    records: Iterable[MomentRecord], *, present: Callable[[Sequence[MomentRecord]], frozenset[str]]
) -> tuple[Moment, ...]:
    """一个行为的每一刻（``units.connected``）：有一条自然命中就是来了（分量取命中里最重的），没有命中而有看得见的落空就是没来
    （分量取落空里最重的），其余不知道、不算。分量 = 原始强度（打折在读的时候算）；在场的那一组由 ``present`` 给。"""

    found = []
    for group in connected(records):
        hits = [item.strength for item in group if item.came is True]
        misses = [item.strength for item in group if item.came is False]
        if hits:
            came, strength = True, max(hits)
        elif misses:
            came, strength = False, max(misses)
        else:
            continue
        past = PastMoment(min(item.formed_at for item in group), present(group), came, strength)
        found.append(Moment(past, tuple(sorted(item.expectation_id for item in group))))
    return tuple(found)


def build_history(
    lane: str,
    night: date,
    records: Iterable[MomentRecord],
    *,
    table: OpportunityTable,
    line: LaneTimeline,
    statuses: Iterable[str],
) -> tuple[UnitHistory, Mapping[str, tuple[Moment, ...]]]:
    """这一晚每个行为历史上的每一刻（在场的输入按当晚的关系表重标）；连同每一刻由哪几条预期组成（退出用）。"""

    wanted = tuple(statuses)
    by_kind: dict[str, list[MomentRecord]] = {}
    for item in records:
        by_kind.setdefault(item.kind_token, []).append(item)

    def present(group: Sequence[MomentRecord]) -> frozenset[str]:
        return present_inputs(group, table=table, line=line, statuses=wanted)

    labelled = {
        kind: found for kind, items in sorted(by_kind.items()) if (found := labelled_moments(items, present=present))
    }
    return UnitHistory(lane, night, labelled), labelled


def retired(moments: Sequence[Moment], *, window: int, before: datetime) -> frozenset[str]:
    """一个行为里可以删的那几刻的预期身份：早于 ``before``（诊断够得着的最早一刻），并且在它所属的每一层里，排在它之后的都已经
    有 ``window`` 刻（单元每层只读最近 M 刻，``units.unit_reading``）——删了它，哪一层的读数都不变。

    它所属的层：同一组（在场的那一组与它相同）、同样多（抬高、压低各几个都一样），以及"同样多里除掉别的某一组"（问那一组时
    拿来当先验）。最后一种不用单独查：它总含着同一组里排在后面的全部，同一组够了它就够了。少见的组合因此留得久——别的组合
    再多再密，也挤不掉它自己那几刻。"""

    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ForesightError("window must be a positive integer")
    found: set[str] = set()
    for moment in moments:
        past = moment.past
        if past.at >= before:
            continue
        layer = [other.past for other in moments if (other.past.ups, other.past.downs) == (past.ups, past.downs)]
        order = {
            id(item.item): item.rank for item in rank_by(layer, key=lambda item: (item.at, sorted(item.signature)))
        }
        newer = [other for other in layer if order[id(other)] > order[id(past)]]
        if len(newer) >= window and sum(1 for other in newer if other.signature == past.signature) >= window:
            found.update(moment.expectation_ids)
    return frozenset(found)


def encode_history(history: UnitHistory) -> str:
    payload: dict[str, Any] = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "lane": history.lane,
        "night": history.night.isoformat(),
        "moments": {
            kind: [
                {
                    "at": item.past.at.isoformat(),
                    "signature": sorted(item.past.signature),
                    "came": item.past.came,
                    "weight": item.past.weight,
                    "expectation_ids": list(item.expectation_ids),
                }
                for item in found
            ]
            for kind, found in sorted(history.moments.items())
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_history(text: str, *, lane: str, night: date) -> UnitHistory:
    try:
        raw = json.loads(text)
        if raw.get("schema_version") != HISTORY_SCHEMA_VERSION:
            raise ValueError(f"schema_version {raw.get('schema_version')!r}")
        if raw["lane"] != lane or date.fromisoformat(raw["night"]) != night:
            raise ValueError("lane or night does not match the file")
        found = {
            str(kind): tuple(
                Moment(
                    PastMoment(
                        datetime.fromisoformat(item["at"]),
                        frozenset(str(token) for token in item["signature"]),
                        bool(item["came"]),
                        float(item["weight"]),
                    ),
                    tuple(str(token) for token in item["expectation_ids"]),
                )
                for item in items
            )
            for kind, items in raw["moments"].items()
        }
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ForesightError(f"unit history of lane {lane!r} for night {night} is malformed: {exc}") from exc
    return UnitHistory(lane, night, found)


__all__ = [
    "HISTORY_SCHEMA_VERSION",
    "Moment",
    "MomentRecord",
    "UnitHistory",
    "build_history",
    "decode_history",
    "decode_record",
    "encode_history",
    "encode_record",
    "labelled_moments",
    "present_inputs",
    "record_of",
    "retired",
]
