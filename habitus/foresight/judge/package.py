"""模型小包的材料（纯函数、对模型零知识）：每个候选带今天这次、历史那几次（原话、时刻、隔多久）、时窗、前因清单、账的级。

原话从行为树的行来（``Seen.name`` / ``summary``），由组合根按 URI 读好交进来；读不到的只给 URI 的尾巴。
日型（当地日历对那一天的说法，"调休上班日"；外部条件包给，``conditions.day_note``）给今天与历史每一次各标一句，由模型在
"像不像"里判，系统不按日型增删候选（裁定 52）；没有放假数据的日子标"日型未知"，不让模型当成工作日。锚在日历上的规律不靠这句：
它是语义树上的日历情境关系，走情境触发（裁定 59）。
历史只给最近 ``max_history`` 次（小包要小：几千字），次数与机会数另给整数。扎堆合并成一次的几条记录，给的是第一条的原话
（机会表上前因与后果的出处都是那一次的第一条）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from habitus.conditions import DAY_TYPE_KEY, UNKNOWN_DAY_NOTE, WEEKDAY_KEY, WEEKDAY_NAMES
from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.model import Expectation, anchor_uri, relation_backed
from habitus.foresight.model import Seen
from habitus.foresight.opportunities.model import Opportunity, OpportunityTable, RelationOpportunities
from habitus.foundation.text import clean_line

SEGMENT_TEXT = {
    "chain": "紧接着",
    "rest_of_day": "当天之后",
    "next_day": "次日",
    "days_2_7": "之后 2–7 天内",
    "situation": "当天",
}


@dataclass(frozen=True)
class HistoryRow:
    """历史上的一次：哪天、前因原话（情境是那个日历情境的名字，日子没有）、后果原话、后果几点、隔了多久（分钟；按天的段
    是隔几天；情境是"当天"）。"""

    number: int
    day: date
    antecedent: str
    consequent: str
    consequent_at: datetime
    gap: str
    uri: str
    note: str | None = None


@dataclass(frozen=True)
class Candidate:
    number: int
    expectation_id: str
    label: str
    source: str
    trigger_text: str
    window: tuple[datetime, datetime]
    opportunities: int
    hits: int
    history: tuple[HistoryRow, ...]
    antecedents: tuple[str, ...]
    level: int
    today_note: str | None = None
    #: 这一刻这个行为单元的全部输入（裁定 56 第 12 条），每条一句：是什么、它那本账从它出发说中几次、语义树上有它 / 没它时的发生率。
    inputs: tuple[str, ...] = ()
    #: 在场的事件前因里成了依据的那几件的原话（两件以上时话里一起说出来："你订了机票、也订了酒店"）。
    causes: tuple[str, ...] = ()
    #: 今天处在的、语义树判成原因的日历情境（"周三"）：已成立的日历关系也是原因，和事件原因分开放，话里说"往常周三"。
    situations: tuple[str, ...] = ()

    def uri_of(self, number: int) -> str | None:
        return next((row.uri for row in self.history if row.number == number), None)


@dataclass(frozen=True)
class Package:
    at: datetime
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ForesightError("a judge package needs at least one candidate")
        if [item.number for item in self.candidates] != list(range(1, len(self.candidates) + 1)):
            raise ForesightError("judge package candidates must be numbered 1..n")

    def by_number(self, number: int) -> Candidate | None:
        return next((item for item in self.candidates if item.number == number), None)


def build_package(
    expectations: Sequence[Expectation],
    table: OpportunityTable | None,
    rows: Mapping[str, Seen],
    *,
    now: datetime,
    levels: Mapping[str, int],
    max_history: int = 6,
    day_note: Callable[[date], str | None] | None = None,
    units: Mapping[str, Sequence[UnitInput]] | None = None,
) -> Package:
    """``rows``：出处（URI 或 ``unsealed:<开始时刻>``）→ 那一行；``levels``：预期身份 → 这本账的级；``day_note``：当地日历
    对某一天的说法（没有就不标）；``units``：依据那条预期的身份 → 这一刻单元的全部输入（含它自己）。"""

    if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 1:
        raise ForesightError("max_history must be a positive integer")
    candidates = []
    for number, item in enumerate(expectations, start=1):
        relation = _relation(table, item) if relation_backed(item.source) else None
        label = _label(table, item, relation)
        history = _history(item, relation, rows, max_history)
        if day_note is not None:
            history = tuple(
                HistoryRow(
                    row.number,
                    row.day,
                    row.antecedent,
                    row.consequent,
                    row.consequent_at,
                    row.gap,
                    row.uri,
                    day_note(row.day),
                )
                for row in history
            )
        candidates.append(
            Candidate(
                number=number,
                expectation_id=item.expectation_id,
                label=label,
                source=item.source,
                trigger_text=_trigger_text(
                    item, relation, rows, now, today_note=day_note(now.date()) if day_note is not None else None
                ),
                window=item.window,
                opportunities=item.opportunities,
                hits=item.hits,
                history=history,
                antecedents=_antecedents(table, item, rows),
                level=levels.get(item.expectation_id, 0),
                today_note=day_note(now.date()) if day_note is not None else None,
                inputs=tuple(
                    _input_line(table, unit, rows, now) for unit in (units or {}).get(item.expectation_id, ())
                ),
                causes=_causes(table, (units or {}).get(item.expectation_id, ()), rows),
                situations=_situations(table, (units or {}).get(item.expectation_id, ())),
            )
        )
    return Package(at=now, candidates=tuple(candidates))


def sources_of(
    expectations: Sequence[Expectation],
    table: OpportunityTable | None,
    *,
    max_history: int = 6,
    units: Mapping[str, Sequence[UnitInput]] | None = None,
) -> frozenset[str]:
    """装小包要读的出处：今天的触发、历史那几次（与 ``build_package`` 选的是同一批）、前因清单里在场的那一条。
    组合根按它读行，前因原话与后果原话就不会因为两处各取各的而对不上（对不上时小包里只剩 URI 的尾巴）。"""

    found: set[str] = set()
    for item in expectations:
        found.add(anchor_uri(item.trigger_instance))
        found |= {antecedent.source for antecedent in item.antecedents if antecedent.source}
        relation = _relation(table, item) if relation_backed(item.source) else None
        if relation is not None:
            for opportunity in _hits(relation)[-max_history:]:
                if relation.segment != "situation":  # 情境机会的前因是那一天本身（calendar:<日子>），不是一条记录
                    found.add(opportunity.antecedent_uri)
                if opportunity.consequent_uri is not None:
                    found.add(opportunity.consequent_uri)
        else:
            found |= set(item.basis[-max_history:])
        for unit in (units or {}).get(item.expectation_id, ()):
            found.add(anchor_uri(unit.expectation.trigger_instance))
    return frozenset(found)


@dataclass(frozen=True)
class UnitInput:
    """单元的一个输入给小包的样子：那条预期与它那本账此刻的预测轴读数（说中几次、几次机会）。"""

    expectation: Expectation
    hits: float
    chances: float


def _input_line(table: OpportunityTable | None, unit: UnitInput, rows: Mapping[str, Seen], now: datetime) -> str:
    item = unit.expectation
    relation = _relation(table, item) if relation_backed(item.source) else None
    if relation is not None and relation.segment == "situation":
        what = f"{_today_in(relation, now)}（日历情境）"
    elif relation is not None:
        what = f"{_name(rows, item.trigger_instance)}（{relation.antecedent_label}，{SEGMENT_TEXT.get(relation.segment, '')}）"
    else:
        what = "离上次够久"
    line = f"{what}：从它出发说中 {unit.hits:.0f} / {unit.chances:.0f} 次"
    if relation is not None and relation.reading is not None:
        line += f"；有它时 {relation.reading.treated:.0%}、没它时 {relation.reading.control:.0%}"
    return line


def _causes(table: OpportunityTable | None, units: Sequence[UnitInput], rows: Mapping[str, Seen]) -> tuple[str, ...]:
    """在场的事件前因里语义树判成原因（效应够大、抬升为正、关系成立）的那几件的原话。"""

    found = []
    for unit in units:
        relation = _relation(table, unit.expectation) if unit.expectation.source == "事件" else None
        if relation is not None and _is_cause(relation):
            found.append(_name(rows, unit.expectation.trigger_instance))
    return tuple(dict.fromkeys(found))


def _situations(table: OpportunityTable | None, units: Sequence[UnitInput]) -> tuple[str, ...]:
    """今天处在的日历情境里语义树判成原因的那几个（"周三"）。"""

    found = []
    for unit in units:
        relation = _relation(table, unit.expectation) if unit.expectation.source == "情境" else None
        if relation is not None and _is_cause(relation):
            found.append(relation.antecedent_label)
    return tuple(dict.fromkeys(found))


def _is_cause(relation: RelationOpportunities) -> bool:
    """关系成立、语义树判效应够大、抬升为正。"""

    reading = relation.reading
    return (
        relation.status == "established" and reading is not None and reading.large_enough is True and reading.lift > 0.0
    )


def plain_text(candidate: Candidate) -> str:
    """模型不可用时的措辞：跟证据走（一次说「上次」，多次说「往常」）；说的时候这件事还没做（做了的不递），所以带「还没」。"""

    label = candidate.label
    once = candidate.hits <= 1
    if len(candidate.causes) >= 2:
        return f"你{'、'.join(candidate.causes)}了，往常接着你会{label}，这次还没有，要吗？"
    if candidate.source == "事件":
        if candidate.history:
            antecedent = candidate.history[-1].antecedent or "那件事"
            return (
                f"上次{antecedent}之后你{label}了，这次还没有，要吗？"
                if once
                else f"往常{antecedent}之后你会{label}，这次还没有，要吗？"
            )
        return f"要{label}吗？"
    if candidate.source == "情境":
        when = "、".join(candidate.situations) or (candidate.history[-1].antecedent if candidate.history else "")
        return f"往常{when}你会{label}，今天还没有。" if when else f"往常这会儿你会{label}，今天还没有。"
    return f"上次隔这么久你{label}了，这次还没有。" if once else f"往常隔这么久你会{label}，这次还没有。"


def _relation(table: OpportunityTable | None, item: Expectation) -> RelationOpportunities | None:
    if table is None:
        return None
    return table.relation(item.trigger)


def _label(table: OpportunityTable | None, item: Expectation, relation: RelationOpportunities | None) -> str:
    if relation is not None:
        return relation.consequent_label
    kind = table.kind(item.kind_token) if table is not None else None
    return kind.label if kind is not None else item.kind_token


def _row(rows: Mapping[str, Seen], source: str) -> Seen | None:
    """按出处找行；事件触发的实例是前因的开始时刻（UTC ISO），按时刻找。"""

    row = rows.get(source)
    if row is not None:
        return row
    return next((item for item in rows.values() if item.started_at.astimezone(UTC).isoformat() == source), None)


def _name(rows: Mapping[str, Seen], source: str) -> str:
    """原话（融合从对话里写出的文本）折成一行：换行、控制字符去掉，不让一段原话在小包里伪造出一节候选。"""

    row = _row(rows, source)
    if row is not None and row.name:
        return clean_line(row.name) or "（无名）"
    tail = source.rsplit("/", 1)[-1]
    return clean_line(tail.split("--", 1)[0]) or clean_line(source)


def _hits(relation: RelationOpportunities) -> list[Opportunity]:
    return [opportunity for opportunity in relation.items if opportunity.came and opportunity.consequent_at is not None]


def _history(
    item: Expectation, relation: RelationOpportunities | None, rows: Mapping[str, Seen], limit: int
) -> tuple[HistoryRow, ...]:
    found: list[HistoryRow] = []
    if relation is not None:
        situation = relation.segment == "situation"
        for opportunity in _hits(relation)[-limit:]:
            assert opportunity.consequent_uri is not None and opportunity.consequent_at is not None
            if situation:
                gap = "当天"
            elif relation.segment in ("chain", "rest_of_day"):
                gap = f"{int((opportunity.consequent_at - opportunity.ended_at).total_seconds() // 60)} 分钟后"
            else:
                gap = f"第 {(opportunity.consequent_at.date() - opportunity.day).days} 天"
            found.append(
                HistoryRow(
                    0,
                    opportunity.day,
                    relation.antecedent_label if situation else _name(rows, opportunity.antecedent_uri),
                    _name(rows, opportunity.consequent_uri),
                    opportunity.consequent_at,
                    gap,
                    opportunity.consequent_uri,
                )
            )
    else:
        for uri in item.basis[-limit:]:
            row = rows.get(uri)
            at = row.started_at if row is not None else item.window[0]
            found.append(HistoryRow(0, at.date(), "", _name(rows, uri), at, "", uri))
    return tuple(
        HistoryRow(number, row.day, row.antecedent, row.consequent, row.consequent_at, row.gap, row.uri)
        for number, row in enumerate(found, start=1)
    )


def _today_in(relation: RelationOpportunities, now: datetime) -> str:
    """ "今天是周三"；周几集合说今天是哪一天、属于哪个集合——"今天是周一、周三、周六"不对，今天只是其中一天（2026-10-18 四方评审）。"""

    if relation.calendar_key == WEEKDAY_KEY and len(relation.calendar_values) > 1:
        return f"今天是{WEEKDAY_NAMES[now.weekday()]}（属于「{relation.antecedent_label}」）"
    return f"今天是{relation.antecedent_label}"


def _trigger_text(
    item: Expectation,
    relation: RelationOpportunities | None,
    rows: Mapping[str, Seen],
    now: datetime,
    *,
    today_note: str | None = None,
) -> str:
    if item.source == "事件":
        row = _row(rows, item.trigger_instance)
        when = f"{row.started_at.astimezone(now.tzinfo):%m-%d %H:%M}" if row is not None else ""
        segment = SEGMENT_TEXT.get(relation.segment, "") if relation is not None else ""
        return (
            f"刚发生：{_name(rows, item.trigger_instance)}（{when}）；关系说{segment}会{_label(None, item, relation)}"
        )
    if item.source == "情境":
        today = _today_in(relation, now) if relation is not None else "今天处在这个日历情境里"
        if relation is not None and relation.calendar_key == DAY_TYPE_KEY and today_note == UNKNOWN_DAY_NOTE:
            # 缺放假数据时日型是按名义周末给的：照实说，不让触发句与抬头的"日型未知"自相矛盾（2026-10-18 四方评审）
            today = f"今天按名义周末算是{relation.antecedent_label}（缺放假数据，日型未知）"
        return (
            f"处在情境里：{today}；关系说当天会{_label(None, item, relation)}，"
            f"往常 {item.window[0]:%H:%M}–{item.window[1]:%H:%M}"
        )
    last = _row(rows, anchor_uri(item.trigger_instance))
    since = (
        f"离上次（{last.started_at:%m-%d}）已 {(now.date() - last.day).days} 天" if last is not None else "离上次够久"
    )
    return since


def _antecedents(table: OpportunityTable | None, item: Expectation, rows: Mapping[str, Seen]) -> tuple[str, ...]:
    if table is None:
        return ()
    lines = []
    for antecedent in item.antecedents:
        relation = table.relation(antecedent.relation)
        if relation is None:
            continue
        arrow = "↑" if antecedent.upward else "↓"
        if relation.segment == "situation":
            state = (
                f"今天是{relation.antecedent_label}" if antecedent.present else f"今天不是{relation.antecedent_label}"
            )
            lines.append(f"{'在场' if antecedent.present else '不在场'} {arrow}：{state}（日历情境）")
            continue
        if antecedent.present:
            lines.append(
                f"在场 {arrow}：{relation.antecedent_label}（{_name(rows, antecedent.source)}）{SEGMENT_TEXT.get(relation.segment, '')}"
            )
        else:
            lines.append(f"不在场 {arrow}：{relation.antecedent_label}{SEGMENT_TEXT.get(relation.segment, '')}")
    return tuple(lines)


__all__ = [
    "SEGMENT_TEXT",
    "Candidate",
    "HistoryRow",
    "Package",
    "UnitInput",
    "build_package",
    "plain_text",
    "sources_of",
]
