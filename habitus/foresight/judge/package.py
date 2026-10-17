"""模型小包的材料（纯函数、对模型零知识）：每个候选带今天这次、历史那几次（原话、时刻、隔多久）、时窗、前因清单、账的级。

原话从行为树的行来（``Seen.name`` / ``summary``），由组合根按 URI 读好交进来；读不到的只给 URI 的尾巴。
日型（当地日历对那一天的说法，"补班日，按周一上班"）给今天与历史每一次各标一句，由模型在"像不像"里判，系统不按日型增删候选
（``scene.calendar`` 的既有裁定：日型是摆在判断者面前的事实，不参与筛选；裁定 52）。没有日历数据时恒为空，不出现在小包里。
历史只给最近 ``max_history`` 次（小包要小：几千字），次数与机会数另给整数。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from habitus.foresight.errors import ForesightError
from habitus.foresight.ledger.model import Expectation, anchor_uri
from habitus.foresight.model import Seen
from habitus.foresight.opportunities.model import Opportunity, OpportunityTable, RelationOpportunities
from habitus.foresight.triggers.days import DAY_OF_MONTH
from habitus.foundation.text import clean_line

SEGMENT_TEXT = {"chain": "紧接着", "rest_of_day": "当天之后", "next_day": "次日", "days_2_7": "之后 2–7 天内"}


@dataclass(frozen=True)
class HistoryRow:
    """历史上的一次：哪天、前因原话（钟面 / 日子没有）、后果原话、后果几点、隔了多久（分钟；按天的段是隔几天）。"""

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
        relation = _relation(table, item) if item.source == "事件" else None
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
                trigger_text=_trigger_text(item, relation, rows, now),
                window=item.window,
                opportunities=item.opportunities,
                hits=item.hits,
                history=history,
                antecedents=_antecedents(table, item, rows),
                level=levels.get(item.expectation_id, 0),
                today_note=day_note(now.date()) if day_note is not None else None,
                inputs=tuple(_input_line(table, unit, rows) for unit in (units or {}).get(item.expectation_id, ())),
                causes=_causes(table, (units or {}).get(item.expectation_id, ()), rows),
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
        relation = _relation(table, item) if item.source == "事件" else None
        if relation is not None:
            for opportunity in _hits(relation)[-max_history:]:
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


def _input_line(table: OpportunityTable | None, unit: UnitInput, rows: Mapping[str, Seen]) -> str:
    item = unit.expectation
    relation = _relation(table, item) if item.source == "事件" else None
    if relation is not None:
        what = f"{_name(rows, item.trigger_instance)}（{relation.antecedent_label}，{SEGMENT_TEXT.get(relation.segment, '')}）"
    elif item.source == "钟面":
        what = f"周{'一二三四五六日'[item.window[0].weekday()]}这会儿"
    elif item.trigger == DAY_OF_MONTH:
        what = f"每月 {item.window[0].day} 号"
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
        reading = relation.reading if relation is not None else None
        if relation is None or reading is None or relation.status != "established":
            continue
        if reading.large_enough is True and reading.lift > 0.0:
            found.append(_name(rows, unit.expectation.trigger_instance))
    return tuple(dict.fromkeys(found))


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
    if candidate.source == "钟面":
        return f"上次这会儿你{label}了，今天还没有。" if once else f"往常这会儿你会{label}，今天还没有。"
    if candidate.trigger_text.startswith("到了日子"):
        day = candidate.window[0].day
        return f"上个月 {day} 号你{label}了，今天还没有。" if once else f"往常每月 {day} 号你会{label}，今天还没有。"
    return f"上次隔这么久你{label}了，这次还没有。" if once else f"往常隔这么久你会{label}，这次还没有。"


def _relation(table: OpportunityTable | None, item: Expectation) -> RelationOpportunities | None:
    if table is None:
        return None
    return next((relation for relation in table.relations if relation.key == item.trigger), None)


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
        for opportunity in _hits(relation)[-limit:]:
            assert opportunity.consequent_uri is not None and opportunity.consequent_at is not None
            if relation.segment in ("chain", "rest_of_day"):
                gap = f"{int((opportunity.consequent_at - opportunity.ended_at).total_seconds() // 60)} 分钟后"
            else:
                gap = f"第 {(opportunity.consequent_at.date() - opportunity.day).days} 天"
            found.append(
                HistoryRow(
                    0,
                    opportunity.day,
                    _name(rows, opportunity.antecedent_uri),
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


def _trigger_text(
    item: Expectation, relation: RelationOpportunities | None, rows: Mapping[str, Seen], now: datetime
) -> str:
    if item.source == "事件":
        row = _row(rows, item.trigger_instance)
        when = f"{row.started_at.astimezone(now.tzinfo):%m-%d %H:%M}" if row is not None else ""
        segment = SEGMENT_TEXT.get(relation.segment, "") if relation is not None else ""
        return (
            f"刚发生：{_name(rows, item.trigger_instance)}（{when}）；关系说{segment}会{_label(None, item, relation)}"
        )
    if item.source == "钟面":
        return f"到了钟点：周{'一二三四五六日'[item.window[0].weekday()]} {item.window[0]:%H:%M}–{item.window[1]:%H:%M} 往常这会儿做"
    if item.trigger == DAY_OF_MONTH:
        return f"到了日子：往月 {item.window[0].day} 号做过"
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
        relation = next((r for r in table.relations if r.key == antecedent.relation), None)
        if relation is None:
            continue
        arrow = "↑" if antecedent.upward else "↓"
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
