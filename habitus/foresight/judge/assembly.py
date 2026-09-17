"""把模型的判断输出装配成 ``Judgement``：只校验我们自己产物的自洽，违反者**降级留信号**。

沿关联装配层的纪律：引用一张包里没有的卡、`next` 写一个所引卡"之后"里没有的名字、时窗落在今天之外
或整个早于此刻、起槽早于此刻（截到此刻）、「不会」却带时窗、「会」却一张卡都不引——这些是组合式记账，
不是语义判断，一律降级（丢那一项、置 null、截断、降成说不准）而不整批拒。候选没答到的降成「说不准」
并留信号；``day_state`` 不是两者之一就记 None（"没答"，不是"正常"）。只有整个输出不成形（不是对象、
没有 verdicts、或者一条都对不上任何候选）才整批不可用，让结构层重试。

**不校验**现实的形状：「会」却没有时窗、`next` 为空、`note` 为空——全都照收。

``ASSEMBLY_VERSION`` 随这里的降级规则一起改：它是判断者版本的一部分——同一份模型答复经不同的装配
纪律会得到不同的判断，存下来的判断要能说清是哪一版装出来的。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from habitus.foresight.assemble import CandidateEvidence, EvidencePack
from habitus.foresight.cards import HistoryCard
from habitus.foresight.errors import ForesightError
from habitus.foresight.judge.model import DAY_STATES, VERDICTS, CandidateVerdict, Judgement
from habitus.foundation.text import clean_line

ASSEMBLY_VERSION = "assembly_v2"


class JudgeAssemblyError(ValueError):
    """模型输出的形状连降级都救不回来。派生自 ValueError：结构层只把 ValueError/TypeError 当"答错了，纠正重问"。"""


@dataclass
class _Signals:
    notes: list[str] = field(default_factory=list)

    def add(self, note: str) -> None:
        self.notes.append(note)


def assemble_judgement(parsed: object, pack: EvidencePack, *, judged_at: datetime, version: str) -> Judgement:
    """一条流水线：按候选收条目 → 逐条整理判决、依据、接下来、时窗 → 补上没答到的 → 今天整体。

    产物类型的任何自洽错误（``ForesightError``）在这里都归成 ``JudgeAssemblyError``：那是"这份答复装不出
    合格的判断"，该让结构层纠正重问，而不是让整拍失败。
    """

    try:
        return _assemble(parsed, pack, judged_at=judged_at, version=version)
    except ForesightError as exc:
        raise JudgeAssemblyError(f"judgement output does not assemble: {exc}") from exc


def _assemble(parsed: object, pack: EvidencePack, *, judged_at: datetime, version: str) -> Judgement:
    if not isinstance(parsed, Mapping):
        raise JudgeAssemblyError("judgement output must be an object")
    raw = parsed.get("verdicts")
    if not isinstance(raw, list):
        raise JudgeAssemblyError("judgement output must carry a verdicts array")
    expanded = {item.kind_token: item for item in pack.expanded}
    signals = _Signals()
    entries = _entries_by_kind(raw, expanded, signals)
    if expanded and not entries:
        raise JudgeAssemblyError("judgement output says nothing about any candidate in the pack")
    verdicts = []
    for kind in sorted(expanded):
        item = entries.get(kind)
        if item is None:
            signals.add(f"unanswered: {kind} got no verdict; recorded as 说不准")
            verdicts.append(
                CandidateVerdict(kind_token=kind, verdict="说不准", window=None, next=(), basis=(), note="")
            )
        else:
            verdicts.append(_verdict(item, expanded[kind], pack, signals))
    day_state, day_note = _day(parsed, signals)
    return Judgement(
        judged_at=judged_at,
        generation=pack.generation,
        moment=pack.moment,
        verdicts=tuple(verdicts),
        day_state=day_state,
        day_note=day_note,
        judge_version=version,
        signals=tuple(signals.notes),
    )


def _entries_by_kind(
    raw: list[Any], expanded: Mapping[str, CandidateEvidence], signals: _Signals
) -> dict[str, Mapping[str, Any]]:
    """每个摊开的候选至多一条：不认识的名字丢、重复的丢后来的。"""

    entries: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            signals.add("verdict_dropped: malformed entry")
            continue
        kind = item.get("kind_token")
        if not isinstance(kind, str) or kind not in expanded:
            signals.add(f"verdict_dropped: {kind!r} is not an expanded candidate in this pack")
            continue
        if kind in entries:
            signals.add(f"verdict_dropped: {kind} appears more than once")
            continue
        entries[kind] = item
    return entries


def _verdict(
    item: Mapping[str, Any], evidence: CandidateEvidence, pack: EvidencePack, signals: _Signals
) -> CandidateVerdict:
    kind = evidence.kind_token
    verdict = item.get("verdict")
    if verdict not in VERDICTS:
        signals.add(f"verdict_degraded: {kind} says {verdict!r}; recorded as 说不准")
        verdict = "说不准"
    cited = _cited_cards(item.get("basis"), evidence, signals)
    if verdict == "会" and not cited:
        # 只看数字不看卡的"会"无从核对："此刻像哪几次"是这一层的全部依据。
        signals.add(f"verdict_degraded: {kind} says 会 but cites no card; recorded as 说不准")
        verdict = "说不准"
    return CandidateVerdict(
        kind_token=kind,
        verdict=verdict,
        window=_window(item.get("window"), kind, verdict, pack, signals),
        next=_next(item.get("next"), kind, cited, signals),
        basis=tuple(card.uri for card in cited),
        note=clean_line(item.get("note")),
    )


def _cited_cards(value: object, evidence: CandidateEvidence, signals: _Signals) -> tuple[HistoryCard, ...]:
    """卡的编号换回卡本身：编号按这个候选自己的卡序，从 1 起。"""

    kind = evidence.kind_token
    cards = evidence.background.cards
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        signals.add(f"basis_dropped: {kind} basis is not an array")
        return ()
    kept: list[int] = []
    for raw in value:
        number = _index(raw, signals, f"{kind} basis")
        if number is None or number > len(cards):
            signals.add(f"basis_dropped: {kind} cites card {raw!r}, which is not in this pack")
        elif number in kept:
            signals.add(f"basis_dropped: {kind} repeats card #{number}")
        else:
            kept.append(number)
    return tuple(cards[number - 1] for number in sorted(kept))


def _next(value: object, kind: str, cited: tuple[HistoryCard, ...], signals: _Signals) -> tuple[str, ...]:
    """接下来只能是所引卡"之后"那段里出现过的名字。"""

    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        signals.add(f"next_dropped: {kind} next is not an array")
        return ()
    seen = {row.name for card in cited for row in card.after}
    kept: list[str] = []
    for raw in value:
        name = clean_line(raw)
        if not name or name not in seen:
            signals.add(f"next_dropped: {kind} names {raw!r}, which follows none of the cited cards")
        elif name not in kept:
            kept.append(name)
    return tuple(kept)


def _window(value: object, kind: str, verdict: str, pack: EvidencePack, signals: _Signals) -> tuple[int, int] | None:
    """时窗要落在今天；整个早于此刻的置 None，起槽早于此刻的截到此刻；「不会」没有时窗。"""

    if value is None:
        return None
    if verdict == "不会":
        signals.add(f"window_dropped: {kind} is judged 不会 yet carries a window")
        return None
    if not isinstance(value, Mapping):
        signals.add(f"window_dropped: {kind} window is not an object")
        return None
    start = _slot(value.get("from_slot"), signals, f"{kind} window from_slot")
    end = _slot(value.get("to_slot"), signals, f"{kind} window to_slot")
    if start is None or end is None or start > end or end >= pack.slots_per_day:
        signals.add(f"window_dropped: {kind} window {value!r} does not fit today's clock face")
        return None
    if end < pack.moment.slot:
        signals.add(f"window_dropped: {kind} window {value!r} ends before the current slot")
        return None
    if start < pack.moment.slot:
        # 时窗说的是"从此刻起什么时候开始"，此刻之前那一截没有意义；截掉、留信号，不整条作废。
        signals.add(f"window_clamped: {kind} window {value!r} started before the current slot; clamped")
        start = pack.moment.slot
    return start, end


def _day(parsed: Mapping[str, Any], signals: _Signals) -> tuple[str | None, str | None]:
    """今天整体正不正常。答不上来就记 None——"没答"不能被写成"正常"。"""

    state = parsed.get("day_state")
    if state not in DAY_STATES:
        signals.add(f"day_state_dropped: {state!r} is not a day state; recorded as not answered")
        state = None
    note = clean_line(parsed.get("day_note")) or None
    return state, note


def _index(value: object, signals: _Signals, label: str) -> int | None:
    """编号的宽容解析：接受整数，也接受 ``3.0`` 这种整值浮点（JSON 修复路径够得着）。"""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer() and value > 0:
        signals.add(f"index_coerced: {label} arrived as {value!r}")
        return int(value)
    return None


def _slot(value: object, signals: _Signals, label: str) -> int | None:
    """槽号从 0 起，与编号不同。"""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        signals.add(f"index_coerced: {label} arrived as {value!r}")
        return int(value)
    return None


__all__ = ["ASSEMBLY_VERSION", "JudgeAssemblyError", "assemble_judgement"]
