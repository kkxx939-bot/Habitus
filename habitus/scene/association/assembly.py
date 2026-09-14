"""把模型的关联输出装配成可落盘的草稿：只校验我们自己产物的自洽，违反者**降级留信号**。

沿归组装配层的既定纪律：记账疏漏（引用一个输入里没有的编号、前因不在依据里、前因比这次还晚、
指向一个不存在的情境、把同一条前提兑现两次）是组合式记账、不是语义判断，实测提示词压不住，
一律降级——丢那一项、置 null——不整批拒，整批拒要白烧一次完整调用。只有**穷尽性**被破坏
（一条草稿都没剩下）才整批不可用：那不是某一项写坏了，是这次调用没回答问题。

**不校验**现实的形状：一次发生只引用一条记录、看不出前因、情境只覆盖过一次、``left`` 为空——
全都照收。特别地，``consumed_by`` 不拿 kinds 词表核对：一个还没在树上出现过的种类是完全可能的，
拿词表把它判掉就是在规定现实该长什么样。它也**不是检索键**——这次用掉了哪几条前提由模型返回的
``consumed`` 编号决定，``consumed_by`` 只是渲染给人和模型看的提示。

引用用的是**一套编号**：当天的流占 1..N，前因候选接在后面（见 ``AssociationInput.citable``）。
草稿里原样保留这套编号，换成 URI 是编排层的事——装配只判断"这个编号在输入里有没有、它是不是
真的更早"。

**编排层将来要做的映射**（现在写死在这里，免得第 4 步现场发明）：``causes`` → ``results_from``
边，指向那条行为的 URI（当天的流取 ``AssociationInput.row(no).uri``，前因候选先经 ``cause_no``
还原再取 ``CauseRow.uri``）；``consumed`` → ``needs`` 边，指向 ``PendingRow.producer_uri``。
``cites`` 只作提示词纪律与本层校验用，不落盘。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from habitus.foundation.ids import canonical_text_identity
from habitus.scene.association.model import AssociationAssembly, AssociationDraft, AssociationInput
from habitus.scene.text import clean_line

#: ``consumed_by`` 与前提正文的字节上界，以及一次发生最多铺几条前提。
#: 这三道约束的是"我们自己存不存得下、渲不渲染得出"，不是现实该长什么样——记录整体有
#: ``MAX_RECORD_BYTES`` 这道硬闸，但撞上它是整条记录写不进去、那一天永远完不成，而调用方
#: 对"是 left 太多把记录顶爆了"一无所知。降级要发生在这里：丢那一项、留信号。
MAX_CONSUMED_BY_BYTES = 200
MAX_PREMISE_TEXT_BYTES = 2_000
MAX_PREMISES_PER_OCCURRENCE = 16


class AssociationAssemblyError(ValueError):
    """模型输出的形状连降级都救不回来（一条草稿都没剩下）。"""


@dataclass
class _Signals:
    notes: list[str] = field(default_factory=list)

    def add(self, note: str) -> None:
        self.notes.append(note)


def assemble_association(parsed: object, payload: AssociationInput) -> AssociationAssembly:
    """一条流水线：收条目 → 逐条整理引用、前因、情境、前提 → 收拢没回答的目标。"""

    if not isinstance(parsed, Mapping):
        raise AssociationAssemblyError("association output must be an object")
    raw = parsed.get("entries")
    if not isinstance(raw, list):
        raise AssociationAssemblyError("association output must carry an entries array")
    signals = _Signals()
    entries = _entries_by_target(raw, payload, signals)
    drafts: list[AssociationDraft] = []
    unanswered: list[int] = []
    # 前提是跨条目的记账：一条只能被兑现一次，先到先得，顺序按目标本身的顺序。
    unspent = {row.no for row in payload.pending}
    for target in payload.targets:
        draft = _draft(entries[target], payload, unspent, signals)
        if draft is None:
            unanswered.append(target)
        else:
            drafts.append(draft)
    if not drafts:
        raise AssociationAssemblyError("association output left every target without a usable entry")
    return AssociationAssembly(drafts=tuple(drafts), unanswered=tuple(sorted(unanswered)), signals=tuple(signals.notes))


def _entries_by_target(raw: list[Any], payload: AssociationInput, signals: _Signals) -> dict[int, Mapping[str, Any]]:
    """穷尽性：每个目标恰好一条。重号丢后来的，缺号整批不可用。"""

    entries: dict[int, Mapping[str, Any]] = {}
    targets = set(payload.targets)
    for item in raw:
        if not isinstance(item, Mapping):
            signals.add("entry_dropped: malformed entry")
            continue
        no = _index(item.get("occurrence_no"), signals, "entry occurrence_no")
        if no is None:
            signals.add(f"entry_dropped: occurrence_no {item.get('occurrence_no')!r} is not an integer")
            continue
        if no not in targets:
            signals.add(f"entry_dropped: #{no} is not one of this call's targets")
            continue
        if no in entries:
            signals.add(f"entry_dropped: #{no} appears more than once")
            continue
        entries[no] = item
    missing = sorted(targets - set(entries))
    if missing:
        raise AssociationAssemblyError(f"association output says nothing about {', '.join(f'#{no}' for no in missing)}")
    return entries


def _draft(
    item: Mapping[str, Any], payload: AssociationInput, unspent: set[int], signals: _Signals
) -> AssociationDraft | None:
    no = int(item["occurrence_no"])
    context = clean_line(item.get("context"))
    if not context:
        signals.add(f"entry_dropped: #{no} has no usable context")
        return None
    cites = _numbers(item.get("cites"), payload.citable, f"#{no} cites", "not available here", signals)
    if not cites:
        # 依据是硬要求：一句话如果指不回任何一条记录，它就无从核对，留着等于留一句编出来的话。
        signals.add(f"entry_dropped: #{no} cites nothing that exists in the input")
        return None
    causes = _numbers(
        item.get("causes"), _earlier_than(no, cites, payload), f"#{no} causes", "not an earlier cited row", signals
    )
    situation_no, new_situation = _situation(item, no, payload, signals)
    consumed = _numbers(item.get("consumed"), frozenset(unspent), f"#{no} consumed", "not an unspent premise", signals)
    unspent -= set(consumed)
    return AssociationDraft(
        occurrence_no=no,
        context=context,
        cites=cites,
        causes=causes,
        situation_no=situation_no,
        new_situation=new_situation,
        consumed=consumed,
        left=_left(item.get("left"), no, signals),
    )


def _earlier_than(no: int, cites: tuple[int, ...], payload: AssociationInput) -> frozenset[int]:
    """前因只能是**不晚于**这次、且不是这次自己的那些依据。

    编号本身不足以定序：前因候选的编号接在当天的流后面，而它们发生在更早的日子。所以按时刻比，
    同刻放行（与边的 ``lag >= 0`` 同口径）。
    """

    instants = payload.instants
    here = instants[no]
    return frozenset(cited for cited in cites if cited != no and instants[cited] <= here)


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


def _numbers(value: object, allowed: frozenset[int], label: str, reason: str, signals: _Signals) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        signals.add(f"references_dropped: {label} is not an array")
        return ()
    kept: set[int] = set()
    for item in value:
        number = _index(item, signals, label)
        if number is None or number not in allowed:
            signals.add(f"reference_dropped: {label} names {item!r}, which is {reason}")
            continue
        if number in kept:
            signals.add(f"reference_dropped: {label} repeats #{number}")
            continue
        kept.add(number)
    return tuple(sorted(kept))


def _situation(
    item: Mapping[str, Any], no: int, payload: AssociationInput, signals: _Signals
) -> tuple[int | None, str | None]:
    """情境二选一：指一个已有的，或说一个新的。两个都给按已有的算，两个都没有就不归。"""

    raw = item.get("situation_no")
    text = clean_line(item.get("new_situation"))
    known = {row.no: row.text for row in payload.situations}
    if raw is not None:
        number = _index(raw, signals, f"#{no} situation_no")
        if number is not None and number in known:
            if text:
                signals.add(f"situation_degraded: #{no} names situation {number} and a new one; kept {number}")
            return number, None
        signals.add(f"situation_degraded: #{no} names unknown situation {raw!r}")
    if text:
        restated = _same_situation(text, known)
        if restated is not None:
            # 规范身份相同就是同一个地址上的同一件事——这是我们自己产物的自洽，不是"像不像"。
            signals.add(f"situation_degraded: #{no} restated existing situation {restated}")
            return restated, None
        return None, text
    signals.add(f"situation_degraded: #{no} belongs to no situation")
    return None, None


def _same_situation(text: str, known: Mapping[int, str]) -> int | None:
    try:
        identity = canonical_text_identity(text, "situation text")
    except (TypeError, ValueError):
        return None
    for number, existing in sorted(known.items()):
        try:
            if canonical_text_identity(existing, "situation text") == identity:
                return number
        except (TypeError, ValueError):
            continue
    return None


def _left(value: object, no: int, signals: _Signals) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list | tuple):
        if value is not None:
            signals.add(f"left_dropped: #{no} left is not an array")
        return ()
    kept: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            signals.add(f"left_dropped: #{no} has a malformed pending effect")
            continue
        text, waits_for = clean_line(item.get("text")), clean_line(item.get("consumed_by"))
        if not text or not waits_for:
            signals.add(f"left_dropped: #{no} pending effect {item.get('text')!r} says nothing or waits for nothing")
        elif len(waits_for.encode("utf-8")) > MAX_CONSUMED_BY_BYTES:
            signals.add(f"left_dropped: #{no} pending effect waits for something too long to store")
        elif len(text.encode("utf-8")) > MAX_PREMISE_TEXT_BYTES:
            signals.add(f"left_dropped: #{no} pending effect is too long to store")
        elif len(kept) >= MAX_PREMISES_PER_OCCURRENCE:
            signals.add(f"left_dropped: #{no} already left {MAX_PREMISES_PER_OCCURRENCE} premises")
        elif (text, waits_for) in kept:
            signals.add(f"left_dropped: #{no} repeats pending effect «{text}»")
        else:
            kept.append((text, waits_for))
    return tuple(kept)


__all__ = [
    "MAX_CONSUMED_BY_BYTES",
    "MAX_PREMISES_PER_OCCURRENCE",
    "MAX_PREMISE_TEXT_BYTES",
    "AssociationAssemblyError",
    "assemble_association",
]


# TODO(ASSOC-001): 跑真实数据之前还要定的两件事（2026-09-14 更新：接线已完成，``left`` 的落点、
# 超限进 blocked、重放正门 ``AssociationRefresher.reset`` 都已经做完）。
#
# 1. **``max_targets_per_call`` 的数值**。现在的 12 是拍的，而且单位与归组的 400 不是一回事
#    （那是一整天的行为数，这是一个候选一天发生几次）。预测树算出来的候选恰恰是高频习惯行为，
#    「操作手机」「使用电脑」一天几十次，会天天触发 ``AssociationLimitError`` 并被封锁一轮。
#    闸要留（保护闸不因数值不合适就去掉），数值必须在真实预测树上按"每候选每天发生次数分布"
#    重定，脚本走桌面实验目录、不进仓库。
# 2. **"一条前提只能被一次发生用掉"这条约束散在三处各说一遍**：``_draft`` 里的 ``unspent``、
#    提示词正文、schema 的 ``consumed`` 描述。要改成"一条前提可以服务多次"（办了健身卡 → 健身
#    很多次）时，三处都得改，而改提示词按纪律要跑真实模型对照。应当收到一处——按一个显式参数
#    记账，提示词那句由 ``build_request`` 按同一参数渲染。**在跑第一批真实数据、定下提示词基线
#    之前做**，否则改它就要重跑整条对照实验。影响小。
