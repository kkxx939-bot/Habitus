"""此刻 vs 历史：逐槽三值表（对上 / 没对上 / 缺信息）+ 命中值 + 证据 URI。不加权、不打分——权重归判决层。

"没对上"的含义是**今天没观测到**，不是"没发生"（观测天然不完整）；判决层按"未见"读它。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC
from enum import Enum

from habitus.scene.views.model import COMPARED_SLOTS, SLOT_LABELS, ActionRef, ContextView


class Verdict(str, Enum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SlotComparison:
    """一行：此刻的值（展示用文本）、判定、命中了什么（观测事实，对比实际用到的值）、证据（历史行为 URI）。"""

    slot: str
    verdict: Verdict
    now: tuple[str, ...]
    matched: tuple[str, ...]
    evidence: tuple[str, ...]
    note: str = ""

    @property
    def label(self) -> str:
        return SLOT_LABELS[self.slot]


@dataclass(frozen=True)
class ComparisonTable:
    kind_token: str
    now: ContextView
    rows: tuple[SlotComparison, ...]
    history_count: int

    def row(self, slot: str) -> SlotComparison:
        for item in self.rows:
            if item.slot == slot:
                return item
        raise KeyError(slot)


def compare(now: ContextView, history: tuple[ContextView, ...]) -> ComparisonTable:
    """每个语义槽：此刻的值在历史视图里出现过 → 对上（命中值 + 出现过的那些历史行为 URI）；历史有可比的值而
    此刻的值没出现过 → 没对上；任一方没有可比的值 → 缺信息。历史视图必须早于此刻——把候选自己的未来当历史
    是我们自己输入的不自洽。"""

    if not isinstance(now, ContextView):
        raise TypeError("now must be a ContextView")
    if any(view.kind_token != now.kind_token for view in history):
        raise ValueError("history views must belong to the same kind_token as now")
    if any(view.at.astimezone(UTC) >= now.at.astimezone(UTC) for view in history):
        raise ValueError("history views must precede now")
    rows = tuple(_ROW_BUILDERS[slot](now, history) for slot in COMPARED_SLOTS)
    return ComparisonTable(kind_token=now.kind_token, now=now, rows=rows, history_count=len(history))


def _scene(now: ContextView, history: tuple[ContextView, ...]) -> SlotComparison:
    """此刻没有情景文档，本层不推断；把候选历史上所属的事全部带出作证据，判决层自己读。"""

    labelled = [view for view in history if view.scene is not None]
    labels = tuple(dict.fromkeys(view.scene.label for view in labelled if view.scene is not None))
    return SlotComparison("scene", Verdict.UNKNOWN, (), labels, _uris(labelled), "now has no scene document")


def _prior_steps(now: ContextView, history: tuple[ContextView, ...]) -> SlotComparison:
    kinds = {step.kind_token for step in now.prior_steps}
    with_steps = [view for view in history if view.prior_steps]
    if not kinds or not with_steps:
        return SlotComparison("prior_steps", Verdict.UNKNOWN, tuple(sorted(kinds)), (), ())
    hits = [(view, kinds & {step.kind_token for step in view.prior_steps}) for view in with_steps]
    return _row("prior_steps", tuple(sorted(kinds)), hits)


def _preceding(now: ContextView, history: tuple[ContextView, ...]) -> SlotComparison:
    """紧邻上一条（预测树转移边的口径：转移窗口内时间上紧邻的那条，不论属于哪件事）。

    与"此前步骤"分开比：后者是同一件事里更早的成员，前者是钟面上挨着的那条——"吃完饭就去打球"的
    转移证据只在这一槽对得上。三态里"确认没有"（∅）是可比的值：历史上做这件事之前常常什么都没做，
    今天也什么都没做，就是对上；"删失"与"没算"是缺信息。"""

    value = None if now.preceding is None else now.preceding.value
    comparable = [(view, view.preceding.value) for view in history if view.preceding is not None and view.preceding.value is not None]
    if value is None or not comparable:
        return SlotComparison("preceding", Verdict.UNKNOWN, () if value is None else (value,), (), ())
    hits = [(view, {value} if seen == value else set()) for view, seen in comparable]
    return _row("preceding", (value,), hits)


def _preconditions(now: ContextView, history: tuple[ContextView, ...]) -> SlotComparison:
    """历史前提今天成立了没有：前提落到的 kind 对（今日已发生的 kind ∪ 待用清单产生方的 kind），或文本对
    待用清单。没有可比形式（target_kinds 为空）的历史前提不计入；全都不可比即缺信息。"""

    pending_kinds = {kind for item in now.preconditions for kind in item.target_kinds}
    pending_texts = {item.text for item in now.preconditions}
    available = now.today_kinds | pending_kinds
    value = tuple(sorted(pending_texts))
    comparable = [
        (view, [item for item in view.preconditions if item.target_kinds])
        for view in history
        if any(item.target_kinds for item in view.preconditions)
    ]
    if not comparable:
        return SlotComparison("preconditions", Verdict.UNKNOWN, value, (), (), "history has no comparable preconditions")
    if not available and not pending_texts:
        return SlotComparison("preconditions", Verdict.UNKNOWN, value, (), ())
    hits = [
        (
            view,
            {kind for item in items for kind in item.target_kinds if kind in available}
            | {item.text for item in items if item.text in pending_texts},
        )
        for view, items in comparable
    ]
    return _row("preconditions", value, hits)


def _causes(now: ContextView, history: tuple[ContextView, ...]) -> SlotComparison:
    """起因只比起因本身（此刻 = 最近一条行为）；历史起因是一件事时落到它留下的待用前提产生方 kind。"""

    now_kinds = {cause.kind_token for cause in now.causes if isinstance(cause, ActionRef)}
    value = tuple(sorted(now_kinds))
    comparable = [
        view
        for view in history
        if any(isinstance(cause, ActionRef) or cause.kinds for cause in view.causes)
    ]
    if not now_kinds or not comparable:
        return SlotComparison("causes", Verdict.UNKNOWN, value, (), ())
    hits = [
        (
            view,
            {
                kind
                for cause in view.causes
                for kind in ((cause.kind_token,) if isinstance(cause, ActionRef) else cause.kinds)
                if kind in now_kinds
            },
        )
        for view in comparable
    ]
    return _row("causes", value, hits)


def _last_time(now: ContextView, history: tuple[ContextView, ...]) -> SlotComparison:
    """上一次只比"当时所在的事"是不是同一件事；距今天数随值带出，不作判据（那是间隔统计，归预测树）。"""

    comparable = [view for view in history if view.last_time is not None and view.last_time.scene is not None]
    if now.last_time is None:
        return SlotComparison("last_time", Verdict.UNKNOWN, (), (), ())
    value = (f"days_ago={now.last_time.days_ago}",) + (() if now.last_time.scene is None else (now.last_time.scene.label,))
    if now.last_time.scene is None or not comparable:
        return SlotComparison("last_time", Verdict.UNKNOWN, value, (), ())
    label = now.last_time.scene.label
    hits = [
        (
            view,
            {label} if view.last_time is not None and view.last_time.scene is not None and view.last_time.scene.label == label else set(),
        )
        for view in comparable
    ]
    return _row("last_time", value, hits)


def _concurrent(now: ContextView, history: tuple[ContextView, ...]) -> SlotComparison:
    kinds = {item.kind_token for item in now.concurrent}
    with_concurrent = [view for view in history if view.concurrent]
    if not kinds or not with_concurrent:
        return SlotComparison("concurrent", Verdict.UNKNOWN, tuple(sorted(kinds)), (), ())
    hits = [(view, kinds & {item.kind_token for item in view.concurrent}) for view in with_concurrent]
    return _row("concurrent", tuple(sorted(kinds)), hits)


def _subjects(now: ContextView, history: tuple[ContextView, ...]) -> SlotComparison:
    """和谁：主体之外的人；此刻或历史都没有别人即缺信息（"独自"不是可对的值，那是观测不完整的常态）。"""

    with_subjects = [view for view in history if view.subjects]
    if not now.subjects or not with_subjects:
        return SlotComparison("subjects", Verdict.UNKNOWN, tuple(now.subjects), (), ())
    hits = [(view, set(view.subjects) & set(now.subjects)) for view in with_subjects]
    return _row("subjects", tuple(now.subjects), hits)


def _fact(now: ContextView, history: tuple[ContextView, ...]) -> SlotComparison:
    """此刻还没有这条事实（行为尚未发生）：永远缺信息，历史摘要作为证据交给判决层阅读。"""

    return SlotComparison("fact", Verdict.UNKNOWN, (), (), _uris(view for view in history if view.summary))


def _row(slot: str, value: tuple[str, ...], hits: list[tuple[ContextView, set[str]]]) -> SlotComparison:
    matched = tuple(sorted({item for _, found in hits for item in found}))
    evidence = _uris(view for view, found in hits if found)
    return SlotComparison(slot, Verdict.MATCHED if evidence else Verdict.UNMATCHED, value, matched, evidence)


def _uris(views: Iterable[ContextView]) -> tuple[str, ...]:
    return tuple(view.occurrence_uri for view in views if view.occurrence_uri is not None)


_ROW_BUILDERS = {
    "scene": _scene,
    "prior_steps": _prior_steps,
    "preceding": _preceding,
    "preconditions": _preconditions,
    "causes": _causes,
    "last_time": _last_time,
    "concurrent": _concurrent,
    "subjects": _subjects,
    "fact": _fact,
}


def render_table(table: ComparisonTable) -> str:
    """给人看的三值表（Markdown）。"""

    symbols = {Verdict.MATCHED: "对上", Verdict.UNMATCHED: "没对上", Verdict.UNKNOWN: "缺信息"}
    lines = ["| 槽 | 判定 | 此刻 | 命中 | 证据 |", "|---|---|---|---|---|"]
    for row in table.rows:
        now_text = "；".join(row.now) if row.now else "—"
        matched = "；".join(row.matched) if row.matched else "—"
        evidence = ", ".join(uri.rsplit("/", 1)[-1] for uri in row.evidence[:4])
        if len(row.evidence) > 4:
            evidence += f" …（共 {len(row.evidence)}）"
        lines.append(f"| {row.label} | {symbols[row.verdict]} | {now_text} | {matched} | {evidence or '—'} |")
    return "\n".join(lines)


__all__ = ["ComparisonTable", "SlotComparison", "Verdict", "compare", "render_table"]
