"""归约里的词表一步：封口的判断交白天归类，发布后把待定条目放进待定池（裁定 7 的范围）。

- 归类看原话 + 概要 + 目标 + 步骤，不只看名字；
- ``kind_token`` 写编号或占位标记（「待定」「非事件」），树上照常记录；
- 待定条目在发布之后才进待定池（要用 occurrence 地址做键），按地址幂等，检查点重放不会多算；
- 「非事件」留 ``kind_not_event`` 信号：融合在运行时读不到它（词表在融合之后），它是离线评价融合质量的计数（裁定 13）；
- 词表从空开始（裁定 29：没有预置清单）：一条 lane 还没有类时白天归类照样问，只判是不是一件事、给真事一个提议名，
  然后进待定池；类由每晚新增从池里按复现长出来。

lane 由组合根给：两条 lane 各是一整套（各自的归约、词表、行为树，见 ``runtime/behavior.py``），
这一套归的就是它自己那条 lane 的类。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from habitus.behavior.kinds.calls import CallFailure
from habitus.behavior.kinds.classify import (
    ClassifyRequest,
    ClassifyResult,
    DaytimeClassifier,
    OccurrenceContent,
    Outcome,
    classify_in_chunks,
)
from habitus.behavior.kinds.ids import Lane, lane_of_token
from habitus.behavior.kinds.model import Vocabulary
from habitus.behavior.kinds.pending import PendingEntry
from habitus.behavior.kinds.store import BehaviorKindStore
from habitus.behavior.reduction.record import ReducibleJudgement
from habitus.behavior.retry import RetryCause

#: 归类没问出结论的那几条换成「错误」时进堆的原因：内容安全拦下的不重试、健康面单列；输入超长是我们自己处理不了的
#: （材料上限与模型窗口没对上，裁定 94 错误分类）；模型整个用不了的放一边、夜里再归一次；别的是 bug，当场报、不白试。
SET_ASIDE: Mapping[CallFailure, RetryCause] = {
    CallFailure.REFUSED: RetryCause.REFUSED,
    CallFailure.TOO_LARGE: RetryCause.DETERMINISTIC,
    CallFailure.UNAVAILABLE: RetryCause.UNAVAILABLE,
    CallFailure.BUG: RetryCause.DETERMINISTIC,
}


@dataclass(frozen=True)
class PendingNote:
    """检查点里随 occurrence 一起定格的待定信息：发布后据此进待定池。"""

    proposed: str
    content: str

    def to_mapping(self) -> dict[str, str]:
        return {"proposed": self.proposed, "content": self.content}

    @classmethod
    def from_mapping(cls, raw: object) -> PendingNote:
        if not isinstance(raw, Mapping) or set(raw) != {"proposed", "content"}:
            raise ValueError("kind pending note must have proposed/content")
        return cls(str(raw["proposed"]), str(raw["content"]))


@dataclass(frozen=True)
class KindStamps:
    """一批判断的归类结果，按它们落成的节点身份（``chain_digest``）给。"""

    tokens: Mapping[str, str]
    pending: Mapping[str, PendingNote]
    requests: int
    classified: int
    not_events: int
    model_calls: int
    signals: tuple[str, ...]
    #: 没有归类结果、要换成「错误」放一边的（按节点身份）与为什么（定夜里试不试）。
    set_aside: Mapping[str, CallFailure] = field(default_factory=dict)


class KindStamping:
    def __init__(self, store: BehaviorKindStore, classifier: DaytimeClassifier, *, lane: Lane) -> None:
        if not isinstance(store, BehaviorKindStore):
            raise TypeError("store must be BehaviorKindStore")
        if not isinstance(classifier, DaytimeClassifier):
            raise TypeError("classifier must be DaytimeClassifier")
        self.store = store
        self.classifier = classifier
        self.lane = Lane(lane)

    async def classify(self, records: Sequence[ReducibleJudgement], *, checkpoint: Callable[[], object]) -> KindStamps:
        requests: list[ClassifyRequest] = []
        for record in records:
            content = judgement_content(record)
            if content is not None:
                requests.append(ClassifyRequest(record.chain_digest, self.lane, content))
        result = await self.classify_requests(requests, self.store.read(), checkpoint=checkpoint)
        by_key = {request.key: request for request in requests}
        signals = list(result.signals)
        pending: dict[str, PendingNote] = {}
        for key, verdict in result.verdicts.items():
            if key in result.unanswerable:
                continue  # 没问出结论：换成「错误」，不落成待定
            if verdict.outcome is Outcome.PENDING:
                content = by_key[key].content
                rendered = content.render(max_steps=self.classifier.config.prompt_max_steps)
                pending[key] = PendingNote(verdict.proposed or content.name, rendered)
            elif verdict.outcome is Outcome.NOT_EVENT:
                signals.append(f"kind_not_event {key[:12]} {by_key[key].content.name!r}: {verdict.reason}")
        outcomes = [verdict.outcome for verdict in result.verdicts.values()]
        return KindStamps(
            tokens={key: verdict.token for key, verdict in result.verdicts.items() if key not in result.unanswerable},
            pending=pending,
            requests=len(requests),
            classified=outcomes.count(Outcome.CLASS),
            not_events=outcomes.count(Outcome.NOT_EVENT),
            model_calls=result.model_calls,
            signals=tuple(signals),
            set_aside=dict(result.unanswerable),
        )

    async def classify_requests(
        self, requests: Sequence[ClassifyRequest], vocabulary: Vocabulary, *, checkpoint: Callable[[], object]
    ) -> ClassifyResult:
        """按批大小切块归类，块与块之间续一次 sweep 租约（积压时几百条也不会把租约耗死）。"""

        return await classify_in_chunks(self.classifier, requests, vocabulary, checkpoint=checkpoint)

    def record_published(self, documents: Sequence[object]) -> PendingAdmission:
        """发布了的文档里带 ``kind_pending`` 的，按 occurrence 地址放进待定池（重放同一批不多算）。"""

        entries: list[PendingEntry] = []
        for item in documents:
            if not isinstance(item, Mapping) or "kind_pending" not in item:
                continue
            note = PendingNote.from_mapping(item["kind_pending"])
            payload = item["payload"]
            day = date.fromisoformat(str(payload["occurred_on"]))
            lane = lane_of_token(str(payload["kind_token"]))
            # 进池那天就是这条落树的那天；读侧取 ``entered_on or day`` 也回落到它，这里显式写下，读侧改口径也不跟着变
            entries.append(
                PendingEntry(str(item["ledger"]["uri"]), lane, day, note.proposed, note.content, entered_on=day)
            )
        return self.record_pending(entries)

    def record_pending(self, entries: Sequence[PendingEntry]) -> PendingAdmission:
        """放进待定池。池满时挤掉进池最早的旧条目（裁定 60 第 2 条，10-19 改）：新习惯恰恰从新条目里长出来，拒收新条目
        就再也长不出新类。挤掉的只是索引，树上的 occurrence 不动；留信号，绝不抛错卡住归约。"""

        if not entries:
            return PendingAdmission(0, ())
        snapshot = self.store.read_pending()
        fresh = {entry.occurrence: entry for entry in entries if entry.occurrence not in snapshot.pool.entries}
        if not fresh:
            return PendingAdmission(0, ())
        merged = snapshot.pool.with_entries(fresh.values())
        overflow = max(0, len(merged.entries) - self.store.config.max_pending)
        evicted = tuple(entry.occurrence for entry in sorted(merged.entries.values(), key=_entered)[:overflow])
        self.store.replace_pending(merged.without(evicted), expected_revision=snapshot.revision)
        return PendingAdmission(len(fresh.keys() - set(evicted)), evicted)


def _entered(entry: PendingEntry) -> tuple[date, date, str]:
    """进池先后：进池那天，同一天按 occurrence 的日子、再按地址（确定性）。"""

    return (entry.entered_on or entry.day, entry.day, entry.occurrence)


@dataclass(frozen=True)
class PendingAdmission:
    added: int
    evicted: tuple[str, ...]

    def signals(self) -> tuple[str, ...]:
        notes = [f"kind_pending_added {self.added}"] if self.added else []
        if self.evicted:
            notes.append(f"kind_pending_pool_full {len(self.evicted)} oldest entries evicted")
        return tuple(notes)


def content_from_fields(fields: Mapping[str, object]) -> OccurrenceContent:
    """树上一条 occurrence 给归类看的内容（与 ``judgement_content`` 同一形状：原话 + 概要 + 目标 + 步骤）。"""

    summary = str(fields.get("summary") or "").strip()
    goal = str(fields.get("goal") or "").strip() or None
    steps = tuple(
        str(step["semantics"]).strip()
        for step in fields.get("basis") or ()  # type: ignore[attr-defined]
        if isinstance(step, Mapping) and str(step.get("semantics") or "").strip()
    )
    return OccurrenceContent(name=str(fields["name"]), summaries=(summary,) if summary else (), goal=goal, steps=steps)


def is_countable(fields: Mapping[str, object]) -> bool:
    """撞车消歧的重复记录（``original_name`` 非空）不算一次独立的发生——与事件序列（``series.reader``）同一口径。"""

    return fields.get("original_name") is None


def judgement_content(record: ReducibleJudgement) -> OccurrenceContent | None:
    """一条可读判断给归类看的内容；没有可读名字（绕过融合守卫的坏数据）时给 ``None``。"""

    name = record.behavior
    if not isinstance(name, str) or not name.strip():
        return None
    summaries = (record.summary.strip(),) if record.summary and record.summary.strip() else ()
    goal = record.goal.strip() if record.goal and record.goal.strip() else None
    steps = tuple(semantics.strip() for semantics, _ in record.basis if semantics.strip())
    return OccurrenceContent(name=name.strip(), summaries=summaries, goal=goal, steps=steps)


__all__ = [
    "KindStamping",
    "KindStamps",
    "PendingAdmission",
    "PendingNote",
    "content_from_fields",
    "is_countable",
    "judgement_content",
]
