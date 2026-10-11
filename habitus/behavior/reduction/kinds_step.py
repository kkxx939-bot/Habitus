"""归约里的词表一步：封口的链交白天归类，发布后把待定条目放进待定池（裁定 7 的范围）。

- 归类看整条链（原话 + 每段概要 + 目标 + 步骤），不只看链头名字；
- ``kind_token`` 写编号或占位标记（「待定」「非事件」），树上照常记录；
- 待定条目在发布之后才进待定池（要用 occurrence 地址做键），按地址幂等，检查点重放不会多算；
- 「非事件」留 ``kind_not_event`` 信号：融合在运行时读不到它（词表在融合之后），它是离线评价融合质量的计数（裁定 13）；
- 词表从空开始（裁定 29：没有预置清单）：一条 lane 还没有类时白天归类照样问，只判是不是一件事、给真事一个提议名，
  然后进待定池；类由每晚新增从池里按复现长出来。

lane 由组合根给（``kinds_default_lane``）：融合还没产出 ``source_lane``，等它有了改为逐条读字段。

TODO(BHV-SESSION-LANE-001)：lane 改成每条记录上的事实（方案已定，裁定 35；这一轮只改融合，没动归约与行为树）。
一条数据属于哪条 lane 由上游送进来时带的来源标识认定，在入口写定，随判断带到 occurrence 与空白上；归类按链
自己的 lane 取候选，``kinds_default_lane`` 删掉。现状：整个部署只能设一个 lane（默认 session）。只跑会话 lane 时
结果是对的；两条 lane 同时跑时，物理 lane 的记录会被拿去和会话 lane 的类比。会话 lane 的判断现在可以从凭据的
来源身份（``session/<会话>``）认出来，但行为树上还没有放 lane 的字段。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

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
from habitus.behavior.reduction.chains import BehaviorChain


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
    """一批链的归类结果，按链身份（``chain_digest``）给。"""

    tokens: Mapping[str, str]
    pending: Mapping[str, PendingNote]
    requests: int
    classified: int
    not_events: int
    model_calls: int
    signals: tuple[str, ...]


class KindStamping:
    def __init__(self, store: BehaviorKindStore, classifier: DaytimeClassifier, *, lane: Lane) -> None:
        if not isinstance(store, BehaviorKindStore):
            raise TypeError("store must be BehaviorKindStore")
        if not isinstance(classifier, DaytimeClassifier):
            raise TypeError("classifier must be DaytimeClassifier")
        self.store = store
        self.classifier = classifier
        self.lane = Lane(lane)

    async def classify(self, chains: Sequence[BehaviorChain], *, checkpoint: Callable[[], object]) -> KindStamps:
        requests: list[ClassifyRequest] = []
        for chain in chains:
            content = chain_content(chain)
            if content is not None:
                requests.append(ClassifyRequest(chain.chain_digest, self.lane, content))
        result = await self.classify_requests(requests, self.store.read(), checkpoint=checkpoint)
        by_key = {request.key: request for request in requests}
        signals = list(result.signals)
        pending: dict[str, PendingNote] = {}
        for key, verdict in result.verdicts.items():
            if verdict.outcome is Outcome.PENDING:
                content = by_key[key].content
                rendered = content.render(max_steps=self.classifier.config.prompt_max_steps)
                pending[key] = PendingNote(verdict.proposed or content.name, rendered)
            elif verdict.outcome is Outcome.NOT_EVENT:
                signals.append(f"kind_not_event {key[:12]} {by_key[key].content.name!r}: {verdict.reason}")
        outcomes = [verdict.outcome for verdict in result.verdicts.values()]
        return KindStamps(
            tokens={key: verdict.token for key, verdict in result.verdicts.items()},
            pending=pending,
            requests=len(requests),
            classified=outcomes.count(Outcome.CLASS),
            not_events=outcomes.count(Outcome.NOT_EVENT),
            model_calls=result.model_calls,
            signals=tuple(signals),
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
            entries.append(PendingEntry(str(item["ledger"]["uri"]), lane, day, note.proposed, note.content))
        return self.record_pending(entries)

    def record_pending(self, entries: Sequence[PendingEntry]) -> PendingAdmission:
        """放进待定池；池满时超出的这批不入池（留信号，由生命周期清理），绝不抛错卡住归约。"""

        if not entries:
            return PendingAdmission(0, ())
        snapshot = self.store.read_pending()
        fresh = [entry for entry in entries if entry.occurrence not in snapshot.pool.entries]
        room = max(0, self.store.config.max_pending - len(snapshot.pool.entries))
        admitted = sorted(fresh, key=lambda entry: entry.occurrence)[:room]
        refused = tuple(entry.occurrence for entry in fresh if entry not in admitted)
        if admitted:
            self.store.replace_pending(snapshot.pool.with_entries(admitted), expected_revision=snapshot.revision)
        return PendingAdmission(len(admitted), refused)


@dataclass(frozen=True)
class PendingAdmission:
    added: int
    refused: tuple[str, ...]

    def signals(self) -> tuple[str, ...]:
        notes = [f"kind_pending_added {self.added}"] if self.added else []
        if self.refused:
            notes.append(f"kind_pending_pool_full {len(self.refused)} entries not admitted")
        return tuple(notes)


def content_from_fields(fields: Mapping[str, object]) -> OccurrenceContent:
    """树上一条 occurrence 给归类看的内容（与 ``chain_content`` 同一形状：原话 + 概要 + 目标 + 步骤）。"""

    summary = str(fields.get("summary") or "").strip()
    goal = str(fields.get("goal") or "").strip() or None
    steps = tuple(
        str(step["semantics"]).strip()
        for step in fields.get("basis") or ()  # type: ignore[attr-defined]
        if isinstance(step, Mapping) and str(step.get("semantics") or "").strip()
    )
    return OccurrenceContent(name=str(fields["name"]), summaries=(summary,) if summary else (), goal=goal, steps=steps)


def is_countable(fields: Mapping[str, object]) -> bool:
    """撞车消歧的重复记录（``original_name`` 非空）不算一次独立的发生——与预测树同一口径。"""

    return fields.get("original_name") is None


def chain_content(chain: BehaviorChain) -> OccurrenceContent | None:
    """一条链给归类看的内容；链头没有可读名字（绕过融合守卫的坏数据）时给 ``None``。"""

    name = chain.head.behavior
    if not isinstance(name, str) or not name.strip():
        return None
    summaries = tuple(item.summary.strip() for item in chain.view if item.summary and item.summary.strip())
    goals = tuple(dict.fromkeys(item.goal.strip() for item in chain.view if item.goal and item.goal.strip()))
    steps = tuple(semantics.strip() for item in chain.view for semantics, _ in item.basis if semantics.strip())
    return OccurrenceContent(name=name.strip(), summaries=summaries, goal="；".join(goals) or None, steps=steps)


__all__ = [
    "KindStamping",
    "KindStamps",
    "PendingAdmission",
    "PendingNote",
    "chain_content",
    "content_from_fields",
    "is_countable",
]
