"""映射器：把一条 occurrence 判到概念上。语义树在这一刀里唯一的模型触点。

流程（《语义树重构》③，2026-09-26 按评审修订）：

1. **召回**  occurrence 的名字 + 概要做 embedding，对**叶子**行为概念的定义向量取 top-K，并上这个 kind 以前
   命中过的概念（有闸）——召回只影响漏，不影响错。祖先概念不参与：命中「打球」算不算「运动」由读侧沿
   parent 链聚合。
2. **机械判据**  带 ``MechanicalRule`` 的候选先由算法算数值（"算法控制数值"）：数值不满足就不问模型；
   满足了再问模型判据句那道语义门——规则说"比常态就寝晚 7.5 小时"对一碗 07:00 的面也成立，
   "这是不是入睡"只有模型答得了。模型从不碰数字，算法从不碰语义。
3. **材料核对**  要当天时间线的（``context=DAY``）没给时间线、要常态值的没给常态——那一条记成**未决**，
   不让模型替我们答成 false。
4. **判定**  剩下的候选 LLM 一次一条 occurrence，对每个候选只答是/否；模型这一次没答成（传输、配额、
   结构两轮都不成形）→ 这些候选记未决、留信号，**不让一条 occurrence 把整天塌掉**，模型层的异常不出本模块。
5. **定档**  算法按概念定义里的数值规则定档。
6. **情境**  那一刻成立的情境概念由调用方按算法填进来，这里核对它们是情境概念、档是定义过的。
7. **留痕**  第几轮答对、JSON 是不是修出来的、与这个 kind 以前的判法有没有翻转，都进 ``signals``。

**映射者看不到任何假设、任何账**：输入只有 occurrence 的事实 + 当天时间线 + 候选概念的定义 + 常态值。
架构测试钉死本包不许（传递性地）触达 ``hypotheses`` / ``ledger``。

**旁册空着不许映射**：构造时核对旁册覆盖了全部叶子行为概念，缺就拒——否则一整天会零调用地映射成
"零命中"并盖上完成标记，永不重做。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from types import MappingProxyType
from typing import Any, cast

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.model import BehaviorKind
from habitus.behavior.tree import BehaviorTree
from habitus.behavior.uri import BehaviorURI
from habitus.foundation.integrity import canonical_digest
from habitus.foundation.text import clean_line
from habitus.model_client import (
    ChatMessage,
    ChatRequest,
    Embedder,
    ModelClientError,
    ModelTransportError,
    StructuredChatClient,
)
from habitus.scene.concepts.model import (
    ConceptDefinition,
    ConceptError,
    ConceptSet,
    ContextScope,
    GradeMeasure,
    concept_identity,
)
from habitus.scene.concepts.vectors import ConceptVectorIndex
from habitus.scene.occurrences.model import ConceptHit, ConceptHits, ConceptHitsError
from habitus.scene.occurrences.situations import SituationOutcome
from habitus.scene.occurrences.store import ConceptHitStore

MAPPER_PROMPT_VERSION = "scene_concept_mapper_prompt_v3"
_WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

MAPPER_SYSTEM_PROMPT = """你在判断一条被观测到的行为属于哪些概念。

给你的材料：这条行为的事实（名字、一句话概要、目标、步骤、开始时刻、最后所见时刻、同在的人、地点），可能还有
这个人的常态值，以及它所在那一天的时间线（当某个候选的判据要看前后发生了什么时才给）。然后是若干候选概念，
每个一句判据句。对每个候选，回答这条行为**是不是**判据句说的那件事：是 → yes，不是 → no，**看不到、判不了 → unknown**。

规则：
- 只按判据句判，不按概念名字的联想判；判据句没说到的条件不要自己补。
- 候选后面写了「数值条件已由算法判定成立」的，那部分**已经算过了，不要再自己算**——只判剩下的语义部分
  （这条行为到底是不是那件事）。
- 判据句里提到"常态"、而候选后面**没有**写「数值条件已由算法判定成立」的，才用材料里给出的常态值自己算偏离。
- 判据句里提到别的事件（"起床后""第一次"）的，到当天时间线里找。时间线上标着「未观测」/「没读懂」的那几段是
  **观测空白**：要找的事件本该落在空白里、时间线上又没有 → 答 unknown（看不到，不是没发生）。时间线覆盖到了
  那段、里面确实没有，才答 no。
- 材料里没写的事实一律当作不知道；判据只差这一条事实时答 unknown，**不要当成 no**——"没看到"和"没发生"是两件事，
  后者会被当成反面证据。
- 每个候选恰好答一次，不多不少，也不要新增候选。"""


class ConceptMapperError(ValueError):
    """映射器的输入与概念集或旁册矛盾。"""


#: 三态判定。``unknown`` 是"材料里看不到、判不了"（用户 09-27"没看到就不算"）——它不是第三种结果，
#: 落到现成的 ``unresolved`` 里：不进分母也不进分子。没有它的时候模型只能答 no，而 no 在账本里是反面证据。
VERDICTS = ("yes", "no", "unknown")


def mapper_json_schema(candidate_names: Sequence[str]) -> dict[str, Any]:
    """候选名字钉进 enum：模型答不出候选之外的名字，也答不了别的候选；判定三态（是 / 不是 / 看不到）。

    **严格模式吃得下的形状**（2026-09-29 探针实测）：真实后端（codex 的 ``--output-schema``、
    OpenAI structured outputs）不认 ``minItems`` / ``maxItems``，带着它们会 400。条数由
    ``assemble_verdicts`` 管——它本来就在管（少了哪个候选、重复了哪个都报），所以 schema 只说
    "有哪些字段、取值属于哪个集合"。
    """

    names = list(candidate_names)
    if not names:
        raise ConceptMapperError("the mapper schema needs at least one candidate")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["concept", "verdict"],
                    "properties": {
                        "concept": {"type": "string", "enum": names, "description": "候选概念的名字，照抄。"},
                        "verdict": {
                            "type": "string",
                            "enum": list(VERDICTS),
                            "description": "这条行为是不是这个概念判据句说的那件事：yes / no / unknown（材料看不到，判不了）。",
                        },
                    },
                },
                "description": f"每个候选恰好一条，共 {len(names)} 条，不多不少。",
            }
        },
    }


SCHEMA_FINGERPRINT = canonical_digest(mapper_json_schema(("候选",)))[:12]
MAPPER_VERSION = f"{MAPPER_PROMPT_VERSION}+schema{SCHEMA_FINGERPRINT}"


@dataclass(frozen=True)
class TimelineEntry:
    """当天时间线上的一行：给要看前后事件的判据用。``gap`` 为真表示这一行是**观测空白**而不是一件行为。

    空白段必须列出来（用户 09-27"没看到就不算"）：不列的话时间线看上去是连续的，模型按"时间线里没有就是没发生"
    把被空白盖住的早餐答成 no，而 no 在账本里是反面证据——七d-2 的假负样本就是这么从映射口进来的。
    """

    uri: str
    name: str
    started_at: datetime
    last_observed_at: datetime
    summary: str
    gap: bool = False

    @classmethod
    def from_document(cls, document: BehaviorDocument) -> TimelineEntry:
        facts = OccurrenceFacts.from_document(document)
        return cls(uri=facts.uri, name=facts.name, started_at=facts.started_at, last_observed_at=facts.last_observed_at, summary=facts.summary)

    @classmethod
    def from_gap(cls, document: BehaviorDocument) -> TimelineEntry:
        """行为树上的一段空白（``gap_kind`` 是「未观测」或「没读懂」）；两者对映射都一样：那段发生了什么不知道。"""

        if document.address.kind is not BehaviorKind.GAP:
            raise ConceptMapperError("from_gap needs a gap document")
        started, ended, kind = (document.fields.get(key) for key in ("started_at", "ended_at", "gap_kind"))
        if not isinstance(started, str) or not isinstance(ended, str) or not isinstance(kind, str):
            raise ConceptMapperError("a gap document carries started_at, ended_at and gap_kind")
        return cls(
            uri=str(BehaviorURI.from_address(document.address)),
            name=kind,
            started_at=datetime.fromisoformat(started),
            last_observed_at=datetime.fromisoformat(ended),
            summary="这段时间没有可用的观测",
            gap=True,
        )


@dataclass(frozen=True)
class OccurrenceFacts:
    """映射者能看到的全部事实。字段名不出本模块：行为文档的形状变了只改 ``from_document``。"""

    uri: str
    name: str
    summary: str
    kind_token: str
    started_at: datetime
    last_observed_at: datetime
    subjects: tuple[str, ...]
    place: str | None
    goal: str | None
    steps: tuple[str, ...]

    @classmethod
    def from_document(cls, document: BehaviorDocument) -> OccurrenceFacts:
        if not isinstance(document, BehaviorDocument) or document.kind is not BehaviorKind.OCCURRENCE:
            raise ConceptMapperError("the mapper reads occurrence documents only")
        fields = document.fields
        started_at = document.address.started_at
        last_observed_at = datetime.fromisoformat(str(fields["last_observed_at"])).astimezone(started_at.tzinfo)
        if last_observed_at < started_at:
            raise ConceptMapperError("occurrence last_observed_at precedes its start; the behaviour tree is inconsistent")
        place, goal = fields.get("place"), fields.get("goal")
        steps = tuple(
            clean_line(step.get("semantics")) for step in fields.get("basis", ()) if isinstance(step, Mapping) and clean_line(step.get("semantics"))
        )
        return cls(
            uri=str(BehaviorURI.from_address(document.address)),
            name=str(fields["name"]),
            summary=str(fields["summary"]),
            kind_token=str(fields["kind_token"]),
            started_at=started_at,
            last_observed_at=last_observed_at,
            subjects=tuple(str(item) for item in fields["subjects"]),
            place=None if place is None else str(place),
            goal=None if goal is None else str(goal),
            steps=steps,
        )

    @property
    def duration_minutes(self) -> float:
        return (self.last_observed_at - self.started_at).total_seconds() / 60.0

    def measures(self) -> Mapping[GradeMeasure, float]:
        return MappingProxyType(
            {
                GradeMeasure.START_MINUTE_OF_DAY: float(self.started_at.hour * 60 + self.started_at.minute),
                GradeMeasure.DURATION_MINUTES: self.duration_minutes,
            }
        )

    def embedding_query(self) -> str:
        return f"{self.name}\n{self.summary}"

    def render(self, *, subject: str | None = None) -> str:
        others = tuple(item for item in self.subjects if item != subject) if subject else self.subjects
        started = self.started_at
        lines = [
            f"行为：{self.name}",
            f"概要：{self.summary}",
            f"目标：{self.goal if self.goal else '（说不出目标）'}",
        ]
        if self.steps:
            lines.append("步骤：" + " → ".join(self.steps))
        lines += [
            f"开始：{started.strftime('%Y-%m-%d %H:%M')}（{_WEEKDAYS[started.weekday()]}）",
            f"最后所见：{self.last_observed_at.strftime('%H:%M')}（历时约 {round(self.duration_minutes)} 分钟）",
            f"同在：{'、'.join(others) if others else '（无）'}",
            f"地点：{self.place if self.place else '（未知）'}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class MapperConfig:
    """K、以前命中的并集上限、时间线行数上限都是保护闸；数值在重放上定，这里只是默认。"""

    recall_limit: int = 10
    previous_hits_limit: int = 10
    max_timeline_rows: int = 80
    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        for label in ("recall_limit", "previous_hits_limit", "max_timeline_rows"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if isinstance(self.transient_retries, bool) or not isinstance(self.transient_retries, int) or self.transient_retries < 0:
            raise ValueError("transient_retries must be a non-negative integer")
        if (
            isinstance(self.transient_retry_delay_seconds, bool)
            or not isinstance(self.transient_retry_delay_seconds, int | float)
            or self.transient_retry_delay_seconds < 0
        ):
            raise ValueError("transient_retry_delay_seconds must be a non-negative number")


def build_request(
    facts: OccurrenceFacts,
    candidates: Sequence[ConceptDefinition],
    baseline: Mapping[str, str],
    *,
    subject: str | None = None,
    timeline: Sequence[TimelineEntry] = (),
    max_timeline_rows: int = 80,
    settled: Mapping[str, str] = MappingProxyType({}),
) -> ChatRequest:
    """``settled``：概念名 → 算法已经判成立的数值条件的人读说法（机械判据那一半，模型不许再算）。"""

    if not candidates:
        raise ConceptMapperError("a mapping request needs at least one candidate")
    sections = ["## 这条行为", facts.render(subject=subject)]
    if baseline:
        sections += ["", "## 这个人的常态", *(f"- {key}：{value}" for key, value in baseline.items())]
    if any(item.context is ContextScope.DAY for item in candidates):
        if not timeline:
            raise ConceptMapperError("a candidate needs the day timeline but none was given")
        rows = _render_timeline(facts, timeline, max_timeline_rows)
        heading = "## 当天时间线（按时刻；标【未观测】/【没读懂】的是观测空白，那几段里发生过什么不知道）"
        sections += ["", heading, *rows]
    sections += ["", "## 候选概念（每个答 yes / no / unknown）"]
    for item in candidates:
        line = f"- {item.name}：{item.definition}"
        if item.name in settled:
            # C-8：数值那一半算法已经算过了。不说清楚的话模型会自己再算一遍常态偏移，而它算不准，
            # 算错就把一个规则已经判成立的候选答成 no。
            line += f" ← 数值条件已由算法判定成立（{settled[item.name]}）：只判这条行为是不是「{item.name}」说的那件事"
        sections += [line]
    return ChatRequest(
        messages=(
            ChatMessage(role="system", content=MAPPER_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n".join(sections)),
        )
    )


def _render_timeline(facts: OccurrenceFacts, timeline: Sequence[TimelineEntry], limit: int) -> list[str]:
    ordered = sorted(timeline, key=lambda row: (row.started_at.astimezone(UTC), row.uri))
    if len(ordered) > limit:
        # 保护闸：围着这一条取最近的 limit 行，两头各砍。
        index = next((i for i, row in enumerate(ordered) if row.uri == facts.uri), len(ordered) // 2)
        start = max(0, min(index - limit // 2, len(ordered) - limit))
        ordered = ordered[start : start + limit]
    lines = []
    for row in ordered:
        span = f"{row.started_at.strftime('%H:%M')}–{row.last_observed_at.strftime('%H:%M')}"
        if row.gap:
            lines.append(f"- {span} 【{row.name}】{row.summary}，这段里发生过什么不知道")
            continue
        marker = "  ← 这条" if row.uri == facts.uri else ""
        lines.append(f"- {span} {row.name}（{row.summary}）{marker}")
    return lines


def assemble_verdicts(parsed: object, candidates: Sequence[ConceptDefinition]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """把模型的答复核对成 (命中的概念身份, 答"看不到"的概念身份)，都按候选顺序。

    每个候选恰好一条，否则整份不成形、交结构层重试。``unknown`` 不是"没命中"：它回到 ``unresolved``，
    账本那边"没看到就不算"（不进分母不进分子）。
    """

    if not isinstance(parsed, Mapping):
        raise ValueError("mapper output must be an object")
    verdicts = parsed.get("verdicts")
    if not isinstance(verdicts, list):
        raise ValueError("mapper output must carry a verdicts list")
    by_name = {item.name: item for item in candidates}
    answered: dict[str, str] = {}
    for verdict in verdicts:
        if not isinstance(verdict, Mapping):
            raise ValueError("each verdict must be an object")
        name = _match_candidate(verdict.get("concept"), by_name)
        if name is None:
            raise ValueError(f"verdict names a concept that was not a candidate: {verdict.get('concept')!r}")
        if name in answered:
            raise ValueError(f"verdict repeats candidate {name!r}")
        decision = verdict.get("verdict")
        if decision not in VERDICTS:
            raise ValueError(f"verdict must be one of {list(VERDICTS)}, got {decision!r}")
        answered[name] = str(decision)
    missing = [name for name in by_name if name not in answered]
    if missing:
        raise ValueError(f"verdicts are missing candidates: {missing}")
    return (
        tuple(by_name[name].identity for name in by_name if answered[name] == "yes"),
        tuple(by_name[name].identity for name in by_name if answered[name] == "unknown"),
    )


def _match_candidate(raw: object, by_name: Mapping[str, ConceptDefinition]) -> str | None:
    """先按原文精确对，再退到"洗掉多余空白后唯一匹配"——两个只差内部空白的名字不能被并成一个。"""

    if isinstance(raw, str) and raw in by_name:
        return raw
    cleaned = clean_line(raw)
    matches = [name for name in by_name if clean_line(name) == cleaned]
    return matches[0] if len(matches) == 1 else None


class ConceptMapper:
    def __init__(
        self,
        client: StructuredChatClient,
        embedder: Embedder,
        concepts: ConceptSet,
        vectors: ConceptVectorIndex,
        *,
        config: MapperConfig | None = None,
        previous_hits: Callable[[str], Iterable[str]] | None = None,
        subject: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        if not isinstance(concepts, ConceptSet):
            raise TypeError("concepts must be a ConceptSet")
        if not isinstance(vectors, ConceptVectorIndex):
            raise TypeError("vectors must be a ConceptVectorIndex")
        resolved = config or MapperConfig()
        if not isinstance(resolved, MapperConfig):
            raise TypeError("config must be MapperConfig")
        self.leaves = concepts.behavior_leaves()
        stale = vectors.stale(concepts, among=self.leaves)
        if stale:
            raise ConceptMapperError(f"concept vectors are missing or stale for {list(stale)}; refresh the sidecar before mapping")
        self.client = client
        self.embedder = embedder
        self.concepts = concepts
        self.vectors = vectors
        self.config = resolved
        self.previous_hits = previous_hits
        self.subject = subject
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    @property
    def version(self) -> str:
        """出处：提示词 + schema 指纹 + embedding 模型 + 判定模型 + 概念集指纹。任一变了，命中就不是同一口径判的。"""

        return f"{MAPPER_VERSION}+emb:{self.vectors.model}+llm:{self.client.client.model}+concepts:{self.concepts.fingerprint}"

    async def _recall(self, facts: OccurrenceFacts) -> tuple[tuple[ConceptDefinition, ...], frozenset[str]]:
        """召回：向量 top-K ∪ 这个 kind 以前命中过的（有闸），只在叶子行为概念里；按身份排序，顺序与模型无关。"""

        chosen: set[str] = set()
        prior: set[str] = set()
        if self.leaves:
            query = await self.embedder.embed_query(facts.embedding_query())
            chosen.update(identity for identity, _score in self.vectors.nearest(query.values, limit=self.config.recall_limit, among=self.leaves))
        if self.previous_hits is not None:
            for name in self.previous_hits(facts.kind_token):
                try:
                    identity = concept_identity(name)
                except ConceptError:
                    continue
                if identity in self.leaves:
                    prior.add(identity)
            chosen.update(sorted(prior)[: self.config.previous_hits_limit])
        return tuple(self.concepts[identity] for identity in sorted(chosen)), frozenset(prior)

    async def map(
        self,
        document: BehaviorDocument,
        *,
        situation_hits: Sequence[ConceptHit] = (),
        situations_checked: Sequence[str] = (),
        baseline: Mapping[str, str] | None = None,
        timeline: Sequence[TimelineEntry] = (),
    ) -> ConceptHits:
        facts = OccurrenceFacts.from_document(document)
        situations = self._situations(situation_hits)
        checked = tuple(dict.fromkeys((*situations_checked, *(hit.concept for hit in situations))))
        snapshot: dict[str, str] = dict(baseline) if baseline else {}
        measures = facts.measures()
        candidates, prior = await self._recall(facts)
        mapped_at = self._clock()
        hits: dict[str, str | None] = {}
        unresolved: list[str] = []
        signals: list[str] = []

        askable: list[ConceptDefinition] = []
        settled: dict[str, str] = {}
        for concept in candidates:
            missing = [key for key in concept.required_baseline_keys if key not in snapshot]
            if missing:
                unresolved.append(concept.name)
                signals.append(f"unresolved: {concept.name} 缺常态 {', '.join(missing)}")
                continue
            if concept.is_mechanical:
                # 数值先判：规则说"不是"就不问模型；规则说"是"再问语义门（这条到底是不是入睡）。
                # 一条 07:00 的「吃了碗面」比常态就寝晚 7.5 小时，规则会通过——只有模型能说它不是入睡。
                decision = concept.decide(measures, snapshot)
                if decision.hit is None:
                    unresolved.append(concept.name)
                    signals.append(f"unresolved: {concept.name} 常态值解不出")
                    continue
                if not decision.hit:
                    continue
                if concept.rule is not None and decision.value is not None:
                    # 规则已经把数值那一半判成立了；渲染时告诉模型别再算（评审 C-8）。
                    settled[concept.name] = f"{concept.rule.criterion()}，这一条实际是 {decision.value:+.0f}"
            if concept.context is ContextScope.DAY and not timeline:
                unresolved.append(concept.name)
                signals.append(f"unresolved: {concept.name} 需要当天时间线")
                continue
            askable.append(concept)

        if askable:
            try:
                (answered, unseen), notes = await self._ask(facts, askable, snapshot, timeline, settled)
            except ModelClientError as exc:
                # 模型层的失败不出本模块：这些候选记未决，一条 occurrence 不把整天塌掉。
                unresolved.extend(concept.name for concept in askable)
                signals.append(f"model: {type(exc).__name__}: {clean_line(str(exc))[:120]}")
            else:
                signals.extend(notes)
                for concept in askable:
                    if concept.identity in answered:
                        hits[concept.identity] = concept.grade_for(measures, snapshot)
                    elif concept.identity in unseen:
                        # 模型说"看不到"（判据引用的事件落在观测空白里）→ 现成的未决，不是"没命中"。
                        unresolved.append(concept.name)
                        signals.append(f"unresolved: {concept.name} 模型答看不到（材料里判不了）")

        unresolved_identities = {concept_identity(name) for name in unresolved}
        for concept in candidates:
            if concept.identity in prior and concept.identity not in hits and concept.identity not in unresolved_identities:
                signals.append(f"flip: kind {facts.kind_token} 以前命中过 {concept.name}，这次没有")

        try:
            return ConceptHits(
                occurrence_uri=facts.uri,
                kind_token=facts.kind_token,
                last_observed_at=facts.last_observed_at,
                hits=tuple(ConceptHit(concept=self.concepts[identity].name, grade=grade) for identity, grade in hits.items()),
                situation_hits=situations,
                situations_checked=checked,
                unresolved=tuple(unresolved),
                baseline_snapshot=snapshot,
                mapper=self.version,
                mapped_at=mapped_at,
                signals=tuple(signals),
            )
        except ConceptHitsError as exc:
            raise ConceptMapperError(str(exc)) from exc

    async def _ask(
        self,
        facts: OccurrenceFacts,
        candidates: Sequence[ConceptDefinition],
        baseline: Mapping[str, str],
        timeline: Sequence[TimelineEntry],
        settled: Mapping[str, str],
    ) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], tuple[str, ...]]:
        request = build_request(
            facts,
            candidates,
            baseline,
            subject=self.subject,
            timeline=timeline,
            max_timeline_rows=self.config.max_timeline_rows,
            settled=settled,
        )
        schema = mapper_json_schema([item.name for item in candidates])
        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(
                    request,
                    schema=schema,
                    name="scene_concept_mapping",
                    validator=lambda parsed: assemble_verdicts(parsed, candidates),
                )
            except ModelTransportError:
                # 只重试传输层的瞬态错；答复本身不合格由结构层纠正。
                if attempt >= self.config.transient_retries:
                    raise
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
                continue
            notes: list[str] = []
            if response.validation_attempts > 1:
                notes.append(f"structured: answered on attempt {response.validation_attempts}")
            if response.parse_mode != "strict":
                notes.append(f"structured: json parsed via {response.parse_mode}")
            return cast("tuple[tuple[str, ...], tuple[str, ...]]", response.value), tuple(notes)
        raise AssertionError("unreachable")  # pragma: no cover

    def _situations(self, hits: Sequence[ConceptHit]) -> tuple[ConceptHit, ...]:
        resolved = tuple(hits)
        for hit in resolved:
            if not isinstance(hit, ConceptHit):
                raise ConceptMapperError("situation_hits must be ConceptHit values")
            if hit.identity not in self.concepts or not self.concepts[hit.identity].role.is_situation:
                raise ConceptMapperError(f"{hit.concept!r} is not a situation concept in this concept set")
            if hit.grade is not None and hit.grade not in {grade.name for grade in self.concepts[hit.identity].grades}:
                raise ConceptMapperError(f"concept {hit.concept!r} defines no grade {hit.grade!r}")
        return resolved


@dataclass(frozen=True)
class DayMappingReport:
    day: date
    mapped: int
    resumed: int
    duplicates_skipped: int
    stale_removed: int
    unresolved: int
    #: 盘上原本就有、这次又重新判了一遍的记录数（口径变了、或输入变了）。它加上 ``stale_removed`` 不为零，
    #: 就说明这一天的命中**变了**——靶它开的承诺与结算已经是脏的，夜批要撤了重开（评审 A-12 / B-5）。
    rewritten: int = 0

    @property
    def changed(self) -> bool:
        return self.rewritten > 0 or self.stale_removed > 0


async def map_closed_day(
    tree: BehaviorTree,
    store: ConceptHitStore,
    mapper: ConceptMapper,
    day: date,
    *,
    now: datetime,
    situation_for: Callable[[BehaviorDocument], SituationOutcome | Sequence[ConceptHit]],
    baseline_for: Callable[[BehaviorDocument], Mapping[str, str]],
    force: bool = False,
) -> DayMappingReport:
    """把已封口的一天从行为树映射到 ``occurrences/``，最后落完成标记。

    - 撞车消歧的重复（``original_name`` 非空）跳过，与读侧和预测夜批同口径；
    - 时间线里**带上这一天的空白段**（gap，未观测 / 没读懂）：不带的话时间线看上去连续，模型按"里面没有就是没发生"
      把被空白盖住的事件答成"没有"，而那在账本里是反面证据（用户 09-27"没看到就不算"）；
    - 先撤标记、清掉上轮多出来的叶子，再逐条映射——**续跑**：同口径、且输入没变的记录不重问模型（全判成了的；
      或只因缺常态而未决、常态表又没变的）；
    - 行为树这一天比盘上少（读不到、或缩水了）时拒绝清掉多出来的记录（多半是根指错或封口日算错），除非 ``force``；
    - ``situation_for`` / ``baseline_for`` 是必填的：情境与常态是映射的材料，缺了就是空栏和"永假"，
      要空就显式传 ``lambda _d: ()`` / ``lambda _d: {}``。
    """

    all_documents = tree.read_day(BehaviorKind.OCCURRENCE, day)
    documents = [document for document in all_documents if document.fields.get("original_name") is None]
    on_disk = len(store.read_day(day))
    if len(documents) < on_disk and not force:
        # 盘上比树上多：树读不到（根指错 / 封口日算错）或缩水了。静默清掉多出来的再盖章，就是"做完了但内容不对"。
        raise ConceptMapperError(
            f"the behaviour tree shows {len(documents)} occurrences on {day} but the hit store holds {on_disk}; pass force=True to drop the extra records"
        )
    keep = frozenset(document.address.identity_name for document in documents)
    removed = store.retain_only(day, keep)
    timeline = tuple(TimelineEntry.from_document(document) for document in documents) + tuple(
        TimelineEntry.from_gap(document) for document in tree.read_day(BehaviorKind.GAP, day)
    )
    resumed = rewritten = 0
    for document in documents:
        baseline = baseline_for(document)
        if store.exists(document.address):
            existing = store.read(document.address)
            if existing.mapper == mapper.version and _inputs_unchanged(existing, baseline):
                resumed += 1
                continue
            rewritten += 1
        situations = situation_for(document)
        if isinstance(situations, SituationOutcome):
            record = await mapper.map(
                document, situation_hits=situations.hits, situations_checked=situations.checked, baseline=baseline, timeline=timeline
            )
        else:
            record = await mapper.map(document, situation_hits=situations, baseline=baseline, timeline=timeline)
        store.write(record)
    marker = store.complete_day(day, records=len(documents), completed_at=now, mapper=mapper.version)
    return DayMappingReport(
        day=day,
        mapped=len(documents) - resumed,
        resumed=resumed,
        duplicates_skipped=len(all_documents) - len(documents),
        stale_removed=len(removed),
        unresolved=marker.unresolved,
        rewritten=rewritten,
    )


def _inputs_unchanged(existing: ConceptHits, baseline: Mapping[str, str]) -> bool:
    """续跑判据是"输入没变"，不是"没有未决"。

    全判成了当然不重问。带未决的记录只在**材料真的换了**时才重问：未决是因为缺常态、而常态表还是那一份 → 再问一遍
    答案不会变，每晚重问只是烧调用；模型这一次没答成（``model:`` 信号）、或当时没给时间线 → 材料现在有了，要重问。
    """

    if not existing.unresolved:
        return True
    if any(signal.startswith("model:") or "需要当天时间线" in signal for signal in existing.signals):
        return False
    return dict(existing.baseline_snapshot) == dict(baseline)


__all__ = [
    "MAPPER_PROMPT_VERSION",
    "MAPPER_SYSTEM_PROMPT",
    "MAPPER_VERSION",
    "SCHEMA_FINGERPRINT",
    "ConceptMapper",
    "ConceptMapperError",
    "DayMappingReport",
    "MapperConfig",
    "OccurrenceFacts",
    "VERDICTS",
    "TimelineEntry",
    "assemble_verdicts",
    "build_request",
    "map_closed_day",
    "mapper_json_schema",
]
