"""映射器：把一条 occurrence 判到细分概念上。语义树在映射这一步唯一的模型触点。

流程（裁定 20）：

1. **基础概念不判**  这条在行为树上的编号就是它的基础概念（``ConceptHits.graded_hits`` 从编号现读）。
   「非事件」不映射、不留记录；「待定」只记情境（没有类，挂不上细分概念）。
2. **候选 = 挂在这个类上的细分概念**  不召回、不按名字联想：源类之外的细分概念与这条无关。
3. **材料核对**  细分概念声明了要什么材料：常态值（``baseline_keys`` / 规则引用的常态）、当天时间线
   （``context=DAY``）、近几天同类记录（``context=RECENT``）。声明了却没给到 → 那一条记**未决**，
   不让模型替我们答成 false。
4. **数值区别由算法判**  带 ``MechanicalRule`` 的细分概念只由算法判（"晚睡 = 睡觉 + 比近期常态晚 ≥120 分钟"）：
   这条已经是睡觉，"是不是入睡"不用再问；算法从不碰语义，模型从不碰数字。
5. **语义区别由模型判**  剩下的一次一条 occurrence 交模型，问"这条已经是〔源类〕，它是否满足〔区别〕"，
   每个候选只答 yes / no / unknown；模型这一次没答成 → 这些候选记未决、留信号，模型层的异常不出本模块。
   **每条判两次**，第二次候选倒过来排：两次都 yes 才算命中、都 no 才算没有；答得不一样记未决（``INCONSISTENT``），
   两边都不计（冒烟实测同输入两遍 7 条里 3 处不同——不判两次，这些翻转会直接进关系检验的计数）。
6. **定档**  算法按细分概念的数值规则定档。
7. **情境**  那一刻成立的情境概念由调用方按算法填进来，这里核对它们是情境概念、档是定义过的。

**映射者看不到任何关系、任何统计**：输入只有 occurrence 的事实 + 声明的材料 + 候选的区别判据 + 常态值。
架构测试钉死本包只（传递性地）引用 ``concepts``。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, cast

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.model import BehaviorKind
from habitus.behavior.uri import BehaviorURI
from habitus.conditions import WEEKDAY_NAMES
from habitus.foundation.integrity import canonical_digest
from habitus.foundation.text import clean_line
from habitus.model_client import ChatMessage, ChatRequest, StructuredChatClient
from habitus.model_client.asking import Answered, Failed, ask
from habitus.scene.concepts.model import (
    BaselineKey,
    ConceptDefinition,
    ConceptSet,
    ContextScope,
    GradeMeasure,
)
from habitus.scene.occurrences.model import (
    MAX_SIGNAL_CHARS,
    ConceptHit,
    ConceptHits,
    ConceptHitsError,
    UnresolvedReason,
)
from habitus.series import SeriesGap, SeriesRecord

MAPPER_PROMPT_VERSION = "scene_concept_mapper_prompt_v4"

MAPPER_SYSTEM_PROMPT = """你在判断一条被观测到的行为满足哪些细分条件。

这条行为已经归到一个行为类（材料里写明了是哪一类），这一点不用再判。给你的材料：这条行为的事实（名字、
一句话概要、目标、步骤、开始时刻、最后所见时刻、同在的人、地点），可能还有这个人的常态值、它所在那一天的
时间线、或近几天同一类的记录（某个候选的区别要看这些时才给）。然后是若干候选细分概念，每个一句区别判据。
对每个候选，回答这条行为**满不满足**那句区别：满足 → yes，不满足 → no，**看不到、判不了 → unknown**。

规则：
- 只按区别判据判，不按概念名字的联想判；判据没说到的条件不要自己补。
- 判据里提到"常态"的，用材料里给出的常态值算偏离。
- 判据里提到别的事件（"起床后""第一次"）的，到当天时间线里找。时间线上标着「未观测」/「没读懂」的那几段是
  **观测空白**：要找的事件本该落在空白里、时间线上又没有 → 答 unknown（看不到，不是没发生）。时间线覆盖到了
  那段、里面确实没有，才答 no。
- 判据里提到"前几天""刚做过"的，到近几天同类记录里找；没列出的日子就是那几天没有这一类的记录。
- 材料里没写的事实一律当作不知道；判据只差这一条事实时答 unknown，**不要当成 no**——"没看到"和"没发生"是两件事，
  后者会被当成反面证据。
- 每个候选恰好答一次，不多不少，也不要新增候选。"""


class ConceptMapperError(ValueError):
    """映射器的输入与概念集矛盾。"""


#: 三态判定。``unknown`` 是"材料里看不到、判不了"（用户 09-27"没看到就不算"）——它不是第三种结果，
#: 落到现成的 ``unresolved`` 里：不进分母也不进分子。没有它的时候模型只能答 no，而 no 在关系检验里是反面证据。
VERDICTS = ("yes", "no", "unknown")


def mapper_json_schema(candidate_names: Sequence[str]) -> dict[str, Any]:
    """候选名字钉进 enum：模型答不出候选之外的名字，也答不了别的候选；判定三态（是 / 不是 / 看不到）。

    **严格模式吃得下的形状**（2026-09-29 探针实测）：真实后端（codex 的 ``--output-schema``、
    OpenAI structured outputs）不认 ``minItems`` / ``maxItems``，带着它们会 400。条数由
    ``assemble_verdicts`` 管，所以 schema 只说"有哪些字段、取值属于哪个集合"。
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
                        "concept": {"type": "string", "enum": names, "description": "候选细分概念的名字，照抄。"},
                        "verdict": {
                            "type": "string",
                            "enum": list(VERDICTS),
                            "description": "这条行为满不满足这个候选的区别判据：yes / no / unknown（材料看不到，判不了）。",
                        },
                    },
                },
                "description": f"每个候选恰好一条，共 {len(names)} 条，不多不少。",
            }
        },
    }


SCHEMA_FINGERPRINT = canonical_digest(mapper_json_schema(("候选",)))[:12]
#: 判法也是口径的一部分：判一次与判两次得出的命中不是同一种东西，版本跟着变，续跑时旧记录会重判。
#: 时间线截在这一条最后所见（裁定 27 第 10 条，E6）：给模型看这一条之后的事，细分标签会取决于之后发生了什么，
#: 而"同一条链""当天剩下的"检验的正是之后——等于把后果放进了前因的材料里。
JUDGEMENT = "judged-twice+timeline-to-self"
#: 提示词正文的指纹进版本：正文改了而忘了手动升版本号，缓存 / 续跑也不会拿旧答案冒充新提示词的答案（第四轮评审 E15）。
PROMPT_FINGERPRINT = canonical_digest({"prompt": MAPPER_SYSTEM_PROMPT})[:8]
MAPPER_VERSION = f"{MAPPER_PROMPT_VERSION}+prompt{PROMPT_FINGERPRINT}+schema{SCHEMA_FINGERPRINT}+{JUDGEMENT}"


@dataclass(frozen=True)
class TimelineEntry:
    """当天时间线上的一行：给要看前后事件的判据用。``gap`` 为真表示这一行是**观测空白**而不是一件行为。

    空白段必须列出来（用户 09-27"没看到就不算"）：不列的话时间线看上去是连续的，模型按"时间线里没有就是没发生"
    把被空白盖住的早餐答成 no，而 no 在关系检验里是反面证据——七d-2 的假负样本就是这么从映射口进来的。
    """

    uri: str
    name: str
    started_at: datetime
    last_observed_at: datetime
    summary: str
    gap: bool = False

    @classmethod
    def from_record(cls, record: SeriesRecord) -> TimelineEntry:
        return cls(
            uri=record.uri,
            name=record.name,
            started_at=record.started_at,
            last_observed_at=record.last_observed_at,
            summary=record.summary,
        )

    @classmethod
    def from_gap(cls, gap: SeriesGap) -> TimelineEntry:
        """一段观测空白（「未观测」或「没读懂」）；两者对映射都一样：那段发生了什么不知道。"""

        return cls(
            uri=gap.uri,
            name=gap.kind.value,
            started_at=gap.started_at,
            last_observed_at=gap.ended_at,
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
        if document.kind is not BehaviorKind.OCCURRENCE:
            raise ConceptMapperError("the mapper reads occurrence documents only")
        fields = document.fields
        started_at = document.address.started_at
        last_observed_at = datetime.fromisoformat(str(fields["last_observed_at"])).astimezone(started_at.tzinfo)
        if last_observed_at < started_at:
            raise ConceptMapperError(
                "occurrence last_observed_at precedes its start; the behaviour tree is inconsistent"
            )
        place, goal = fields.get("place"), fields.get("goal")
        steps = tuple(
            clean_line(step.get("semantics"))
            for step in fields.get("basis", ())
            if isinstance(step, Mapping) and clean_line(step.get("semantics"))
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
            f"开始：{started.strftime('%Y-%m-%d %H:%M')}（{WEEKDAY_NAMES[started.weekday()]}）",
            f"最后所见：{self.last_observed_at.strftime('%H:%M')}（历时约 {round(self.duration_minutes)} 分钟）",
            f"同在：{'、'.join(others) if others else '（无）'}",
            f"地点：{self.place if self.place else '（未知）'}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class MapperConfig:
    """时间线行数、近几天记录的天数与行数都是保护闸；数值在重放上定，这里只是默认。"""

    max_timeline_rows: int = 80
    recent_days: int = 7
    max_recent_rows: int = 40
    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        for label in ("max_timeline_rows", "recent_days", "max_recent_rows"):
            value = getattr(self, label)
            if value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.transient_retries < 0:
            raise ValueError("transient_retries must be a non-negative integer")
        if self.transient_retry_delay_seconds < 0:
            raise ValueError("transient_retry_delay_seconds must be a non-negative number")


@dataclass(frozen=True)
class MappingMaterial:
    """一条 occurrence 映射时可用的材料。``recent`` 为 None 是"没给"，``()`` 是"给了，近几天没有同类记录"。"""

    baseline: Mapping[str, str] = MappingProxyType({})
    timeline: Sequence[TimelineEntry] = ()
    recent: Sequence[TimelineEntry] | None = None

    def has(self, scope: ContextScope) -> bool:
        if scope is ContextScope.DAY:
            return bool(self.timeline)
        if scope is ContextScope.RECENT:
            return self.recent is not None
        return True


def build_request(
    facts: OccurrenceFacts,
    source_label: str,
    candidates: Sequence[ConceptDefinition],
    material: MappingMaterial,
    *,
    subject: str | None = None,
    max_timeline_rows: int = 80,
    max_recent_rows: int = 40,
) -> ChatRequest:
    """``source_label``：这条已经归到的类的名字——问模型的是"已经是它，满不满足区别"。"""

    if not candidates:
        raise ConceptMapperError("a mapping request needs at least one candidate")
    sections = [f"## 这条行为（已归为「{source_label}」）", facts.render(subject=subject)]
    if material.baseline:
        sections += ["", "## 这个人的常态", *(f"- {key}：{value}" for key, value in material.baseline.items())]
    scopes = {item.context for item in candidates}
    if ContextScope.DAY in scopes:
        if not material.timeline:
            raise ConceptMapperError("a candidate needs the day timeline but none was given")
        heading = "## 当天时间线（按时刻；标【未观测】/【没读懂】的是观测空白，那几段里发生过什么不知道）"
        sections += ["", heading, *_render_timeline(facts, material.timeline, max_timeline_rows)]
    if ContextScope.RECENT in scopes:
        if material.recent is None:
            raise ConceptMapperError("a candidate needs the recent records of this class but none were given")
        sections += ["", f"## 近几天「{source_label}」的记录", *_render_recent(material.recent, max_recent_rows)]
    sections += ["", "## 候选细分概念（每个答 yes / no / unknown）"]
    sections += [f"- {item.name}：{item.definition}" for item in candidates]
    return ChatRequest(
        messages=(
            ChatMessage(role="system", content=MAPPER_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n".join(sections)),
        )
    )


def _render_recent(rows: Sequence[TimelineEntry], limit: int) -> list[str]:
    if not rows:
        return ["（这几天没有）"]
    ordered = sorted(rows, key=lambda row: (row.started_at.astimezone(UTC), row.uri))[-limit:]
    return [
        f"- {row.started_at.strftime('%m-%d %H:%M')}（{WEEKDAY_NAMES[row.started_at.weekday()]}）{row.name}（{row.summary}）"
        for row in ordered
    ]


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


def assemble_verdicts(
    parsed: object, candidates: Sequence[ConceptDefinition]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """把模型的答复核对成 (命中的概念身份, 答"看不到"的概念身份)，都按候选顺序。

    每个候选恰好一条，否则整份不成形、交结构层重试。``unknown`` 不是"没命中"：它回到 ``unresolved``，
    关系检验那边"没看到就不算"（不进分母不进分子）。
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
        concepts: ConceptSet,
        *,
        config: MapperConfig | None = None,
        subject: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        resolved = config or MapperConfig()
        self.client = client
        self.concepts = concepts
        self.config = resolved
        self.subject = subject
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    def version_for(self, kind_token: str) -> str:
        """一条记录的口径：提示词 + 模型 + **挂在它那个类上的**细分概念的指纹。

        概念集里别处变了（别的类多了个细分概念、情境概念改了），这一条的候选没变，就不必再问模型——回填只花在真受影响的记录上
        （语义树新方案第四节："新增细分概念要把源类全部历史重新映射，走有预算的回填队列"）。情境由算法算，变了就地重算、不问模型。
        """

        candidates = [self.concepts[name].fingerprint for name in self.concepts.refinements_on(kind_token)]
        return f"{MAPPER_VERSION}+llm:{self.client.client.model}+candidates:{canonical_digest(candidates)[:16]}"

    def day_version(self, kind_tokens: Iterable[str]) -> str:
        """一天的完成标记的口径：那天出现的各个类各自的口径 + 情境概念的指纹（情境由算法就地重算、不问模型，但变了要回去重算）。
        别的 lane、别的类改了概念，这一天的口径不变，不被拉回去重做（第四轮评审 E9：以前用整个概念集的指纹，一处变动全部历史
        都成了"待做"，每晚预算 14 天，历史长了欠账永远还不完）。"""

        classes = sorted({self.version_for(token) for token in kind_tokens})
        situations = sorted(self.concepts[name].fingerprint for name in self.concepts.situations())
        return f"{MAPPER_VERSION}+day:{canonical_digest({'classes': classes, 'situations': situations})[:16]}"

    def needs(self, kind_token: str) -> frozenset[ContextScope]:
        """映射这个类的一条要哪些材料（调用方据此决定读不读近几天的记录）。"""

        return frozenset(self.concepts[name].context for name in self.concepts.refinements_on(kind_token))

    async def map(
        self,
        document: BehaviorDocument,
        *,
        lane: str,
        classified: bool = True,
        situation_hits: Sequence[ConceptHit] = (),
        situations_checked: Sequence[str] = (),
        material: MappingMaterial | None = None,
    ) -> ConceptHits:
        """``classified=False`` 是「待定」：只记情境。``lane`` 是这条所在的 lane（调用方从事件序列读）。"""

        facts = OccurrenceFacts.from_document(document)
        situations = self._situations(situation_hits)
        checked = tuple(dict.fromkeys((*situations_checked, *(hit.concept for hit in situations))))
        given = material or MappingMaterial()
        snapshot = dict(given.baseline)
        candidates = (
            tuple(self.concepts[name] for name in self.concepts.refinements_on(facts.kind_token)) if classified else ()
        )
        mapped_at = self._clock()
        verdict = await self._judge(facts, candidates, given)
        try:
            return ConceptHits(
                occurrence_uri=facts.uri,
                kind_token=facts.kind_token,
                classified=classified,
                last_observed_at=facts.last_observed_at,
                hits=tuple(
                    ConceptHit(concept=self.concepts[identity].name, grade=grade)
                    for identity, grade in verdict.hits.items()
                ),
                situation_hits=situations,
                situations_checked=checked,
                unresolved=verdict.unresolved,
                baseline_snapshot=snapshot,
                mapper=self.version_for(facts.kind_token),
                mapped_at=mapped_at,
                lane=lane,
                signals=verdict.signals,
                recent_digest=None if given.recent is None else recent_digest(given.recent),
                timeline_digest=timeline_digest(given.timeline)
                if classified and ContextScope.DAY in self.needs(facts.kind_token)
                else None,
            )
        except ConceptHitsError as exc:
            raise ConceptMapperError(str(exc)) from exc

    async def _judge(
        self, facts: OccurrenceFacts, candidates: Sequence[ConceptDefinition], material: MappingMaterial
    ) -> _Verdict:
        measures = facts.measures()
        verdict = _Verdict()
        askable: list[ConceptDefinition] = []
        for concept in candidates:
            missing = [key for key in concept.required_baseline_keys if key not in material.baseline]
            if missing:
                verdict.unsure(concept, UnresolvedReason.BASELINE_MISSING, f"缺常态 {', '.join(missing)}")
            elif concept.is_mechanical:
                self._decide(concept, measures, material.baseline, verdict)
            elif not material.has(concept.context):
                verdict.unsure(concept, *_MISSING_MATERIAL[concept.context])
            else:
                askable.append(concept)
        if askable:
            await self._ask_model(facts, askable, material, verdict)
        return verdict

    @staticmethod
    def _decide(
        concept: ConceptDefinition,
        measures: Mapping[GradeMeasure, float],
        baseline: Mapping[str, str],
        verdict: _Verdict,
    ) -> None:
        """数值区别：这条已经是源类，规则成立就是命中，不再问模型。"""

        decision = concept.decide(measures, baseline)
        if decision.hit is None:
            verdict.unsure(concept, UnresolvedReason.BASELINE_UNPARSEABLE, "常态值解不出")
        elif decision.hit:
            verdict.hits[concept.identity] = concept.grade_for(measures, baseline)

    async def _ask_model(
        self,
        facts: OccurrenceFacts,
        candidates: Sequence[ConceptDefinition],
        material: MappingMaterial,
        verdict: _Verdict,
    ) -> None:
        first = await self._ask(facts, candidates, material)
        second = (
            await self._ask(facts, tuple(reversed(candidates)), material) if not isinstance(first, Failed) else first
        )
        for answer in (first, second):
            if isinstance(answer, Failed):
                # 模型层的失败不出本模块：这些候选记未决，一条 occurrence 不把整天塌掉。
                verdict.unresolved.update((concept.name, UnresolvedReason.MODEL_FAILED) for concept in candidates)
                verdict.note(answer.signal)
                return
        assert isinstance(first, Answered) and isinstance(second, Answered)
        readings = [
            _readings(cast("tuple[tuple[str, ...], tuple[str, ...]]", answer.value)) for answer in (first, second)
        ]
        verdict.note(*first.notes, *second.notes)
        measures = facts.measures()
        for concept in candidates:
            once, twice = (reading(concept.identity) for reading in readings)
            if once != twice:
                verdict.unsure(concept, UnresolvedReason.INCONSISTENT, f"两次答得不一样（{once} / {twice}）")
            elif once == "yes":
                verdict.hits[concept.identity] = concept.grade_for(measures, material.baseline)
            elif once == "unknown":
                # 模型说"看不到"（区别引用的事件落在观测空白里）→ 现成的未决，不是"没命中"。
                verdict.unsure(concept, UnresolvedReason.MODEL_UNSEEN, "模型答看不到（材料里判不了）")

    async def _ask(
        self,
        facts: OccurrenceFacts,
        candidates: Sequence[ConceptDefinition],
        material: MappingMaterial,
    ) -> Answered | Failed:
        request = build_request(
            facts,
            self.concepts.class_label(facts.kind_token),
            candidates,
            replace(material, baseline=self._shown_baseline(material.baseline)),
            subject=self.subject,
            max_timeline_rows=self.config.max_timeline_rows,
            max_recent_rows=self.config.max_recent_rows,
        )
        return await ask(
            self.client,
            request,
            schema=mapper_json_schema([item.name for item in candidates]),
            name="scene_concept_mapping",
            validator=lambda parsed: assemble_verdicts(parsed, candidates),
            retries=self.config.transient_retries,
            delay_seconds=self.config.transient_retry_delay_seconds,
        )

    def _shown_baseline(self, baseline: Mapping[str, str]) -> Mapping[str, str]:
        """常态表给模型看的样子：键里的概念显示名字（类名），不显示编号（裁定 21-1）。"""

        shown: dict[str, str] = {}
        for text, value in baseline.items():
            key = BaselineKey.parse(text)
            shown[f"{self.concepts.label_of(key.concept)}的{key.window.label}{key.statistic.quantity}"] = value
        return shown

    def _situations(self, hits: Sequence[ConceptHit]) -> tuple[ConceptHit, ...]:
        resolved = tuple(hits)
        for hit in resolved:
            if hit.identity not in self.concepts or not self.concepts[hit.identity].role.is_situation:
                raise ConceptMapperError(f"{hit.concept!r} is not a situation concept in this concept set")
            if hit.grade is not None and hit.grade not in {grade.name for grade in self.concepts[hit.identity].grades}:
                raise ConceptMapperError(f"concept {hit.concept!r} defines no grade {hit.grade!r}")
        return resolved


def _readings(value: tuple[tuple[str, ...], tuple[str, ...]]) -> Callable[[str], str]:
    """（命中的, 看不到的）→ 一个候选的读法：yes / unknown / no。"""

    answered, unseen = value

    def reading(identity: str) -> str:
        return "yes" if identity in answered else "unknown" if identity in unseen else "no"

    return reading


_MISSING_MATERIAL: dict[ContextScope, tuple[UnresolvedReason, str]] = {
    ContextScope.DAY: (UnresolvedReason.TIMELINE_MISSING, "需要当天时间线"),
    ContextScope.RECENT: (UnresolvedReason.RECENT_MISSING, "需要近几天同类记录"),
}


class _Verdict:
    """一条 occurrence 的判定累积：命中（身份 → 档）、未决（名字 → 原因）、信号。"""

    def __init__(self) -> None:
        self.hits: dict[str, str | None] = {}
        self.unresolved: dict[str, UnresolvedReason] = {}
        self.signals: tuple[str, ...] = ()

    def unsure(self, concept: ConceptDefinition, reason: UnresolvedReason, note: str) -> None:
        self.unresolved[concept.name] = reason
        self.note(f"unresolved: {concept.name} {note}")

    def note(self, *texts: str) -> None:
        """留痕截到记录的单条上限之内：模型失败的信号带着核对报错，可能比上限长，截掉尾巴也不能让整条记录写不出。"""

        self.signals += tuple(clean_line(text)[:MAX_SIGNAL_CHARS] for text in texts)


def timeline_digest(rows: Sequence[TimelineEntry]) -> str:
    """ "当天时间线"这份材料的摘要：每一行的地址、时刻与说法。同一天后来补进一条记录（或一段空白），这里就会变。"""

    return canonical_digest(
        [
            [row.uri, row.name, row.started_at.isoformat(), row.last_observed_at.isoformat(), row.summary, row.gap]
            for row in rows
        ]
    )[:16]


def recent_digest(rows: Sequence[TimelineEntry]) -> str:
    """ "近几天同类记录"这份材料的摘要：哪几条（地址）。别的日子被迁移改了编号，这里就会变。"""

    return canonical_digest(sorted(row.uri for row in rows))[:16]


__all__ = [
    "MAPPER_PROMPT_VERSION",
    "MAPPER_SYSTEM_PROMPT",
    "MAPPER_VERSION",
    "SCHEMA_FINGERPRINT",
    "VERDICTS",
    "ConceptMapper",
    "ConceptMapperError",
    "MapperConfig",
    "MappingMaterial",
    "OccurrenceFacts",
    "TimelineEntry",
    "assemble_verdicts",
    "build_request",
    "mapper_json_schema",
    "recent_digest",
    "timeline_digest",
]
