"""触点③：算法扫出"不一样"，交给模型解释，它提结构假设——闭环的那半圈。

前半圈是算法的：稳定性扫出某条关系在两个情境下**分得开**（`stability.layers` 里两层区间不重叠），
或者某个概念的近期常态与历来常态**在漂**。两者都只是"不一样"这个事实，算法说不出**为什么**。

后半圈交给模型：给它两边各几天的 occurrence 叙事，让它读出那几天到底有什么不同，然后提 0～N 条
**结构假设**——三种之一：

- **直接因**：那个情境本身就是个原因（"赶工"让他不吃早饭），写成一条 ``{赶工中} → 后件`` 的假设；
- **调节体**：它不是原因，是让别的原因变强/变弱的条件——**情境进前件集合**（``{晚睡, 赶工中} → 后件``，
  设计稿七f-7 的原意"真正成立的是 {熬夜, 在家} → 打球"）。旧写法"前件还是原来那些、在 split_by 里写它"
  写出来与原假设同身份，永远被当成重复丢掉（评审 A-5 / C-10）；
- **链**：它经由第三件事起作用（"赶工 → 晚睡 → 不吃早饭"），写成中间那一段 ``{赶工中} → 晚睡``——
  所以后件的取值集合是**全部行为概念**，不能钉死成事实里的后件（评审 C-10 ②）。

**分型与逐峰是算法的**（与触点② 同一段代码 ``expand_draft``）：后件一天三个峰就三本账，没有节律就无节律型。
触点③ 自己不写峰号。

**B8：统计不回流。** 给模型的材料里**没有**强度、区间、兑现率、样本数——只有"这两层不一样"
（或"这个常态在往后漂"）这个事实加上两边的叙事。让模型看见数字，它下一轮写的假设就会去迎合数字，
先验与读数互相自证，账本就白记了。方向可以给（"往后漂"），大小不给。

**C3：从写入日起攒账、不回填。** 这是"不自证"的关键——**用来发现它的那批观测不算它的证据**。
所以它写出来的假设 ``created_at`` 是**夜批的时刻**（不是墙钟：重放 7 月的历史时墙钟是 9 月，
闸一加上就永远开不了账，评审 B-7），账本开账时按写入时刻拦（``ledger.opening.backfills``，裁定四）。

⚠ 提示词与 schema 是**草稿**，措辞要拿真实模型跑对照再定。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any, cast

from habitus.foundation.integrity import canonical_digest
from habitus.foundation.text import clean_line
from habitus.model_client import ChatMessage, ChatRequest, ModelClientError, ModelTransportError, StructuredChatClient
from habitus.scene.concepts.model import (
    ConceptDefinition,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ConceptSource,
    concept_identity,
)
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.hypotheses.author import (
    DEFAULT_LIMITS,
    NO_RHYTHMS,
    NO_TYPE_PRIOR,
    ExpansionLimits,
    HypothesisAuthorConfig,
    HypothesisDraft,
    expand_draft,
)
from habitus.scene.hypotheses.model import (
    MAX_ANTECEDENTS,
    Antecedent,
    Aspect,
    Direction,
    Hypothesis,
    HypothesisError,
    HypothesisOrigin,
    HypothesisSource,
    TypePrior,
)

CLOSURE_PROMPT_VERSION = "scene_closure_prompt_v1"
#: 一次触发最多提几条新假设。闭环每晚都会扫出一批"不一样"，不设闸的话假设集会自己膨胀，
#: 而每条新假设都要攒几周才知道成不成立。**待定值**。
MAX_CLOSURE_HYPOTHESES = 3
#: 给模型看的叙事最多几行（每边）。多了提示词会被一天几十条 occurrence 淹掉。
MAX_NARRATIVE_LINES = 20


class ClosureError(ValueError):
    """闭环的输入自相矛盾（例如两边一条叙事都没有）。"""


class Structure(str, Enum):
    """模型对"为什么不一样"的三种回答。写进假设的理由里，读的人要知道它当时是怎么想的。"""

    DIRECT = "direct"
    MODERATOR = "moderator"
    CHAIN = "chain"

    @property
    def label(self) -> str:
        return {Structure.DIRECT: "直接因", Structure.MODERATOR: "调节体", Structure.CHAIN: "链"}[self]


@dataclass(frozen=True)
class SplitFact:
    """算法扫出来的一个调节：这条关系按某个情境分层之后，两层分得开。

    **不带任何数字**（B8）。``narrative`` 是那几天的 occurrence 一行一条，模型靠它看出两边差在哪。
    """

    consequent: str
    antecedents: tuple[str, ...]
    situation: str
    with_situation: tuple[str, ...] = ()
    without_situation: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (self.consequent, self.situation, *self.antecedents):
            concept_identity(name)
        if not self.antecedents:
            raise ClosureError("a split fact names the antecedents of the relation it splits")
        if not self.with_situation or not self.without_situation:
            raise ClosureError("a split fact carries the narrative of both layers; one side alone explains nothing")

    def render(self) -> str:
        items = "、".join(self.antecedents)
        return (
            f"关系：{{{items}}} → {self.consequent}\n"
            f"分层：按「{self.situation}」分成两边之后，两边**明显不一样**（具体差多少不告诉你，也不要猜）。\n\n"
            f"### 「{self.situation}」在场的那几天\n"
            + "\n".join(f"- {line}" for line in self.with_situation[:MAX_NARRATIVE_LINES])
            + f"\n\n### 「{self.situation}」不在场的那几天\n"
            + "\n".join(f"- {line}" for line in self.without_situation[:MAX_NARRATIVE_LINES])
        )


@dataclass(frozen=True)
class DriftFact:
    """算法扫出来的一个漂移：某个概念的常态在往一个方向移。

    同样不带大小——只有"往后/往前"这个方向（与"基准只给方向不给大小"同向）。
    """

    concept: str
    quantity: str
    direction: str
    recent: tuple[str, ...] = ()
    earlier: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        concept_identity(self.concept)
        if not clean_line(self.quantity) or not clean_line(self.direction):
            raise ClosureError("a drift fact says which quantity moved and which way")
        if not self.recent or not self.earlier:
            raise ClosureError("a drift fact carries the narrative of both stretches")

    def render(self) -> str:
        return (
            f"漂移：「{self.concept}」的{self.quantity}在**往{self.direction}移**（移了多少不告诉你，也不要猜）。\n\n"
            f"### 近期那几天\n"
            + "\n".join(f"- {line}" for line in self.recent[:MAX_NARRATIVE_LINES])
            + "\n\n### 早先那几天\n"
            + "\n".join(f"- {line}" for line in self.earlier[:MAX_NARRATIVE_LINES])
        )


@dataclass(frozen=True)
class ClosureProposal:
    explanation: str = ""
    hypotheses: tuple[Hypothesis, ...] = ()
    signals: tuple[str, ...] = field(default_factory=tuple)

    @property
    def wrote_nothing(self) -> bool:
        return not self.hypotheses


CLOSURE_SYSTEM_PROMPT = """算法发现了一件"不一样"的事，但它说不出为什么。你来读那几天到底发生了什么，然后说清楚。

你会看到：一件"不一样"的事实，以及**两边各若干天的行为记录**（一行一条，按时刻）。
**没有任何数字**——不要问、也不要猜"差多少""涨了几成"，那与你要做的事无关。

你要做两件事：

一、**解释**：读两边的记录，说出它们到底差在哪（一两句，具体到行为，不要"可能有关"这种话）。

二、**提假设**：0～{max_hypotheses} 条。每条要说清是下面哪一种：
- direct（直接因）：这件事本身就是个原因 —— 写成 {{它}} → 后件；
- moderator（调节体）：它不是原因，是让别的原因变强或变弱的条件 —— 把它**加进原来的前件集合**，
  写成 {{原来的前件, 它}} → 后件（"晚睡只在赶工时才让他不吃早饭"就写 {{晚睡, 赶工中}} → 早餐）；
- chain（链）：它经由第三件事起作用（甲 → 乙 → 后件）—— 那就写**中间那一段**（{{甲}} → 乙，乙是另一个行为），
  因为后半段（乙 → 后件）多半已经有账了。

规则：
- **说不出来就一条都不提**。宁可只给解释、不给假设——每条假设都要再攒几周才知道成不成立，
  提错了是在浪费几周。
- 前件与后件只能用给你的概念名字，照抄，不要新造。**第几次机会不用你猜**，算法按后件的节律定。
- 前件集合里**至少要有一个行为**（情境是状态，当不了账的锚）：情境要当直接因或链的起点，就把它和
  触发它的那个行为一起写（{{赶工中, 修改代码}} → 晚睡）。
- 只说方向（更多/更少、更晚/更早），**不说大小**。
- 后件自己不能出现在前件里；前件也不能是后件的上级或下级。"""


def closure_json_schema(concepts: ConceptSet) -> dict[str, Any]:
    """概念名字钉进 enum。与另外两个触点同一条纪律：严格模式（``required`` 列全部属性、不用边界关键字）。

    后件的取值集合是**全部行为概念**：链要写中间那一段（{甲} → 乙），乙不是事实里的后件。
    "直接因 / 调节体的后件必须是事实里的后件"由 ``assemble_closure`` 按结构核对，不进 schema。
    """

    names = sorted(concepts[identity].name for identity in concepts)
    behaviours = sorted(concepts[identity].name for identity in concepts.behaviors())
    if not behaviours:
        raise ClosureError("a closure schema needs at least one behaviour concept")
    antecedent = {
        "type": "object",
        "additionalProperties": False,
        "required": ["concept", "grade"],
        "properties": {
            "concept": {"type": "string", "enum": names},
            "grade": {"type": ["string", "null"]},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["explanation", "hypotheses"],
        "properties": {
            "explanation": {"type": "string", "description": "两边到底差在哪，一两句，具体到行为。"},
            "hypotheses": {
                "type": "array",
                "description": f"0～{MAX_CLOSURE_HYPOTHESES} 条；说不出来就给 []。",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["structure", "antecedents", "consequent", "aspect", "direction", "type_prior", "why"],
                    "properties": {
                        "structure": {"type": "string", "enum": [item.value for item in Structure]},
                        "antecedents": {
                            "type": "array",
                            "items": antecedent,
                            "description": f"1–{MAX_ANTECEDENTS} 个前件。",
                        },
                        "consequent": {"type": "string", "enum": behaviours, "description": "direct / moderator 填事实里的后件；chain 填中间那个行为。"},
                        "aspect": {"type": "string", "enum": [item.value for item in Aspect]},
                        "direction": {"type": "string", "enum": [item.value for item in Direction]},
                        "type_prior": {"type": "string", "enum": [*(item.value for item in TypePrior), NO_TYPE_PRIOR]},
                        "why": {"type": "string"},
                    },
                },
            },
        },
    }


def _fingerprint_set() -> ConceptSet:
    """给 schema 指纹用的最小概念集；指纹随 schema 的形状变，不随概念集变。"""

    source = ConceptSource(ConceptOrigin.BASELINE)
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    return ConceptSet(
        (
            ConceptDefinition(
                name="甲", definition="样例", role=ConceptRole.BEHAVIOR, source=source, created_at=moment
            ),
            ConceptDefinition(name="乙", definition="样例", role=ConceptRole.STATE, source=source, created_at=moment),
        )
    )


CLOSURE_SCHEMA_FINGERPRINT = canonical_digest(closure_json_schema(_fingerprint_set()))[:12]
CLOSURE_VERSION = f"{CLOSURE_PROMPT_VERSION}+schema{CLOSURE_SCHEMA_FINGERPRINT}"


def build_closure_request(fact: SplitFact | DriftFact, concepts: ConceptSet) -> ChatRequest:
    """材料三段：那件"不一样"的事 + 两边的叙事 + 能用的概念名字。**没有数字**。"""

    if not isinstance(concepts, ConceptSet):
        raise TypeError("concepts must be a ConceptSet")
    lines = ["## 算法发现的不一样", fact.render(), "", "## 能用的概念（名字照抄）"]
    for identity in sorted(concepts):
        item = concepts[identity]
        lines.append(f"- {item.name}（{item.role.value}）：{item.definition}")
    prompt = CLOSURE_SYSTEM_PROMPT.format(max_hypotheses=MAX_CLOSURE_HYPOTHESES)
    return ChatRequest(
        messages=(ChatMessage(role="system", content=prompt), ChatMessage(role="user", content="\n".join(lines)))
    )


def assemble_closure(
    parsed: object,
    concepts: ConceptSet,
    fact: SplitFact | DriftFact | None = None,
    *,
    now: datetime,
    known: Sequence[Hypothesis] = (),
    rhythms: Mapping[str, Rhythm] = NO_RHYTHMS,
    limits: ExpansionLimits = DEFAULT_LIMITS,
) -> tuple[str, tuple[Hypothesis, ...], tuple[str, ...]]:
    """核对成 (解释, 新假设, 留痕)。已经有的假设不重复写；写不成的单条丢掉并报出来，不否决整批。

    - **按结构核对**：直接因 / 调节体的后件必须是事实里的后件，前件里要有那个情境（不然与原假设同身份）；
      链的后件必须是**别的**行为（中间那一段）。``structure`` 不再只是标签。
    - **按节律展开**：走触点② 的 ``expand_draft``——后件有节律就逐峰各一条、前因有节律也按峰分，没节律就无节律型，
      上限由 ``limits`` 守。
    """

    if not isinstance(parsed, Mapping):
        raise ValueError("closure output must be an object")
    explanation = clean_line(parsed.get("explanation"))
    if not explanation:
        raise ValueError("closure output must carry an explanation")
    entries = parsed.get("hypotheses")
    if not isinstance(entries, list):
        raise ValueError("closure output must carry a hypotheses list")
    if len(entries) > MAX_CLOSURE_HYPOTHESES:
        raise ValueError(f"a closure round proposes at most {MAX_CLOSURE_HYPOTHESES} hypotheses, got {len(entries)}")
    seen = {item.identity for item in known}
    built: list[Hypothesis] = []
    signals: list[str] = []
    for entry in entries:
        draft, consequent, structure, why = _draft(entry, concepts, fact)
        if draft is None or consequent is None or structure is None:
            signals.append(f"dropped: {why}")
            continue
        try:
            expanded, notes = expand_draft(draft, consequent, rhythms, concepts, now=now, origin=HypothesisOrigin.MODERATION, limits=limits)
            for hypothesis in expanded:
                hypothesis.validate_against(concepts)
        except (HypothesisError, ValueError) as exc:
            signals.append(f"dropped: 写不成一条假设：{clean_line(str(exc))[:140]}")
            continue
        signals.extend(notes)
        for hypothesis in expanded:
            stamped = _stamp(hypothesis, structure)
            if stamped.identity in seen:
                signals.append(f"dropped: {stamped.label()} 已经有账了，不重复写")
                continue
            seen.add(stamped.identity)
            built.append(stamped)
    return explanation, tuple(built), tuple(signals)


def _stamp(hypothesis: Hypothesis, structure: Structure) -> Hypothesis:
    """来源标成闭环、理由前面写上它是哪一种结构（读的人要知道当时是怎么想的）。"""

    return replace(
        hypothesis,
        note=f"{structure.label}：{hypothesis.note}"[:400],
        source=HypothesisSource(origin=HypothesisOrigin.MODERATION, note=f"闭环 {CLOSURE_VERSION}｜{structure.value}"),
    )


def _draft(
    entry: object, concepts: ConceptSet, fact: SplitFact | DriftFact | None
) -> tuple[HypothesisDraft | None, str | None, Structure | None, str]:
    if not isinstance(entry, Mapping):
        return None, None, None, "答复里有一条不是对象"
    raw_structure = entry.get("structure")
    if raw_structure not in {item.value for item in Structure}:
        return None, None, None, f"structure 只能是 {[item.value for item in Structure]}，给的是 {raw_structure!r}"
    structure = Structure(raw_structure)
    raw = entry.get("antecedents")
    if not isinstance(raw, list) or not raw:
        return None, None, None, "前件集合是空的"
    prior = entry.get("type_prior")
    try:
        antecedents = tuple(
            Antecedent(
                concept=clean_line(item.get("concept")),
                grade=None if item.get("grade") in (None, "") else clean_line(item.get("grade")),
            )
            for item in raw
            if isinstance(item, Mapping)
        )
        consequent = clean_line(entry.get("consequent"))
        if not consequent:
            return None, None, None, "后件是空的"
        aspect = Aspect(entry.get("aspect"))
        draft = HypothesisDraft(
            antecedents=antecedents,
            aspect=aspect,
            direction=Direction(entry.get("direction")),
            type_prior=None if prior in (None, "", NO_TYPE_PRIOR) else TypePrior(prior),
            released_by=(),
            split_by=(),
            why=clean_line(entry.get("why")) or "（没给理由）",
        )
    except (HypothesisError, ValueError) as exc:
        return None, None, None, f"写不成一条假设：{clean_line(str(exc))[:140]}"
    problem = _structure_mismatch(structure, antecedents, consequent, fact)
    if problem is not None:
        return None, None, None, problem
    return draft, consequent, structure, ""


def _structure_mismatch(
    structure: Structure, antecedents: Sequence[Antecedent], consequent: str, fact: SplitFact | DriftFact | None
) -> str | None:
    """这一条与它自称的结构对不对得上。对不上就丢那一条——``structure`` 不是标签。"""

    if fact is None:
        return None
    subject = fact.situation if isinstance(fact, SplitFact) else fact.concept
    target = fact.consequent if isinstance(fact, SplitFact) else fact.concept
    names = {item.identity for item in antecedents}
    if structure is Structure.CHAIN:
        if concept_identity(consequent) == concept_identity(target):
            return f"链要写中间那一段（{{…}} → 别的行为 → {target}），后件不能还是 {target}"
        return None
    if concept_identity(consequent) != concept_identity(target):
        return f"{structure.label}的后件必须是 {target}，给的是 {consequent}"
    if isinstance(fact, SplitFact):
        if concept_identity(subject) not in names:
            return f"{structure.label}的前件里要有「{subject}」——不然与原假设同一条账"
        if structure is Structure.MODERATOR and not (names & {concept_identity(name) for name in fact.antecedents}):
            return f"调节体要把「{subject}」加进原来的前件集合 {set(fact.antecedents)}，不是另起一条"
    return None


class ClosureAuthor:
    """触点③。一次一个"不一样"，失败这一轮不写。"""

    def __init__(
        self,
        client: StructuredChatClient,
        *,
        config: HypothesisAuthorConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        self.client = client
        self.config = config or HypothesisAuthorConfig()
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    @property
    def version(self) -> str:
        return f"{CLOSURE_VERSION}+llm:{self.client.client.model}"

    async def propose(
        self,
        fact: SplitFact | DriftFact,
        concepts: ConceptSet,
        *,
        known: Sequence[Hypothesis] = (),
        rhythms: Mapping[str, Rhythm] = NO_RHYTHMS,
        now: datetime | None = None,
    ) -> ClosureProposal:
        """``now`` 是夜批的时刻——写下的假设从这一刻起攒账。不传才退到自己的时钟（离线脚本）。"""

        request = build_closure_request(fact, concepts)
        schema = closure_json_schema(concepts)
        moment = now if now is not None else self._clock()

        def validate(parsed: object) -> tuple[str, tuple[Hypothesis, ...], tuple[str, ...]]:
            return assemble_closure(parsed, concepts, fact, now=moment, known=known, rhythms=rhythms, limits=self.config.limits)

        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(
                    request, schema=schema, name="scene_closure", validator=validate
                )
            except ModelTransportError:
                if attempt >= self.config.transient_retries:
                    return ClosureProposal(signals=("model: transport failed; 这一轮不写",))
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
                continue
            except ModelClientError as exc:
                cause = exc.__cause__
                detail = f"；{clean_line(str(cause))[:220]}" if cause is not None else ""
                return ClosureProposal(signals=(f"model: {type(exc).__name__}: {clean_line(str(exc))[:120]}{detail}",))
            explanation, built, notes = cast("tuple[str, tuple[Hypothesis, ...], tuple[str, ...]]", response.value)
            signals = [f"closure: {self.version}", *notes]
            if response.validation_attempts > 1:
                signals.append(f"structured: answered on attempt {response.validation_attempts}")
            return ClosureProposal(explanation=explanation, hypotheses=built, signals=tuple(signals))
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = [
    "CLOSURE_PROMPT_VERSION",
    "CLOSURE_SYSTEM_PROMPT",
    "CLOSURE_VERSION",
    "MAX_CLOSURE_HYPOTHESES",
    "MAX_NARRATIVE_LINES",
    "ClosureAuthor",
    "ClosureError",
    "ClosureProposal",
    "DriftFact",
    "SplitFact",
    "Structure",
    "assemble_closure",
    "build_closure_request",
    "closure_json_schema",
]
