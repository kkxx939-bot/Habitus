"""触点②：给一个后件写它的前因假设——凭常识说"哪些行为可能影响它、往哪边"。

一次一个后件（与映射器一次一条 occurrence 同一个节奏）：提示词短、能把"这个后件的节律"讲清楚，
而"一个后件最多几条"这个闸也就自然落在一次调用上。后果面不用另问——每条假设都有一个后件，
把所有后件都问一遍，某个行为的"它导致了什么"就是别的后件那几轮的产物（``views.behaviours`` 拼两面）。

**输入只有两样**（B8：基准要和观测独立）：概念集（定义 + 层级）、每个行为概念的节律。
**不看观测、不看账、不看强度**——否则基准就成了对已有统计的复述，"先验" 与"读数"互相自证。

**不许穷举连线**（门槛能定 3 的前提）。83 个概念两两连就是几千条，那等于把"n≥3 就出数"的
误报率推到九成（实测：常识对七成时假的占 1/6，纯随机配对时占 9/10）。所以：一个后件最多
``MAX_DRAFTS_PER_CONSEQUENT`` 条、说不出理由就不写，宁少勿多；提示词明说会拿无关的对来测它
（安慰剂检验就是量这把筛子准不准的尺子）。

**分型与峰由算法定，不由模型定**（2026-09-27 裁定"分型看节律"；2026-09-30 裁定一 / 二-6；2026-10-01 定稿）：

- 后件**有节律**（树上天天有峰）→ **逐峰各一条**假设、各一本账（身份里带后果峰号）：三个咖啡峰的平时概率不同
  （0.55 / 0.40 / 0.30），前置条件也不同。
- 后件**无节律**（打球、就诊）→ 无节律型一条：``consequent_peak=None``，不数峰、没有时效，只有"来了"与
  "``released_by`` 命中"两种结法。
- **前因有节律也按峰分**：下午那杯与晚上那杯咖啡是两个前因，各一条假设、各算对晚睡的影响；峰外归 ``#0`` 不丢。
  多前因集合里**每个行为元素都分**（用户 10-01："多体下要看到每个行为对后果的影响"）。
- 次数方面不分后果峰：一条，``horizon`` = 一天的峰数（"晚睡当天咖啡喝几杯"）；前因照分。
- 峰的钟面时段从节律口抄进假设（``windows``），以后树重建不改。
- 上限由**配置**守（``ExpansionLimits``，用户 10-01"需要配置化"）：一个后件最多几本账、一个前因最多分几个峰；
  超了先不分前因峰、再截后果峰，截了报出来——不把算法的展开算成模型答错。

模型**不再猜"第几次机会"**：峰是钟面上的事实，不是常识能猜的。

**失败不写半批**：与触点① 同一条纪律——传输、结构、核对任一不过，这一轮一条都不写、留信号。

⚠ 提示词与 schema 是**草稿**，措辞要拿真实模型跑对照再定；调用那一层是实验脚本，不在仓库里。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
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
from habitus.scene.concepts.rhythm import Rhythm, rhythm_of
from habitus.scene.hypotheses.model import (
    MAX_ANTECEDENTS,
    Antecedent,
    Aspect,
    Direction,
    Hypothesis,
    HypothesisError,
    HypothesisOrigin,
    HypothesisSource,
    PeakWindow,
    TypePrior,
)

HYPOTHESIS_AUTHOR_PROMPT_VERSION = "scene_hypothesis_author_prompt_v1"
#: 一个后件最多写几条（**模型答的条数**，不是展开后的条数）。这是"不许穷举"那条裁定的闸，
#: 也是常识筛子的工作点；数值待定，重放时按安慰剂检验的结果复核。
MAX_DRAFTS_PER_CONSEQUENT = 6
#: 逐峰展开之后一个后件最多几本账的**缺省**（前因峰 × 后果峰：4 组前因 × 3 × 3 = 36）。真正用的数在 ``ExpansionLimits`` 里，
#: 由组合根从配置传（用户 10-01"需要配置化"）。
MAX_HYPOTHESES_PER_CONSEQUENT = 36
#: 一个前因最多按几个峰分；峰更多的前因不分（整条当一个前因）。**待定值**。
MAX_ANTECEDENT_PEAKS = 4
#: 严格模式下 ``type_prior`` 不能既是 enum 又可为 null，所以"没有先验"用一个显式取值表示。
NO_TYPE_PRIOR = "none"


class HypothesisAuthorError(ValueError):
    """假设作者的输入与概念集矛盾（例如后件不是行为概念）。"""


@dataclass(frozen=True)
class HypothesisProposal:
    """一个后件这一轮的产物：展开后的假设 + 留痕。``hypotheses`` 空 = 一条都不写。"""

    consequent: str
    hypotheses: tuple[Hypothesis, ...] = ()
    signals: tuple[str, ...] = ()

    @property
    def wrote_nothing(self) -> bool:
        return not self.hypotheses


HYPOTHESIS_AUTHOR_SYSTEM_PROMPT = """你在为一个行为找它的前因：哪些别的行为或情境，可能让它更容易/更不容易发生、
更晚/更早发生、发生得更多/更少。

给你的是这个人的一套概念（每个带判据句）、其中一个当**后件**，以及相关行为的作息节律。
你要输出 0 条或几条假设。每条假设 = 一组前件 + 影响哪个方面 + 往哪边。

最重要的一条：**只写说得通的**。
- 说不出"为什么这会影响那"的，不要写。写少了没关系，写错了会污染整套账。
- **不要把概念两两连一遍**。我们会拿编出来的无关配对来测你（例如"清理无用代码 → 就诊"），
  那种你应该一条都不给。
- 一个后件最多 {max_drafts} 条。挑最说得通的那几条。

每条假设要填的：
- antecedents：1–{max_antecedents} 个概念。**多个前件表示"这几件事一起才有这个影响"**（晚睡 + 出差中），
  不是"各自都有影响"——那是两条假设。前件里至少要有一个行为概念（它给这条账定锚点：从它开始算）。
  某个前件只在某一档时才有影响（晚睡里只有"重"那一档），在 grade 里写档名。
- aspect：probability（会不会发生）/ timing（什么时候发生）/ count（发生几次）。
- direction：probability 用 up/down（更容易/更不容易）；timing 用 up=更晚、down=更早；count 用 up=更多。
- type_prior：probability 必填 —— enabling（没有它根本不会发生）/ promoting（更容易）/ inhibiting（更不容易）；
  方向要和它一致（inhibiting 配 down，其余配 up）。timing 与 count 不用填。
- released_by：只在后件**没有节律**时才有意义 —— 哪些行为一发生就说明这个前提作废了
  （再挂一次号取代上一次）。不知道就留空。
- split_by：你怀疑这条影响只在某种情境下成立时，写那个情境概念（我们会分层看，不会替你下结论）。
- why：一句话，为什么这说得通。这句会被存下来给人看。

不要写的东西：
- **强度**（"影响很大""压 40%"）——大小由账算，你只说方向。
- 后件自己不能出现在前件里；前件也不能是后件的上级或下级（「打球 → 运动」是重言）。
- 概念名字照抄给你的，不要改写、不要新造概念。"""


def hypothesis_author_json_schema(concepts: ConceptSet, consequent: str) -> dict[str, Any]:
    """概念名字钉进 enum：模型编不出新概念，也连不到概念集之外的东西。

    与触点① 同一条纪律（2026-09-29 探针实测）：**严格模式**的后端（本机 codex 的 ``--output-schema``、
    OpenAI structured outputs）要求 ``required`` 列出每一个属性、且不认 ``minItems`` 这类边界关键字。
    所以 schema 只管字段与取值集合，条数与边界由 validator 管（``MAX_DRAFTS_PER_CONSEQUENT``、
    ``Hypothesis`` 自己的 1–4 个前件都在那边）。"可选"用可为 null / 可为空表来表达。
    """

    names = sorted(concepts[identity].name for identity in concepts)
    antecedent_names = [name for name in names if concept_identity(name) != concept_identity(consequent)]
    behaviours = sorted(concepts[identity].name for identity in concepts.behaviors())
    situations = sorted(concepts[identity].name for identity in concepts.situations())
    if not antecedent_names:
        raise HypothesisAuthorError("a hypothesis authoring schema needs at least one candidate antecedent")
    antecedent = {
        "type": "object",
        "additionalProperties": False,
        "required": ["concept", "grade"],
        "properties": {
            "concept": {"type": "string", "enum": antecedent_names},
            "grade": {"type": ["string", "null"], "description": "只在某一档才有影响时填档名，否则 null。"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypotheses"],
        "properties": {
            "hypotheses": {
                "type": "array",
                "description": f"0 条或最多 {MAX_DRAFTS_PER_CONSEQUENT} 条；说不出理由就给 []。",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["antecedents", "aspect", "direction", "type_prior", "released_by", "split_by", "why"],
                    "properties": {
                        "antecedents": {
                            "type": "array",
                            "items": antecedent,
                            "description": f"1–{MAX_ANTECEDENTS} 个前件：这几件事**一起**才有这个影响。",
                        },
                        "aspect": {"type": "string", "enum": [item.value for item in Aspect]},
                        "direction": {"type": "string", "enum": [item.value for item in Direction]},
                        "type_prior": {
                            "type": "string",
                            "enum": [*(item.value for item in TypePrior), NO_TYPE_PRIOR],
                            "description": f"probability 必填三者之一；timing 与 count 填 {NO_TYPE_PRIOR!r}。",
                        },
                        "released_by": {
                            "type": "array",
                            "items": {"type": "string", "enum": behaviours},
                            "description": "无节律的后件才用：谁一发生就说明前提作废；不知道填 []。",
                        },
                        "split_by": {
                            "type": "array",
                            "items": {"type": "string", "enum": situations},
                            "description": "怀疑只在某种情境下成立时写那个情境概念；否则 []。",
                        },
                        "why": {"type": "string", "description": "一句话：为什么这说得通。"},
                    },
                },
            }
        },
    }


def _fingerprint_set() -> ConceptSet:
    """给 schema 指纹用的最小概念集；指纹只要随 schema 形状变，不随概念集变。"""

    source = ConceptSource(ConceptOrigin.BASELINE)
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    return ConceptSet(
        (
            ConceptDefinition(name="甲", definition="样例", role=ConceptRole.BEHAVIOR, source=source, created_at=moment),
            ConceptDefinition(name="乙", definition="样例", role=ConceptRole.BEHAVIOR, source=source, created_at=moment),
        )
    )


HYPOTHESIS_AUTHOR_SCHEMA_FINGERPRINT = canonical_digest(hypothesis_author_json_schema(_fingerprint_set(), "甲"))[:12]
HYPOTHESIS_AUTHOR_VERSION = f"{HYPOTHESIS_AUTHOR_PROMPT_VERSION}+schema{HYPOTHESIS_AUTHOR_SCHEMA_FINGERPRINT}"
NO_RHYTHMS: Mapping[str, Rhythm] = MappingProxyType({})


def build_hypothesis_request(
    concepts: ConceptSet,
    consequent: str,
    rhythms: Mapping[str, Rhythm] = NO_RHYTHMS,
) -> ChatRequest:
    """材料三段：后件是哪个（含它的节律）、可选的前件有哪些、它们各自的作息。"""

    target = concepts[consequent]
    if not target.role.is_behavior:
        raise HypothesisAuthorError(f"consequent {consequent!r} must be a behaviour concept")
    own = rhythm_of(rhythms, target.name)
    sections = [
        "## 后件（要找它的前因）",
        f"- {target.name}：{target.definition}",
        f"- 它的节律：{own.render()}",
        "",
        "## 可以当前件的概念",
    ]
    for identity in sorted(concepts):
        item = concepts[identity]
        if identity == target.identity:
            continue
        grades = "；档：" + "/".join(grade.name for grade in item.grades) if item.grades else ""
        sections.append(f"- {item.name}（{item.role.value}{'，上级 ' + item.parent if item.parent else ''}）：{item.definition}{grades}")
    others = [rhythm.render() for name, rhythm in rhythms.items() if rhythm.peaks and concept_identity(name) != target.identity]
    if others:
        sections += ["", "## 这些前件平常的作息", *(f"- {line}" for line in others)]
    prompt = HYPOTHESIS_AUTHOR_SYSTEM_PROMPT.format(max_drafts=MAX_DRAFTS_PER_CONSEQUENT, max_antecedents=MAX_ANTECEDENTS)
    return ChatRequest(messages=(ChatMessage(role="system", content=prompt), ChatMessage(role="user", content="\n".join(sections))))


@dataclass(frozen=True)
class HypothesisDraft:
    """模型答复里的一条，已按形状核对过、还没按节律展开。触点③ 也用它——闭环提的假设走同一段"按节律分型、逐峰展开"。"""

    antecedents: tuple[Antecedent, ...]
    aspect: Aspect
    direction: Direction
    type_prior: TypePrior | None
    released_by: tuple[str, ...]
    split_by: tuple[str, ...]
    why: str


@dataclass(frozen=True)
class ExpansionLimits:
    """展开的两道闸（配置化）：一个后件最多几本账；一个前因最多按几个峰分（0 = 前因一律不分峰）。"""

    max_accounts_per_consequent: int = MAX_HYPOTHESES_PER_CONSEQUENT
    max_antecedent_peaks: int = MAX_ANTECEDENT_PEAKS

    def __post_init__(self) -> None:
        if isinstance(self.max_accounts_per_consequent, bool) or not isinstance(self.max_accounts_per_consequent, int) or self.max_accounts_per_consequent <= 0:
            raise ValueError("max_accounts_per_consequent must be a positive integer")
        if isinstance(self.max_antecedent_peaks, bool) or not isinstance(self.max_antecedent_peaks, int) or self.max_antecedent_peaks < 0:
            raise ValueError("max_antecedent_peaks must be a non-negative integer")


DEFAULT_LIMITS = ExpansionLimits()


def assemble_hypotheses(
    parsed: object,
    concepts: ConceptSet,
    consequent: str,
    rhythms: Mapping[str, Rhythm] = NO_RHYTHMS,
    *,
    now: datetime,
    origin: HypothesisOrigin = HypothesisOrigin.BASELINE,
    limits: ExpansionLimits = DEFAULT_LIMITS,
) -> tuple[tuple[Hypothesis, ...], tuple[str, ...]]:
    """核对 + 按节律展开成能落盘的假设，最后对着概念集逐条核对；不合格就整份不成形，交结构层重试。

    展开是**算法的活**：后件有节律就逐峰各一条，前因有节律也逐峰各一条（含峰外 ``#0``），没节律的一条；
    次数方面一条带 ``horizon``。一个后件的账数上限由 ``limits`` 守，按模型写了几组前因均摊。
    """

    drafts = _drafts(parsed)
    if not drafts:
        return (), ("author: 模型一条都没写（这在无关配对上是对的）",)
    hypotheses: list[Hypothesis] = []
    signals: list[str] = []
    allowance = max(1, limits.max_accounts_per_consequent // max(1, len(drafts)))
    for draft in drafts:
        built, notes = expand_draft(
            draft,
            consequent,
            rhythms,
            concepts,
            now=now,
            origin=origin,
            limits=ExpansionLimits(max_accounts_per_consequent=allowance, max_antecedent_peaks=limits.max_antecedent_peaks),
        )
        hypotheses.extend(built)
        signals.extend(notes)
    validate_all(hypotheses, concepts)
    identities = [item.identity for item in hypotheses]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        raise ValueError(f"two proposed hypotheses share one identity: {duplicates}")
    return tuple(hypotheses), tuple(signals)


def windows_of(rhythm: Rhythm) -> tuple[PeakWindow, ...]:
    """节律口的峰 → 写进假设的钟面时段表。"""

    return tuple(PeakWindow(peak.ordinal, peak.start_minute, peak.end_minute) for peak in rhythm.peaks)


def expand_draft(
    draft: HypothesisDraft,
    consequent: str,
    rhythms: Mapping[str, Rhythm],
    concepts: ConceptSet,
    *,
    now: datetime,
    origin: HypothesisOrigin,
    limits: ExpansionLimits = DEFAULT_LIMITS,
) -> tuple[tuple[Hypothesis, ...], tuple[str, ...]]:
    """一条草稿 → 前因峰 × 后果峰 本账。分型与峰在这里由节律定；模型只说了"谁影响谁、往哪边"。

    上限（``limits``）超了先**不分前因峰**（峰最多的前因先合回一条），再**截后果峰**（按钟面顺序留前几个），
    截了报出来——不把算法的展开算成模型答错。
    """

    label = "{" + ", ".join(item.label() for item in draft.antecedents) + "}"
    signals: list[str] = []
    own = rhythm_of(rhythms, consequent)
    windows: dict[str, tuple[PeakWindow, ...]] = {}
    # 后果这一侧
    if not own.has_rhythm:
        if draft.aspect is not Aspect.PROBABILITY:
            raise ValueError(
                f"{consequent!r} has no rhythm on the tree, so {draft.aspect.value} cannot be measured; only probability hypotheses fit it"
            )
        consequent_peaks: list[int | None] = [None]
    else:
        windows[concept_identity(consequent)] = windows_of(own)
        consequent_peaks = [1] if draft.aspect is Aspect.COUNT else list(range(1, own.opportunities_per_day + 1))
    horizon = own.opportunities_per_day if (draft.aspect is Aspect.COUNT and own.has_rhythm) else 1
    # 前因这一侧：每个有节律的行为元素按峰分（含峰外 0）
    variants: dict[str, list[int | None]] = {}
    for item in draft.antecedents:
        rhythm = rhythm_of(rhythms, item.concept)
        is_behaviour = item.identity in concepts and concepts[item.identity].role.is_behavior
        if is_behaviour and rhythm.has_rhythm and rhythm.opportunities_per_day <= limits.max_antecedent_peaks:
            windows[item.identity] = windows_of(rhythm)
            variants[item.identity] = [*range(1, rhythm.opportunities_per_day + 1), 0]
        else:
            if is_behaviour and rhythm.has_rhythm:
                signals.append(f"rhythm: {item.concept} 一天 {rhythm.opportunities_per_day} 个峰，超过前因分峰的上限 {limits.max_antecedent_peaks} → 不分峰")
            variants[item.identity] = [None]

    def total(current: Mapping[str, list[int | None]], peaks: Sequence[int | None]) -> int:
        count = len(peaks)
        for options in current.values():
            count *= len(options)
        return count

    # 超上限：先把峰最多的前因合回一条，再截后果峰
    while total(variants, consequent_peaks) > limits.max_accounts_per_consequent and any(len(v) > 1 for v in variants.values()):
        widest = max((name for name in variants if len(variants[name]) > 1), key=lambda name: len(variants[name]))
        signals.append(f"rhythm: {label} → {consequent} 展开超过上限 {limits.max_accounts_per_consequent} 本 → {widest} 不分峰")
        variants[widest] = [None]
        windows.pop(widest, None)
    if total(variants, consequent_peaks) > limits.max_accounts_per_consequent:
        keep = max(1, limits.max_accounts_per_consequent // max(1, total(variants, [None])))
        signals.append(f"rhythm: {label} → {consequent} 一天 {len(consequent_peaks)} 个峰，超过上限 → 这一轮只建前 {keep} 个")
        consequent_peaks = consequent_peaks[:keep]
    if own.has_rhythm and len(consequent_peaks) > 1:
        signals.append(f"rhythm: {label} → {consequent} 一天 {len(consequent_peaks)} 个峰，逐峰各一条")
    built: list[Hypothesis] = []
    for combination in _combinations(draft.antecedents, variants):
        for peak in consequent_peaks:
            built.append(_build(draft, combination, consequent, windows, now=now, origin=origin, consequent_peak=peak, horizon=horizon))
    return tuple(built), tuple(signals)


def _combinations(antecedents: Sequence[Antecedent], variants: Mapping[str, list[int | None]]) -> tuple[tuple[Antecedent, ...], ...]:
    """前因集合的每个元素各取一个峰号（或不分），做笛卡尔积。"""

    combos: list[tuple[Antecedent, ...]] = [()]
    for item in antecedents:
        combos = [(*prefix, Antecedent(item.concept, item.grade, peak)) for prefix in combos for peak in variants[item.identity]]
    return tuple(combos)


def _build(
    draft: HypothesisDraft,
    antecedents: tuple[Antecedent, ...],
    consequent: str,
    windows: Mapping[str, tuple[PeakWindow, ...]],
    *,
    now: datetime,
    origin: HypothesisOrigin,
    consequent_peak: int | None,
    horizon: int,
) -> Hypothesis:
    try:
        return Hypothesis(
            antecedents=antecedents,
            consequent=consequent,
            aspect=draft.aspect,
            direction=draft.direction,
            note=draft.why[:400],
            source=HypothesisSource(origin=origin, note=HYPOTHESIS_AUTHOR_VERSION),
            created_at=now,
            type_prior=draft.type_prior,
            split_by=draft.split_by,
            consequent_peak=consequent_peak,
            windows={name: items for name, items in windows.items() if name == concept_identity(consequent) or any(a.identity == name and a.peak is not None for a in antecedents)},
            horizon=horizon,
            released_by=draft.released_by if consequent_peak is None else (),
        )
    except HypothesisError as exc:
        raise ValueError(f"hypothesis {draft.why!r} is not usable: {exc}") from exc


def validate_all(hypotheses: Sequence[Hypothesis], concepts: ConceptSet) -> None:
    """对着概念集核对每一条（后件是行为、前件都在、档定义过、不是重言、分账项是情境）。"""

    for item in hypotheses:
        try:
            item.validate_against(concepts)
        except HypothesisError as exc:
            raise ValueError(f"{item.label()} does not hold against the concept set: {exc}") from exc


def _drafts(parsed: object) -> tuple[HypothesisDraft, ...]:
    if not isinstance(parsed, Mapping):
        raise ValueError("hypothesis author output must be an object")
    entries = parsed.get("hypotheses")
    if not isinstance(entries, list):
        raise ValueError("hypothesis author output must carry a hypotheses list")
    if len(entries) > MAX_DRAFTS_PER_CONSEQUENT:
        raise ValueError(f"write at most {MAX_DRAFTS_PER_CONSEQUENT} hypotheses for one consequent, got {len(entries)}")
    return tuple(_draft(entry) for entry in entries)


def _draft(entry: object) -> HypothesisDraft:
    if not isinstance(entry, Mapping):
        raise ValueError("each proposed hypothesis must be an object")
    raw = entry.get("antecedents")
    if not isinstance(raw, list) or not raw:
        raise ValueError("a hypothesis needs a non-empty antecedent set")
    antecedents = tuple(_antecedent(item) for item in raw)
    aspect = _enum(Aspect, entry.get("aspect"), "aspect")
    prior = entry.get("type_prior")
    if prior == NO_TYPE_PRIOR:
        prior = None
    return HypothesisDraft(
        antecedents=antecedents,
        aspect=aspect,
        direction=_enum(Direction, entry.get("direction"), "direction"),
        type_prior=None if prior in (None, "") else _enum(TypePrior, prior, "type_prior"),
        released_by=_names(entry.get("released_by"), "released_by"),
        split_by=_names(entry.get("split_by"), "split_by"),
        why=_text(entry.get("why"), "why"),
    )


def _antecedent(item: object) -> Antecedent:
    if not isinstance(item, Mapping):
        raise ValueError("each antecedent must be an object")
    grade = item.get("grade")
    try:
        return Antecedent(concept=_text(item.get("concept"), "antecedent concept"), grade=None if grade in (None, "") else _text(grade, "grade"))
    except HypothesisError as exc:
        raise ValueError(f"antecedent is not usable: {exc}") from exc


def _names(raw: object, label: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list of concept names")
    return tuple(_text(item, label) for item in raw)


@dataclass(frozen=True)
class HypothesisAuthorConfig:
    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0
    #: 展开的两道闸（配置化）：一个后件最多几本账、一个前因最多分几个峰。
    limits: ExpansionLimits = DEFAULT_LIMITS

    def __post_init__(self) -> None:
        if isinstance(self.transient_retries, bool) or not isinstance(self.transient_retries, int) or self.transient_retries < 0:
            raise ValueError("transient_retries must be a non-negative integer")
        if not isinstance(self.limits, ExpansionLimits):
            raise ValueError("limits must be ExpansionLimits")
        if (
            isinstance(self.transient_retry_delay_seconds, bool)
            or not isinstance(self.transient_retry_delay_seconds, int | float)
            or self.transient_retry_delay_seconds < 0
        ):
            raise ValueError("transient_retry_delay_seconds must be a non-negative number")


class HypothesisAuthor:
    """触点②。一次一个后件；失败这一轮一条都不写。"""

    def __init__(
        self,
        client: StructuredChatClient,
        *,
        config: HypothesisAuthorConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        resolved = config or HypothesisAuthorConfig()
        if not isinstance(resolved, HypothesisAuthorConfig):
            raise TypeError("config must be HypothesisAuthorConfig")
        self.client = client
        self.config = resolved
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    @property
    def version(self) -> str:
        return f"{HYPOTHESIS_AUTHOR_VERSION}+llm:{self.client.client.model}"

    async def propose(
        self,
        concepts: ConceptSet,
        consequent: str,
        rhythms: Mapping[str, Rhythm] = NO_RHYTHMS,
        *,
        origin: HypothesisOrigin = HypothesisOrigin.BASELINE,
    ) -> HypothesisProposal:
        if not isinstance(concepts, ConceptSet):
            raise TypeError("concepts must be a ConceptSet")
        name = concepts[consequent].name
        request = build_hypothesis_request(concepts, name, rhythms)
        schema = hypothesis_author_json_schema(concepts, name)
        now = self._clock()

        def validate(parsed: object) -> tuple[tuple[Hypothesis, ...], tuple[str, ...]]:
            return assemble_hypotheses(parsed, concepts, name, rhythms, now=now, origin=origin, limits=self.config.limits)

        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(
                    request, schema=schema, name="scene_hypothesis_authoring", validator=validate
                )
            except ModelTransportError:
                if attempt >= self.config.transient_retries:
                    return HypothesisProposal(consequent=name, signals=("model: transport failed; 这一轮不写",))
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
                continue
            except ModelClientError as exc:
                return HypothesisProposal(consequent=name, signals=(_failure(exc),))
            hypotheses, notes = cast("tuple[tuple[Hypothesis, ...], tuple[str, ...]]", response.value)
            signals = [f"author: {self.version}", *notes]
            if response.validation_attempts > 1:
                signals.append(f"structured: answered on attempt {response.validation_attempts}")
            if response.parse_mode != "strict":
                signals.append(f"structured: json parsed via {response.parse_mode}")
            return HypothesisProposal(consequent=name, hypotheses=hypotheses, signals=tuple(signals))
        raise AssertionError("unreachable")  # pragma: no cover


def _failure(exc: ModelClientError) -> str:
    """失败信号要说出**哪条规则不过**。

    结构层把 validator 的报错挂在 ``__cause__`` 上，只报 "model failed domain validation after 2 attempts"
    的话，看日志的人既不知道是形状不对还是核对不过，也不知道该改提示词的哪一句——探针第二轮就卡在这里。
    """

    cause = exc.__cause__
    detail = f"；{clean_line(str(cause))[:220]}" if cause is not None else ""
    return f"model: {type(exc).__name__}: {clean_line(str(exc))[:120]}{detail}"


def _text(value: object, label: str) -> str:
    text = clean_line(value)
    if not text:
        raise ValueError(f"{label} must be non-empty text")
    return text


def _enum(kind: type[Enum], value: object, label: str) -> Any:
    try:
        return kind(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be one of {[item.value for item in kind]}, got {value!r}") from exc


__all__ = [
    "HYPOTHESIS_AUTHOR_PROMPT_VERSION",
    "HYPOTHESIS_AUTHOR_SYSTEM_PROMPT",
    "HYPOTHESIS_AUTHOR_VERSION",
    "DEFAULT_LIMITS",
    "MAX_ANTECEDENT_PEAKS",
    "MAX_DRAFTS_PER_CONSEQUENT",
    "NO_TYPE_PRIOR",
    "MAX_HYPOTHESES_PER_CONSEQUENT",
    "NO_RHYTHMS",
    "ExpansionLimits",
    "HypothesisAuthor",
    "HypothesisAuthorConfig",
    "HypothesisAuthorError",
    "HypothesisProposal",
    "HypothesisDraft",
    "assemble_hypotheses",
    "build_hypothesis_request",
    "expand_draft",
    "windows_of",
    "hypothesis_author_json_schema",
    "validate_all",
]
