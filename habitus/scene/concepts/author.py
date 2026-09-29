"""触点①：把行为树上那一批散名字（kind）写成一组可判的概念定义。

语义树的输入是概念，而上游给的是 kind——上游按自己的粒度切的名字（实测 45 天 83 个 kind / 603 条
occurrence，其中 44 个 kind 只出现 1–2 次、合起来只占 5%）。这一口把那批名字翻成"能对着一条
occurrence 答是/否"的判据句，映射器再拿它去判。

**粒度三档**（2026-09-28 裁定，写进提示词）：

1. ``≥10`` 次 → 给它一个叶子概念，自己一本账；
2. ``3–9`` 次 → 不单独定义，找语义相近的几个**凑成一个概念**（合计凑到 ≥10）；
3. ``1–2`` 次 → 不定义，留在残差，等残差升级判据自己捞。

10 是待定值（实测 ≥10 次的 15 个 kind 盖住 72% 的 occurrence；n=10 能抓到 75% 的真效应、n=3 只抓到 55%），
重放时按真实分布复核，而且**要按 lane 各自量**（这批是 coding agent lane，生活侧密度不同）。

**概念可以比 kind 粗，不能比 kind 细**：上游把很多不同的事折进一个 kind 时语义分辨率就钉在那儿了，
概念层级救不回来（那要改上游的折叠，属事件融合那条线）。往粗的上限不是样本数而是"这个概念当后件时
说不出一条说得通的前因"——那就是定宽了，该拆。

**模型写什么、算法写什么**：模型写名字、判据句、角色、层级、认领哪几个 kind，以及"和常态比多少"
这种**定义层面**的数值规则；算法核对（覆盖、认领不重、上级存在、不成环、规则引用的常态键有主），
并且**每一条 occurrence 的数值仍由算法算**（``ConceptDefinition.decide``），模型永不碰。

**失败不写半批**：模型这一次没答成（传输、配额、结构两轮都不成形）或答复过不了核对，整批一个都不写、
留信号；那些 kind 留在残差里，下一次再问。半批写进去的后果是概念集指纹变了、全部命中要重算，
而缺的那几个 kind 又得再来一轮。

⚠ 提示词与 schema 是**草稿**：措辞要用真实模型跑对照再定（本机 codex `gpt-5.6-terra` 为主，
回退 Claude Code 的 sonnet）。调用那一层是实验脚本，不在仓库里。
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
    MAX_DEFINITION_CHARS,
    MAX_GRADES,
    MIN_GRADES,
    BaselineKey,
    ConceptDefinition,
    ConceptError,
    ConceptGrade,
    ConceptOrigin,
    ConceptRole,
    ConceptSet,
    ConceptSource,
    ContextScope,
    GradeMeasure,
    MechanicalRule,
    concept_identity,
)
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.concepts.situation import SituationBasis, SituationError, SituationRule

CONCEPT_AUTHOR_PROMPT_VERSION = "scene_concept_author_prompt_v2"  # v2：2026-09-29 探针后加"跨场合复发"那一条
#: 三档的两条线。都是待定值，重放时按真实分布与 lane 各自复核。
MIN_LEAF_OCCURRENCES = 10
MIN_GROUPED_OCCURRENCES = 3
#: 一批最多写多少个概念。一次写几十个的话核对不过来、也说明 kind 清单没先筛过（保护闸）。
MAX_CONCEPTS_PER_BATCH = 40
MAX_EXAMPLES_PER_KIND = 5
#: 空的节律表 / 空的认领表；冻结的默认值不能是可变字典。
NO_RHYTHMS: Mapping[str, Rhythm] = MappingProxyType({})
_NO_CLAIMS: Mapping[str, tuple[str, ...]] = MappingProxyType({})


class ConceptAuthorError(ValueError):
    """概念作者的输入自相矛盾（例如同一个 kind 给了两份摘要）。"""


class KindTier(str, Enum):
    """一个 kind 该怎么处置，由次数定，算法算，不问模型。"""

    LEAF = "leaf"
    GROUP = "group"
    RESIDUE = "residue"

    @property
    def instruction(self) -> str:
        if self is KindTier.LEAF:
            return "给它一个概念"
        return "和相近的凑成一个概念" if self is KindTier.GROUP else "不要定义，留在残差"


@dataclass(frozen=True)
class KindBrief:
    """一个 kind 交给模型看的全部材料：名字、次数、跨几天、几条真例子。

    例子是**真的** occurrence 的一行说法（名字 + 概要），由调用方从行为树取——模型凭名字联想会把
    「清理无用代码」写成"打扫"。``days`` 用来分辨"3 次挤在同一天"和"3 周各一次"。
    """

    kind_token: str
    occurrences: int
    days: int
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind_token", clean_line(self.kind_token))
        if not self.kind_token:
            raise ConceptAuthorError("a kind brief needs a kind token")
        for label in ("occurrences", "days"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConceptAuthorError(f"kind brief {label} must be a positive integer")
        if self.days > self.occurrences:
            raise ConceptAuthorError("a kind cannot span more days than it has occurrences")
        examples = tuple(clean_line(item) for item in self.examples if clean_line(item))
        object.__setattr__(self, "examples", examples[:MAX_EXAMPLES_PER_KIND])

    @property
    def tier(self) -> KindTier:
        if self.occurrences >= MIN_LEAF_OCCURRENCES:
            return KindTier.LEAF
        return KindTier.GROUP if self.occurrences >= MIN_GROUPED_OCCURRENCES else KindTier.RESIDUE

    def render(self) -> str:
        examples = "；".join(self.examples) if self.examples else "（没给例子）"
        return f"- {self.kind_token}：45 天里 {self.occurrences} 次、跨 {self.days} 天 → {self.tier.instruction}｜例：{examples}"


@dataclass(frozen=True)
class ConceptProposal:
    """一批提案：写成的定义 + 每个概念认领了哪几个 kind + 留痕。

    ``definitions`` 空 = 这一批一个都没写（模型失败或核对没过）；调用方据此**什么都不落盘**。
    ``claims`` 不进概念文件（``ConceptSource`` 只在残差升级那一支记 ``kind_token``），它是给人看的
    核对材料：认领关系在盘上由命中记录反推（``views.kinds.concept_kinds``）。
    """

    definitions: tuple[ConceptDefinition, ...] = ()
    claims: Mapping[str, tuple[str, ...]] = _NO_CLAIMS
    signals: tuple[str, ...] = ()
    #: 被丢掉的概念与理由（它们认领的 kind 退回残差）。不静默：下一轮还要靠它知道为什么少了一个概念。
    dropped: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", MappingProxyType(dict(self.claims)))

    @property
    def wrote_nothing(self) -> bool:
        return not self.definitions


CONCEPT_AUTHOR_SYSTEM_PROMPT = """你在给一个人的行为定义一套概念，供后面逐条判定使用。

材料是上游切好的行为名字（kind），每个带出现次数、跨了几天、几条真实例子，可能还带它的作息节律。
你要输出一组概念，每个概念一句**判据句**。

判据句的要求：对着一条具体的行为记录（名字、概要、目标、步骤、开始时刻、历时、同在的人、地点）
能答"是"或"不是"。写"这是一次为了修复缺陷而改动代码的行为"，不要写"与编程有关的活动"。

粒度按材料里标好的处置来，不要自己改：
- 标「给它一个概念」的：单独一个概念，认领它自己。
- 标「和相近的凑成一个概念」的：几个语义相近的合成一个概念，一起认领（例：排查 CI 失败 + 排查性能问题 +
  排查设备故障 → 一个概念「排查问题」）。凑不出相近的就别勉强，留着不认领。
- 标「不要定义，留在残差」的：一个都不要认领，在 skipped 里说一句为什么。

其他规则：
- 一个 kind 只能被一个概念认领；不要认领材料里没给的名字。
- 概念可以比 kind **粗**（几个 kind 凑一个概念，或几个概念共一个上级），**不能比 kind 细**——
  上游把不同的事折进了同一个名字，你也分不开，不要假装分得开。
- 层级：细的各自一个概念，想要粗的那一层就给它们同一个 parent（上级自己不认领任何 kind）。
  上级只在"细的几个确实是同一类"时才写，不要为了整齐造上级。
- **行为种类要能跨场合复发**。上游给的名字常常带着当时那个项目/仓库/对象（"为 Tagent 添加 ReAct 支持"、
  "重构 MemoryOS 的 API 目录"），那是**那一次**的说法，不是行为种类。概念要写成换个项目也还成立的那一层
  （"改代码"、"重构目录结构"、"调研实现"）——否则项目一换，这个概念再也不会命中，它的账就永远攒不起来。
  项目、仓库、工具、人这些"跟谁/在哪"属**情境**（role 填 object），不要写进行为概念的名字或判据里。
- **持续状况不是行为**："出差中""赶工中""生病中"这类持续的状态写成情境概念（role 填 state / object /
  day_type），它们不认领 kind。行为概念（role = behavior）只能来自上游给的 kind。
- 数值：你可以写一条"和常态比"的规则（例：入睡时刻比常态就寝晚 120 分钟以上），那是**定义**；
  每一条记录的数值由算法算，你不要算、也不要在判据句里写具体某一天的数字。
  写了规则才能写档（2–3 档，按同一个量切开、不重叠）。
- 判据要看当天前后发生了什么的（"起床后的第一次进食"），context 填 day；要用到某个行为的常态值的，
  在 baseline_keys 里写成 `<概念>:usual_start:recent` 或 `<概念>:usual_duration:recent`（三段：谁的、
  哪种统计、哪个窗）。窗一律填 `recent`（他近期的习惯）——`all`（历来常态）只用来读"他在漂移"，
  不能当判据。
- 已经有的概念不要重写，也不要改名；新概念的名字不能和已有的重名。

情境概念还要写一条 situation —— 算法照它算，所以只能用下面这几种说法之一：
- weekdays：名义上的周几（0=周一 … 6=周日），例：周末 = [5, 6]。
- calendar_note：当地日历那句话里含某个词，例：调休日 = "补班"。
- subject：同在的人里有谁。  · place：地点是哪儿。
- open_claim：某个行为"该做还没做"（它还有一条没结的承诺），例：约了球还没打 = 盯「打球」。
- streak：某个概念**连着 N 天**命中（N ≥ 2），例：赶工中 = 「写代码」连着 3 天。
- yesterday：昨天命中过某个概念（可以指定档），例：昨晚晚睡 = 昨天命中「晚睡」的重档。
role 填 derived 的**必须**写 situation（派生的定义就是"算法从历史算"）。写不成上面任何一种的状态
（"出差中""生病中"）就**别写 situation**：我们暂时算不出它，留着以后接数据源，不要编一个凑合的规则。"""


def concept_author_json_schema(briefs: Sequence[KindBrief]) -> dict[str, Any]:
    """kind 名字钉进 enum：模型认领不了材料里没有的名字，也编不出新 kind。

    **写成严格模式吃得下的形状**（2026-09-29 探针实测）：本机 codex 的 ``--output-schema`` 与
    OpenAI 的 structured outputs 都要求 ``required`` 列出 ``properties`` 里的**每一个**键，
    并且不认 ``minItems`` / ``maxItems`` / ``maxLength`` / ``minimum`` 这类边界关键字。
    不这么写，真实后端会直接拒（探针第一轮就是这么连挂三次、回退到另一个后端的）。

    所以分工是：**schema 只管"有哪些字段、取值属于哪个集合"，边界与条数一律由 validator 管**
    （它本来就在管：条数、长度、区间都在 ``assemble_concepts`` 与 ``ConceptDefinition`` 里）。
    可选的字段用"可为 null / 可为空表"表达，不靠"不出现在 required 里"表达。
    """

    tokens = [brief.kind_token for brief in briefs]
    if not tokens:
        raise ConceptAuthorError("the concept author schema needs at least one kind brief")
    if len(set(tokens)) != len(tokens):
        raise ConceptAuthorError("kind briefs must name each kind once")
    rule = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": ["measure", "lower_minutes", "upper_minutes", "relative_to"],
        "properties": {
            "measure": {"type": "string", "enum": [item.value for item in GradeMeasure]},
            "lower_minutes": {"type": ["integer", "null"]},
            "upper_minutes": {"type": ["integer", "null"]},
            "relative_to": {
                "type": ["string", "null"],
                "description": "比的是谁的常态，形如 就寝:usual_start:recent（窗必须是 recent）；比绝对值就填 null。",
            },
        },
    }
    grade = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "lower_minutes", "upper_minutes"],
        "properties": {
            "name": {"type": "string"},
            "lower_minutes": {"type": "integer"},
            "upper_minutes": {"type": "integer"},
        },
    }
    situation = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": ["basis", "weekdays", "value", "concept", "grade", "days"],
        "properties": {
            "basis": {"type": "string", "enum": [item.value for item in SituationBasis]},
            "weekdays": {"type": "array", "items": {"type": "integer"}, "description": "日型用：0=周一 … 6=周日；其他族填 []。"},
            "value": {"type": ["string", "null"], "description": "calendar_note / subject / place 要匹配的值。"},
            "concept": {"type": ["string", "null"], "description": "open_claim / streak / yesterday 盯的那个概念。"},
            "grade": {"type": ["string", "null"], "description": "只在某一档才算时填档名。"},
            "days": {"type": ["integer", "null"], "description": "streak 连着几天（≥2）；其他族填 null。"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["concepts", "skipped"],
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "definition", "role", "parent", "claims", "context", "baseline_keys", "rule", "grades", "situation", "why"],
                    "properties": {
                        "name": {"type": "string", "description": "概念的名字，短、像个名词。"},
                        "definition": {"type": "string", "description": f"判据句：对着一条记录能答是/否，{MAX_DEFINITION_CHARS} 字以内。"},
                        "role": {"type": "string", "enum": [item.value for item in ConceptRole]},
                        "parent": {"type": ["string", "null"], "description": "上级概念的名字；没有就填 null。"},
                        "claims": {"type": "array", "items": {"type": "string", "enum": tokens}, "description": "这个概念认领的 kind；情境概念填 []。"},
                        "context": {"type": "string", "enum": [item.value for item in ContextScope]},
                        "baseline_keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "判据句要用到的常态值，形如 就寝:usual_start:recent（三段：概念、统计、窗；窗填 recent）；不用就填 []。",
                        },
                        "rule": rule,
                        "grades": {"type": "array", "items": grade, "description": f"只有写了 rule 才能写档；写就写 {MIN_GRADES}–{MAX_GRADES} 档，按同一个量切开、不重叠；不写填 []。"},
                        "situation": situation,
                        "why": {"type": "string", "description": "一句话：为什么这么切。"},
                    },
                },
                "description": f"最多 {MAX_CONCEPTS_PER_BATCH} 个；一个都不该定义时给 []。",
            },
            "skipped": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind", "why"],
                    "properties": {"kind": {"type": "string", "enum": tokens}, "why": {"type": "string"}},
                },
                "description": "没有认领的 kind 与理由；没有就填 []。",
            },
        },
    }


CONCEPT_AUTHOR_SCHEMA_FINGERPRINT = canonical_digest(concept_author_json_schema((KindBrief("样例", 1, 1),)))[:12]
CONCEPT_AUTHOR_VERSION = f"{CONCEPT_AUTHOR_PROMPT_VERSION}+schema{CONCEPT_AUTHOR_SCHEMA_FINGERPRINT}"


def build_concept_request(
    briefs: Sequence[KindBrief],
    existing: ConceptSet,
    rhythms: Mapping[str, Rhythm] = NO_RHYTHMS,
) -> ChatRequest:
    """材料三段：要处置的 kind、已有的概念（不要重写）、相关行为的节律。"""

    if not briefs:
        raise ConceptAuthorError("a concept authoring request needs at least one kind brief")
    sections = ["## 上游给的行为名字", *(brief.render() for brief in briefs)]
    if len(existing):
        sections += ["", "## 已经有的概念（不要重写、不要重名，可以当 parent）"]
        sections += [f"- {existing[identity].name}（{existing[identity].role.value}）：{existing[identity].definition}" for identity in sorted(existing)]
    lines = [rhythm.render() for rhythm in rhythms.values() if rhythm.peaks]
    if lines:
        sections += ["", "## 这些行为平常的作息（一天几次、几点）", *(f"- {line}" for line in lines)]
    return ChatRequest(
        messages=(
            ChatMessage(role="system", content=CONCEPT_AUTHOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n".join(sections)),
        )
    )


@dataclass(frozen=True)
class _Draft:
    """模型答复里的一条，已按形状核对过、还没变成 ``ConceptDefinition``。"""

    name: str
    definition: str
    role: ConceptRole
    parent: str | None
    claims: tuple[str, ...]
    context: ContextScope
    baseline_keys: tuple[str, ...]
    rule: MechanicalRule | None
    grades: tuple[ConceptGrade, ...]
    situation: SituationRule | None
    why: str | None


def assemble_concepts(
    parsed: object,
    briefs: Sequence[KindBrief],
    existing: ConceptSet,
    *,
    now: datetime,
    origin: ConceptOrigin = ConceptOrigin.BASELINE,
) -> tuple[tuple[ConceptDefinition, ...], Mapping[str, tuple[str, ...]], tuple[str, ...]]:
    """核对成一组能落盘的定义；返回 (定义, 认领, 丢掉了什么)。

    **单个概念不合格就丢那一个，不否决整批**（2026-09-29 探针实测的教训）：真实模型一次写 25 个概念，
    其中一个把两个 kind 凑成 8 次（差 2 次不到 10）——按"整批否决"的老写法，另外 24 个合格的概念连同
    那一次调用全废。丢掉的那个概念认领的 kind 退回残差，这正是残差的定义（"攒够了再升级"），
    而"够不够样本"本来就该算法算、不该让模型负责。

    **仍然整批否决的只有整体不自洽**：答复不成形、``≥10`` 次的 kind 没有独占概念（那是真的丢语义，
    要模型重答）、概念集本身构不成（上级不存在、成环、与已有重名）。
    """

    drafts = _drafts(parsed, briefs)
    if not drafts:
        return (), {}, ()
    by_kind = {brief.kind_token: brief for brief in briefs}
    kept: list[_Draft] = []
    dropped: list[str] = []
    claimed: dict[str, str] = {}
    parents = {concept_identity(draft.parent) for draft in drafts if draft.parent is not None}
    for draft in drafts:
        reason = _unusable(draft, by_kind, claimed, parents)
        if reason is not None:
            dropped.append(f"dropped: {draft.name}（{reason}）→ 它认领的 {list(draft.claims) or '—'} 退回残差")
            continue
        for kind in draft.claims:
            claimed[kind] = draft.name
        kept.append(draft)
    _require_coverage(kept, by_kind, claimed)
    definitions = _definitions(kept, existing, now=now, origin=origin)
    return definitions, {draft.name: draft.claims for draft in kept if draft.claims}, tuple(dropped)


def _unusable(draft: _Draft, by_kind: Mapping[str, KindBrief], claimed: Mapping[str, str], parents: frozenset[str] | set[str]) -> str | None:
    """这一个概念为什么不能用；能用就返回 None。"""

    for kind in draft.claims:
        if kind in claimed:
            return f"{kind} 已经被 {claimed[kind]} 认领了"
    if not draft.role.is_behavior and draft.claims:
        return "情境概念不认领 kind"
    if draft.role.is_behavior and not draft.claims and draft.rule is None and concept_identity(draft.name) not in parents:
        # 行为概念有三条命中路径：认领 kind、带数值规则、或它是上级。三条都没有就永远不会被命中。
        return "既不认领 kind、又没有数值规则、也不是谁的上级，永远不会命中"
    leaves = [kind for kind in draft.claims if by_kind[kind].tier is KindTier.LEAF]
    if not leaves and draft.claims:
        total = sum(by_kind[kind].occurrences for kind in draft.claims)
        if total < MIN_LEAF_OCCURRENCES:
            return f"凑出来只有 {total} 次，不到 {MIN_LEAF_OCCURRENCES}"
    return None


def _require_coverage(drafts: Sequence[_Draft], by_kind: Mapping[str, KindBrief], claimed: Mapping[str, str]) -> None:
    """留下来的这些必须满足**整体性**的两条，不满足就整批重答（2026-09-28 裁定的三档）。

    1. ``≥10`` 次的 kind 必须有主，而且**自己一个概念**——把 98 次的「修改代码」和 26 次的「审查代码」
       并成一个就把语义丢了（用户 2026-09-28 纠正的正是这一点）；要粗的那一层靠 ``parent``。
    2. 一个概念不能既认领 ``≥10`` 次的 kind、又捎上别的（同上：那是并组）。

    "凑出来不到 10 次"不在这里——那条按单个概念丢掉、kind 退残差（见 ``_unusable``）。
    """

    missing = [token for token, brief in by_kind.items() if brief.tier is KindTier.LEAF and token not in claimed]
    if missing:
        raise ValueError(f"these kinds occur at least {MIN_LEAF_OCCURRENCES} times and each needs a concept of its own: {missing}")
    for draft in drafts:
        leaves = [kind for kind in draft.claims if by_kind[kind].tier is KindTier.LEAF]
        if leaves and len(draft.claims) > 1:
            raise ValueError(
                f"concept {draft.name!r} claims {leaves} (at least {MIN_LEAF_OCCURRENCES} occurrences each) together with "
                f"{[kind for kind in draft.claims if kind not in leaves]}; give each such kind its own concept and group them under a parent"
            )


def _drafts(parsed: object, briefs: Sequence[KindBrief]) -> tuple[_Draft, ...]:
    if not isinstance(parsed, Mapping):
        raise ValueError("concept author output must be an object")
    entries = parsed.get("concepts")
    if not isinstance(entries, list):
        raise ValueError("concept author output must carry a concepts list")
    if not entries:
        # 空答复是合法的一种答复（"这批一个都不该定义"）；调用方据此什么都不写。
        return ()
    tokens = {brief.kind_token for brief in briefs}
    drafts = tuple(_draft(entry, tokens) for entry in entries)
    identities = [concept_identity(draft.name) for draft in drafts]
    if len(set(identities)) != len(identities):
        raise ValueError("two proposed concepts share one name")
    return drafts


def _draft(entry: object, tokens: frozenset[str] | set[str]) -> _Draft:
    if not isinstance(entry, Mapping):
        raise ValueError("each proposed concept must be an object")
    claims = _list(entry.get("claims"), "claims")
    if any(item not in tokens for item in claims):
        raise ValueError(f"concept claims must name the given kinds, got {claims!r}")
    if len(set(claims)) != len(claims):
        raise ValueError("a concept claims each kind once")
    keys = _list(entry.get("baseline_keys"), "baseline_keys")
    rule = _rule(entry.get("rule"))
    return _Draft(
        name=_text(entry.get("name"), "concept name"),
        definition=_text(entry.get("definition"), "concept definition"),
        role=_enum(ConceptRole, entry.get("role"), "role"),
        parent=None if entry.get("parent") in (None, "") else _text(entry.get("parent"), "parent"),
        claims=tuple(str(item) for item in claims),
        context=_enum(ContextScope, entry.get("context") or ContextScope.OCCURRENCE.value, "context"),
        baseline_keys=tuple(_text(key, "baseline key") for key in keys),
        rule=rule,
        grades=_grades(_list(entry.get("grades"), "grades"), rule),
        situation=_situation(entry.get("situation")),
        why=clean_line(entry.get("why")) or None,
    )


def _list(value: object, label: str) -> list[Any]:
    """缺省当空表；给了别的东西就是答复不成形，交结构层重试。"""

    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return list(value)


def _rule(raw: object) -> MechanicalRule | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("rule must be an object or null")
    relative = raw.get("relative_to")
    try:
        return MechanicalRule(
            measure=_enum(GradeMeasure, raw.get("measure"), "rule measure"),
            lower=_optional_int(raw.get("lower_minutes"), "rule lower_minutes"),
            upper=_optional_int(raw.get("upper_minutes"), "rule upper_minutes"),
            relative_to=None if relative in (None, "") else _text(relative, "rule relative_to"),
        )
    except ConceptError as exc:
        raise ValueError(f"rule is not usable: {exc}") from exc


def _grades(raw: Sequence[Any], rule: MechanicalRule | None) -> tuple[ConceptGrade, ...]:
    if not raw:
        return ()
    if rule is None:
        # 无规则的档没有"比的是什么"这个参照；v1 只支持规则带档（档比的就是规则算出的那个量）。
        raise ValueError("grades need a rule to say what quantity they split; drop the grades or add the rule")
    try:
        return tuple(
            ConceptGrade(
                name=_text(item.get("name") if isinstance(item, Mapping) else None, "grade name"),
                measure=rule.measure,
                lower=_required_int(item.get("lower_minutes") if isinstance(item, Mapping) else None, "grade lower_minutes"),
                upper=_required_int(item.get("upper_minutes") if isinstance(item, Mapping) else None, "grade upper_minutes"),
                relative=rule.is_relative,
            )
            for item in raw
        )
    except ConceptError as exc:
        raise ValueError(f"grades are not usable: {exc}") from exc


def _situation(raw: object) -> SituationRule | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("situation must be an object or null")
    weekdays = _list(raw.get("weekdays"), "situation weekdays")
    if any(isinstance(day, bool) or not isinstance(day, int) for day in weekdays):
        raise ValueError("situation weekdays must be integers 0–6")
    days = raw.get("days")
    try:
        return SituationRule(
            basis=_enum(SituationBasis, raw.get("basis"), "situation basis"),
            weekdays=tuple(weekdays),
            value=None if raw.get("value") in (None, "") else _text(raw.get("value"), "situation value"),
            concept=None if raw.get("concept") in (None, "") else _text(raw.get("concept"), "situation concept"),
            grade=None if raw.get("grade") in (None, "") else _text(raw.get("grade"), "situation grade"),
            days=1 if days is None else _required_int(days, "situation days"),
        )
    except SituationError as exc:
        raise ValueError(f"situation is not computable: {exc}") from exc


def _definitions(
    drafts: Sequence[_Draft],
    existing: ConceptSet,
    *,
    now: datetime,
    origin: ConceptOrigin,
) -> tuple[ConceptDefinition, ...]:
    """变成 ``ConceptDefinition``、核对常态键有主，最后整组构造一次 ``ConceptSet``（上级与环在那里抛）。"""

    names = {concept_identity(draft.name): draft.name for draft in drafts}
    for identity, name in names.items():
        if identity in existing:
            raise ValueError(f"concept {name!r} already exists; do not rewrite it")
    built: list[ConceptDefinition] = []
    for draft in drafts:
        watched = draft.situation.concept if draft.situation is not None else None
        if watched is not None and concept_identity(watched) not in existing and concept_identity(watched) not in names:
            raise ValueError(f"the situation rule of {draft.name!r} watches {watched!r}, which is not a concept here")
        for key in _baseline_keys(draft):
            owner = concept_identity(key.concept)
            if owner not in existing and owner not in names:
                raise ValueError(f"concept {draft.name!r} names a baseline of {key.concept!r}, which is not a concept here")
        note = f"{CONCEPT_AUTHOR_VERSION}｜{draft.why}" if draft.why else CONCEPT_AUTHOR_VERSION
        kind_token = draft.claims[0] if origin is ConceptOrigin.RESIDUE and draft.claims else None
        if origin is ConceptOrigin.RESIDUE and len(draft.claims) > 1:
            raise ValueError(f"a residue upgrade claims one kind; {draft.name!r} claims {list(draft.claims)}")
        try:
            built.append(
                ConceptDefinition(
                    name=draft.name,
                    definition=draft.definition,
                    role=draft.role,
                    source=ConceptSource(origin=origin, note=note[:400], kind_token=kind_token),
                    created_at=now,
                    parent=draft.parent,
                    grades=draft.grades,
                    rule=draft.rule,
                    context=draft.context,
                    baseline_keys=draft.baseline_keys,
                    situation=draft.situation,
                )
            )
        except ConceptError as exc:
            raise ValueError(f"concept {draft.name!r} is not usable: {exc}") from exc
    try:
        ConceptSet([*(existing[identity] for identity in sorted(existing)), *built])
    except ConceptError as exc:
        raise ValueError(f"the proposed concepts do not form a consistent set: {exc}") from exc
    return _parents_first(built)


def _baseline_keys(draft: _Draft) -> tuple[BaselineKey, ...]:
    texts = list(draft.baseline_keys)
    if draft.rule is not None and draft.rule.relative_to is not None:
        texts.append(draft.rule.relative_to)
    try:
        return tuple(BaselineKey.parse(text) for text in texts)
    except ConceptError as exc:
        raise ValueError(f"concept {draft.name!r} names an unusable baseline key: {exc}") from exc


def _parents_first(definitions: Sequence[ConceptDefinition]) -> tuple[ConceptDefinition, ...]:
    """``ConceptStore.write`` 要求上级先落盘，所以这里就把顺序排好，调用方按序写。"""

    remaining = list(definitions)
    ordered: list[ConceptDefinition] = []
    while remaining:
        pending = {item.identity for item in remaining}
        ready = [item for item in remaining if item.parent_identity is None or item.parent_identity not in pending]
        if not ready:  # pragma: no cover - ConceptSet 已经拒了环
            raise ValueError("the proposed hierarchy has a cycle")
        ordered.extend(ready)
        remaining = [item for item in remaining if item not in ready]
    return tuple(ordered)


@dataclass(frozen=True)
class AuthorConfig:
    """瞬态重试的两个数；与映射器同一套（传输层重试，答复不合格由结构层纠正）。"""

    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.transient_retries, bool) or not isinstance(self.transient_retries, int) or self.transient_retries < 0:
            raise ValueError("transient_retries must be a non-negative integer")
        if (
            isinstance(self.transient_retry_delay_seconds, bool)
            or not isinstance(self.transient_retry_delay_seconds, int | float)
            or self.transient_retry_delay_seconds < 0
        ):
            raise ValueError("transient_retry_delay_seconds must be a non-negative number")


class ConceptAuthor:
    """触点①。一次一批 kind，失败就整批不写。"""

    def __init__(
        self,
        client: StructuredChatClient,
        *,
        config: AuthorConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        resolved = config or AuthorConfig()
        if not isinstance(resolved, AuthorConfig):
            raise TypeError("config must be AuthorConfig")
        self.client = client
        self.config = resolved
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)

    @property
    def version(self) -> str:
        return f"{CONCEPT_AUTHOR_VERSION}+llm:{self.client.client.model}"

    async def propose(
        self,
        briefs: Sequence[KindBrief],
        existing: ConceptSet,
        rhythms: Mapping[str, Rhythm] = NO_RHYTHMS,
        *,
        origin: ConceptOrigin = ConceptOrigin.BASELINE,
    ) -> ConceptProposal:
        if not isinstance(existing, ConceptSet):
            raise TypeError("existing must be a ConceptSet")
        askable = tuple(briefs)
        if not any(brief.tier is not KindTier.RESIDUE for brief in askable):
            # 全是一两次的长尾：凑不出一个够 10 次的概念，一个调用都不必花。
            return ConceptProposal(signals=(f"skipped: 全部 {len(askable)} 个 kind 都不到 {MIN_GROUPED_OCCURRENCES} 次，留在残差",))
        request = build_concept_request(askable, existing, rhythms)
        schema = concept_author_json_schema(askable)
        now = self._clock()
        for attempt in range(self.config.transient_retries + 1):
            try:
                response = await self.client.complete_json_async(
                    request,
                    schema=schema,
                    name="scene_concept_authoring",
                    validator=lambda parsed: assemble_concepts(parsed, askable, existing, now=now, origin=origin),
                )
            except ModelTransportError:
                if attempt >= self.config.transient_retries:
                    return ConceptProposal(signals=("model: transport failed; 这一批不写，kind 留在残差",))
                await asyncio.sleep(self.config.transient_retry_delay_seconds * (attempt + 1))
                continue
            except ModelClientError as exc:
                # 结构两轮都没成形、或核对没过：整批不写，下一次再问。
                return ConceptProposal(signals=(_failure(exc),))
            definitions, claims, dropped = cast(
                "tuple[tuple[ConceptDefinition, ...], Mapping[str, tuple[str, ...]], tuple[str, ...]]", response.value
            )
            signals = [f"author: {self.version}"]
            if not definitions:
                signals.append("author: 模型说这批一个都不该定义，什么都不写")
            if response.validation_attempts > 1:
                signals.append(f"structured: answered on attempt {response.validation_attempts}")
            if response.parse_mode != "strict":
                signals.append(f"structured: json parsed via {response.parse_mode}")
            signals.extend(dropped)
            return ConceptProposal(definitions=definitions, claims=claims, signals=tuple(signals), dropped=dropped)
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


def _optional_int(value: object, label: str) -> int | None:
    return None if value is None else _required_int(value, label)


def _required_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer number of minutes")
    return value


__all__ = [
    "CONCEPT_AUTHOR_PROMPT_VERSION",
    "CONCEPT_AUTHOR_SYSTEM_PROMPT",
    "CONCEPT_AUTHOR_VERSION",
    "MAX_CONCEPTS_PER_BATCH",
    "MIN_GROUPED_OCCURRENCES",
    "MIN_LEAF_OCCURRENCES",
    "NO_RHYTHMS",
    "AuthorConfig",
    "ConceptAuthor",
    "ConceptAuthorError",
    "ConceptProposal",
    "KindBrief",
    "KindTier",
    "assemble_concepts",
    "build_concept_request",
    "concept_author_json_schema",
]
