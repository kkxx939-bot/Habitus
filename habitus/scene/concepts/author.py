"""触点①：在基础词表的类之上写细分概念、汇总概念与情境概念（裁定 20）。

基础概念不在这里写：词表里每个类同步时自动生成一个基础概念（身份 = 类编号），不调模型。
这一口只写三样：

1. **细分概念** = 一个源类 + 一句区别：提醒句与源类相同，只是这一次的条件不同（「晚睡」= 睡觉里比近期常态
   晚两小时以上的；「返工」= 修改代码里改的是前两天刚改过的同一处）。区别是数值的写成规则由算法判；
   是语义的写成一句区别判据，并声明判它要什么材料（这一条 / 当天时间线 / 近几天同类记录）。
   **提醒句不同的不是细分概念**——那是词表里该有的类（早饭不是吃饭的细分，它就是一个类）。
2. **汇总概念** = 同一条 lane 里几个类放在一起说得通（「写代码」= 修改代码 + 排查问题 + 验证测试）。
   模型只答"放一起说得通吗"，曲线由成员类相加，不判、不命中。
3. **情境概念** = 当时的状况（周几、日历、同在的人、地点、连续 N 天、24 小时内），算法照说明算。

模型看到的是类名、判据、例子，**不看次数**（裁定七：按次数定粒度太死板）。

**单个概念不合格就丢那一个，不否决整批**：形状不对、跨 lane 汇总、引用了不存在的概念——丢那一个并说清理由，
引用它的连带丢；整批否决只留给答复本身不成形（交结构层重试）。

⚠ 提示词与 schema 是**草稿**：措辞要用真实模型跑对照再定（本机 codex 为主，回退 Claude Code 的 sonnet）。
调用那一层是实验脚本，不在仓库里。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, cast

from habitus.foundation.integrity import canonical_digest
from habitus.foundation.text import clean_line
from habitus.model_client import ChatMessage, ChatRequest, StructuredChatClient
from habitus.scene.concepts.model import (
    BASELINE_KEY_SEPARATOR,
    MAX_DEFINITION_CHARS,
    MAX_GRADES,
    MIN_GRADES,
    BaselineKey,
    ConceptDefinition,
    ConceptError,
    ConceptGrade,
    ConceptKind,
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
from habitus.scene.llm import Failed, ask

CONCEPT_AUTHOR_PROMPT_VERSION = "scene_concept_author_prompt_v5"
#: 一批最多写多少个概念（保护闸）。
MAX_CONCEPTS_PER_BATCH = 40
MAX_EXAMPLES_PER_CLASS = 5
#: 空的节律表；冻结的默认值不能是可变字典。
NO_RHYTHMS: Mapping[str, Rhythm] = MappingProxyType({})


class ConceptAuthorError(ValueError):
    """概念作者的输入自相矛盾（例如同一个类给了两份材料）。"""


@dataclass(frozen=True)
class ClassBrief:
    """一个词表类交给模型看的材料：类名、判据、lane、几条真例子（编号只给程序用，不给模型看，裁定 21-1）。

    例子是**真的** occurrence 的一行说法（名字 + 概要），由调用方从行为树取——模型只看类名会凭联想写区别。
    """

    class_id: str
    name: str
    criterion: str
    lane: str
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label in ("class_id", "name", "criterion", "lane"):
            value = clean_line(getattr(self, label))
            if not value:
                raise ConceptAuthorError(f"a class brief needs a {label}")
            object.__setattr__(self, label, value)
        examples = tuple(clean_line(item) for item in self.examples if clean_line(item))
        object.__setattr__(self, "examples", examples[:MAX_EXAMPLES_PER_CLASS])

    def render(self, label: str) -> str:
        """``label`` 是给模型看的名字（一般就是类名；两条 lane 有同名类时带上 lane）。"""

        examples = "；".join(self.examples) if self.examples else "（没给例子）"
        lane = "" if label.endswith(f"（{self.lane}）") else f"（{self.lane}）"
        return f"- {label}{lane}：{self.criterion}｜例：{examples}"


def class_labels(briefs: Sequence[ClassBrief]) -> Mapping[str, str]:
    """给模型看的类名 → 类编号。同名的类（只会出现在不同 lane）带上 lane 区分。同一个类给两份材料是调用方的错。"""

    if len({brief.class_id for brief in briefs}) != len(briefs):
        raise ConceptAuthorError("class briefs must name each class once")

    counts: dict[str, int] = {}
    for brief in briefs:
        counts[brief.name] = counts.get(brief.name, 0) + 1
    return MappingProxyType(
        {
            (brief.name if counts[brief.name] == 1 else f"{brief.name}（{brief.lane}）"): brief.class_id
            for brief in briefs
        }
    )


@dataclass(frozen=True)
class ConceptProposal:
    """一批提案：写成的定义 + 留痕。``definitions`` 空 = 这一批一个都没写；调用方据此**什么都不落盘**。"""

    definitions: tuple[ConceptDefinition, ...] = ()
    signals: tuple[str, ...] = ()
    #: 被丢掉的概念与理由。不静默：下一轮还要靠它知道为什么少了一个概念。
    dropped: tuple[str, ...] = ()
    #: 模型答成了没有（答成了、一个都没写也算答成）。没答成的调用方下一回再问。
    answered: bool = True

    @property
    def wrote_nothing(self) -> bool:
        return not self.definitions


CONCEPT_AUTHOR_SYSTEM_PROMPT = """你在一个人的行为词表之上定义概念，供后面找因果关系用。

材料是词表里的行为类：类名、判据、属于哪条 lane（session 会话 / physical 物理）、几条真实例子，
可能还带它的作息节律；以及已经有的概念。每个类已经自动有一个同名的基础概念，**不要再为类本身写概念**。
你只写下面三种：

1. 细分概念（role = behavior，kind = refinement，classes 填一个源类）：
   这个类里**提醒的话不变、只是这一次的条件不同**的那一部分。例：
   - 「晚睡」= 睡觉里开始时刻比近期常态晚 120 分钟以上的 —— 写成 rule（数值由算法算），definition 写一句人读的说明；
   - 「返工」= 修改代码里改的是前两天刚改过的同一处 —— 写不成数值，definition 写成对着材料能答是/否的
     **区别**（不要重复"这是修改代码"，那已经确定了），context 填 recent（要看近几天同类的记录）。
   要看当天前后发生了什么的，context 填 day；只看这一条本身的，填 occurrence。写了 rule 的 context 必须是 occurrence。
   **提醒的话不一样的不是细分**：早饭和吃饭提醒的话不同（"该吃早饭了"），那是词表里该有的类，不要写成细分概念。
2. 汇总概念（role = behavior，kind = group，classes 填两个以上成员类）：
   几个类放在一起说得通、一起看有意义（例：「写代码」= 修改代码 + 排查问题 + 验证测试）。
   成员类必须在**同一条 lane**；汇总概念不写 rule、grades，context 填 occurrence，definition 写一句它们共同是什么。
3. 情境概念（role 填 state / object / day_type / derived，kind 填 null，classes 填 []）：当时的状况，不是一件事。

其他规则：
- 只写说得通、以后用得上的；不要为了凑数把每个类都细分一遍。
- 数值：rule 是**定义**（"比常态晚 120 分钟以上"），每一条记录的数值由算法算，你不要算。
  写了 rule 才能写档（2–3 档，按同一个量切开、不重叠）。
- 要用到常态值的，在 baseline_keys（或 rule 的 relative_to）里写成 `<谁>:usual_start:recent` 或
  `<谁>:usual_duration:recent`；`<谁>` 是类名（例 睡觉）或一个细分概念的名字。窗一律填 `recent`。
- 已经有的概念不要重写、不要重名。

情境概念还要写一条 situation —— 算法照它算，所以只能用下面这几种说法之一：
- weekdays：名义上的周几（0=周一 … 6=周日），例：周末 = [5, 6]。
- calendar_note：当地日历那句话里含某个词，例：调休日 = "补班"。
- subject：同在的人里有谁。  · place：地点是哪儿。
- streak：某个概念（类名或概念名）**往前连着 N 个 24 小时**都命中（N ≥ 2），例：赶工中 = 「写代码」连着 3 天。
- yesterday：这条行为开始之前 **24 小时内**命中过某个概念（可以指定档），例：昨晚晚睡 = 之前 24 小时内命中「晚睡」的重档。
role 填 derived 的**必须**写 situation。写不成上面任何一种的状态（"出差中""生病中"）就别写 situation，
我们暂时算不出它；"在做哪个项目""心情如何"这类算不出来的，一个都不要写。"""


def concept_author_json_schema(briefs: Sequence[ClassBrief]) -> dict[str, Any]:
    """类名钉进 enum：模型引用不了材料里没有的类（答复里的类名由 ``assemble_concepts`` 换回编号）。

    **写成严格模式吃得下的形状**（2026-09-29 探针实测）：``required`` 列出 ``properties`` 里的每一个键，
    不用 ``minItems`` / ``maxItems`` / ``maxLength`` 这类边界关键字；边界与条数一律由 validator 管。
    """

    ids = list(class_labels(briefs))
    if not ids:
        raise ConceptAuthorError("the concept author schema needs at least one class brief")
    if len(set(ids)) != len(ids):
        raise ConceptAuthorError("class briefs must name each class once")
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
            "weekdays": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "日型用：0=周一 … 6=周日；其他族填 []。",
            },
            "value": {"type": ["string", "null"], "description": "calendar_note / subject / place 要匹配的值。"},
            "concept": {"type": ["string", "null"], "description": "streak / yesterday 盯的那个概念。"},
            "grade": {"type": ["string", "null"], "description": "只在某一档才算时填档名。"},
            "days": {"type": ["integer", "null"], "description": "streak 连着几天（≥2）；其他族填 null。"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["concepts"],
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name",
                        "definition",
                        "role",
                        "kind",
                        "classes",
                        "context",
                        "baseline_keys",
                        "rule",
                        "grades",
                        "situation",
                        "why",
                    ],
                    "properties": {
                        "name": {"type": "string", "description": "概念的名字，短、像个名词。"},
                        "definition": {
                            "type": "string",
                            "description": f"细分概念写区别（对着材料能答是/否），汇总与情境写一句它是什么；{MAX_DEFINITION_CHARS} 字以内。",
                        },
                        "role": {"type": "string", "enum": [item.value for item in ConceptRole]},
                        "kind": {
                            "type": ["string", "null"],
                            "enum": [ConceptKind.REFINEMENT.value, ConceptKind.GROUP.value, None],
                            "description": "行为概念填 refinement（细分）或 group（汇总）；情境概念填 null。",
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "string", "enum": ids},
                            "description": "细分概念填一个源类；汇总概念填两个以上同 lane 的成员类；情境概念填 []。",
                        },
                        "context": {"type": "string", "enum": [item.value for item in ContextScope]},
                        "baseline_keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "区别要用到的常态值，形如 p-k0015:usual_start:recent；不用就填 []。",
                        },
                        "rule": rule,
                        "grades": {
                            "type": "array",
                            "items": grade,
                            "description": f"只有写了 rule 才能写档；写就写 {MIN_GRADES}–{MAX_GRADES} 档；不写填 []。",
                        },
                        "situation": situation,
                        "why": {"type": "string", "description": "一句话：为什么要这个概念。"},
                    },
                },
                "description": f"最多 {MAX_CONCEPTS_PER_BATCH} 个；没有值得写的就给 []。",
            },
        },
    }


CONCEPT_AUTHOR_SCHEMA_FINGERPRINT = canonical_digest(
    concept_author_json_schema((ClassBrief("s-k0001", "样例", "样例判据", "session"),))
)[:12]
#: 提示词正文的指纹进版本：正文改了而忘了手动升版本号，缓存 / 续跑也不会拿旧答案冒充新提示词的答案（第四轮评审 E15）。
PROMPT_FINGERPRINT = canonical_digest({"prompt": CONCEPT_AUTHOR_SYSTEM_PROMPT})[:8]
CONCEPT_AUTHOR_VERSION = f"{CONCEPT_AUTHOR_PROMPT_VERSION}+prompt{PROMPT_FINGERPRINT}+schema{CONCEPT_AUTHOR_SCHEMA_FINGERPRINT}"


def build_concept_request(
    briefs: Sequence[ClassBrief],
    existing: ConceptSet,
    rhythms: Mapping[str, Rhythm] = NO_RHYTHMS,
) -> ChatRequest:
    """材料三段：词表的类、已有的概念（不要重写）、相关类的作息。全部显示名字，编号只作引用。"""

    if not briefs:
        raise ConceptAuthorError("a concept authoring request needs at least one class brief")
    labels = {class_id: label for label, class_id in class_labels(briefs).items()}
    sections = ["## 词表里的行为类", *(brief.render(labels[brief.class_id]) for brief in briefs)]
    authored = [identity for identity in sorted(existing) if existing[identity].kind is not ConceptKind.BASE]
    if authored:
        sections += ["", "## 已经有的概念（不要重写、不要重名）"]
        sections += [_render_existing(existing, identity) for identity in authored]
    lines = [rhythm.render() for rhythm in rhythms.values() if rhythm.peaks]
    if lines:
        sections += ["", "## 这些行为平常的作息（一天几次、几点）", *(f"- {line}" for line in lines)]
    return ChatRequest(
        messages=(
            ChatMessage(role="system", content=CONCEPT_AUTHOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n".join(sections)),
        )
    )


def _render_existing(existing: ConceptSet, identity: str) -> str:
    item = existing[identity]
    classes = "、".join(existing.class_label(class_id) for class_id in item.classes)
    kind = item.kind.value if item.kind is not None else item.role.value
    return f"- {item.label}（{kind}{'：' + classes if classes else ''}）：{item.definition}"


@dataclass(frozen=True)
class _Draft:
    """模型答复里的一条，已按形状核对过、还没变成 ``ConceptDefinition``。"""

    name: str
    definition: str
    role: ConceptRole
    kind: ConceptKind | None
    classes: tuple[str, ...]
    context: ContextScope
    baseline_keys: tuple[str, ...]
    rule: MechanicalRule | None
    grades: tuple[ConceptGrade, ...]
    situation: SituationRule | None
    why: str | None


def assemble_concepts(
    parsed: object,
    briefs: Sequence[ClassBrief],
    existing: ConceptSet,
    *,
    now: datetime,
) -> tuple[tuple[ConceptDefinition, ...], tuple[str, ...]]:
    """核对成一组能落盘的定义；返回 (定义, 丢掉了什么)。

    答复不成形（形状、重名）整份交结构层重试；单个概念立不住（跨 lane 汇总、与已有重名、规则不自洽、
    引用了不存在的概念）丢那一个并说清理由，引用它的连带丢。
    """

    drafts = _drafts(parsed, briefs)
    lanes = {brief.class_id: brief.lane for brief in briefs}
    built: dict[str, ConceptDefinition] = {}
    dropped: list[str] = []
    for draft in drafts:
        definition, reason = _build(draft, lanes, existing, now=now)
        if definition is None:
            dropped.append(f"dropped: {draft.name}（{reason}）")
        else:
            built[definition.identity] = definition
    kept = _cascade(built, existing, dropped)
    try:
        ConceptSet([*(existing[identity] for identity in sorted(existing)), *kept])
    except ConceptError as exc:
        raise ValueError(f"the proposed concepts do not form a consistent set: {exc}") from exc
    return tuple(kept), tuple(dropped)


def _build(
    draft: _Draft, lanes: Mapping[str, str], existing: ConceptSet, *, now: datetime
) -> tuple[ConceptDefinition | None, str]:
    """一个草稿变成定义；立不住就返回 (None, 理由)。"""

    if concept_identity(draft.name) in existing or draft.name in {item.label for item in existing.values()}:
        return None, "已经有同名概念（含与类名同名）"
    lane = None
    if draft.role.is_behavior:
        members = {lanes[item] for item in draft.classes}
        if len(members) > 1:
            return None, f"成员类跨了 lane（{sorted(members)}），汇总只在一条 lane 内"
        lane = members.pop() if members else None
    note = f"{CONCEPT_AUTHOR_VERSION}｜{draft.why}" if draft.why else CONCEPT_AUTHOR_VERSION
    try:
        return (
            ConceptDefinition(
                name=draft.name,
                definition=draft.definition,
                role=draft.role,
                source=ConceptSource(origin=ConceptOrigin.AUTHOR, note=note[:400]),
                created_at=now,
                kind=draft.kind,
                classes=draft.classes,
                lane=lane,
                grades=draft.grades,
                rule=draft.rule,
                context=draft.context,
                baseline_keys=draft.baseline_keys,
                situation=draft.situation,
            ),
            "",
        )
    except ConceptError as exc:
        return None, str(exc)


def _cascade(
    built: Mapping[str, ConceptDefinition], existing: ConceptSet, dropped: list[str]
) -> list[ConceptDefinition]:
    """连带丢弃：情境说明盯的、常态键的主不在（已有 ∪ 留下的）里的，丢掉并说清是因为谁。"""

    alive = dict(built)
    changed = True
    while changed:
        changed = False
        for identity, item in list(alive.items()):
            missing = next(
                (
                    name
                    for name in _references(item)
                    if concept_identity(name) not in existing and concept_identity(name) not in alive
                ),
                None,
            )
            if missing is None:
                continue
            del alive[identity]
            dropped.append(f"dropped: {item.name}（引用的「{missing}」不是已有或这批留下的概念）")
            changed = True
    return list(alive.values())


def _references(item: ConceptDefinition) -> tuple[str, ...]:
    names = [BaselineKey.parse(key).concept for key in item.required_baseline_keys]
    if item.situation is not None and item.situation.concept is not None:
        names.append(item.situation.concept)
    return tuple(names)


def _drafts(parsed: object, briefs: Sequence[ClassBrief]) -> tuple[_Draft, ...]:
    if not isinstance(parsed, Mapping):
        raise ValueError("concept author output must be an object")
    entries = parsed.get("concepts")
    if not isinstance(entries, list):
        raise ValueError("concept author output must carry a concepts list")
    if len(entries) > MAX_CONCEPTS_PER_BATCH:
        raise ValueError(f"write at most {MAX_CONCEPTS_PER_BATCH} concepts in one batch")
    drafts = tuple(_draft(entry, class_labels(briefs)) for entry in entries)
    identities = [concept_identity(draft.name) for draft in drafts]
    if len(set(identities)) != len(identities):
        raise ValueError("two proposed concepts share one name")
    return drafts


def _draft(entry: object, labels: Mapping[str, str]) -> _Draft:
    """一条答复 → 草稿。模型写的是类名，这里换回类编号：成员类、常态键的"谁"、情境说明盯的概念、规则比的常态键。"""

    if not isinstance(entry, Mapping):
        raise ValueError("each proposed concept must be an object")
    classes = _list(entry.get("classes"), "classes")
    if any(item not in labels for item in classes):
        raise ValueError(f"concept classes must name the given classes, got {classes!r}")
    keys = [
        _owned_key(_text(key, "baseline key"), labels) for key in _list(entry.get("baseline_keys"), "baseline_keys")
    ]
    rule = _rule(entry.get("rule"), labels)
    kind = entry.get("kind")
    return _Draft(
        name=_text(entry.get("name"), "concept name"),
        definition=_text(entry.get("definition"), "concept definition"),
        role=_enum(ConceptRole, entry.get("role"), "role"),
        kind=None if kind in (None, "") else _enum(ConceptKind, kind, "kind"),
        classes=tuple(labels[item] for item in classes),
        context=_enum(ContextScope, entry.get("context") or ContextScope.OCCURRENCE.value, "context"),
        baseline_keys=tuple(keys),
        rule=rule,
        grades=_grades(_list(entry.get("grades"), "grades"), rule),
        situation=_situation(entry.get("situation"), labels),
        why=clean_line(entry.get("why")) or None,
    )


def _list(value: object, label: str) -> list[Any]:
    """缺省当空表；给了别的东西就是答复不成形，交结构层重试。"""

    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return list(value)


def _rule(raw: object, labels: Mapping[str, str]) -> MechanicalRule | None:
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
            relative_to=None if relative in (None, "") else _owned_key(_text(relative, "rule relative_to"), labels),
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
                lower=_required_int(
                    item.get("lower_minutes") if isinstance(item, Mapping) else None, "grade lower_minutes"
                ),
                upper=_required_int(
                    item.get("upper_minutes") if isinstance(item, Mapping) else None, "grade upper_minutes"
                ),
                relative=rule.is_relative,
            )
            for item in raw
        )
    except ConceptError as exc:
        raise ValueError(f"grades are not usable: {exc}") from exc


def _resolved(text: str, labels: Mapping[str, str]) -> str:
    return labels.get(text, text)


def _owned_key(text: str, labels: Mapping[str, str]) -> str:
    """常态键 ``<谁>:<统计>:<窗>`` 里的"谁"是类名时换成类编号（基础概念的身份）；细分概念的名字原样。"""

    owner, separator, rest = text.partition(BASELINE_KEY_SEPARATOR)
    return f"{labels.get(owner, owner)}{separator}{rest}"


def _situation(raw: object, labels: Mapping[str, str]) -> SituationRule | None:
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
            concept=None
            if raw.get("concept") in (None, "")
            else _resolved(_text(raw.get("concept"), "situation concept"), labels),
            grade=None if raw.get("grade") in (None, "") else _text(raw.get("grade"), "situation grade"),
            days=1 if days is None else _required_int(days, "situation days"),
        )
    except SituationError as exc:
        raise ValueError(f"situation is not computable: {exc}") from exc


@dataclass(frozen=True)
class AuthorConfig:
    """瞬态重试的两个数；与映射器同一套（传输层重试，答复不合格由结构层纠正）。"""

    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.transient_retries, bool)
            or not isinstance(self.transient_retries, int)
            or self.transient_retries < 0
        ):
            raise ValueError("transient_retries must be a non-negative integer")
        if (
            isinstance(self.transient_retry_delay_seconds, bool)
            or not isinstance(self.transient_retry_delay_seconds, int | float)
            or self.transient_retry_delay_seconds < 0
        ):
            raise ValueError("transient_retry_delay_seconds must be a non-negative number")


class ConceptAuthor:
    """触点①。一次一批类，失败就整批不写。"""

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
        briefs: Sequence[ClassBrief],
        existing: ConceptSet,
        rhythms: Mapping[str, Rhythm] = NO_RHYTHMS,
    ) -> ConceptProposal:
        if not isinstance(existing, ConceptSet):
            raise TypeError("existing must be a ConceptSet")
        askable = tuple(briefs)
        request = build_concept_request(askable, existing, rhythms)
        schema = concept_author_json_schema(askable)
        now = self._clock()
        answer = await ask(
            self.client,
            request,
            schema=schema,
            name="scene_concept_authoring",
            validator=lambda parsed: assemble_concepts(parsed, askable, existing, now=now),
            retries=self.config.transient_retries,
            delay_seconds=self.config.transient_retry_delay_seconds,
        )
        if isinstance(answer, Failed):
            # 传输、结构或核对没过：整批不写，下一次再问。
            return ConceptProposal(signals=(answer.signal,), answered=False)
        definitions, dropped = cast("tuple[tuple[ConceptDefinition, ...], tuple[str, ...]]", answer.value)
        signals = [f"author: {self.version}"]
        if not definitions:
            signals.append("author: 这一批没有写成的概念，什么都不写")
        signals.extend(answer.notes)
        signals.extend(dropped)
        return ConceptProposal(definitions=definitions, signals=tuple(signals), dropped=dropped)


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
    "NO_RHYTHMS",
    "AuthorConfig",
    "ClassBrief",
    "ConceptAuthor",
    "ConceptAuthorError",
    "ConceptProposal",
    "assemble_concepts",
    "class_labels",
    "build_concept_request",
    "concept_author_json_schema",
]
