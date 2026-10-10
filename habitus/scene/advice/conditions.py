"""候选调节条件（语义树新方案 ``13`` ②）：一个前因"可能被什么情境左右"。模型触点。

- **读 A 那一条与它之前的事，不看 B 有没有发生**：每个够样本的前因挑几条（按时间均匀取），每条给它自己的时刻、概要、目标、
  地点、同在的人，以及当天它之前的几条（时刻、类名、概要）。后果一个字都不给——条件必须在 A 发生那一刻就判得出来。
- **必须编译成三种可算的模板之一**（``relations.conditions.compile_proposal``），落不成的进已拒清单、带原因；
  已收下的与已拒的都写进提示词，不每晚换个说法反复试。
- 第 N 晚提出的条件只在 N 之后的数据上检验（关系检验里做）。
- 只在概念集变了（或前因刚够样本）时问；按输入摘要缓存，回放只读缓存。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, cast

from habitus.foundation.integrity import canonical_digest
from habitus.foundation.text import clean_line
from habitus.model_client import ChatMessage, ChatRequest, StructuredChatClient
from habitus.scene.advice.model import Proposal
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.llm import Answered, Failed, ask
from habitus.scene.relations.conditions import ConditionError, Field, Template, compile_proposal

CONDITION_PROMPT_VERSION = "scene_relation_conditions_prompt_v1"
#: 一次最多收几条提议（多了说明模型在凑数）。
MAX_PROPOSALS = 5

CONDITION_SYSTEM_PROMPT = """你在为一个人的某一类行为找"它可能受什么情境左右"，供后面的统计检验去验证。

给你这类行为的几次记录：每次的时刻、概要、目标、地点、同在的人，以及当天在它之前的几件事。请提出最多 5 个
"这一类行为在某种情境下，之后的走向可能不一样"的情境条件。条件必须能写成下面三种之一，否则不要提：

1. field（结构字段）：weekend（是否周末）、calendar（当天的日历备注，value 写备注原文，如"调休"）、
   place（地点，value 写地点原文）、with（同在的某个人，value 写名字原文）；
2. timeline（时间线）：这件事之前不久（同一段连续的工作里）出现过某一类行为，concept 写那一类的名字；
3. concept（情境概念）：这次记录被标为某个情境，concept 写情境的名字。

规则：
- 只根据给你的记录与名单提，不要编名单之外的名字、地点或人；
- 条件要在这件事发生那一刻就判得出来，不能依赖之后发生了什么；
- 已经收下或已经被拒的条件不要再提；
- why 用一句话说为什么觉得这个情境可能有影响。"""


@dataclass(frozen=True)
class Occasion:
    """给模型看的一次 A：它自己与当天之前的几件事（都已渲染成文字，调用方从事件序列取）。"""

    moment: str
    summary: str
    goal: str | None = None
    place: str | None = None
    subjects: tuple[str, ...] = ()
    before: tuple[str, ...] = ()

    def render(self, number: int) -> str:
        lines = [
            f"### 第 {number} 次（{self.moment}）",
            f"概要：{self.summary}",
            f"目标：{self.goal or '（说不出目标）'}",
            f"地点：{self.place or '（未知）'}",
            f"同在：{'、'.join(self.subjects) or '（无）'}",
        ]
        lines.append("之前：" + ("；".join(self.before) if self.before else "（当天之前没有记录）"))
        return "\n".join(lines)


@dataclass(frozen=True)
class ConditionQuestion:
    lane: str
    antecedent: str
    occasions: tuple[Occasion, ...]
    behaviours: tuple[str, ...]
    situations: tuple[str, ...]
    accepted: tuple[str, ...]
    refused: tuple[str, ...]
    labels: Mapping[str, str]
    definitions: Mapping[str, str]

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "version": CONDITION_VERSION,
                "lane": self.lane,
                "antecedent": [self.labels[self.antecedent], self.definitions[self.antecedent]],
                "occasions": [occasion.render(index) for index, occasion in enumerate(self.occasions, start=1)],
                "behaviours": sorted(self.labels[name] for name in self.behaviours),
                "situations": sorted([self.labels[name], self.definitions[name]] for name in self.situations),
                "accepted": sorted(self.accepted),
                "refused": sorted(self.refused),
            }
        )


def question_for(
    concepts: ConceptSet,
    lane: str,
    antecedent: str,
    occasions: Sequence[Occasion],
    *,
    accepted: Sequence[str] = (),
    refused: Sequence[str] = (),
) -> ConditionQuestion:
    behaviours = tuple(name for name in concepts.behaviors() if name != antecedent and concepts.lane_of(name) == lane)
    situations = tuple(name for name in concepts.situations() if concepts.lane_of(name) in (lane, None))
    names = (antecedent, *behaviours, *situations)
    return ConditionQuestion(
        lane=lane,
        antecedent=antecedent,
        occasions=tuple(occasions),
        behaviours=behaviours,
        situations=situations,
        accepted=tuple(accepted),
        refused=tuple(refused),
        labels={name: concepts.label_of(name) for name in names},
        definitions={name: clean_line(concepts[name].definition) for name in names},
    )


def condition_json_schema(names: Sequence[str]) -> dict[str, Any]:
    """名字钉进 enum（行为类与情境概念的名字）；字段与模板也是 enum。可空的用 ``[类型, "null"]``。"""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["conditions"],
        "properties": {
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["template", "field", "value", "concept", "why"],
                    "properties": {
                        "template": {"type": "string", "enum": [item.value for item in Template]},
                        "field": {"type": ["string", "null"], "enum": [*(item.value for item in Field), None]},
                        "value": {"type": ["string", "null"]},
                        "concept": {"type": ["string", "null"], "enum": [*names, None]},
                        "why": {"type": "string"},
                    },
                },
                "description": f"最多 {MAX_PROPOSALS} 条；没有像样的就给空列表。",
            }
        },
    }


CONDITION_SCHEMA_FINGERPRINT = canonical_digest(condition_json_schema(("样例",)))[:12]
#: 提示词正文的指纹进版本：正文改了而忘了手动升版本号，缓存 / 续跑也不会拿旧答案冒充新提示词的答案（第四轮评审 E15）。
PROMPT_FINGERPRINT = canonical_digest({"prompt": CONDITION_SYSTEM_PROMPT})[:8]
CONDITION_VERSION = f"{CONDITION_PROMPT_VERSION}+prompt{PROMPT_FINGERPRINT}+schema{CONDITION_SCHEMA_FINGERPRINT}"


def build_condition_request(question: ConditionQuestion) -> ChatRequest:
    labels, definitions = question.labels, question.definitions
    lines = [
        f"## 这一类行为：{labels[question.antecedent]}（{definitions[question.antecedent]}）",
        "",
        *(occasion.render(index) for index, occasion in enumerate(question.occasions, start=1)),
        "",
        "## 可以用的行为类（timeline 模板的 concept 只能从这里选）",
        "、".join(labels[name] for name in question.behaviours) or "（无）",
        "",
        "## 可以用的情境（concept 模板的 concept 只能从这里选）",
        *([f"- {labels[name]}：{definitions[name]}" for name in question.situations] or ["（无）"]),
    ]
    if question.accepted:
        lines += ["", "## 已经收下的条件（不要再提）", *(f"- {item}" for item in question.accepted)]
    if question.refused:
        lines += ["", "## 已经被拒的提法（不要再提）", *(f"- {item}" for item in question.refused)]
    return ChatRequest(
        messages=(
            ChatMessage(role="system", content=CONDITION_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n".join(lines)),
        )
    )


def assemble_proposals(
    parsed: object, question: ConditionQuestion, concepts: ConceptSet, *, night: date
) -> tuple[Proposal, ...]:
    """核对形状（不多于上限、每条字段齐）；能编译的收下，编译不了的记拒绝原因——编译失败不是答复不合格，不叫模型重答。"""

    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("conditions"), list):
        raise ValueError("condition output must carry a conditions list")
    items = cast("list[object]", parsed["conditions"])
    if len(items) > MAX_PROPOSALS:
        raise ValueError(f"at most {MAX_PROPOSALS} conditions, got {len(items)}")
    found: list[Proposal] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("each condition must be an object")
        why = clean_line(item.get("why"))
        if not why:
            raise ValueError("each condition needs a reason")
        raw = {key: item.get(key) for key in ("template", "field", "value", "concept")}
        try:
            condition = compile_proposal(raw, concepts, lane=question.lane)
        except ConditionError as exc:
            found.append(Proposal(question.antecedent, night, why, raw=raw, refused=str(exc)))
            continue
        if condition.template is Template.TIMELINE and condition.value == question.antecedent:
            found.append(
                Proposal(
                    question.antecedent, night, why, raw=raw, refused="a condition cannot be the antecedent itself"
                )
            )
            continue
        found.append(Proposal(question.antecedent, night, why, condition=condition.text))
    return tuple(found)


@dataclass(frozen=True)
class ConditionConfig:
    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0


@dataclass(frozen=True)
class ConditionAnswer:
    proposals: tuple[Proposal, ...] = ()
    signals: tuple[str, ...] = ()
    #: 模型答成了没有（答成了、一条都没提也算答成）。
    answered: bool = True


class ConditionProposer:
    def __init__(self, client: StructuredChatClient, *, config: ConditionConfig | None = None) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        self.client = client
        self.config = config or ConditionConfig()

    @property
    def version(self) -> str:
        return f"{CONDITION_VERSION}+llm:{self.client.client.model}"

    async def propose(self, question: ConditionQuestion, concepts: ConceptSet, *, night: date) -> ConditionAnswer:
        names = [question.labels[name] for name in (*question.behaviours, *question.situations)]
        answer = await ask(
            self.client,
            build_condition_request(question),
            schema=condition_json_schema(names),
            name="scene_relation_conditions",
            validator=lambda parsed: assemble_proposals(parsed, question, concepts, night=night),
            retries=self.config.transient_retries,
            delay_seconds=self.config.transient_retry_delay_seconds,
        )
        if isinstance(answer, Failed):
            return ConditionAnswer(signals=(answer.signal,), answered=False)
        assert isinstance(answer, Answered)
        proposals = tuple(
            replace(item, version=self.version, model=answer.model) for item in cast("tuple[Proposal, ...]", answer.value)
        )
        refused = sum(1 for item in proposals if item.refused is not None)
        return ConditionAnswer(
            proposals=proposals,
            signals=(
                f"conditions: {self.version}",
                f"conditions: 提了 {len(proposals)} 条、编译不了 {refused} 条",
                *answer.notes,
            ),
        )


__all__ = [
    "CONDITION_SYSTEM_PROMPT",
    "CONDITION_VERSION",
    "MAX_PROPOSALS",
    "ConditionAnswer",
    "ConditionConfig",
    "ConditionProposer",
    "ConditionQuestion",
    "Occasion",
    "assemble_proposals",
    "build_condition_request",
    "condition_json_schema",
    "question_for",
]
