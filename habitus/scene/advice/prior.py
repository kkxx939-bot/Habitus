"""先验（语义树新方案 ``13`` ②）：常识上 A 之后 B 会不会更容易 / 更难发生。模型触点。

- **输入只给词表**：类名、判据句（细分 / 汇总概念给它们的区别或定义）、lane。**不给概要、计数、节律或任何统计量**——
  先验必须和检验数据独立，否则它就不是先验，而是把同一份数据看了两遍。
- 两种问法（实时化阶段 2「先验混合问法」），一对（前因, 后果）只归其中一题（归哪一题在 ``scene.advice.pairing``）：
  **旧问法**固定一个后果、列前因（``rate``）；**新问法**固定一个前因、列后果（``rate_consequents``）——新概念出现时只问两题，
  调用量不随概念总数增长。两种都分三档，**乱序问三遍**（原序、倒序、对半轮转），取中位，差两档的给中性。
  新问法的提示词、请求拼法、校验照原型（``数据/先验问法对照/run_compare.py``）落下来，在旧提示词上只做"后果 / 前因"角色互换；
  旧问法那一半的提示词、版本号一个字不动（盘上已有的答案照用），新问法另起版本号、另一份缓存。
- 按输入摘要缓存（由调用方存）：词表与概念不变，就不重问；回放只读缓存。
- 先验只调第 1 道的权重；每晚同时报不加权的结果，只靠先验才成立的打"靠先验"标签（状态折叠里做）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from habitus.foundation.integrity import canonical_digest
from habitus.foundation.text import clean_line
from habitus.model_client import ChatMessage, ChatRequest, StructuredChatClient
from habitus.model_client.asking import Answered, Failed, ask
from habitus.scene.advice.model import PriorLevel, settle
from habitus.scene.concepts.model import ConceptSet

# v1 → v2（2026-10-08，裁定 27 第 8 条）：例子"讨论完方案之后动手改代码"正是这份数据上唯一成立的那一对——等于把检验结果写进了先验；
# 换成与数据无关的例子，判据一字未动。没跑真实模型对照（去掉的是泄漏，不是调措辞），对照放到提示词那一轮。
PRIOR_PROMPT_VERSION = "scene_relation_prior_prompt_v2"

PRIOR_SYSTEM_PROMPT = """你在给一个人的行为之间可能的关联打先验分，供后面的统计检验加权用。

给你一个"后果"行为和一组"前因"行为（都是这个人自己记录里的行为类别，附判据）。对每个前因判断：
这个人做了前因之后（几分钟到几周之内），后果会不会因此**明显更容易或更难**发生——方向不论，只问有没有关联。

三档：
- likely：按常识很可能有关联（例如做完饭之后吃饭、收到快递之后拆包装）；
- unsure：说不准，可能有也可能没有；
- unlikely：按常识不太可能有关联。

规则：
- 只按常识与类别本身的意思判断，不要假设这个人的具体习惯；
- 不知道就答 unsure，不要因为想不到就答 unlikely；
- 每个前因恰好答一次，不多不少，照抄名字。"""


#: 新问法（固定前因、列后果）：另起版本号，缓存与旧问法的分开。
PRIOR_BY_ANTECEDENT_PROMPT_VERSION = "scene_relation_prior_by_antecedent_prompt_v1"

PRIOR_BY_ANTECEDENT_SYSTEM_PROMPT = """你在给一个人的行为之间可能的关联打先验分，供后面的统计检验加权用。

给你一个"前因"行为和一组"后果"行为（都是这个人自己记录里的行为类别，附判据）。对每个后果判断：
这个人做了前因之后（几分钟到几周之内），这个后果会不会因此**明显更容易或更难**发生——方向不论，只问有没有关联。

三档：
- likely：按常识很可能有关联（例如做完饭之后吃饭、收到快递之后拆包装）；
- unsure：说不准，可能有也可能没有；
- unlikely：按常识不太可能有关联。

规则：
- 只按常识与类别本身的意思判断，不要假设这个人的具体习惯；
- 不知道就答 unsure，不要因为想不到就答 unlikely；
- 每个后果恰好答一次，不多不少，照抄名字。"""


class PriorError(ValueError):
    """先验的输入与概念集矛盾。"""


@dataclass(frozen=True)
class PriorQuestion:
    """问一次先验的全部输入：后果与它可比的前因（都按概念身份），以及给模型看的名字与判据。"""

    lane: str
    consequent: str
    antecedents: tuple[str, ...]
    labels: Mapping[str, str]
    definitions: Mapping[str, str]


def question_for(concepts: ConceptSet, lane: str, consequent: str, antecedents: Sequence[str]) -> PriorQuestion:
    names = (consequent, *antecedents)
    for name in names:
        if name not in concepts:
            raise PriorError(f"{name!r} is not in the concept set")
    labels = {name: concepts.label_of(name) for name in names}
    if len(set(labels[name] for name in antecedents)) != len(antecedents):
        raise PriorError("antecedents shown to the model must have distinct names")
    return PriorQuestion(
        lane=lane,
        consequent=consequent,
        antecedents=tuple(antecedents),
        labels=labels,
        definitions={name: clean_line(concepts[name].definition) for name in names},
    )


@dataclass(frozen=True)
class ConsequentsQuestion:
    """新问法问一次的全部输入：前因与归这一题的后果（都按概念身份），以及给模型看的名字与判据。"""

    lane: str
    antecedent: str
    consequents: tuple[str, ...]
    labels: Mapping[str, str]
    definitions: Mapping[str, str]


def consequents_question_for(
    concepts: ConceptSet, lane: str, antecedent: str, consequents: Sequence[str]
) -> ConsequentsQuestion:
    names = (antecedent, *consequents)
    for name in names:
        if name not in concepts:
            raise PriorError(f"{name!r} is not in the concept set")
    labels = {name: concepts.label_of(name) for name in names}
    if len(set(labels[name] for name in consequents)) != len(consequents):
        raise PriorError("consequents shown to the model must have distinct names")
    return ConsequentsQuestion(
        lane=lane,
        antecedent=antecedent,
        consequents=tuple(consequents),
        labels=labels,
        definitions={name: clean_line(concepts[name].definition) for name in names},
    )


def prior_json_schema(names: Sequence[str]) -> dict[str, Any]:
    """前因名字钉进 enum；条数由核对管（真实后端的严格模式不认 minItems / maxItems）。"""

    return _ratings_schema(names, "前因")


def prior_by_antecedent_json_schema(names: Sequence[str]) -> dict[str, Any]:
    """新问法的 schema：后果名字钉进 enum。"""

    return _ratings_schema(names, "后果")


def _ratings_schema(names: Sequence[str], role: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["ratings"],
        "properties": {
            "ratings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["concept", "level"],
                    "properties": {
                        "concept": {"type": "string", "enum": list(names), "description": f"{role}的名字，照抄。"},
                        "level": {"type": "string", "enum": [level.value for level in PriorLevel]},
                    },
                },
                "description": f"每个{role}恰好一条，共 {len(names)} 条。",
            }
        },
    }


PRIOR_SCHEMA_FINGERPRINT = canonical_digest(prior_json_schema(("样例",)))[:12]
#: 提示词正文的指纹进版本：正文改了而忘了手动升版本号，缓存 / 续跑也不会拿旧答案冒充新提示词的答案（第四轮评审 E15）。
PROMPT_FINGERPRINT = canonical_digest({"prompt": PRIOR_SYSTEM_PROMPT})[:8]
PRIOR_VERSION = f"{PRIOR_PROMPT_VERSION}+prompt{PROMPT_FINGERPRINT}+schema{PRIOR_SCHEMA_FINGERPRINT}"
PRIOR_BY_ANTECEDENT_VERSION = (
    f"{PRIOR_BY_ANTECEDENT_PROMPT_VERSION}"
    f"+prompt{canonical_digest({'prompt': PRIOR_BY_ANTECEDENT_SYSTEM_PROMPT})[:8]}"
    f"+schema{canonical_digest(prior_by_antecedent_json_schema(('样例',)))[:12]}"
)


def build_prior_request(question: PriorQuestion, order: Sequence[str]) -> ChatRequest:
    labels, definitions = question.labels, question.definitions
    lines = [
        "## 后果",
        f"{labels[question.consequent]}：{definitions[question.consequent]}",
        "",
        "## 前因（每个答一档）",
        *(f"- {labels[name]}：{definitions[name]}" for name in order),
    ]
    return ChatRequest(
        messages=(
            ChatMessage(role="system", content=PRIOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n".join(lines)),
        )
    )


def build_prior_by_antecedent_request(question: ConsequentsQuestion, order: Sequence[str]) -> ChatRequest:
    labels, definitions = question.labels, question.definitions
    lines = [
        "## 前因",
        f"{labels[question.antecedent]}：{definitions[question.antecedent]}",
        "",
        "## 后果（每个答一档）",
        *(f"- {labels[name]}：{definitions[name]}" for name in order),
    ]
    return ChatRequest(
        messages=(
            ChatMessage(role="system", content=PRIOR_BY_ANTECEDENT_SYSTEM_PROMPT),
            ChatMessage(role="user", content="\n".join(lines)),
        )
    )


def assemble_ratings(parsed: object, question: PriorQuestion) -> dict[str, PriorLevel]:
    """核对：每个前因恰好一条、档在三档里；名字换回概念身份。"""

    return _assemble(parsed, question.antecedents, question.labels, "an antecedent", "antecedents")


def assemble_consequent_ratings(parsed: object, question: ConsequentsQuestion) -> dict[str, PriorLevel]:
    """新问法的核对：每个后果恰好一条、档在三档里；名字换回概念身份。"""

    return _assemble(parsed, question.consequents, question.labels, "a consequent", "consequents")


def _assemble(
    parsed: object, asked: Sequence[str], labels: Mapping[str, str], one: str, many: str
) -> dict[str, PriorLevel]:
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("ratings"), list):
        raise ValueError("prior output must carry a ratings list")
    by_label = {labels[name]: name for name in asked}
    found: dict[str, PriorLevel] = {}
    for item in cast("list[object]", parsed["ratings"]):
        if not isinstance(item, Mapping):
            raise ValueError("each rating must be an object")
        name = by_label.get(clean_line(item.get("concept")))
        if name is None:
            raise ValueError(f"rating names {one} that was not asked: {item.get('concept')!r}")
        if name in found:
            raise ValueError(f"rating repeats {labels[name]!r}")
        try:
            found[name] = PriorLevel(item.get("level"))
        except ValueError as exc:
            raise ValueError(f"level must be one of {[level.value for level in PriorLevel]}") from exc
    missing = [labels[name] for name in asked if name not in found]
    if missing:
        raise ValueError(f"ratings are missing {many}: {missing}")
    return found


@dataclass(frozen=True)
class PriorAnswer:
    """一个后果的先验：每个前因一档（三遍取中位后）；没答成是空、带信号。"""

    levels: Mapping[str, PriorLevel]
    signals: tuple[str, ...] = ()
    #: 出处：提示词与 schema 的版本、这几遍实际作答的模型（与缓存一起存，回看时分得清是谁答的）。
    version: str = ""
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class PriorConfig:
    transient_retries: int = 1
    transient_retry_delay_seconds: float = 5.0
    #: 乱序问几遍（方案：三遍取中位）。
    rounds: int = 3


class PriorAdvisor:
    def __init__(self, client: StructuredChatClient, *, config: PriorConfig | None = None) -> None:
        self.client = client
        self.config = config or PriorConfig()

    @property
    def version(self) -> str:
        return f"{PRIOR_VERSION}+llm:{self.client.client.model}"

    @property
    def version_by_antecedent(self) -> str:
        """新问法（固定前因、列后果）的版本：缓存键里用它，与旧问法的缓存分开。"""

        return f"{PRIOR_BY_ANTECEDENT_VERSION}+llm:{self.client.client.model}"

    async def rate(self, question: PriorQuestion) -> PriorAnswer:
        """旧问法：固定后果、列前因。乱序问 ``rounds`` 遍；任何一遍没答成，这一题这一回就不给先验（权重按 1），下次再问。"""

        return await self._rate(
            question.antecedents,
            lambda order: build_prior_request(question, order),
            lambda order: prior_json_schema([question.labels[name] for name in order]),
            lambda parsed: assemble_ratings(parsed, question),
            name="scene_relation_prior",
            version=self.version,
            role="前因",
        )

    async def rate_consequents(self, question: ConsequentsQuestion) -> PriorAnswer:
        """新问法：固定前因、列后果；``levels`` 的键是后果。同样乱序问 ``rounds`` 遍取中位。"""

        return await self._rate(
            question.consequents,
            lambda order: build_prior_by_antecedent_request(question, order),
            lambda order: prior_by_antecedent_json_schema([question.labels[name] for name in order]),
            lambda parsed: assemble_consequent_ratings(parsed, question),
            name="scene_relation_prior_by_antecedent",
            version=self.version_by_antecedent,
            role="后果",
        )

    async def _rate(
        self,
        asked: Sequence[str],
        request: Callable[[Sequence[str]], ChatRequest],
        schema: Callable[[Sequence[str]], dict[str, Any]],
        validator: Callable[[object], dict[str, PriorLevel]],
        *,
        name: str,
        version: str,
        role: str,
    ) -> PriorAnswer:
        if not asked:
            return PriorAnswer(levels={})
        rounds: list[dict[str, PriorLevel]] = []
        notes: list[str] = []
        models: set[str] = set()
        for order in _orders(asked, self.config.rounds):
            answer = await ask(
                self.client,
                request(order),
                schema=schema(order),
                name=name,
                validator=validator,
                retries=self.config.transient_retries,
                delay_seconds=self.config.transient_retry_delay_seconds,
            )
            if isinstance(answer, Failed):
                return PriorAnswer(levels={}, signals=(answer.signal,))
            assert isinstance(answer, Answered)
            rounds.append(cast("dict[str, PriorLevel]", answer.value))
            notes.extend(answer.notes)
            models.add(answer.model)
        levels = {item: settle([answers[item] for answers in rounds]) for item in asked}
        split = sum(1 for item in asked if len({answers[item] for answers in rounds}) > 1)
        apart = sum(
            1
            for item in asked
            if max(answers[item].rank for answers in rounds) - min(answers[item].rank for answers in rounds) >= 2
        )
        return PriorAnswer(
            levels=levels,
            version=version,
            models=tuple(sorted(models)),
            signals=(
                f"prior: {version}",
                f"prior: {len(asked)} 个{role}里 {split} 个几遍答得不一样、{apart} 个差了两档（给中性）",
                *notes,
            ),
        )


def _orders(names: Sequence[str], rounds: int) -> list[tuple[str, ...]]:
    """原序、倒序、对半轮转……（确定性的"乱序"：同样的输入问出同样的顺序，缓存才有意义）。"""

    base = tuple(names)
    half = len(base) // 2
    candidates = [base, tuple(reversed(base)), base[half:] + base[:half]]
    return [candidates[index % len(candidates)] for index in range(rounds)]


__all__ = [
    "PRIOR_BY_ANTECEDENT_SYSTEM_PROMPT",
    "PRIOR_BY_ANTECEDENT_VERSION",
    "PRIOR_SYSTEM_PROMPT",
    "PRIOR_VERSION",
    "ConsequentsQuestion",
    "PriorAdvisor",
    "PriorAnswer",
    "PriorConfig",
    "PriorError",
    "PriorQuestion",
    "assemble_consequent_ratings",
    "assemble_ratings",
    "build_prior_by_antecedent_request",
    "build_prior_request",
    "consequents_question_for",
    "prior_by_antecedent_json_schema",
    "prior_json_schema",
    "question_for",
]
