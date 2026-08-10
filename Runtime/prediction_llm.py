"""行为预测的离线 LLM 蒸馏与在线语义顾问（组合根侧）。

prediction 不得 import ModelClient，全部 LLM 编排住在组合根：离线批处理
把语义理解蒸馏成版本化的确定性表（行为先验、偏好立场），在线顾问只在
判决不确定时被调用、只对既有候选重排、影响有界且进入 predictor 身份。

TODO(PRED-RUNTIME-001): prediction 主链装配（蒸馏批的调度周期、advisor
接入点与配置边界）待 prediction 接入 Runtime 生命周期时另行设计确认；
本模块的函数与类在装配前已可被显式调用并测试。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

from foundation.integrity import canonical_digest, canonical_json
from memory.document import MemoryDocument
from memory.model import MemoryKind
from ModelClient import ChatCallContext, ChatMessage, ChatRequest, StructuredChatClient
from prediction import PredictionContext, PredictionRun, PredictionRunSourceBinding
from prediction.learning import (
    PredictionBehaviorPrior,
    PredictionBehaviorVocabulary,
    prior_entry,
)
from prediction.learning.keys import sequence_state_identity
from prediction.predictor import PatternGraphPredictor, PredictionDecision, PredictionMemorySignal
from Runtime.prediction_bridge import MemorySignalBridgeError, PredictionStanceTable

_ADVISOR_CONTRACT = "prediction-llm-advisor-v1"
_ADVISOR_TRIGGER_GATES = frozenset({"probability_below_threshold", "margin_below_threshold"})
_MAX_OUTPUT_TOKENS = 800
_MAX_CATALOG_ENTRIES = 64
_MAX_STANCE_BULLETS = 64
_MAX_HISTORY_STEPS = 8


async def distill_behavior_prior(
    client: StructuredChatClient,
    *,
    persona_documents: Sequence[MemoryDocument],
    branch_catalog: Mapping[str, Mapping[str, Any]],
    level: str = "action",
    domain: str,
    strength: float = 2.0,
    vocabulary: PredictionBehaviorVocabulary | None = None,
) -> PredictionBehaviorPrior:
    """把人物画像蒸馏成同域根状态的行为先验伪计数表。

    LLM 只能对目录中已知的行为给份额（Schema 枚举强制），不能发明行为；
    份额之和超一在领域校验层拒绝并触发有界重试。``vocabulary`` 必须与
    学习器同一份：domain 与先验身份经它归一，否则先验键与学习器状态键
    失配、先验静默零生效。``branch_catalog`` 也应以同一词表构建。产出的
    先验表是版本化确定性对象，进入 generation 身份，学习聚合保持零 LLM。
    """

    if not isinstance(client, StructuredChatClient):
        raise TypeError("client must be StructuredChatClient")
    resolved_vocabulary = vocabulary or PredictionBehaviorVocabulary()
    if not isinstance(resolved_vocabulary, PredictionBehaviorVocabulary):
        raise TypeError("vocabulary must be PredictionBehaviorVocabulary")
    if level not in {"action", "event"}:
        raise ValueError("prior distillation level must be action or event")
    domain = resolved_vocabulary.canonical_token(domain, "prior target_domain")
    catalog = dict(branch_catalog)
    if not catalog:
        return PredictionBehaviorPrior(entries=(), strength=strength)
    if len(catalog) > _MAX_CATALOG_ENTRIES:
        raise ValueError("behavior catalog exceeds the distillation bound")
    behaviors = sorted(catalog)
    persona = _persona_bullets(persona_documents)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["shares"],
        "properties": {
            "shares": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["behavior", "share"],
                    "properties": {
                        "behavior": {"type": "string", "enum": behaviors},
                        "share": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            }
        },
    }
    payload = canonical_json(
        {"persona": persona, "domain": domain, "known_behaviors": behaviors}
    )
    response = await client.complete_json_async(
        ChatRequest(
            messages=(
                ChatMessage(
                    role="system",
                    content=(
                        "你是行为先验蒸馏器。根据人物画像，估计这样的人在该场景下、没有任何"
                        "上下文时接下来做各已知行为的倾向份额。只能使用给出的已知行为，"
                        "不得发明新行为；share 为 0 到 1，总和不得超过 1，留白表示其他未知"
                        "行为的可能性。画像没有依据的行为给低份额或省略，不要均匀分配。"
                    ),
                ),
                ChatMessage(role="user", content="请输出严格的行为先验 JSON：\n" + payload),
            ),
            temperature=0.0,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
        ),
        schema=schema,
        validator=_validate_shares,
        name="prediction_behavior_prior",
        context=ChatCallContext(prompt_version="prediction_behavior_prior_v1"),
    )
    shares = cast(dict[str, float], response.value)
    if not shares:
        return PredictionBehaviorPrior(entries=(), strength=strength)
    entry = prior_entry(
        sequence_state_identity(level, domain, None),
        tuple((catalog[behavior], share) for behavior, share in sorted(shares.items())),
    )
    return PredictionBehaviorPrior(entries=(entry,), strength=strength)


async def derive_stance_table(
    client: StructuredChatClient,
    bullets: Sequence[str],
) -> PredictionStanceTable:
    """把偏好/画像条目批量解析为结构化立场表。

    标记词表只能识别表层否定；这里由 LLM 判断每条是"希望发生"还是
    "希望避免"，条目枚举强制来自输入，结果落成版本化表供桥接消费。
    """

    if not isinstance(client, StructuredChatClient):
        raise TypeError("client must be StructuredChatClient")
    unique = [item for item in dict.fromkeys(text.strip() for text in bullets) if item]
    if not unique:
        return PredictionStanceTable()
    if len(unique) > _MAX_STANCE_BULLETS:
        raise ValueError("stance parsing exceeds the batch bound")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["stances"],
        "properties": {
            "stances": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "stance"],
                    "properties": {
                        "text": {"type": "string", "enum": unique},
                        "stance": {"type": "string", "enum": ["support", "oppose"]},
                    },
                },
            }
        },
    }
    response = await client.complete_json_async(
        ChatRequest(
            messages=(
                ChatMessage(
                    role="system",
                    content=(
                        "你是偏好立场解析器。逐条判断每个条目表达的是希望发生/倾向去做"
                        "（support），还是希望避免/不要做（oppose）。注意条件否定与双重"
                        "否定的真实语义；每个条目只输出一次。"
                    ),
                ),
                ChatMessage(
                    role="user",
                    content="请输出严格的立场 JSON：\n" + canonical_json({"items": unique}),
                ),
            ),
            temperature=0.0,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
        ),
        schema=schema,
        validator=_validate_stances,
        name="prediction_stance_table",
        context=ChatCallContext(prompt_version="prediction_stance_table_v1"),
    )
    return PredictionStanceTable(stances=cast(dict[str, str], response.value))


class PatternGraphLLMAdvisor:
    """判决不确定时的在线语义顾问；只重排既有候选，不发明行为。"""

    def __init__(self, client: StructuredChatClient) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be StructuredChatClient")
        self.client = client
        route = client.client.config.route
        self.advisor_digest = canonical_digest(
            {
                "contract": _ADVISOR_CONTRACT,
                "prompt_version": "v1",
                "provider": route.provider,
                "adapter": route.adapter,
                "model": route.model,
                "structured_output_mode": client.client.config.structured_output_mode,
            }
        )

    async def advise(self, run: PredictionRun) -> dict[str, float]:
        """对一次预测的候选给出偏好权重（branch_key → [0,1]）。"""

        if not isinstance(run, PredictionRun):
            raise TypeError("run must be PredictionRun")
        candidates = [
            {
                "branch_key": str(item.payload["branch_key"]),
                "semantics": item.semantics,
                "probability": item.probability,
            }
            for item in run.candidates
        ]
        if not candidates:
            return {}
        keys = sorted(str(item["branch_key"]) for item in candidates)
        history = run.context["input"]["behavior_history"]
        recent = [
            str(step["semantics"])
            for step in (*history["completed_actions"], *history["completed_events"])
        ][-_MAX_HISTORY_STEPS:]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["preferences"],
            "properties": {
                "preferences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["branch_key", "weight"],
                        "properties": {
                            "branch_key": {"type": "string", "enum": keys},
                            "weight": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                }
            },
        }
        payload = canonical_json(
            {
                "scope": run.context["prediction_scope"],
                "recent_steps": recent,
                "candidates": candidates,
            }
        )
        response = await self.client.complete_json_async(
            ChatRequest(
                messages=(
                    ChatMessage(
                        role="system",
                        content=(
                            "你是行为预测的语义顾问。行为统计对下一步没有把握，请根据情境"
                            "语义判断哪些候选更合理，给出 0 到 1 的偏好权重。只能引用给出的"
                            "branch_key，不得发明行为；都不合理时输出空数组。你的权重只做"
                            "有界重排，不构成执行授权。"
                        ),
                    ),
                    ChatMessage(role="user", content="请输出严格的偏好 JSON：\n" + payload),
                ),
                temperature=0.0,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            ),
            schema=schema,
            validator=_validate_preferences,
            name="prediction_advisor_preferences",
            context=ChatCallContext(prompt_version=f"{_ADVISOR_CONTRACT}"),
        )
        return cast(dict[str, float], response.value)


async def predict_with_advice(
    predictor: PatternGraphPredictor,
    advisor: PatternGraphLLMAdvisor,
    context: PredictionContext,
    *,
    predicted_at: datetime,
    horizon_seconds: float,
    source_bindings: Sequence[PredictionRunSourceBinding],
    memory_signals: Sequence[PredictionMemorySignal] = (),
    signal_provenance: str | None = None,
) -> PredictionDecision:
    """不确定性门控的两段式预测：先纯统计判决，被概率/边际门拦下才问顾问。

    行为数据有把握（执行放行或惯例明确）时不调用 LLM；风险门拦下的
    判决也不问——顾问重排改变不了历史负向结果。第二次判决携带顾问
    权重，predictor 的 advisor_digest 保证结果可审计。
    """

    if not isinstance(predictor, PatternGraphPredictor):
        raise TypeError("predictor must be PatternGraphPredictor")
    if not isinstance(advisor, PatternGraphLLMAdvisor):
        raise TypeError("advisor must be PatternGraphLLMAdvisor")
    if predictor.advisor_digest != advisor.advisor_digest:
        raise MemorySignalBridgeError("predictor advisor_digest does not match the advisor")
    first = predictor.predict(
        context,
        predicted_at=predicted_at,
        horizon_seconds=horizon_seconds,
        source_bindings=source_bindings,
        memory_signals=memory_signals,
        signal_provenance=signal_provenance,
    )
    if first.execute or not first.run.candidates:
        return first
    if not set(first.blocked_gates) & _ADVISOR_TRIGGER_GATES:
        return first
    weights = await advisor.advise(first.run)
    if not weights:
        return first
    return predictor.predict(
        context,
        predicted_at=predicted_at,
        horizon_seconds=horizon_seconds,
        source_bindings=source_bindings,
        memory_signals=memory_signals,
        advisor_weights=weights,
        signal_provenance=signal_provenance,
    )


def _persona_bullets(documents: Sequence[MemoryDocument]) -> list[str]:
    bullets: list[str] = []
    for document in documents:
        if not isinstance(document, MemoryDocument):
            raise TypeError("persona documents must be MemoryDocument values")
        if document.kind not in {MemoryKind.PROFILE, MemoryKind.PREFERENCE}:
            raise ValueError("persona distillation accepts only profile and preference memory")
        content = str(document.fields["content"])
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")) and stripped[2:].strip():
                bullets.append(stripped[2:].strip())
    return list(dict.fromkeys(bullets))


def _validate_shares(value: object) -> dict[str, float]:
    payload = cast(Mapping[str, Any], value)
    shares: dict[str, float] = {}
    total = 0.0
    for item in payload["shares"]:
        behavior = str(item["behavior"])
        if behavior in shares:
            raise ValueError(f"behavior share repeated: {behavior}")
        share = float(item["share"])
        if share <= 0:
            continue
        shares[behavior] = share
        total += share
    if total > 1.000000001:
        raise ValueError("behavior shares cannot exceed one")
    return shares


def _validate_stances(value: object) -> dict[str, str]:
    payload = cast(Mapping[str, Any], value)
    stances: dict[str, str] = {}
    for item in payload["stances"]:
        text = str(item["text"])
        if text in stances:
            raise ValueError(f"stance repeated for one item: {text}")
        stances[text] = str(item["stance"])
    return stances


def _validate_preferences(value: object) -> dict[str, float]:
    payload = cast(Mapping[str, Any], value)
    preferences: dict[str, float] = {}
    for item in payload["preferences"]:
        key = str(item["branch_key"])
        if key in preferences:
            raise ValueError(f"preference repeated for one branch: {key}")
        preferences[key] = float(item["weight"])
    return preferences


__all__ = [
    "PatternGraphLLMAdvisor",
    "derive_stance_table",
    "distill_behavior_prior",
    "predict_with_advice",
]
