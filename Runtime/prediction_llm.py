"""行为预测的离线 LLM 蒸馏与在线语义裁量（组合根侧）。

prediction 不得 import ModelClient，全部 LLM 编排住在组合根。离线批处理
把语义理解蒸馏成版本化的确定性表（行为先验）；在线有两个受控位置：
不确定时的顾问（只对既有候选重排）与执行前的约束检查（对将执行的
候选批量判定是否违反用户表达过的约束，fail-closed）。两处都以记忆
条目原文 + 当前情境为输入，在调用现场做语义裁量——立场与相关性不做
任何预计算近似。

TODO(PRED-RUNTIME-001): prediction 主链装配（蒸馏批的调度周期、advisor
与约束检查的接入点、配置边界）待 prediction 接入 Runtime 生命周期时
另行设计确认；本模块的函数与类在装配前已可被显式调用并测试。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

from foundation.integrity import canonical_digest, canonical_json
from memory.document import MemoryDocument
from memory.model import MemoryKind
from ModelClient import (
    ChatCallContext,
    ChatMessage,
    ChatRequest,
    ModelClientError,
    StructuredChatClient,
)
from prediction import PredictionContext, PredictionRun, PredictionRunSourceBinding
from prediction.learning import (
    PredictionBehaviorPrior,
    PredictionBehaviorVocabulary,
    prior_entry,
)
from prediction.learning.keys import sequence_state_identity
from prediction.predictor import PatternGraphPredictor, PredictionDecision
from Runtime.prediction_bridge import MemoryBridgeError

_ADVISOR_CONTRACT = "prediction-llm-advisor-v1"
_CHECKER_CONTRACT = "prediction-llm-constraint-checker-v1"
_ADVISOR_TRIGGER_GATES = frozenset({"probability_below_threshold", "margin_below_threshold"})
_MAX_OUTPUT_TOKENS = 800
_MAX_CATALOG_ENTRIES = 64
_MAX_HISTORY_STEPS = 8
_MAX_CONTEXT_ENTRIES = 24


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


class PatternGraphLLMAdvisor:
    """判决不确定时的在线语义顾问；只重排既有候选，不发明行为。

    输入包含记忆条目原文与当前情境：意图是否仍然相关、偏好在此刻是
    趋向还是回避，都由顾问在语境里裁量，而不是由预计算标签决定。
    """

    def __init__(self, client: StructuredChatClient) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be StructuredChatClient")
        self.client = client
        route = client.client.config.route
        self.advisor_digest = canonical_digest(
            {
                "contract": _ADVISOR_CONTRACT,
                "prompt_version": "v2",
                "provider": route.provider,
                "adapter": route.adapter,
                "model": route.model,
                "structured_output_mode": client.client.config.structured_output_mode,
            }
        )

    async def advise(
        self,
        run: PredictionRun,
        *,
        memory_context: Sequence[str] = (),
    ) -> dict[str, float]:
        """对一次预测的候选给出偏好权重（branch_key → [0,1]）。"""

        if not isinstance(run, PredictionRun):
            raise TypeError("run must be PredictionRun")
        entries = _context_entries(memory_context)
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
                "memory_context": entries,
                "candidates": candidates,
            }
        )
        response = await self.client.complete_json_async(
            ChatRequest(
                messages=(
                    ChatMessage(
                        role="system",
                        content=(
                            "你是行为预测的语义顾问。行为统计对下一步没有把握，请结合当前"
                            "情境与用户的记忆条目（未完成事项、偏好、画像），判断哪些候选"
                            "更合理，给出 0 到 1 的偏好权重。注意记忆条目是否与此刻相关："
                            "过期的意图、不适用于当前时段或条件的偏好不应影响判断。只能"
                            "引用给出的 branch_key，不得发明行为；都不合理时输出空数组。"
                            "你的权重只做有界重排，不构成执行授权。"
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


class PredictionConstraintChecker:
    """直接执行前的批量约束检查；无人在环路径的最后一道语义闸。

    对将要执行的全部候选一次调用、逐候选判定"此刻执行是否违反用户
    表达过的约束"。条件性约束（"晚上九点后不用洗衣机""除非下雨否则
    浇水"）在这里天然正确——检查器看得到当前情境。建议档不经过本
    检查：用户本人就是建议的裁决者。
    """

    def __init__(self, client: StructuredChatClient) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be StructuredChatClient")
        self.client = client
        route = client.client.config.route
        self.checker_digest = canonical_digest(
            {
                "contract": _CHECKER_CONTRACT,
                "prompt_version": "v1",
                "provider": route.provider,
                "adapter": route.adapter,
                "model": route.model,
                "structured_output_mode": client.client.config.structured_output_mode,
            }
        )

    async def review(
        self,
        run: PredictionRun,
        *,
        memory_context: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """逐候选返回 {branch_key: {"allowed": bool, "reason": str}}。"""

        if not isinstance(run, PredictionRun):
            raise TypeError("run must be PredictionRun")
        entries = _context_entries(memory_context)
        candidates = [
            {
                "branch_key": str(item.payload["branch_key"]),
                "semantics": item.semantics,
            }
            for item in run.candidates
        ]
        if not candidates or not entries:
            return {}
        keys = sorted(str(item["branch_key"]) for item in candidates)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["verdicts"],
            "properties": {
                "verdicts": {
                    "type": "array",
                    "minItems": len(keys),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["branch_key", "allowed", "reason"],
                        "properties": {
                            "branch_key": {"type": "string", "enum": keys},
                            "allowed": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                    },
                }
            },
        }
        expected_keys = frozenset(keys)

        def _covering_validator(value: object) -> dict[str, dict[str, Any]]:
            return _validate_verdicts(value, expected_keys)
        history = run.context["input"]["behavior_history"]
        recent = [
            str(step["semantics"])
            for step in (*history["completed_actions"], *history["completed_events"])
        ][-_MAX_HISTORY_STEPS:]
        payload = canonical_json(
            {
                "scope": run.context["prediction_scope"],
                "predicted_at": run.predicted_at,
                "recent_steps": recent,
                "memory_context": entries,
                "candidates": candidates,
            }
        )
        response = await self.client.complete_json_async(
            ChatRequest(
                messages=(
                    ChatMessage(
                        role="system",
                        content=(
                            "你是行为执行前的约束检查器。系统即将自动执行某个行为，请逐个"
                            "候选判断：结合当前时刻与情境，执行它是否违反用户在记忆条目里"
                            "表达过的约束或回避（注意条件性约束的适用条件是否此刻成立）。"
                            "只判断给出的候选；allowed=false 时 reason 必须引用具体条目依据。"
                            "拿不准时倾向 allowed=false——本检查只守自动执行路径，误拦的"
                            "代价是降级为向用户建议。"
                        ),
                    ),
                    ChatMessage(role="user", content="请输出严格的判定 JSON：\n" + payload),
                ),
                temperature=0.0,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            ),
            schema=schema,
            validator=_covering_validator,
            name="prediction_constraint_verdicts",
            context=ChatCallContext(prompt_version=_CHECKER_CONTRACT),
        )
        return cast(dict[str, dict[str, Any]], response.value)


async def predict_with_advice(
    predictor: PatternGraphPredictor,
    advisor: PatternGraphLLMAdvisor,
    context: PredictionContext,
    *,
    predicted_at: datetime,
    horizon_seconds: float,
    source_bindings: Sequence[PredictionRunSourceBinding],
    memory_context: Sequence[str] = (),
    memory_provenance: str | None = None,
) -> PredictionDecision:
    """不确定性门控的两段式预测：先纯统计判决，被概率/边际门拦下才问顾问。

    行为数据有把握（执行放行或惯例明确）时不调用 LLM；风险门在场的
    判决无论是否叠加其他门都不问——顾问重排改变不了历史负向结果，
    也不允许 LLM 输出成为从"风险拦截"到"执行"的因果通路。顾问不可用
    时回退第一次纯统计判决——顾问是有界重排器，其可用性不得决定
    建议/沉默档的成立。第二次判决携带顾问权重并记入 run 的
    ``advisor_adjustment``，与 advisor_digest 一起构成完整审计。
    """

    if not isinstance(predictor, PatternGraphPredictor):
        raise TypeError("predictor must be PatternGraphPredictor")
    if not isinstance(advisor, PatternGraphLLMAdvisor):
        raise TypeError("advisor must be PatternGraphLLMAdvisor")
    if predictor.advisor_digest != advisor.advisor_digest:
        raise MemoryBridgeError("predictor advisor_digest does not match the advisor")
    first = predictor.predict(
        context,
        predicted_at=predicted_at,
        horizon_seconds=horizon_seconds,
        source_bindings=source_bindings,
        memory_provenance=memory_provenance,
    )
    if first.execute or not first.run.candidates:
        return first
    if "negative_outcome_risk" in first.blocked_gates:
        return first
    if not set(first.blocked_gates) & _ADVISOR_TRIGGER_GATES:
        return first
    try:
        weights = await advisor.advise(first.run, memory_context=memory_context)
    except ModelClientError:
        return first
    if not weights:
        return first
    return predictor.predict(
        context,
        predicted_at=predicted_at,
        horizon_seconds=horizon_seconds,
        source_bindings=source_bindings,
        advisor_weights=weights,
        memory_provenance=memory_provenance,
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


def _context_entries(memory_context: Sequence[str]) -> list[str]:
    if isinstance(memory_context, (str, bytes)) or not isinstance(memory_context, Sequence):
        raise TypeError("memory_context must be a sequence of text entries")
    entries: list[str] = []
    for entry in memory_context:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("memory_context entries must be non-empty text")
        entries.append(entry.strip())
    if len(entries) > _MAX_CONTEXT_ENTRIES:
        raise ValueError("memory_context exceeds the bounded entry count")
    return entries


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


def _validate_preferences(value: object) -> dict[str, float]:
    payload = cast(Mapping[str, Any], value)
    preferences: dict[str, float] = {}
    for item in payload["preferences"]:
        key = str(item["branch_key"])
        if key in preferences:
            raise ValueError(f"preference repeated for one branch: {key}")
        preferences[key] = float(item["weight"])
    return preferences


def _validate_verdicts(value: object, expected_keys: frozenset[str]) -> dict[str, dict[str, Any]]:
    """约束判定必须恰好覆盖每个候选一次——缺失即默许等于 fail-open。"""

    payload = cast(Mapping[str, Any], value)
    verdicts: dict[str, dict[str, Any]] = {}
    for item in payload["verdicts"]:
        key = str(item["branch_key"])
        if key in verdicts:
            raise ValueError(f"verdict repeated for one branch: {key}")
        verdicts[key] = {"allowed": bool(item["allowed"]), "reason": str(item["reason"])}
    if set(verdicts) != expected_keys:
        missing = sorted(expected_keys - set(verdicts))
        raise ValueError(f"verdicts must cover every candidate exactly once; missing: {missing}")
    return verdicts


__all__ = [
    "PatternGraphLLMAdvisor",
    "PredictionConstraintChecker",
    "distill_behavior_prior",
    "predict_with_advice",
]
