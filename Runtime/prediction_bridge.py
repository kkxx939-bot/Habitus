"""长期记忆检索结果到行为预测信号的组合根桥接。

prediction 不得 import memory（架构边界强制），memory 也不应知道 prediction；
把记忆命中翻译成 ``PredictionMemorySignal`` 的算法因此只能住在组合根一侧。
本模块是纯函数层：输入是一次记忆检索的完整命中与快照，输出是可注入
``PatternGraphPredictor.predict`` 的信号与来源绑定，不做检索、不落盘。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from types import MappingProxyType

from foundation.integrity import canonical_digest
from memory.document import MemoryDocument
from memory.model import MemoryKind
from memory.retrieval import MemoryMatchedMemory
from memory.snapshot import MemorySnapshot
from memory.uri import MemoryURI
from prediction import PredictionContext, PredictionRunSourceBinding
from prediction.predictor import PredictionMemorySignal


class MemorySignalBridgeError(ValueError):
    """记忆信号桥接的输入或配置违反合同。"""


@dataclass(frozen=True)
class MemorySignalBridgeConfig:
    """记忆命中折算为信号权重的确定性系数。

    权重 = 类型先验 × 状态因子 × 检索相关性 × 新鲜度，并截断到 (0, 1]。
    intention 的先验最高，因为未完成事项是"下一步做什么"最直接的证据；
    preference 表达选择倾向次之；profile 只提供微弱的稳定先验。
    ``relevance_threshold`` 是硬性去噪门槛：与当前行为上下文相关性不足的
    命中直接丢弃而不是向上保底——检索不到相关记忆的行为事件等于不看记忆。
    ``intention_freshness_tau_days`` 按 last_confirmed_at 的未确认天数做
    指数衰减——长期未被对话确认的事项不应继续以全权重影响预测。
    """

    intention_prior: float = 1.0
    preference_prior: float = 0.7
    profile_prior: float = 0.4
    blocked_intention_factor: float = 0.5
    intention_freshness_tau_days: float = 60.0
    relevance_threshold: float = 0.35
    max_bullets_per_document: int = 3
    max_signals: int = 8
    persona_preference_weight: float = 0.5
    persona_profile_weight: float = 0.35
    max_persona_signals: int = 6

    def __post_init__(self) -> None:
        for name in (
            "intention_prior",
            "preference_prior",
            "profile_prior",
            "blocked_intention_factor",
            "relevance_threshold",
            "persona_preference_weight",
            "persona_profile_weight",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        object.__setattr__(
            self,
            "max_persona_signals",
            _positive_int(self.max_persona_signals, "max_persona_signals"),
        )

    def identity_material(self) -> dict[str, float | int]:
        """返回参与信号溯源摘要的完整系数快照。"""

        return {
            "intention_prior": self.intention_prior,
            "preference_prior": self.preference_prior,
            "profile_prior": self.profile_prior,
            "blocked_intention_factor": self.blocked_intention_factor,
            "intention_freshness_tau_days": self.intention_freshness_tau_days,
            "relevance_threshold": self.relevance_threshold,
            "max_bullets_per_document": self.max_bullets_per_document,
            "max_signals": self.max_signals,
            "persona_preference_weight": self.persona_preference_weight,
            "persona_profile_weight": self.persona_profile_weight,
            "max_persona_signals": self.max_persona_signals,
        }
        object.__setattr__(
            self,
            "intention_freshness_tau_days",
            _positive_finite(self.intention_freshness_tau_days, "intention_freshness_tau_days"),
        )
        object.__setattr__(
            self,
            "max_bullets_per_document",
            _positive_int(self.max_bullets_per_document, "max_bullets_per_document"),
        )
        object.__setattr__(self, "max_signals", _positive_int(self.max_signals, "max_signals"))


_SIGNAL_KINDS = {
    MemoryKind.INTENTION: "intention",
    MemoryKind.PREFERENCE: "preference",
    MemoryKind.PROFILE: "profile",
}


def derive_memory_signals(
    memories: Sequence[MemoryMatchedMemory],
    *,
    now: datetime,
    config: MemorySignalBridgeConfig | None = None,
    stance_table: PredictionStanceTable | None = None,
) -> tuple[PredictionMemorySignal, ...]:
    """把一次记忆检索的命中确定性折算为低权重预测信号。

    记忆是行为预测的补充而不是常开输入：低于 ``relevance_threshold`` 的
    命中直接丢弃，检索不到相关记忆的行为事件自然产生零信号。只转换
    intention / preference / profile 三类；entity / tool / event 与
    completed intention 直接丢弃。intention 取 ``next_step``（最具体的
    下一步）与 ``intent_name`` 两条语义；preference / profile 按 Markdown
    bullet 拆分并限量，含否定语义（"不要/避免"类）的条目折算为 oppose
    信号去压制对应行为，绝不加强。同输入必同输出，信号只提示概率重分配。
    """

    if isinstance(memories, (str, bytes)) or not isinstance(memories, Sequence):
        raise MemorySignalBridgeError("memories must be a sequence of MemoryMatchedMemory values")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise MemorySignalBridgeError("now must be a timezone-aware datetime")
    resolved = config or MemorySignalBridgeConfig()
    if not isinstance(resolved, MemorySignalBridgeConfig):
        raise MemorySignalBridgeError("config must be MemorySignalBridgeConfig")
    candidates: list[tuple[float, str, str, str, str]] = []
    for match in memories:
        if not isinstance(match, MemoryMatchedMemory):
            raise MemorySignalBridgeError("memories must contain MemoryMatchedMemory values")
        signal_kind = _SIGNAL_KINDS.get(match.document.kind)
        if signal_kind is None:
            continue
        score = float(match.hit.score)
        if score < resolved.relevance_threshold:
            continue
        relevance = min(1.0, max(0.0, score))
        uri = str(match.uri)
        if match.document.kind is MemoryKind.INTENTION:
            weight, texts = _intention_material(match, now, relevance, resolved)
        else:
            prior = (
                resolved.preference_prior
                if match.document.kind is MemoryKind.PREFERENCE
                else resolved.profile_prior
            )
            weight = prior * relevance
            texts = _bullets(match.document.fields["content"], resolved.max_bullets_per_document)
        weight = min(1.0, weight)
        if weight <= 0:
            continue
        candidates.extend(
            (weight, uri, text, signal_kind, _stance(text, match.document.kind, stance_table))
            for text in texts
        )
    best: dict[tuple[str, str], tuple[float, str, str, str, str]] = {}
    for candidate in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        key = (candidate[4], " ".join(candidate[2].split()).casefold())
        if key not in best:
            best[key] = candidate
    selected = sorted(best.values(), key=lambda item: (-item[0], item[1], item[2]))
    return tuple(
        PredictionMemorySignal(source_uri=uri, kind=kind, semantics=text, weight=weight, stance=stance)
        for weight, uri, text, kind, stance in selected[: resolved.max_signals]
    )


def memory_source_bindings(
    snapshots: Iterable[MemorySnapshot],
) -> tuple[PredictionRunSourceBinding, ...]:
    """把记忆快照折算为 PredictionRun 的来源绑定，按 URI 排序去重。

    绑定使用快照读取器计算的规范 digest；缺失的记忆不能被绑定，
    同一 URI 出现不同版本快照视为调用方错误而不是静默择一。
    """

    bindings: dict[str, PredictionRunSourceBinding] = {}
    for snapshot in snapshots:
        if not snapshot.exists or snapshot.revision is None or snapshot.source_digest is None:
            raise MemorySignalBridgeError("a missing memory snapshot cannot bind a prediction run")
        binding = PredictionRunSourceBinding(
            str(snapshot.identity),
            int(snapshot.revision),
            str(snapshot.source_digest),
        )
        existing = bindings.get(binding.uri)
        if existing is not None and existing != binding:
            raise MemorySignalBridgeError("one memory URI resolved to conflicting snapshots")
        bindings[binding.uri] = binding
    return tuple(bindings[uri] for uri in sorted(bindings))


def _intention_material(
    match: MemoryMatchedMemory,
    now: datetime,
    relevance: float,
    config: MemorySignalBridgeConfig,
) -> tuple[float, tuple[str, ...]]:
    fields = match.document.fields
    status = str(fields["status"])
    if status == "completed":
        return 0.0, ()
    factor = config.blocked_intention_factor if status == "blocked" else 1.0
    last_confirmed_at = match.document.metadata.last_confirmed_at
    if last_confirmed_at is None:
        freshness = 1.0
    else:
        unconfirmed_days = max(0.0, (now - last_confirmed_at).total_seconds() / 86400.0)
        freshness = math.exp(-unconfirmed_days / config.intention_freshness_tau_days)
    texts = tuple(
        text.strip()
        for text in (fields.get("next_step"), fields["intent_name"])
        if isinstance(text, str) and text.strip()
    )
    return config.intention_prior * factor * relevance * freshness, texts


def derive_persona_signals(
    documents: Sequence[MemoryDocument],
    *,
    config: MemorySignalBridgeConfig | None = None,
    stance_table: PredictionStanceTable | None = None,
) -> tuple[PredictionMemorySignal, ...]:
    """把"你是什么人"折算为常驻低权重的人物先验信号。

    与逐事件检索的 :func:`derive_memory_signals` 不同，人物通道由组合根
    定期直接读取 ``profile.md`` 与稳定 preferences 生成，不带检索分数、
    不看当前事件——它表达的是跨事件成立的人物认知，帮助在行为数据
    稀疏的状态下判读"这样的人下一步更可能做什么"。权重固定为较低的
    人物系数；预测器侧的置信度缩放与惯例旁路保证：规律行为足够确定时
    人物先验不产生任何影响。只接受 profile / preference 两类文档。
    """

    resolved = config or MemorySignalBridgeConfig()
    if not isinstance(resolved, MemorySignalBridgeConfig):
        raise MemorySignalBridgeError("config must be MemorySignalBridgeConfig")
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise MemorySignalBridgeError("documents must be a sequence of MemoryDocument values")
    candidates: list[tuple[float, str, str, str, str]] = []
    for document in documents:
        if not isinstance(document, MemoryDocument):
            raise MemorySignalBridgeError("documents must contain MemoryDocument values")
        if document.kind not in {MemoryKind.PROFILE, MemoryKind.PREFERENCE}:
            raise MemorySignalBridgeError("persona signals accept only profile and preference memory")
        uri = str(MemoryURI.from_address(document.address))
        weight = (
            resolved.persona_preference_weight
            if document.kind is MemoryKind.PREFERENCE
            else resolved.persona_profile_weight
        )
        signal_kind = _SIGNAL_KINDS[document.kind]
        for text in _bullets(document.fields["content"], resolved.max_bullets_per_document):
            candidates.append(
                (weight, uri, text, signal_kind, _stance(text, document.kind, stance_table))
            )
    best: dict[tuple[str, str], tuple[float, str, str, str, str]] = {}
    for candidate in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        key = (candidate[4], " ".join(candidate[2].split()).casefold())
        if key not in best:
            best[key] = candidate
    selected = sorted(best.values(), key=lambda item: (-item[0], item[1], item[2]))
    return tuple(
        PredictionMemorySignal(source_uri=uri, kind=kind, semantics=text, weight=weight, stance=stance)
        for weight, uri, text, kind, stance in selected[: resolved.max_persona_signals]
    )


def signal_provenance(
    signals: Sequence[PredictionMemorySignal],
    *,
    now: datetime,
    config: MemorySignalBridgeConfig | None = None,
    stance_table: PredictionStanceTable | None = None,
) -> str:
    """把一次预测实际使用的信号及其全部生成因素折算为溯源摘要。

    桥配置系数、立场表版本与信号内容都影响最终候选概率却不在 predictor
    身份内；本摘要交给 ``PredictionRun.signal_provenance`` 记录，使每次
    判决可回答"当时的记忆信号是什么、由哪版配置和表折算"，记忆参数的
    定档与错误执行的归因才有依据。
    """

    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise MemorySignalBridgeError("now must be a timezone-aware datetime")
    resolved = config or MemorySignalBridgeConfig()
    if not isinstance(resolved, MemorySignalBridgeConfig):
        raise MemorySignalBridgeError("config must be MemorySignalBridgeConfig")
    normalized = []
    for signal in signals:
        if not isinstance(signal, PredictionMemorySignal):
            raise MemorySignalBridgeError("signals must contain PredictionMemorySignal values")
        normalized.append(
            {
                "source_uri": signal.source_uri,
                "kind": signal.kind,
                "semantics": signal.semantics,
                "target_refs": list(signal.target_refs),
                "weight": signal.weight,
                "stance": signal.stance,
            }
        )
    return canonical_digest(
        {
            "contract": "prediction-signal-provenance-v1",
            "bridge_config": resolved.identity_material(),
            "stance_table_version": None if stance_table is None else stance_table.version,
            "signals": normalized,
            "derived_at": now,
        }
    )


def memory_query_for_context(context: PredictionContext) -> str:
    """从预测上下文构造行为条件化的记忆检索查询。

    查询由"刚发生的行为 + 涉及对象 + 活跃目标"组成，使检索本身被当前
    行为事件约束：与此无关的记忆根本不会被召回，等价于该事件不看记忆。
    """

    if not isinstance(context, PredictionContext):
        raise MemorySignalBridgeError("context must be PredictionContext")
    parts: list[str] = []
    history = context.input["behavior_history"]
    level = str(context.anchor["anchor_type"])
    steps = history["completed_actions"] if level == "action" else history["completed_events"]
    if steps:
        last = max(steps, key=lambda step: (step["sequence"] or 0, step["available_at"] or ""))
        parts.append(str(last["semantics"]))
        parts.extend(str(item) for item in last["target_refs"])
    frame = context.input["observation_frame"]
    parts.extend(str(item) for item in frame["active_goals"])
    parts.extend(str(item) for item in frame["subjects"])
    unique = [part for part in dict.fromkeys(part.strip() for part in parts) if part]
    if not unique:
        return "接下来可能进行的日常行为"
    return " ".join(unique)


_STRONG_OPPOSE_MARKERS = ("不喜欢", "讨厌")
_OPPOSE_MARKERS = ("不要", "不用", "不得", "不吃", "避免", "禁止")
_CLAUSE_INITIAL_OPPOSE = ("别", "勿")
_SUPPORT_CUES = ("喜欢", "记得", "别忘", "希望")
_CLAUSE_SEPARATORS = "，,。;；!！ "
_STANCES = frozenset({"support", "oppose"})


@dataclass(frozen=True)
class PredictionStanceTable:
    """离线 LLM 立场解析产出的条目→立场映射。

    标记词表只能识别表层否定，条件否定与双重否定需要语义理解；解析在
    离线批处理完成（组合根调用 LLM），本对象只承载版本化的确定性结果。
    表中缺席的条目回退到标记词表判定。
    """

    stances: Mapping[str, str] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.stances, Mapping):
            raise MemorySignalBridgeError("stance table must be a mapping")
        normalized: dict[str, str] = {}
        for raw_text, stance in self.stances.items():
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise MemorySignalBridgeError("stance table keys must be non-empty text")
            if stance not in _STANCES:
                raise MemorySignalBridgeError("stance table values must be support or oppose")
            normalized[" ".join(raw_text.split()).casefold()] = stance
        object.__setattr__(self, "stances", MappingProxyType(dict(sorted(normalized.items()))))

    @property
    def version(self) -> str:
        return canonical_digest(
            {
                "contract": "prediction-stance-table-v1",
                "stances": dict(self.stances),
            }
        )

    def lookup(self, text: str) -> str | None:
        return self.stances.get(" ".join(text.split()).casefold())


def _stance(text: str, kind: MemoryKind, table: PredictionStanceTable | None = None) -> str:
    """折算一个条目的立场；intention 恒为支持，表命中优先于标记词表。

    标记词表回退遵循"宁可不压制也不反向压制"：无歧义的厌恶词直接反对；
    含正向线索（喜欢/记得/别忘/希望）的条目默认支持；多字否定词全句
    生效，单字"别/勿"只在子句起始生效——避免"特别喜欢""按类别整理"
    这类复合词被整句子串匹配误判成反对。真正的语义判定属于立场表。
    """

    if kind is MemoryKind.INTENTION:
        return "support"
    if table is not None:
        resolved = table.lookup(text)
        if resolved is not None:
            return resolved
    if any(marker in text for marker in _STRONG_OPPOSE_MARKERS):
        return "oppose"
    if any(cue in text for cue in _SUPPORT_CUES):
        return "support"
    if any(marker in text for marker in _OPPOSE_MARKERS):
        return "oppose"
    clauses = [clause for clause in _split_clauses(text) if clause]
    if any(clause.startswith(_CLAUSE_INITIAL_OPPOSE) for clause in clauses):
        return "oppose"
    return "support"


def _split_clauses(text: str) -> list[str]:
    clauses = [text]
    for separator in _CLAUSE_SEPARATORS:
        clauses = [part for clause in clauses for part in clause.split(separator)]
    return [clause.strip() for clause in clauses]


def _bullets(content: object, limit: int) -> tuple[str, ...]:
    if not isinstance(content, str) or not content.strip():
        return ()
    bullets = [
        line.strip()[2:].strip()
        for line in content.splitlines()
        if line.strip().startswith(("- ", "* ")) and line.strip()[2:].strip()
    ]
    if not bullets:
        bullets = [content.strip()]
    return tuple(bullets[:limit])


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemorySignalBridgeError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise MemorySignalBridgeError(f"{label} must be finite")
    return normalized


def _unit(value: object, label: str) -> float:
    normalized = _finite(value, label)
    if not 0 <= normalized <= 1:
        raise MemorySignalBridgeError(f"{label} must be between zero and one")
    return normalized


def _positive_finite(value: object, label: str) -> float:
    normalized = _finite(value, label)
    if normalized <= 0:
        raise MemorySignalBridgeError(f"{label} must be positive")
    return normalized


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MemorySignalBridgeError(f"{label} must be a positive integer")
    return value


__all__ = [
    "MemorySignalBridgeConfig",
    "MemorySignalBridgeError",
    "PredictionStanceTable",
    "derive_memory_signals",
    "derive_persona_signals",
    "memory_query_for_context",
    "memory_source_bindings",
    "signal_provenance",
]
