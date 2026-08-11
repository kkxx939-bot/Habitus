"""长期记忆到行为预测的组合根桥接:检索、上下文供给与审计绑定。

prediction 不得 import memory(架构边界强制),memory 也不应知道 prediction,
桥接因此住在组合根。本模块只做确定性工作:构造行为条件化的检索查询、
按相关性过滤命中、把记忆条目整理成交给两个 LLM 位置(不确定时的顾问、
执行前的约束检查)的原文上下文,并产出可审计的来源绑定与溯源摘要。

这里刻意不做任何语义判断:条目的立场(趋向/回避)、意图与此刻的相关性
都依赖情境,属于 LLM 在调用现场的裁量;领域代码不用关键词、正则或
预计算映射表重新解释自然语言。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from foundation.integrity import canonical_digest
from memory.document import MemoryDocument
from memory.model import MemoryKind
from memory.retrieval import MemoryMatchedMemory
from memory.snapshot import MemorySnapshot
from prediction import PredictionContext, PredictionRunSourceBinding
from prediction.learning.keys import select_last_step

MAX_MEMORY_CONTEXT_ENTRIES = 24
_CONTEXT_KINDS = frozenset({MemoryKind.INTENTION, MemoryKind.PREFERENCE, MemoryKind.PROFILE})
_SENTENCE_SEPARATORS = "。;；!！?？\n"


class MemoryBridgeError(ValueError):
    """记忆桥接的输入或配置违反合同。"""


@dataclass(frozen=True)
class MemoryBridgeConfig:
    """上下文供给的确定性边界。

    ``relevance_threshold`` 是硬性去噪门槛:与当前行为上下文相关性不足的
    命中直接丢弃而不是向上保底——检索不到相关记忆的行为事件等于不看记忆。
    条目数量上限约束交给 LLM 的上下文体积;这里没有任何语义权重系数,
    条目如何影响判断由 LLM 在情境里裁量。
    """

    relevance_threshold: float = 0.35
    max_bullets_per_document: int = 3
    max_context_entries: int = 12

    def __post_init__(self) -> None:
        threshold = self.relevance_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= float(threshold) <= 1
        ):
            raise MemoryBridgeError("relevance_threshold must be between zero and one")
        object.__setattr__(self, "relevance_threshold", float(threshold))
        object.__setattr__(
            self,
            "max_bullets_per_document",
            _positive_int(self.max_bullets_per_document, "max_bullets_per_document"),
        )
        entries_cap = _positive_int(self.max_context_entries, "max_context_entries")
        if entries_cap > MAX_MEMORY_CONTEXT_ENTRIES:
            raise MemoryBridgeError(
                "max_context_entries cannot exceed the shared LLM context bound"
            )
        object.__setattr__(self, "max_context_entries", entries_cap)

    def identity_material(self) -> dict[str, float | int]:
        """返回参与溯源摘要的完整配置快照。"""

        return {
            "relevance_threshold": self.relevance_threshold,
            "max_bullets_per_document": self.max_bullets_per_document,
            "max_context_entries": self.max_context_entries,
        }


def derive_memory_context(
    memories: Sequence[MemoryMatchedMemory],
    *,
    now: datetime,
    config: MemoryBridgeConfig | None = None,
) -> tuple[str, ...]:
    """把一次记忆检索的命中整理为交给 LLM 的原文上下文条目。

    只收 intention / preference / profile;低于相关性门槛的命中直接丢弃。
    intention 渲染为带状态与未确认天数的事实句(新鲜度是事实,如何看待
    这个事实由 LLM 判断);preference / profile 按 bullet 或句子拆分,
    保持条目粒度。不做立场判定、不做加权——条目影响力由调用现场的
    LLM 在当前情境里裁量。
    """

    resolved = config or MemoryBridgeConfig()
    if not isinstance(resolved, MemoryBridgeConfig):
        raise MemoryBridgeError("config must be MemoryBridgeConfig")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise MemoryBridgeError("now must be a timezone-aware datetime")
    if isinstance(memories, (str, bytes)) or not isinstance(memories, Sequence):
        raise MemoryBridgeError("memories must be a sequence of MemoryMatchedMemory values")
    entries: list[str] = []
    for match in memories:
        if not isinstance(match, MemoryMatchedMemory):
            raise MemoryBridgeError("memories must contain MemoryMatchedMemory values")
        document = match.document
        if document.kind not in _CONTEXT_KINDS:
            continue
        if float(match.hit.score) < resolved.relevance_threshold:
            continue
        if document.kind is MemoryKind.INTENTION:
            entry = _intention_entry(document, now)
            if entry is not None:
                entries.append(entry)
        else:
            entries.extend(_bullets(document.fields["content"], resolved.max_bullets_per_document))
    unique = [entry for entry in dict.fromkeys(entries) if entry]
    return tuple(unique[: resolved.max_context_entries])


def persona_context(
    documents: Sequence[MemoryDocument],
    *,
    config: MemoryBridgeConfig | None = None,
) -> tuple[str, ...]:
    """把组合根定期直读的画像与稳定偏好整理为常驻上下文条目。

    与逐事件检索的 :func:`derive_memory_context` 互补:这里回答"你是
    什么人",供顾问在行为数据稀疏时判读、供约束检查识别用户表达过的
    边界。只接受 profile / preference 两类文档。
    """

    resolved = config or MemoryBridgeConfig()
    if not isinstance(resolved, MemoryBridgeConfig):
        raise MemoryBridgeError("config must be MemoryBridgeConfig")
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise MemoryBridgeError("documents must be a sequence of MemoryDocument values")
    entries: list[str] = []
    for document in documents:
        if not isinstance(document, MemoryDocument):
            raise MemoryBridgeError("documents must contain MemoryDocument values")
        if document.kind not in {MemoryKind.PROFILE, MemoryKind.PREFERENCE}:
            raise MemoryBridgeError("persona context accepts only profile and preference memory")
        entries.extend(_bullets(document.fields["content"], resolved.max_bullets_per_document))
    unique = [entry for entry in dict.fromkeys(entries) if entry]
    return tuple(unique[: resolved.max_context_entries])


def memory_provenance(
    entries: Sequence[str],
    *,
    now: datetime,
    config: MemoryBridgeConfig | None = None,
) -> str:
    """把一次预测实际提供给 LLM 的记忆上下文折算为溯源摘要。

    条目集合、桥配置与整理时刻都影响 LLM 的裁量输入却不在 predictor
    身份内;本摘要交给 ``PredictionRun.memory_provenance`` 记录,使每次
    判决可回答"当时给了 LLM 哪些记忆、由哪版配置整理"。
    """

    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise MemoryBridgeError("now must be a timezone-aware datetime")
    resolved = config or MemoryBridgeConfig()
    if not isinstance(resolved, MemoryBridgeConfig):
        raise MemoryBridgeError("config must be MemoryBridgeConfig")
    normalized = []
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise MemoryBridgeError("provenance entries must be non-empty text")
        normalized.append(entry.strip())
    return canonical_digest(
        {
            "contract": "prediction-memory-provenance-v1",
            "bridge_config": resolved.identity_material(),
            "entries": normalized,
            "derived_at": now,
        }
    )


def memory_source_bindings(
    snapshots: Iterable[MemorySnapshot],
) -> tuple[PredictionRunSourceBinding, ...]:
    """把记忆快照转成 PredictionRun 的来源绑定,按 URI 排序去重。

    绑定使用快照读取器计算的规范 digest;缺失的记忆不能被绑定,
    同一 URI 出现不同版本快照视为调用方错误而不是静默择一。
    """

    bindings: dict[str, PredictionRunSourceBinding] = {}
    for snapshot in snapshots:
        if not snapshot.exists or snapshot.revision is None or snapshot.source_digest is None:
            raise MemoryBridgeError("a missing memory snapshot cannot bind a prediction run")
        binding = PredictionRunSourceBinding(
            str(snapshot.identity),
            int(snapshot.revision),
            str(snapshot.source_digest),
        )
        existing = bindings.get(binding.uri)
        if existing is not None and existing != binding:
            raise MemoryBridgeError("one memory URI resolved to conflicting snapshots")
        bindings[binding.uri] = binding
    return tuple(bindings[uri] for uri in sorted(bindings))


def memory_query_for_context(context: PredictionContext) -> str:
    """从预测上下文构造行为条件化的记忆检索查询。

    查询由"刚发生的行为 + 涉及对象 + 活跃目标"组成,使检索本身被当前
    行为事件约束:与此无关的记忆根本不会被召回,等价于该事件不看记忆。
    选步规则与状态匹配共用同一出处。
    """

    if not isinstance(context, PredictionContext):
        raise MemoryBridgeError("context must be PredictionContext")
    parts: list[str] = []
    history = context.input["behavior_history"]
    level = str(context.anchor["anchor_type"])
    steps = history["completed_actions"] if level == "action" else history["completed_events"]
    last = select_last_step(steps)
    if last is not None:
        parts.append(str(last["semantics"]))
        parts.extend(str(item) for item in last["target_refs"])
    frame = context.input["observation_frame"]
    parts.extend(str(item) for item in frame["active_goals"])
    parts.extend(str(item) for item in frame["subjects"])
    unique = [part for part in dict.fromkeys(part.strip() for part in parts) if part]
    if not unique:
        return "接下来可能进行的日常行为"
    return " ".join(unique)


def _intention_entry(document: MemoryDocument, now: datetime) -> str | None:
    fields = document.fields
    status = str(fields["status"])
    if status == "completed":
        return None
    name = str(fields["intent_name"])
    parts = [f"未完成事项「{name}」(状态 {status}"]
    last_confirmed_at = document.metadata.last_confirmed_at
    if last_confirmed_at is not None:
        days = int(max(0.0, (now - last_confirmed_at).total_seconds()) // 86400)
        if days >= 1:
            parts.append(f",{days} 天未确认")
    parts.append(")")
    next_step = fields.get("next_step")
    if isinstance(next_step, str) and next_step.strip():
        parts.append(f";下一步:{next_step.strip()}")
    return "".join(parts)


def _bullets(content: object, limit: int) -> tuple[str, ...]:
    """按 bullet 拆分条目;散文式无 bullet 内容按句拆分而不是整篇一条。

    整篇作为单条上下文会让 LLM 难以逐条引用与裁量,条目粒度是立场与
    相关性判断的正确量纲。
    """

    if not isinstance(content, str) or not content.strip():
        return ()
    bullets = [
        line.strip()[2:].strip()
        for line in content.splitlines()
        if line.strip().startswith(("- ", "* ")) and line.strip()[2:].strip()
    ]
    if not bullets:
        sentences = [content]
        for separator in _SENTENCE_SEPARATORS:
            sentences = [part for sentence in sentences for part in sentence.split(separator)]
        bullets = [sentence.strip() for sentence in sentences if sentence.strip()]
    return tuple(bullets[:limit])


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MemoryBridgeError(f"{label} must be a positive integer")
    return value


__all__ = [
    "MAX_MEMORY_CONTEXT_ENTRIES",
    "MemoryBridgeConfig",
    "MemoryBridgeError",
    "derive_memory_context",
    "memory_provenance",
    "memory_query_for_context",
    "memory_source_bindings",
    "persona_context",
]
