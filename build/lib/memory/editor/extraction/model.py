"""受控检索决策和候选解析结果的严格领域模型。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from infrastructure.editor.snapshot import SnapshotBatch
from memory.editor.candidate import MemoryCandidateBatch
from memory.editor.mutation import MemoryMutationPlan
from memory.editor.page_id import MemoryPageIdMap
from memory.snapshot import MemorySnapshotBatch
from pre.conversation.messages.model import require_sha256

_MAX_REASON_CHARS = 600


class MemoryExtractionError(RuntimeError):
    """Conversation 无法在确定性边界内解析为可靠记忆候选。"""


class MemoryExtractionPermanentError(MemoryExtractionError):
    """相同输入重试不会改变结果的提取失败。"""


class MemoryExtractionCapacityError(MemoryExtractionPermanentError):
    """完整事实无法装入已配置的模型或旧记忆预算。"""


class MemoryRetrievalIncompleteError(MemoryExtractionPermanentError):
    """受控检索达到上限后，旧记忆上下文仍不足。"""


class MemoryRetrievalStatus(str, Enum):
    """Retrieval Grader 对当前完整上下文的判断。"""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    IRRELEVANT = "irrelevant"


class MemoryRetrievalAction(str, Enum):
    """受控 ReAct 每轮唯一允许的只读动作。"""

    FINISH = "finish"
    SEARCH = "memory_search"
    READ = "memory_read"


@dataclass(frozen=True)
class MemoryRetrievalDecision:
    """把检索充分性判断和下一次单动作绑定在一个严格响应中。"""

    status: MemoryRetrievalStatus
    action: MemoryRetrievalAction
    query: str | None
    uri: str | None
    reason: str

    def __post_init__(self) -> None:
        try:
            status = MemoryRetrievalStatus(self.status)
            action = MemoryRetrievalAction(self.action)
        except ValueError as exc:
            raise ValueError("retrieval decision contains an unsupported enum") from exc
        reason = _clean_text(self.reason, "reason", maximum=_MAX_REASON_CHARS)
        query = _optional_clean_text(self.query, "query")
        uri = _optional_clean_text(self.uri, "uri")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "uri", uri)

        if action is MemoryRetrievalAction.FINISH:
            if status is MemoryRetrievalStatus.INSUFFICIENT:
                raise ValueError("insufficient retrieval cannot finish")
            if query is not None or uri is not None:
                raise ValueError("finish retrieval action cannot carry query or uri")
            return
        if status is MemoryRetrievalStatus.SUFFICIENT:
            raise ValueError("sufficient retrieval must finish")
        if action is MemoryRetrievalAction.SEARCH:
            if query is None or uri is not None:
                raise ValueError("memory_search requires query and forbids uri")
            return
        if status is not MemoryRetrievalStatus.INSUFFICIENT:
            raise ValueError("memory_read is only valid for insufficient context")
        if uri is None or query is not None:
            raise ValueError("memory_read requires uri and forbids query")

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        """返回 Retrieval Grader 的无置信度严格 Schema。"""

        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "MemoryRetrievalDecision",
            "description": (
                "判断当前完整旧记忆是否足够支撑记忆候选解析。每轮只能 finish、执行一次"
                " memory_search，或读取一个已经由系统暴露的 memory URI。不得输出多个查询、"
                "记忆候选、置信度、修改或写入操作。"
            ),
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "action", "query", "uri", "reason"],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [status.value for status in MemoryRetrievalStatus],
                },
                "action": {
                    "type": "string",
                    "enum": [action.value for action in MemoryRetrievalAction],
                },
                "query": {
                    "type": ["string", "null"],
                    "minLength": 1,
                    "maxLength": 20_000,
                    "description": "memory_search 的单个语义查询；其他动作必须为 null。",
                },
                "uri": {
                    "type": ["string", "null"],
                    "minLength": 1,
                    "description": "memory_read 的一个完整 memory:// L2 URI；其他动作必须为 null。",
                },
                "reason": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_REASON_CHARS,
                    "description": "只解释检索充分性，不生成或修正记忆内容。",
                },
            },
        }

    @classmethod
    def model_validate(cls, value: object) -> MemoryRetrievalDecision:
        """严格解析 Grader 输出，不执行字段或类型修复。"""

        payload = _strict_object(
            value,
            {"status", "action", "query", "uri", "reason"},
            "retrieval decision",
        )
        status = MemoryRetrievalStatus(_clean_text(payload["status"], "status"))
        action = MemoryRetrievalAction(_clean_text(payload["action"], "action"))
        return cls(
            status=status,
            action=action,
            query=_optional_clean_text(payload["query"], "query"),
            uri=_optional_clean_text(payload["uri"], "uri"),
            reason=_clean_text(payload["reason"], "reason"),
        )

    def require_action_allowed(self, *, allow_action: bool) -> None:
        """最终检索轮禁止继续调用工具。"""

        if not isinstance(allow_action, bool):
            raise TypeError("allow_action must be boolean")
        if not allow_action and self.action is not MemoryRetrievalAction.FINISH:
            raise MemoryRetrievalIncompleteError(
                "retrieval grader requested another action after the configured final round"
            )


@dataclass(frozen=True)
class MemoryRetrievalObservation:
    """一次只读动作的有界审计结果，不保存文档正文副本。"""

    iteration: int
    action: MemoryRetrievalAction
    input_value: str
    result_uris: tuple[str, ...]
    added_uris: tuple[str, ...]
    relation_expanded_uris: tuple[str, ...]
    cached: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int) or self.iteration <= 0:
            raise ValueError("retrieval observation iteration must be positive")
        action = MemoryRetrievalAction(self.action)
        if action is MemoryRetrievalAction.FINISH:
            raise ValueError("finish does not produce a retrieval observation")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "input_value", _clean_text(self.input_value, "input_value"))
        for name in ("result_uris", "added_uris", "relation_expanded_uris"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, str) or not item for item in values):
                raise TypeError(f"{name} must be a tuple of non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} cannot contain duplicate URIs")
        if not isinstance(self.cached, bool):
            raise TypeError("retrieval observation cached must be boolean")

    def to_dict(self) -> dict[str, object]:
        """返回供下一轮 Grader 阅读的紧凑结果。"""

        return {
            "iteration": self.iteration,
            "action": self.action.value,
            "input": self.input_value,
            "result_uris": list(self.result_uris),
            "added_uris": list(self.added_uris),
            "relation_expanded_uris": list(self.relation_expanded_uris),
            "cached": self.cached,
        }


@dataclass(frozen=True)
class MemoryExtractionResult:
    """通过检索、严格校验和初步字段计划的候选解析结果。"""

    conversation_id: str
    segment_id: str
    source_segment_digest: str
    candidates: MemoryCandidateBatch
    mutations: MemoryMutationPlan
    old_memories: MemorySnapshotBatch
    page_ids: MemoryPageIdMap
    retrieval_decisions: tuple[MemoryRetrievalDecision, ...]
    retrieval_observations: tuple[MemoryRetrievalObservation, ...]

    def __post_init__(self) -> None:
        for name in ("conversation_id", "segment_id"):
            _clean_text(getattr(self, name), name)
        require_sha256(self.source_segment_digest, "source_segment_digest")
        if not isinstance(self.candidates, MemoryCandidateBatch):
            raise TypeError("candidates must be a MemoryCandidateBatch")
        if not isinstance(self.mutations, MemoryMutationPlan):
            raise TypeError("mutations must be a MemoryMutationPlan")
        if not isinstance(self.old_memories, SnapshotBatch):
            raise TypeError("old_memories must be a MemorySnapshotBatch")
        if self.mutations.read_set.old_memories != self.old_memories:
            raise ValueError("preliminary mutation plan does not use the extracted old-memory snapshots")
        candidate_page_ids = {candidate.page_id for candidate in self.candidates.iter_candidates()}
        mutation_page_ids = {mutation.match.candidate.page_id for mutation in self.mutations.mutations}
        if mutation_page_ids != candidate_page_ids:
            raise ValueError("preliminary mutation plan must cover every memory candidate exactly once")
        if not isinstance(self.page_ids, MemoryPageIdMap):
            raise TypeError("page_ids must be a MemoryPageIdMap")
        if not self.retrieval_decisions or any(
            not isinstance(item, MemoryRetrievalDecision) for item in self.retrieval_decisions
        ):
            raise TypeError("retrieval_decisions must contain at least one decision")
        if any(not isinstance(item, MemoryRetrievalObservation) for item in self.retrieval_observations):
            raise TypeError("retrieval_observations contains an invalid observation")


def _strict_object(
    value: object,
    expected: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    keys = set(value)
    unknown = keys - expected
    missing = expected - keys
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    return value


def _clean_text(value: object, name: str, *, maximum: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    normalized = value.strip()
    if maximum is not None and len(normalized) > maximum:
        raise ValueError(f"{name} exceeds its character limit")
    return normalized


def _optional_clean_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, name)


__all__ = [
    "MemoryExtractionError",
    "MemoryExtractionCapacityError",
    "MemoryExtractionPermanentError",
    "MemoryExtractionResult",
    "MemoryRetrievalAction",
    "MemoryRetrievalDecision",
    "MemoryRetrievalIncompleteError",
    "MemoryRetrievalObservation",
    "MemoryRetrievalStatus",
]
