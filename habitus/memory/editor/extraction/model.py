"""受控检索决策和候选解析结果的严格领域模型。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import cast

from habitus.memory.editor.candidate import MemoryCandidateBatch, MemoryCandidateViolation
from habitus.memory.editor.mutation import MemoryMutationPlan
from habitus.memory.editor.page_id import MemoryPageIdMap
from habitus.memory.snapshot import MemorySnapshotBatch
from habitus.pre.conversation.messages.model import require_sha256

_MAX_REASON_CHARS = 600
_MAX_KEYWORDS = 6  # 与助手查记忆一侧"挑"的检索词个数、字数默认值一致
_MAX_KEYWORD_CHARS = 50


class MemoryExtractionError(RuntimeError):
    """Conversation 无法在确定性边界内解析为可靠记忆候选。"""


class MemoryExtractionPermanentError(MemoryExtractionError):
    """相同输入重试不会改变结果的提取失败。"""


class MemoryExtractionCapacityError(MemoryExtractionPermanentError):
    """完整事实无法装入已配置的模型或旧记忆预算。"""


class MemoryRetrievalStatus(str, Enum):
    """Retrieval Grader 对当前完整上下文的判断。"""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    IRRELEVANT = "irrelevant"


class MemoryRetrievalAction(str, Enum):
    """受控 ReAct 每轮唯一允许的只读动作。"""

    FINISH = "finish"
    LIST = "memory_list"
    SEARCH = "memory_search"
    READ = "memory_read"


@dataclass(frozen=True)
class MemoryRetrievalDecision:
    """把检索充分性判断和下一次单动作绑定在一个严格响应中。"""

    status: MemoryRetrievalStatus
    action: MemoryRetrievalAction
    directory: str | None
    uris: tuple[str, ...]
    reason: str
    keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            status = MemoryRetrievalStatus(self.status)
            action = MemoryRetrievalAction(self.action)
        except ValueError as exc:
            raise ValueError("retrieval decision contains an unsupported enum") from exc
        reason = _clean_text(self.reason, "reason", maximum=_MAX_REASON_CHARS)
        directory = _optional_clean_text(self.directory, "directory")
        uris = tuple(_clean_text(uri, "uri") for uri in self.uris)
        if len(set(uris)) != len(uris):
            raise ValueError("memory_read uris must be unique")
        keywords = tuple(_clean_text(keyword, "keyword", maximum=_MAX_KEYWORD_CHARS) for keyword in self.keywords)
        if len(set(keywords)) != len(keywords) or len(keywords) > _MAX_KEYWORDS:
            raise ValueError(f"memory_search keywords must be unique and at most {_MAX_KEYWORDS}")
        object.__setattr__(self, "keywords", keywords)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "uris", uris)
        self._require_valid_combination()

    def _require_valid_combination(self) -> None:
        status, action, directory, uris = self.status, self.action, self.directory, self.uris
        if action is MemoryRetrievalAction.FINISH:
            if status is MemoryRetrievalStatus.INSUFFICIENT:
                raise ValueError("insufficient retrieval cannot finish")
            if directory is not None or uris or self.keywords:
                raise ValueError("finish retrieval action cannot carry a directory, uris or keywords")
            return
        if status is MemoryRetrievalStatus.SUFFICIENT:
            raise ValueError("sufficient retrieval must finish")
        if action is MemoryRetrievalAction.SEARCH:
            if not self.keywords or uris:
                raise ValueError("memory_search requires keywords and forbids uris")
            return
        if self.keywords:
            raise ValueError("only memory_search carries keywords")
        if action is MemoryRetrievalAction.LIST:
            if directory is None or uris:
                raise ValueError("memory_list requires a directory and forbids uris")
            return
        if status is not MemoryRetrievalStatus.INSUFFICIENT:
            raise ValueError("memory_read is only valid for insufficient context")
        if not uris or directory is not None:
            raise ValueError("memory_read requires uris and forbids a directory")

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        """返回 Retrieval Grader 的无置信度严格 Schema。"""

        return _decision_schema(
            title="MemoryRetrievalDecision",
            description=(
                "判断当前完整旧记忆是否足够支撑记忆候选解析。每轮只能 finish、执行一次 memory_list（列一个目录）、"
                "执行一次 memory_search（按关键词在卡片字段文字里字面查找，可以限定目录），"
                "或执行一次 memory_read（读名字清单、目录列表、搜索结果或一跳关系里出现过的若干个 memory URI）。"
                "不得输出记忆候选、置信度、修改或写入操作。"
            ),
            actions=tuple(MemoryRetrievalAction),
            directory={
                "type": ["string", "null"],
                "minLength": 1,
                "description": (
                    "memory_list 要列的目录，例如 habitus://entities/项目；memory_search 可以用它限定只搜这个目录，"
                    "也可以为 null 搜全部；其他动作必须为 null。"
                ),
            },
            uris={
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": "memory_read 要读的完整 habitus:// L2 URI，可以多个；其他动作必须为空数组。",
            },
            keywords={
                "type": "array",
                "maxItems": _MAX_KEYWORDS,
                "items": {"type": "string", "minLength": 1, "maxLength": _MAX_KEYWORD_CHARS},
                "description": "memory_search 的检索词（字面匹配，不分大小写）；其他动作必须为空数组。",
            },
        )

    @classmethod
    def model_validate(cls, value: object) -> MemoryRetrievalDecision:
        """严格解析 Grader 输出，不执行字段或类型修复。"""

        payload = _strict_object(
            value,
            {"status", "action", "directory", "uris", "keywords", "reason"},
            "retrieval decision",
        )
        status = MemoryRetrievalStatus(_clean_text(payload["status"], "status"))
        action = MemoryRetrievalAction(_clean_text(payload["action"], "action"))
        raw_uris = cast(list[object], payload["uris"])  # JSON Schema 已在解码时限定为数组
        raw_keywords = cast(list[object], payload["keywords"])
        return cls(
            status=status,
            action=action,
            directory=_optional_clean_text(payload["directory"], "directory"),
            uris=tuple(_clean_text(uri, "uri") for uri in raw_uris),
            reason=_clean_text(payload["reason"], "reason"),
            keywords=tuple(_clean_text(keyword, "keyword") for keyword in raw_keywords),
        )


@dataclass(frozen=True)
class MemoryRetrievalFinalDecision(MemoryRetrievalDecision):
    """检索轮次用完后的结束决策：只能 finish，status 如实记录已读材料是否充分。

    这一轮系统不再执行工具，所以 insufficient 也只是结束检索、按已读到的旧记忆继续，
    不构成失败。
    """

    def _require_valid_combination(self) -> None:
        if self.action is not MemoryRetrievalAction.FINISH:
            raise ValueError("final retrieval round can only finish")
        if self.directory is not None or self.uris or self.keywords:
            raise ValueError("finish retrieval action cannot carry a directory, uris or keywords")

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        """返回检索轮次用完后只允许 finish 的严格 Schema。"""

        return _decision_schema(
            title="MemoryRetrievalFinalDecision",
            description=(
                "检索轮次预算已经用完，系统不会再执行 memory_list、memory_search 或 memory_read。只能返回"
                " finish，按已读到的完整旧记忆结束检索；status 如实记录这些材料"
                "是否充分，insufficient 不会导致失败。不得输出目录、URI、记忆候选、置信度、"
                "修改或写入操作。"
            ),
            actions=(MemoryRetrievalAction.FINISH,),
            directory={"type": "null", "description": "最后一轮不能再列目录，必须为 null。"},
            uris={"type": "array", "maxItems": 0, "description": "最后一轮不能再读取，必须为空数组。"},
            keywords={"type": "array", "maxItems": 0, "description": "最后一轮不能再搜索，必须为空数组。"},
        )


@dataclass(frozen=True)
class MemoryRetrievalObservation:
    """一次只读动作的有界审计结果，不保存文档正文副本。"""

    iteration: int
    action: MemoryRetrievalAction
    inputs: tuple[str, ...]
    result_uris: tuple[str, ...]
    added_uris: tuple[str, ...]
    relation_expanded_uris: tuple[str, ...]
    cached: bool = False

    def __post_init__(self) -> None:
        if self.iteration <= 0:
            raise ValueError("retrieval observation iteration must be positive")
        action = MemoryRetrievalAction(self.action)
        if action is MemoryRetrievalAction.FINISH:
            raise ValueError("finish does not produce a retrieval observation")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "inputs", tuple(_clean_text(item, "input") for item in self.inputs))
        if not self.inputs:
            raise ValueError("retrieval observation requires its action inputs")
        for name in ("result_uris", "added_uris", "relation_expanded_uris"):
            values = getattr(self, name)
            if any(not item for item in values):
                raise TypeError(f"{name} must be a tuple of non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} cannot contain duplicate URIs")

    def to_dict(self) -> dict[str, object]:
        """返回供下一轮 Grader 阅读的紧凑结果。"""

        return {
            "iteration": self.iteration,
            "action": self.action.value,
            "inputs": list(self.inputs),
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
    retrieval_ended_at_limit: bool = False
    """检索在轮次上限处结束：最后一轮判定已读材料不足，或最后一轮答复没能通过结构校验。"""
    source_rejections: tuple[MemoryCandidateViolation, ...] = ()
    """修补一轮后仍通不过来源校验、从本批剔掉的条目及原因（含引用被剔候选的关系与身份提议）。"""

    def __post_init__(self) -> None:
        for name in ("conversation_id", "segment_id"):
            _clean_text(getattr(self, name), name)
        require_sha256(self.source_segment_digest, "source_segment_digest")
        if self.mutations.read_set.old_memories != self.old_memories:
            raise ValueError("preliminary mutation plan does not use the extracted old-memory snapshots")
        candidate_page_ids = {candidate.page_id for candidate in self.candidates.iter_candidates()}
        mutation_page_ids = {mutation.match.candidate.page_id for mutation in self.mutations.mutations}
        if mutation_page_ids != candidate_page_ids:
            raise ValueError("preliminary mutation plan must cover every memory candidate exactly once")
        if not self.retrieval_decisions and not self.retrieval_ended_at_limit:
            raise TypeError("retrieval_decisions must contain at least one decision")


def _decision_schema(
    *,
    title: str,
    description: str,
    actions: tuple[MemoryRetrievalAction, ...],
    directory: dict[str, object],
    uris: dict[str, object],
    keywords: dict[str, object],
) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "description": description,
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "action", "directory", "uris", "keywords", "reason"],
        "properties": {
            "status": {
                "type": "string",
                "enum": [status.value for status in MemoryRetrievalStatus],
            },
            "action": {
                "type": "string",
                "enum": [action.value for action in actions],
            },
            "directory": directory,
            "uris": uris,
            "keywords": keywords,
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": _MAX_REASON_CHARS,
                "description": "只解释检索充分性，不生成或修正记忆内容。",
            },
        },
    }


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
    "MemoryRetrievalFinalDecision",
    "MemoryRetrievalObservation",
    "MemoryRetrievalStatus",
]
