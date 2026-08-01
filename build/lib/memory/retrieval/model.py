"""Agent 对外记忆检索使用的严格配置、查询计划和结果模型。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from memory.conversation.indexing import ConversationSummaryMatch
from memory.document import MemoryDocument, MemoryStoredLink
from memory.intention import (
    MemoryIntentionRecallScope,
    MemoryIntentionReview,
    intention_matches_scope,
)
from memory.model import MemoryKind
from memory.retrieval.lifecycle import MemoryTemperature
from memory.uri import MemoryURI, MemoryURINodeType


class MemorySearchError(RuntimeError):
    """记忆搜索无法在完整性或资源边界内完成。"""


@dataclass(frozen=True)
class MemorySearchServiceConfig:
    """限制公开检索、上下文规划、关系扩展和最终上下文大小。"""

    default_limit: int = 10
    max_limit: int = 20
    candidate_multiplier: int = 4
    max_query_chars: int = 5_000
    max_planned_queries: int = 4
    max_planned_query_chars: int = 2_000
    max_query_intent_chars: int = 500
    max_summary_context_chars: int = 30_000
    max_recent_messages: int = 5
    max_recent_message_chars: int = 4_000
    max_target_context_chars: int = 8_000
    max_planner_context_chars: int = 70_000
    planner_max_output_tokens: int = 2_000
    max_target_roots: int = 16
    max_relation_neighbors_per_match: int = 5
    max_relation_neighbors_total: int = 32
    max_context_chars: int = 120_000
    retrieval_grader_max_context_chars: int = 120_000
    retrieval_grader_max_output_tokens: int = 1_000
    retrieval_grader_max_reason_chars: int = 1_000
    retrieval_grader_max_missing_items: int = 3
    retrieval_grader_max_missing_item_chars: int = 1_000
    summary_fallback_enabled: bool = True
    summary_fallback_limit: int = 3
    summary_fallback_max_query_chars: int = 2_000
    summary_fallback_max_context_chars: int = 24_000

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("default_limit", self.default_limit, 1, 1_000),
            ("max_limit", self.max_limit, 1, 1_000),
            ("candidate_multiplier", self.candidate_multiplier, 1, 20),
            ("max_query_chars", self.max_query_chars, 1, 1_000_000),
            ("max_planned_queries", self.max_planned_queries, 1, 8),
            ("max_planned_query_chars", self.max_planned_query_chars, 1, 5_000),
            ("max_query_intent_chars", self.max_query_intent_chars, 1, 1_000),
            ("max_summary_context_chars", self.max_summary_context_chars, 1, 2_000_000),
            ("max_recent_messages", self.max_recent_messages, 1, 1_000),
            ("max_recent_message_chars", self.max_recent_message_chars, 1, 1_000_000),
            ("max_target_context_chars", self.max_target_context_chars, 1, 1_000_000),
            ("max_planner_context_chars", self.max_planner_context_chars, 1_024, 4_000_000),
            ("planner_max_output_tokens", self.planner_max_output_tokens, 128, 65_536),
            ("max_target_roots", self.max_target_roots, 1, 128),
            (
                "max_relation_neighbors_per_match",
                self.max_relation_neighbors_per_match,
                0,
                1_000,
            ),
            ("max_relation_neighbors_total", self.max_relation_neighbors_total, 0, 10_000),
            ("max_context_chars", self.max_context_chars, 1_024, 4_000_000),
            (
                "retrieval_grader_max_context_chars",
                self.retrieval_grader_max_context_chars,
                1_024,
                4_000_000,
            ),
            (
                "retrieval_grader_max_output_tokens",
                self.retrieval_grader_max_output_tokens,
                128,
                65_536,
            ),
            (
                "retrieval_grader_max_reason_chars",
                self.retrieval_grader_max_reason_chars,
                1,
                10_000,
            ),
            (
                "retrieval_grader_max_missing_items",
                self.retrieval_grader_max_missing_items,
                1,
                10,
            ),
            (
                "retrieval_grader_max_missing_item_chars",
                self.retrieval_grader_max_missing_item_chars,
                1,
                10_000,
            ),
            ("summary_fallback_limit", self.summary_fallback_limit, 1, 100),
            (
                "summary_fallback_max_query_chars",
                self.summary_fallback_max_query_chars,
                1,
                5_000,
            ),
            (
                "summary_fallback_max_context_chars",
                self.summary_fallback_max_context_chars,
                1,
                1_000_000,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if self.default_limit > self.max_limit:
            raise ValueError("memory search default_limit cannot exceed max_limit")
        if self.max_planned_query_chars > self.max_query_chars:
            raise ValueError("planned query characters cannot exceed the public query bound")
        if self.max_relation_neighbors_per_match > self.max_relation_neighbors_total:
            raise ValueError("per-match relation limit cannot exceed the total relation limit")
        if not isinstance(self.summary_fallback_enabled, bool):
            raise TypeError("summary_fallback_enabled must be boolean")
        if self.summary_fallback_max_query_chars > self.max_query_chars:
            raise ValueError("Summary fallback query cannot exceed the public query bound")
        if self.summary_fallback_max_context_chars > self.max_context_chars:
            raise ValueError("Summary fallback context cannot exceed the total context bound")
        if self.max_context_chars > self.retrieval_grader_max_context_chars:
            raise ValueError("retrieval grader context must fit the complete assembled Memory context")


class MemoryRetrievalSufficiency(str, Enum):
    """Memory 主召回是否足以支持当前问题。"""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class MemoryRetrievalAssessment:
    """只决定是否启用 Summary 后备，不生成或修改任何记忆。"""

    decision: MemoryRetrievalSufficiency
    reason: str
    missing_information: tuple[str, ...]
    summary_query: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", MemoryRetrievalSufficiency(self.decision))
        object.__setattr__(self, "reason", _bounded_text(self.reason, "retrieval assessment reason", 10_000))
        if not isinstance(self.missing_information, tuple) or any(
            not isinstance(item, str) or not item.strip() or item != item.strip()
            for item in self.missing_information
        ):
            raise ValueError("retrieval assessment missing_information contains invalid text")
        if len(set(self.missing_information)) != len(self.missing_information):
            raise ValueError("retrieval assessment missing_information must be unique")
        if self.decision is MemoryRetrievalSufficiency.SUFFICIENT:
            if self.missing_information or self.summary_query is not None:
                raise ValueError("sufficient Memory retrieval cannot request Summary fallback")
            return
        if not self.missing_information:
            raise ValueError("insufficient Memory retrieval must identify missing information")
        if self.summary_query is None:
            raise ValueError("insufficient Memory retrieval must provide one Summary query")
        object.__setattr__(
            self,
            "summary_query",
            _bounded_text(self.summary_query, "Summary fallback query", 5_000),
        )

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "reason", "missing_information", "summary_query"],
            "properties": {
                "decision": {"type": "string", "enum": [item.value for item in MemoryRetrievalSufficiency]},
                "reason": {"type": "string", "minLength": 1, "maxLength": 10_000},
                "missing_information": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {"type": "string", "minLength": 1, "maxLength": 10_000},
                },
                "summary_query": {
                    "anyOf": [
                        {"type": "string", "minLength": 1, "maxLength": 5_000},
                        {"type": "null"},
                    ]
                },
            },
        }

    @classmethod
    def model_validate(cls, value: object) -> MemoryRetrievalAssessment:
        if not isinstance(value, Mapping):
            raise TypeError("retrieval assessment must be an object")
        expected = {"decision", "reason", "missing_information", "summary_query"}
        if set(value) != expected:
            raise ValueError("retrieval assessment fields do not match its schema")
        raw_missing = value["missing_information"]
        if not isinstance(raw_missing, list | tuple):
            raise TypeError("retrieval assessment missing_information must be a list")
        return cls(
            decision=MemoryRetrievalSufficiency(value["decision"]),
            reason=value["reason"],
            missing_information=tuple(raw_missing),
            summary_query=value["summary_query"],
        )


@dataclass(frozen=True)
class MemorySearchHit:
    """语义搜索返回的一个 L2 URI，以及可审计的阶段分数。"""

    uri: MemoryURI
    score: float
    vector_score: float | None = None
    rerank_score: float | None = None
    lifecycle_semantic_score: float | None = None
    lifecycle_hotness: float | None = None
    lifecycle_temperature: MemoryTemperature | None = None

    def __post_init__(self) -> None:
        parsed = MemoryURI.parse(self.uri)
        if parsed.node_type is not MemoryURINodeType.DOCUMENT:
            raise ValueError("memory search hit must identify an L2 document")
        object.__setattr__(self, "uri", parsed)
        score = _finite_score(self.score, "memory search score")
        object.__setattr__(self, "score", score)
        vector_score = (
            score
            if self.vector_score is None
            else _finite_score(
                self.vector_score,
                "memory search vector_score",
            )
        )
        if not -1.0 <= vector_score <= 1.0:
            raise ValueError("memory search vector_score must be between -1 and 1")
        object.__setattr__(self, "vector_score", vector_score)
        if self.rerank_score is not None:
            rerank_score = _finite_score(self.rerank_score, "memory search rerank_score")
            object.__setattr__(self, "rerank_score", rerank_score)
        lifecycle_values = (
            self.lifecycle_semantic_score,
            self.lifecycle_hotness,
            self.lifecycle_temperature,
        )
        if all(value is None for value in lifecycle_values):
            if self.rerank_score is not None and score != self.rerank_score:
                raise ValueError("memory search final score must equal its rerank score")
            return
        if any(value is None for value in lifecycle_values):
            raise ValueError("memory search lifecycle score fields must be complete")
        semantic_score = _finite_score(
            self.lifecycle_semantic_score,
            "memory search lifecycle_semantic_score",
        )
        hotness = _finite_score(
            self.lifecycle_hotness,
            "memory search lifecycle_hotness",
        )
        if not 0.0 <= hotness <= 1.0:
            raise ValueError("memory search lifecycle_hotness must be between zero and one")
        temperature = MemoryTemperature(self.lifecycle_temperature)
        if self.rerank_score is not None and semantic_score != self.rerank_score:
            raise ValueError("memory search lifecycle semantic score must equal its rerank score")
        if score > semantic_score:
            raise ValueError("memory search lifecycle adjustment cannot improve the semantic score")
        object.__setattr__(self, "lifecycle_semantic_score", semantic_score)
        object.__setattr__(self, "lifecycle_hotness", hotness)
        object.__setattr__(self, "lifecycle_temperature", temperature)


@dataclass(frozen=True)
class MemoryTypedQuery:
    """由当前问题直接构造或由受控查询规划器生成的一条独立查询。"""

    query: str
    intent: str
    priority: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _bounded_text(self.query, "typed query", 5_000))
        object.__setattr__(self, "intent", _bounded_text(self.intent, "query intent", 1_000))
        if isinstance(self.priority, bool) or not isinstance(self.priority, int) or not 1 <= self.priority <= 5:
            raise ValueError("query priority must be between one and five")

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "intent": self.intent,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, value: object) -> MemoryTypedQuery:
        if not isinstance(value, Mapping) or set(value) != {"query", "intent", "priority"}:
            raise ValueError("typed query must contain exactly query, intent, and priority")
        return cls(
            query=value["query"],
            intent=value["intent"],
            priority=value["priority"],
        )


@dataclass(frozen=True)
class MemoryQueryPlanContent:
    """LLM 只能生成查询语义，不得生成 URI、过滤器或记忆操作。"""

    queries: tuple[MemoryTypedQuery, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.queries, tuple) or not 1 <= len(self.queries) <= 8:
            raise ValueError("memory query plan must contain between one and eight queries")
        if any(not isinstance(query, MemoryTypedQuery) for query in self.queries):
            raise TypeError("memory query plan contains an invalid query")
        normalized = {(query.query.casefold(), query.intent.casefold()) for query in self.queries}
        if len(normalized) != len(self.queries):
            raise ValueError("memory query plan contains duplicate queries")

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["queries"],
            "properties": {
                "queries": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query", "intent", "priority"],
                        "properties": {
                            "query": {"type": "string", "minLength": 1, "maxLength": 5_000},
                            "intent": {"type": "string", "minLength": 1, "maxLength": 1_000},
                            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                    },
                }
            },
        }

    @classmethod
    def model_validate(cls, value: object) -> MemoryQueryPlanContent:
        if not isinstance(value, Mapping) or set(value) != {"queries"}:
            raise ValueError("memory query plan must contain exactly queries")
        raw_queries = value["queries"]
        if not isinstance(raw_queries, list | tuple):
            raise TypeError("memory query plan queries must be a list")
        return cls(tuple(MemoryTypedQuery.from_dict(query) for query in raw_queries))


@dataclass(frozen=True)
class MemoryQueryPlan:
    """绑定原始问题和可选 Conversation 的已验证查询计划。"""

    original_query: str
    queries: tuple[MemoryTypedQuery, ...]
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_query", _bounded_text(self.original_query, "original query", 1_000_000))
        MemoryQueryPlanContent(self.queries)
        if self.conversation_id is not None:
            object.__setattr__(
                self,
                "conversation_id",
                _bounded_text(self.conversation_id, "conversation_id", 1_000),
            )

    @property
    def contextual(self) -> bool:
        return self.conversation_id is not None


@dataclass(frozen=True)
class MemoryQueryResult:
    """一条规划查询的只读命中，用于结果溯源。"""

    query: MemoryTypedQuery
    hits: tuple[MemorySearchHit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query, MemoryTypedQuery):
            raise TypeError("query result query must be MemoryTypedQuery")
        if not isinstance(self.hits, tuple) or any(not isinstance(hit, MemorySearchHit) for hit in self.hits):
            raise TypeError("query result hits must contain MemorySearchHit values")
        expected = tuple(sorted(self.hits, key=lambda hit: (-hit.score, str(hit.uri))))
        if self.hits != expected or len({hit.uri for hit in self.hits}) != len(self.hits):
            raise ValueError("query result hits must be unique and relevance-sorted")


@dataclass(frozen=True)
class MemoryRelatedMemory:
    """由一个直接命中的持久关系确定性补充的一跳完整记忆。"""

    relation: MemoryStoredLink
    document: MemoryDocument
    intention_review: MemoryIntentionReview | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation, MemoryStoredLink):
            raise TypeError("related memory relation must be MemoryStoredLink")
        if not isinstance(self.document, MemoryDocument):
            raise TypeError("related memory document must be MemoryDocument")
        uri = MemoryURI.from_address(self.document.address)
        if uri not in {self.relation.from_uri, self.relation.to_uri}:
            raise ValueError("related memory document is not a relation endpoint")
        _validate_intention_review(self.document, self.intention_review)


@dataclass(frozen=True)
class MemoryMatchedMemory:
    """一个直接语义命中的完整 L2 记忆及其有界一跳关系。"""

    hit: MemorySearchHit
    document: MemoryDocument
    matched_queries: tuple[str, ...]
    intention_review: MemoryIntentionReview | None = None
    related: tuple[MemoryRelatedMemory, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.hit, MemorySearchHit):
            raise TypeError("matched memory hit must be MemorySearchHit")
        if not isinstance(self.document, MemoryDocument):
            raise TypeError("matched memory document must be MemoryDocument")
        if MemoryURI.from_address(self.document.address) != self.hit.uri:
            raise ValueError("matched memory document does not match its hit URI")
        if not isinstance(self.matched_queries, tuple) or not self.matched_queries:
            raise ValueError("matched memory must record at least one query")
        if any(not isinstance(query, str) or not query.strip() for query in self.matched_queries):
            raise TypeError("matched memory queries must be non-empty strings")
        if len(set(self.matched_queries)) != len(self.matched_queries):
            raise ValueError("matched memory queries must be unique")
        _validate_intention_review(self.document, self.intention_review)
        if not isinstance(self.related, tuple) or any(
            not isinstance(item, MemoryRelatedMemory) for item in self.related
        ):
            raise TypeError("matched memory related values are invalid")
        identities = tuple(item.relation.identity for item in self.related)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
            raise ValueError("matched memory relations must be unique and sorted")

    @property
    def uri(self) -> MemoryURI:
        return self.hit.uri


class MemorySearchDegradationStage(str, Enum):
    """可选增强失败后已经采用的确定性降级阶段。"""

    QUERY_PLANNER = "query_planner"
    RECALL_LIFECYCLE = "recall_lifecycle"
    RETRIEVAL_GRADER = "retrieval_grader"
    SUMMARY_FALLBACK = "summary_fallback"


@dataclass(frozen=True)
class MemorySearchDegradation:
    """不包含敏感异常正文的检索降级事实。"""

    stage: MemorySearchDegradationStage
    error_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", MemorySearchDegradationStage(self.stage))
        if not isinstance(self.error_type, str) or not self.error_type or len(self.error_type) > 256:
            raise ValueError("search degradation error_type must be non-empty bounded text")


@dataclass(frozen=True)
class MemorySearchResult:
    """可直接交给 Agent 的完整、可溯源、受预算约束的记忆结果。"""

    query: str
    target_roots: tuple[MemoryURI, ...]
    kinds: tuple[MemoryKind, ...]
    intention_scope: MemoryIntentionRecallScope
    plan: MemoryQueryPlan
    query_results: tuple[MemoryQueryResult, ...]
    memories: tuple[MemoryMatchedMemory, ...]
    retrieval_assessment: MemoryRetrievalAssessment | None
    summary_fallback_attempted: bool
    summary_fallbacks: tuple[ConversationSummaryMatch, ...]
    context: str
    budget_exhausted: bool
    degradations: tuple[MemorySearchDegradation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _bounded_text(self.query, "search result query", 1_000_000))
        if not isinstance(self.target_roots, tuple) or not self.target_roots:
            raise ValueError("search result requires target roots")
        if any(root.node_type is not MemoryURINodeType.DIRECTORY for root in self.target_roots):
            raise ValueError("search result roots must identify memory directories")
        if not isinstance(self.kinds, tuple):
            raise TypeError("search result kinds must be a tuple")
        normalized_kinds = tuple(MemoryKind(kind) for kind in self.kinds)
        if len(normalized_kinds) != len(set(normalized_kinds)):
            raise ValueError("search result kinds must be unique")
        object.__setattr__(self, "kinds", normalized_kinds)
        scope = MemoryIntentionRecallScope(self.intention_scope)
        object.__setattr__(self, "intention_scope", scope)
        if scope is MemoryIntentionRecallScope.COMPLETED and normalized_kinds != (
            MemoryKind.INTENTION,
        ):
            raise ValueError("completed Intention search must target only Intention memory")
        if not isinstance(self.plan, MemoryQueryPlan) or self.plan.original_query != self.query:
            raise ValueError("search result query plan does not match the original query")
        if not isinstance(self.query_results, tuple) or any(
            not isinstance(result, MemoryQueryResult) for result in self.query_results
        ):
            raise TypeError("search result query_results are invalid")
        if not isinstance(self.degradations, tuple) or any(
            not isinstance(item, MemorySearchDegradation) for item in self.degradations
        ):
            raise TypeError("search result degradations are invalid")
        stages = tuple(item.stage for item in self.degradations)
        if len(stages) != len(set(stages)):
            raise ValueError("search result degradation stages must be unique")
        if tuple(result.query for result in self.query_results) != self.plan.queries:
            raise ValueError("search result query_results do not match the query plan")
        if not isinstance(self.memories, tuple) or any(
            not isinstance(memory, MemoryMatchedMemory) for memory in self.memories
        ):
            raise TypeError("search result memories are invalid")
        expected = tuple(sorted(self.memories, key=lambda item: (-item.hit.score, str(item.uri))))
        if self.memories != expected or len({item.uri for item in self.memories}) != len(self.memories):
            raise ValueError("search result memories must be unique and relevance-sorted")
        for memory in self.memories:
            if normalized_kinds and memory.document.kind not in normalized_kinds:
                raise ValueError("direct search result is outside its requested memory kinds")
            if memory.document.kind is MemoryKind.INTENTION and not intention_matches_scope(
                memory.document.fields.get("status"),
                scope,
            ):
                raise ValueError("direct Intention result is outside its requested recall scope")
        if self.retrieval_assessment is not None and not isinstance(
            self.retrieval_assessment,
            MemoryRetrievalAssessment,
        ):
            raise TypeError("search result retrieval_assessment is invalid")
        if not isinstance(self.summary_fallback_attempted, bool):
            raise TypeError("search result summary_fallback_attempted must be boolean")
        if not isinstance(self.summary_fallbacks, tuple) or any(
            not isinstance(item, ConversationSummaryMatch) for item in self.summary_fallbacks
        ):
            raise TypeError("search result summary_fallbacks are invalid")
        fallback_expected = tuple(
            sorted(
                self.summary_fallbacks,
                key=lambda item: (-item.score, item.reference.identity),
            )
        )
        if self.summary_fallbacks != fallback_expected or len(
            {item.reference.identity for item in self.summary_fallbacks}
        ) != len(self.summary_fallbacks):
            raise ValueError("Summary fallback results must be unique and relevance-sorted")
        if self.summary_fallbacks and (
            self.retrieval_assessment is None
            or self.retrieval_assessment.decision is not MemoryRetrievalSufficiency.INSUFFICIENT
        ):
            raise ValueError("Summary fallback requires an insufficient Memory assessment")
        if self.summary_fallbacks and not self.summary_fallback_attempted:
            raise ValueError("Summary fallback results require an attempted fallback search")
        if self.summary_fallback_attempted and (
            self.retrieval_assessment is None
            or self.retrieval_assessment.decision is not MemoryRetrievalSufficiency.INSUFFICIENT
        ):
            raise ValueError("Summary fallback can only be attempted after an insufficient assessment")
        if not isinstance(self.context, str):
            raise TypeError("search result context must be text")
        if bool(self.memories or self.summary_fallbacks) != bool(self.context):
            raise ValueError("search result context presence must match its Memory or Summary content")
        if not isinstance(self.budget_exhausted, bool):
            raise TypeError("search result budget_exhausted must be boolean")

    @property
    def total(self) -> int:
        return len(self.memories)


def _bounded_text(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be non-empty text without surrounding whitespace")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds its character bound")
    return value


def _finite_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{label} must be finite")
    return score


def _validate_intention_review(
    document: MemoryDocument,
    review: MemoryIntentionReview | None,
) -> None:
    """确保复核提示只附着在活动 Intention，不变成通用记忆字段。"""

    active = document.kind is MemoryKind.INTENTION and document.fields.get("status") != "completed"
    if not active:
        if review is not None:
            raise ValueError("only active Intention memory accepts a review reminder")
        return
    if not isinstance(review, MemoryIntentionReview):
        raise ValueError("active Intention memory requires a review reminder")
    if review.last_confirmed_at != document.metadata.last_confirmed_at:
        raise ValueError("Intention review reminder does not match document confirmation metadata")


__all__ = [
    "MemoryMatchedMemory",
    "MemoryQueryPlan",
    "MemoryQueryPlanContent",
    "MemoryQueryResult",
    "MemoryRelatedMemory",
    "MemoryRetrievalAssessment",
    "MemoryRetrievalSufficiency",
    "MemorySearchError",
    "MemorySearchDegradation",
    "MemorySearchDegradationStage",
    "MemorySearchHit",
    "MemorySearchResult",
    "MemorySearchServiceConfig",
    "MemoryTypedQuery",
]
