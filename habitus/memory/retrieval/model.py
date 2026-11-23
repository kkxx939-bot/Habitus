"""Agent 记忆检索的严格配置与结果模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from habitus.memory.conversation.summary_reference import ConversationSummaryMatch
from habitus.memory.document import MemoryDocument, MemoryStoredLink
from habitus.memory.model import MemoryKind
from habitus.memory.recall import MemoryTemperature
from habitus.memory.retrieval.records import BehaviorRecord
from habitus.memory.uri import MemoryURI, MemoryURINodeType


class MemorySearchError(RuntimeError):
    """记忆检索无法在完整性或资源边界内完成。"""


@dataclass(frozen=True)
class MemorySearchServiceConfig:
    """限制一次检索的总时间、轮数、各处来源的条数与最终上下文大小。"""

    default_limit: int = 10
    max_limit: int = 20
    max_query_chars: int = 5_000
    time_limit_seconds: float = 12.0
    accounting_reserve_seconds: float = 1.0
    max_rounds: int = 2
    max_catalog_chars: int = 40_000
    max_reads_per_round: int = 8
    max_unlisted_scan: int = 1_000
    max_keywords: int = 6
    max_keyword_chars: int = 50
    max_record_days: int = 92
    max_record_list: int = 80
    max_record_hits: int = 20
    max_summary_hits: int = 5
    max_material_chars: int = 120_000
    max_prompt_chars: int = 200_000
    step_max_output_tokens: int = 1_000
    max_summary_context_chars: int = 30_000
    max_recent_messages: int = 5
    max_recent_message_chars: int = 4_000
    max_relation_neighbors_per_match: int = 5
    max_relation_neighbors_total: int = 32
    max_context_chars: int = 120_000

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("default_limit", self.default_limit, 1, 1_000),
            ("max_limit", self.max_limit, 1, 1_000),
            ("max_query_chars", self.max_query_chars, 1, 1_000_000),
            ("max_rounds", self.max_rounds, 1, 3),
            ("max_catalog_chars", self.max_catalog_chars, 1_024, 2_000_000),
            ("max_reads_per_round", self.max_reads_per_round, 1, 100),
            ("max_unlisted_scan", self.max_unlisted_scan, 0, 100_000),
            ("max_keywords", self.max_keywords, 1, 20),
            ("max_keyword_chars", self.max_keyword_chars, 1, 1_000),
            ("max_record_days", self.max_record_days, 1, 3_660),
            ("max_record_list", self.max_record_list, 1, 10_000),
            ("max_record_hits", self.max_record_hits, 1, 1_000),
            ("max_summary_hits", self.max_summary_hits, 1, 1_000),
            ("max_material_chars", self.max_material_chars, 1_024, 4_000_000),
            ("max_prompt_chars", self.max_prompt_chars, 1_024, 8_000_000),
            ("step_max_output_tokens", self.step_max_output_tokens, 128, 65_536),
            ("max_summary_context_chars", self.max_summary_context_chars, 1, 2_000_000),
            ("max_recent_messages", self.max_recent_messages, 1, 1_000),
            ("max_recent_message_chars", self.max_recent_message_chars, 1, 1_000_000),
            ("max_relation_neighbors_per_match", self.max_relation_neighbors_per_match, 0, 1_000),
            ("max_relation_neighbors_total", self.max_relation_neighbors_total, 0, 10_000),
            ("max_context_chars", self.max_context_chars, 1_024, 4_000_000),
        ):
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        for name, seconds, lower, upper in (
            ("time_limit_seconds", self.time_limit_seconds, 1.0, 600.0),
            ("accounting_reserve_seconds", self.accounting_reserve_seconds, 0.05, 60.0),
        ):
            if not lower <= float(seconds) <= upper:
                raise ValueError(f"{name} must be between {lower:g} and {upper:g}")
        if self.default_limit > self.max_limit:
            raise ValueError("memory search default_limit cannot exceed max_limit")
        if self.accounting_reserve_seconds >= self.time_limit_seconds:
            raise ValueError("memory search accounting reserve must leave time for the search itself")
        if self.max_relation_neighbors_per_match > self.max_relation_neighbors_total:
            raise ValueError("per-match relation limit cannot exceed the total relation limit")


class MemorySearchDegradationStage(str, Enum):
    """某一步没做成、已经按退路继续的阶段。"""

    CONVERSATION_CONTEXT = "conversation_context"
    CATALOG = "catalog"
    PICK = "pick"
    JUDGE = "judge"
    RECORDS = "records"
    SUMMARIES = "summaries"
    RECALL_LIFECYCLE = "recall_lifecycle"
    TIME_LIMIT = "time_limit"


@dataclass(frozen=True)
class MemorySearchDegradation:
    """不包含敏感异常正文的检索降级事实。"""

    stage: MemorySearchDegradationStage
    error_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", MemorySearchDegradationStage(self.stage))
        if not self.error_type or len(self.error_type) > 256:
            raise ValueError("search degradation error_type must be non-empty bounded text")


@dataclass(frozen=True)
class MemoryRelatedMemory:
    """由一张卡片的持久关系确定性补充的一跳完整记忆。"""

    relation: MemoryStoredLink
    document: MemoryDocument

    def __post_init__(self) -> None:
        uri = MemoryURI.from_address(self.document.address)
        if uri not in {self.relation.from_uri, self.relation.to_uri}:
            raise ValueError("related memory document is not a relation endpoint")


@dataclass(frozen=True)
class MemoryMatchedMemory:
    """一张被挑中并用到的完整 L2 卡片及其有界一跳关系。"""

    document: MemoryDocument
    temperature: MemoryTemperature | None = None
    related: tuple[MemoryRelatedMemory, ...] = ()

    def __post_init__(self) -> None:
        identities = tuple(item.relation.identity for item in self.related)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
            raise ValueError("matched memory relations must be unique and sorted")

    @property
    def uri(self) -> MemoryURI:
        return MemoryURI.from_address(self.document.address)


@dataclass(frozen=True)
class MemorySearchResult:
    """可直接交给 Agent 的上下文与它的出处：卡片、每一轮的记录、会话摘要三个区。"""

    query: str
    kinds: tuple[MemoryKind, ...]
    memories: tuple[MemoryMatchedMemory, ...]
    records: tuple[BehaviorRecord, ...]
    summaries: tuple[ConversationSummaryMatch, ...]
    context: str
    budget_exhausted: bool
    rounds: int
    timed_out: bool
    notes: tuple[str, ...] = ()
    degradations: tuple[MemorySearchDegradation, ...] = ()

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("search result query must be non-empty")
        normalized_kinds = tuple(MemoryKind(kind) for kind in self.kinds)
        if len(normalized_kinds) != len(set(normalized_kinds)):
            raise ValueError("search result kinds must be unique")
        object.__setattr__(self, "kinds", normalized_kinds)
        stages = tuple(item.stage for item in self.degradations)
        if len(stages) != len(set(stages)):
            raise ValueError("search result degradation stages must be unique")
        if len({item.uri for item in self.memories}) != len(self.memories):
            raise ValueError("search result memories must be unique")
        if any(item.uri.node_type is not MemoryURINodeType.DOCUMENT for item in self.memories):
            raise ValueError("search result memories must be L2 documents")
        if bool(self.memories or self.records or self.summaries or self.notes) != bool(self.context):
            raise ValueError("search result context presence must match its content")

    @property
    def total(self) -> int:
        return len(self.memories)


__all__ = [
    "MemoryMatchedMemory",
    "MemoryRelatedMemory",
    "MemorySearchDegradation",
    "MemorySearchDegradationStage",
    "MemorySearchError",
    "MemorySearchResult",
    "MemorySearchServiceConfig",
]
