"""HTTP API 独立请求与响应 Schema。"""

from __future__ import annotations

from contextlib import suppress
from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from habitus.foundation.ids import canonical_path_identity
from habitus.memory.conversation.layout import MAX_CONVERSATION_ID_UTF8_BYTES
from habitus.memory.intention import MemoryIntentionRecallScope
from habitus.memory.model import MemoryKind

_ResultT = TypeVar("_ResultT")
PositiveSequence = Annotated[int, Field(strict=True, gt=0)]
NonNegativeSequence = Annotated[int, Field(strict=True, ge=0)]
PositiveLimit = Annotated[int, Field(strict=True, gt=0, le=1_000)]
PositiveTimeout = Annotated[float, Field(strict=True, gt=0, le=3_600)]
JobVersion = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
DeliveryId = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RetryAfterSeconds = Annotated[int, Field(strict=True, ge=0, le=3_600)]


def _conversation_id_within_limit(value: str) -> str:
    """会话号会成为会话目录名：按 UTF-8 字节卡上限，与领域 ``ConversationAddress`` 同一个常数、同一个量法——
    原文与归一身份（目录名，casefold 可能变长）取大的。"""

    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("conversation_id must be valid Unicode text") from exc
    with suppress(ValueError):  # 不能当路径段的会话号交给领域拒，报错与以前一样
        size = max(size, len(canonical_path_identity(value, "conversation_id").encode("utf-8")))
    if size > MAX_CONVERSATION_ID_UTF8_BYTES:
        raise ValueError(f"conversation_id must be at most {MAX_CONVERSATION_ID_UTF8_BYTES} UTF-8 bytes, got {size}")
    return value


ConversationIdText = Annotated[str, Field(min_length=1), AfterValidator(_conversation_id_within_limit)]


class HTTPErrorCode(str, Enum):
    """HTTP Adapter 对外稳定错误码；不得直接暴露内部异常类名。"""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    CONFLICT = "CONFLICT"
    ABORTED = "ABORTED"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    UNAVAILABLE = "UNAVAILABLE"
    INTERNAL = "INTERNAL"


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RememberRequest(StrictSchema):
    delivery_id: DeliveryId | None = None
    conversation_id: ConversationIdText
    started_on: date
    protocol: Annotated[str, Field(min_length=1, max_length=128)]
    payload: Any
    start_sequence: NonNegativeSequence
    occurred_at: datetime
    after_turn: StrictBool | None = None
    wait_timeout_seconds: PositiveTimeout | None = None

    @field_validator("conversation_id", "protocol")
    @classmethod
    def validate_normalized_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("value must be normalized text without surrounding whitespace")
        return value

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone offset")
        return value


class RememberPartRequest(StrictSchema):
    """单独一轮超过请求体上限时分块上传的一块：``data`` 是整批 remember 请求体字节的一段（标准 base64）。"""

    delivery_id: DeliveryId
    part_index: NonNegativeSequence
    part_count: Annotated[int, Field(strict=True, ge=2, le=100_000)]
    body_bytes: PositiveSequence
    body_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    data: Annotated[str, Field(min_length=4)]

    @model_validator(mode="after")
    def validate_part_index(self) -> RememberPartRequest:
        if self.part_index >= self.part_count:
            raise ValueError("part_index must be smaller than part_count")
        return self


class RecallRequest(StrictSchema):
    query: Annotated[str, Field(min_length=1)]
    conversation_id: ConversationIdText | None = None
    started_on: date | None = None
    limit: PositiveLimit | None = None
    kinds: tuple[MemoryKind, ...] = ()
    intention_scope: MemoryIntentionRecallScope = MemoryIntentionRecallScope.ACTIVE

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("conversation_id must not contain surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_conversation_identity(self) -> RecallRequest:
        if (self.conversation_id is None) != (self.started_on is None):
            raise ValueError("conversation_id and started_on must be provided together")
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("kinds must not contain duplicates")
        return self


class FlushRequest(StrictSchema):
    conversation_id: ConversationIdText
    started_on: date
    wait_timeout_seconds: PositiveTimeout | None = None

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("conversation_id must not contain surrounding whitespace")
        return value


class ReminderClaimRequest(StrictSchema):
    """宿主来取提醒（方案 32 第 4.1 条）。``session`` 与插件送转写时的 conversation_id 是同一个身份。"""

    host: Annotated[str, Field(pattern=r"^[a-z0-9](?:[a-z0-9-]{0,62})$")]
    session: Annotated[str, Field(min_length=1, max_length=256)]
    at: datetime
    limit: Annotated[int, Field(strict=True, ge=1, le=20)] = 3

    @field_validator("session")
    @classmethod
    def validate_session(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("session must not contain surrounding whitespace")
        return value

    @field_validator("at")
    @classmethod
    def validate_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("at must include a timezone offset")
        return value


class ReminderResult(StrictSchema):
    request_id: Annotated[str, Field(min_length=1)]
    kind: str
    text: str
    options: list[str]
    deadline: datetime
    lease_until: datetime
    claimed_at: datetime


class ReminderClaimResult(StrictSchema):
    reminders: list[ReminderResult]


class MemoryListResult(StrictSchema):
    """``habitus memory ls``：一个目录的直接内容，只给 ``memory://`` 地址。"""

    uri: str
    directories: list[str]
    documents: list[str]
    abstract: str | None


class MemoryLinkResult(StrictSchema):
    from_uri: str
    to_uri: str
    link_type: str


class MemoryReadResult(StrictSchema):
    """``habitus memory read``：文档给结构化字段与正文；目录的 L0 / L1 层只有 ``text``。"""

    uri: str
    node: Literal["document", "layer"]
    kind: str | None = None
    level: Literal["abstract", "overview"] | None = None
    revision: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_confirmed_at: datetime | None = None
    fields: dict[str, Any] | None = None
    text: str
    links: list[MemoryLinkResult] = []
    backlinks: list[MemoryLinkResult] = []


class ReminderHandoffResult(StrictSchema):
    host: str
    session: str
    claimed_at: datetime
    ended: str | None


class ReminderProgressResult(StrictSchema):
    request_id: str
    kind: str
    text: str
    deadline: datetime
    stage: Literal["待取", "取走了", "说了", "宿主没取", "已发生"]
    awaiting_at: datetime
    spoken_at: datetime | None
    spoken_via: str | None
    response: Literal["正", "中", "负", "已做"] | None
    responded_at: datetime | None
    outcome: str | None
    attempts: int | None
    handoffs: list[ReminderHandoffResult]


class ReminderDayResult(StrictSchema):
    """``habitus reminders``：某天进了提醒出口的每条提醒走到了哪一步，加上 R8 要看的几个数。"""

    day: date
    reminders: list[ReminderProgressResult]
    awaiting: int
    taken: int
    spoken: int
    never_claimed: int
    responded: int


class CapabilitiesResult(StrictSchema):
    api_version: Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+$")]
    service_version: Annotated[str, Field(min_length=1)]
    protocols: list[str]
    features: list[str]


class ServiceCapabilitiesResult(CapabilitiesResult):
    """``/capabilities`` 对外的那一份：服务能力加上 HTTP 层的请求体上限（插件据此把一批按轮拆成几个请求）。"""

    max_request_bytes: PositiveSequence


class RememberJobResult(StrictSchema):
    memory_sequence: PositiveSequence
    conversation_id: str
    started_on: date
    status: str


class ConsistencyResult(StrictSchema):
    memory_sequence: PositiveSequence
    state: str


class RememberResult(StrictSchema):
    ignored_items: NonNegativeSequence
    after_turn: bool
    next_sequence: NonNegativeSequence
    jobs: list[RememberJobResult]
    consistency: list[ConsistencyResult]


class RememberPartResult(StrictSchema):
    """收下一块之后：收齐前 ``remember`` 为空；收齐那一块带回拼回的整批按 ``/memory/remember`` 处理的结果。"""

    received_parts: PositiveSequence
    part_count: PositiveSequence
    remember: RememberResult | None


class FlushResult(StrictSchema):
    jobs: list[RememberJobResult]
    consistency: list[ConsistencyResult]


class ConversationCursorResult(StrictSchema):
    next_sequence: NonNegativeSequence


class RecallMemoryResult(StrictSchema):
    uri: str
    score: float
    matched_queries: list[str]


class RecallSummaryResult(StrictSchema):
    reference: str
    score: float


class RecallDegradationResult(StrictSchema):
    stage: str
    error_type: str


class RecallResult(StrictSchema):
    query: str
    queries: list[str]
    context: str
    memories: list[RecallMemoryResult]
    summaries: list[RecallSummaryResult]
    degradations: list[RecallDegradationResult]
    budget_exhausted: bool


class JobFailureResult(StrictSchema):
    message: str


class JobStatusResult(StrictSchema):
    memory_sequence: PositiveSequence
    conversation_id: str
    started_on: date
    state: str
    job_status: str
    terminal: bool
    attempts: NonNegativeSequence
    next_attempt_at: datetime | None
    last_failure: JobFailureResult | None
    blocking: bool
    manual_action_required: bool
    version: JobVersion
    created_at: datetime
    updated_at: datetime


class JobListResult(StrictSchema):
    jobs: list[JobStatusResult]
    next_before_sequence: PositiveSequence | None


class BlockedJobResult(StrictSchema):
    job: JobStatusResult | None


class RetryJobRequest(StrictSchema):
    expected_version: JobVersion


class RetryJobResult(StrictSchema):
    previous: JobStatusResult
    job: JobStatusResult
    worker_restarted: bool


class AuditEventResult(StrictSchema):
    audit_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    occurred_at: datetime
    category: str
    operation: str
    status: str
    request_id: str | None
    memory_sequence: PositiveSequence | None
    attributes: dict[str, str | int | float | bool]


class AuditListResult(StrictSchema):
    events: list[AuditEventResult]


class HealthCheckResult(StrictSchema):
    name: str
    status: str
    detail: str
    critical: bool


class HealthResult(StrictSchema):
    status: str
    ready: bool
    # 召回所需的就绪：不看只关记忆写入侧的检查（记忆 Job 失败待处置或在等模型恢复时召回照常可用）
    recall_ready: bool
    checked_at: datetime
    checks: list[HealthCheckResult]


class SuccessResponse(BaseModel, Generic[_ResultT]):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    result: _ResultT
    request_id: str


class ErrorInfo(StrictSchema):
    code: HTTPErrorCode
    message: str
    retryable: StrictBool
    retry_after_seconds: RetryAfterSeconds | None = None
    details: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_retry_advice(self) -> ErrorInfo:
        if self.retry_after_seconds is not None and not self.retryable:
            raise ValueError("retry_after_seconds requires retryable=true")
        return self


class ErrorResponse(StrictSchema):
    status: Literal["error"] = "error"
    error: ErrorInfo
    request_id: str


__all__ = [
    "ConversationIdText",
    "ErrorInfo",
    "ErrorResponse",
    "HTTPErrorCode",
    "BlockedJobResult",
    "AuditListResult",
    "ConversationCursorResult",
    "FlushRequest",
    "FlushResult",
    "HealthResult",
    "JobListResult",
    "JobStatusResult",
    "CapabilitiesResult",
    "RecallRequest",
    "RecallResult",
    "RememberPartRequest",
    "RememberPartResult",
    "RememberRequest",
    "RememberResult",
    "ReminderClaimRequest",
    "ReminderClaimResult",
    "ReminderResult",
    "RetryJobRequest",
    "RetryJobResult",
    "ServiceCapabilitiesResult",
    "SuccessResponse",
]
