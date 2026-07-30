"""HTTP API 独立请求与响应 Schema。"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryKind

_ResultT = TypeVar("_ResultT")
PositiveSequence = Annotated[int, Field(strict=True, gt=0)]
NonNegativeSequence = Annotated[int, Field(strict=True, ge=0)]
PositiveLimit = Annotated[int, Field(strict=True, gt=0, le=1_000)]
PositiveTimeout = Annotated[float, Field(strict=True, gt=0, le=3_600)]
JobVersion = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RetryAfterSeconds = Annotated[int, Field(strict=True, ge=0, le=3_600)]


class HTTPErrorCode(str, Enum):
    """HTTP Adapter 对外稳定错误码；不得直接暴露内部异常类名。"""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
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
    conversation_id: Annotated[str, Field(min_length=1, max_length=256)]
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


class RecallRequest(StrictSchema):
    query: Annotated[str, Field(min_length=1)]
    conversation_id: Annotated[str, Field(min_length=1, max_length=256)] | None = None
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


class ProtocolsResult(StrictSchema):
    protocols: list[str]


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
    jobs: list[RememberJobResult]
    consistency: list[ConsistencyResult]


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
    "ErrorInfo",
    "ErrorResponse",
    "HTTPErrorCode",
    "BlockedJobResult",
    "AuditListResult",
    "HealthResult",
    "JobListResult",
    "JobStatusResult",
    "ProtocolsResult",
    "RecallRequest",
    "RecallResult",
    "RememberRequest",
    "RememberResult",
    "RetryJobRequest",
    "RetryJobResult",
    "SuccessResponse",
]
