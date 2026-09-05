"""HTTP 错误码、状态映射、异常分类与重试建议契约。"""

from types import SimpleNamespace
from typing import cast

from habitus.infrastructure.store.contracts.lock import LockLostError
from habitus.integrations.http import HTTPMemoryJobConflictError, HTTPMemoryJobNotFoundError
from habitus.integrations.http_api.app import create_http_app
from habitus.integrations.http_api.errors import (
    ERROR_CODE_TO_HTTP_STATUS,
    documented_error_responses,
    map_exception,
)
from habitus.integrations.http_api.schemas import ErrorResponse, HTTPErrorCode
from habitus.memory.conversation import ConversationJournalError, ConversationWriteConflictError
from habitus.memory.retrieval import MemorySearchError
from habitus.model_client import ModelRateLimitError, ModelTransportError
from habitus.runtime import MemoryConsistencySnapshot, MemoryConsistencyTimeoutError, Runtime, RuntimeStateError


def test_every_public_error_code_has_one_central_http_status() -> None:
    assert set(ERROR_CODE_TO_HTTP_STATUS) == set(HTTPErrorCode)
    assert ERROR_CODE_TO_HTTP_STATUS[HTTPErrorCode.CONFLICT] == 409
    assert ERROR_CODE_TO_HTTP_STATUS[HTTPErrorCode.FAILED_PRECONDITION] == 412
    assert ERROR_CODE_TO_HTTP_STATUS[HTTPErrorCode.RESOURCE_EXHAUSTED] == 429
    assert ERROR_CODE_TO_HTTP_STATUS[HTTPErrorCode.UNAVAILABLE] == 503


def test_openapi_error_responses_publish_the_shared_schema_codes_and_headers() -> None:
    responses = documented_error_responses()
    without_unavailable = documented_error_responses(exclude_statuses=frozenset({503}))

    assert responses[409]["model"] is ErrorResponse
    assert responses[409]["description"] == "Habitus error: ABORTED, CONFLICT"
    assert set(responses[409]["headers"]) == {"Retry-After", "X-Request-ID"}
    assert 401 not in responses
    assert 403 not in responses
    assert 503 not in without_unavailable


def test_actual_openapi_uses_error_schema_but_keeps_readiness_503_as_health_snapshot() -> None:
    runtime = object.__new__(Runtime)
    schema = create_http_app(runtime).openapi()

    remember_responses = schema["paths"]["/api/v1/memory/remember"]["post"]["responses"]
    readiness_responses = schema["paths"]["/ready"]["get"]["responses"]
    assert remember_responses["409"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorResponse"
    }
    assert remember_responses["503"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorResponse"
    }
    assert readiness_responses["503"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SuccessResponse_HealthResult_"
    }


def test_public_job_errors_keep_not_found_and_precondition_semantics() -> None:
    missing = map_exception(HTTPMemoryJobNotFoundError("job is outside its retention window"))
    changed = map_exception(HTTPMemoryJobConflictError("job version changed"))

    assert (missing.code, missing.status_code, missing.retryable) == (
        HTTPErrorCode.NOT_FOUND,
        404,
        False,
    )
    assert (changed.code, changed.status_code, changed.retryable) == (
        HTTPErrorCode.FAILED_PRECONDITION,
        412,
        False,
    )


def test_conversation_conflict_is_not_misreported_as_invalid_argument() -> None:
    spec = map_exception(ConversationWriteConflictError("append would create a gap"))

    assert spec.code is HTTPErrorCode.CONFLICT
    assert spec.status_code == 409
    assert spec.retryable is False
    assert spec.details == {"conflict_type": "conversation_write"}


def test_journal_integrity_error_does_not_leak_internal_message() -> None:
    spec = map_exception(ConversationJournalError("secret path /private/data is corrupt"))

    assert spec.code is HTTPErrorCode.INTERNAL
    assert spec.message == "Conversation journal integrity check failed"
    assert "/private/data" not in spec.message
    assert spec.retryable is False


def test_consistency_timeout_is_retryable_and_keeps_safe_job_identity() -> None:
    snapshot = cast(
        MemoryConsistencySnapshot,
        SimpleNamespace(
            state=SimpleNamespace(value="pending"),
            requested_job=SimpleNamespace(memory_sequence=17),
        ),
    )
    spec = map_exception(MemoryConsistencyTimeoutError(snapshot))

    assert spec.code is HTTPErrorCode.DEADLINE_EXCEEDED
    assert spec.status_code == 504
    assert spec.retryable is True
    assert spec.details == {"state": "pending", "memory_sequence": 17}


def test_lock_loss_and_runtime_unavailability_have_distinct_retry_semantics() -> None:
    lock = map_exception(LockLostError("lease expired at /private/path"))
    runtime = map_exception(RuntimeStateError("runtime has not started"))

    assert (lock.code, lock.status_code, lock.retryable, lock.retry_after_seconds) == (
        HTTPErrorCode.ABORTED,
        409,
        True,
        1,
    )
    assert (runtime.code, runtime.status_code, runtime.retryable) == (
        HTTPErrorCode.UNAVAILABLE,
        503,
        True,
    )
    assert "/private/path" not in lock.message


def test_model_rate_limit_preserves_bounded_retry_after_without_provider_details() -> None:
    spec = map_exception(
        ModelRateLimitError(
            "provider key sk-secret was rate limited",
            retry_after_seconds=1.2,
        )
    )

    assert spec.code is HTTPErrorCode.RESOURCE_EXHAUSTED
    assert spec.status_code == 429
    assert spec.retryable is True
    assert spec.retry_after_seconds == 2
    assert "sk-secret" not in spec.message


def test_memory_search_uses_typed_dependency_cause_instead_of_text_guessing() -> None:
    try:
        raise MemorySearchError("semantic search failed") from ModelTransportError(
            "upstream timeout with local details",
            retry_after_seconds=0.1,
        )
    except MemorySearchError as exc:
        spec = map_exception(exc)

    assert spec.code is HTTPErrorCode.UNAVAILABLE
    assert spec.status_code == 503
    assert spec.retryable is True
    assert spec.retry_after_seconds == 1
    assert spec.message == "Model provider is temporarily unavailable"


def test_unknown_exception_is_safe_non_retryable_internal_error() -> None:
    spec = map_exception(RuntimeError("Bearer secret-token at /private/path"))

    assert (spec.code, spec.status_code, spec.retryable) == (
        HTTPErrorCode.INTERNAL,
        500,
        False,
    )
    assert spec.message == "Internal server error"
