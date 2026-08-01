"""把 m2bOS 边界异常转换为稳定、可观测且不泄密的 HTTP 契约。"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from infrastructure.store.contracts.lock import LockLostError
from infrastructure.vector import (
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreError,
    VectorStoreIntegrityError,
)
from integrations.http import HTTPMemoryJobConflictError, HTTPMemoryJobNotFoundError
from integrations.http_api.body_limit import RequestBodyLimitExceeded
from integrations.http_api.schemas import ErrorInfo, ErrorResponse, HTTPErrorCode
from memory.conversation import ConversationJournalError, ConversationWriteConflictError
from memory.retrieval import MemorySearchError
from memory.workflow import MemoryJobBlockedError, MemoryJobError, MemoryJobNotReadyError
from ModelClient import (
    ModelAuthenticationError,
    ModelClientError,
    ModelConfigurationError,
    ModelContentSafetyError,
    ModelInputTooLargeError,
    ModelPermissionError,
    ModelQuotaError,
    ModelRateLimitError,
    ModelTransportError,
)
from Runtime import (
    MemoryConsistencyTimeoutError,
    RuntimeInitializationError,
    RuntimeStateError,
)

logger = logging.getLogger(__name__)

ERROR_CODE_TO_HTTP_STATUS: Mapping[HTTPErrorCode, int] = MappingProxyType(
    {
        HTTPErrorCode.INVALID_ARGUMENT: 400,
        HTTPErrorCode.NOT_FOUND: 404,
        HTTPErrorCode.METHOD_NOT_ALLOWED: 405,
        HTTPErrorCode.CONFLICT: 409,
        HTTPErrorCode.ABORTED: 409,
        HTTPErrorCode.FAILED_PRECONDITION: 412,
        HTTPErrorCode.REQUEST_TOO_LARGE: 413,
        HTTPErrorCode.RESOURCE_EXHAUSTED: 429,
        HTTPErrorCode.DEADLINE_EXCEEDED: 504,
        HTTPErrorCode.UNAVAILABLE: 503,
        HTTPErrorCode.INTERNAL: 500,
    }
)


@dataclass(frozen=True)
class HTTPErrorSpec:
    """一次具体失败的公开语义；message/details 不得携带秘密或本地路径。"""

    code: HTTPErrorCode
    message: str
    retryable: bool
    details: dict[str, Any] | None = None
    retry_after_seconds: int | None = None

    @property
    def status_code(self) -> int:
        return ERROR_CODE_TO_HTTP_STATUS[self.code]


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) and value else "unavailable"


def _bounded_retry_after(value: object) -> int | None:
    if isinstance(value, str):
        if not value.isascii() or not value.isdigit():
            return None
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        return None
    return min(3_600, math.ceil(resolved))


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    result: list[BaseException] = []
    while pending:
        current = pending.pop(0)
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(current)
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException) and id(nested) not in seen:
                pending.append(nested)
    return tuple(result)


def _model_error_spec(exc: ModelClientError) -> HTTPErrorSpec:
    retry_after = _bounded_retry_after(exc.retry_after_seconds)
    if isinstance(exc, ModelRateLimitError):
        return HTTPErrorSpec(
            HTTPErrorCode.RESOURCE_EXHAUSTED,
            "Model provider rate limit exceeded",
            True,
            retry_after_seconds=retry_after,
        )
    if isinstance(exc, ModelQuotaError):
        return HTTPErrorSpec(
            HTTPErrorCode.RESOURCE_EXHAUSTED,
            "Model provider quota is exhausted",
            False,
        )
    if isinstance(exc, ModelTransportError):
        return HTTPErrorSpec(
            HTTPErrorCode.UNAVAILABLE,
            "Model provider is temporarily unavailable",
            True,
            retry_after_seconds=retry_after,
        )
    if isinstance(exc, ModelAuthenticationError | ModelPermissionError):
        return HTTPErrorSpec(
            HTTPErrorCode.FAILED_PRECONDITION,
            "Model provider credentials cannot authorize this operation",
            False,
        )
    if isinstance(exc, ModelConfigurationError):
        return HTTPErrorSpec(
            HTTPErrorCode.FAILED_PRECONDITION,
            "Model provider is not configured for this operation",
            False,
        )
    if isinstance(exc, ModelInputTooLargeError):
        return HTTPErrorSpec(
            HTTPErrorCode.FAILED_PRECONDITION,
            "Model input exceeds the configured provider capacity",
            False,
        )
    if isinstance(exc, ModelContentSafetyError):
        return HTTPErrorSpec(
            HTTPErrorCode.FAILED_PRECONDITION,
            "Model provider rejected the operation under its content policy",
            False,
        )
    if exc.retryable:
        return HTTPErrorSpec(
            HTTPErrorCode.UNAVAILABLE,
            "Model operation is temporarily unavailable",
            True,
            retry_after_seconds=retry_after,
        )
    return HTTPErrorSpec(HTTPErrorCode.INTERNAL, "Model operation failed", False)


def _vector_error_spec(exc: VectorStoreError) -> HTTPErrorSpec:
    if isinstance(exc, VectorStoreBusyError):
        return HTTPErrorSpec(
            HTTPErrorCode.UNAVAILABLE,
            "Vector store is temporarily busy",
            True,
            details={"conflict_type": "vector_store_busy"},
            retry_after_seconds=1,
        )
    if isinstance(exc, VectorStoreConflictError):
        return HTTPErrorSpec(
            HTTPErrorCode.ABORTED,
            "Vector store state changed during the operation",
            True,
            details={"conflict_type": "vector_store_state"},
        )
    if isinstance(exc, VectorStoreIntegrityError):
        return HTTPErrorSpec(HTTPErrorCode.INTERNAL, "Vector store integrity check failed", False)
    return HTTPErrorSpec(HTTPErrorCode.UNAVAILABLE, "Vector store operation failed", False)


def _wrapped_dependency_spec(exc: BaseException) -> HTTPErrorSpec | None:
    for item in _exception_chain(exc)[1:]:
        if isinstance(item, ModelClientError):
            return _model_error_spec(item)
        if isinstance(item, VectorStoreError):
            return _vector_error_spec(item)
        if isinstance(item, LockLostError):
            return HTTPErrorSpec(
                HTTPErrorCode.ABORTED,
                "Storage lock ownership was lost",
                True,
                details={"conflict_type": "lock_lost"},
                retry_after_seconds=1,
            )
        if isinstance(item, TimeoutError):
            return HTTPErrorSpec(
                HTTPErrorCode.UNAVAILABLE,
                "A required dependency timed out",
                True,
                retry_after_seconds=1,
            )
    return None


def map_exception(exc: Exception) -> HTTPErrorSpec:
    """只按 m2bOS 的真实异常类型映射，不根据错误文本猜测业务语义。"""

    if isinstance(exc, HTTPMemoryJobNotFoundError):
        return HTTPErrorSpec(HTTPErrorCode.NOT_FOUND, str(exc), False)
    if isinstance(exc, HTTPMemoryJobConflictError):
        return HTTPErrorSpec(HTTPErrorCode.FAILED_PRECONDITION, str(exc), False)
    if isinstance(exc, RequestBodyLimitExceeded):
        return HTTPErrorSpec(HTTPErrorCode.REQUEST_TOO_LARGE, "Request body exceeds the configured limit", False)
    if isinstance(exc, MemoryConsistencyTimeoutError):
        return HTTPErrorSpec(
            HTTPErrorCode.DEADLINE_EXCEEDED,
            "Memory consistency wait exceeded its deadline",
            True,
            details={
                "state": exc.snapshot.state.value,
                "memory_sequence": exc.snapshot.requested_job.memory_sequence,
            },
        )
    if isinstance(exc, ConversationWriteConflictError):
        return HTTPErrorSpec(
            HTTPErrorCode.CONFLICT,
            str(exc),
            False,
            details={"conflict_type": "conversation_write"},
        )
    if isinstance(exc, ConversationJournalError):
        return HTTPErrorSpec(HTTPErrorCode.INTERNAL, "Conversation journal integrity check failed", False)
    if isinstance(exc, RuntimeStateError):
        return HTTPErrorSpec(HTTPErrorCode.UNAVAILABLE, "Runtime is not ready for this operation", True)
    if isinstance(exc, RuntimeInitializationError):
        return HTTPErrorSpec(HTTPErrorCode.UNAVAILABLE, "Runtime initialization failed", False)
    if isinstance(exc, MemoryJobNotReadyError):
        return HTTPErrorSpec(
            HTTPErrorCode.UNAVAILABLE,
            "Memory job is still in its retry backoff window",
            True,
            retry_after_seconds=1,
        )
    if isinstance(exc, MemoryJobBlockedError):
        return HTTPErrorSpec(HTTPErrorCode.FAILED_PRECONDITION, "Memory job cannot advance", False)
    if isinstance(exc, MemoryJobError):
        return HTTPErrorSpec(HTTPErrorCode.INTERNAL, "Memory job state is inconsistent", False)
    if isinstance(exc, MemorySearchError):
        return _wrapped_dependency_spec(exc) or HTTPErrorSpec(
            HTTPErrorCode.INTERNAL,
            "Memory search failed its integrity checks",
            False,
        )
    if isinstance(exc, ModelClientError):
        return _model_error_spec(exc)
    if isinstance(exc, VectorStoreError):
        return _vector_error_spec(exc)
    if isinstance(exc, LockLostError):
        return HTTPErrorSpec(
            HTTPErrorCode.ABORTED,
            "Storage lock ownership was lost",
            True,
            details={"conflict_type": "lock_lost"},
            retry_after_seconds=1,
        )
    if isinstance(exc, TimeoutError):
        return HTTPErrorSpec(
            HTTPErrorCode.UNAVAILABLE,
            "Storage operation timed out",
            True,
            details={"conflict_type": "path_busy"},
            retry_after_seconds=1,
        )
    if isinstance(exc, ValueError):
        return HTTPErrorSpec(HTTPErrorCode.INVALID_ARGUMENT, str(exc), False)
    return HTTPErrorSpec(HTTPErrorCode.INTERNAL, "Internal server error", False)


def error_response(
    request: Request,
    *,
    code: HTTPErrorCode,
    message: str,
    retryable: bool,
    details: dict[str, Any] | None = None,
    retry_after_seconds: int | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    resolved_retry_after = _bounded_retry_after(retry_after_seconds)
    payload = ErrorResponse(
        error=ErrorInfo(
            code=code,
            message=message,
            retryable=retryable,
            retry_after_seconds=resolved_retry_after,
            details=details,
        ),
        request_id=_request_id(request),
    )
    response_headers = dict(headers or {})
    if resolved_retry_after is not None:
        response_headers["Retry-After"] = str(resolved_retry_after)
    return JSONResponse(
        status_code=ERROR_CODE_TO_HTTP_STATUS[code],
        content=payload.model_dump(mode="json", exclude_none=True),
        headers=response_headers or None,
    )


def documented_error_responses(
    *,
    exclude_statuses: frozenset[int] = frozenset(),
) -> dict[int | str, dict[str, Any]]:
    """生成 OpenAPI 共用错误响应，确保客户端能发现错误体和关联头。"""

    codes_by_status: dict[int, list[str]] = {}
    for code, status_code in ERROR_CODE_TO_HTTP_STATUS.items():
        codes_by_status.setdefault(status_code, []).append(code.value)
    return {
        status_code: {
            "model": ErrorResponse,
            "description": f"m2bOS error: {', '.join(sorted(codes))}",
            "headers": {
                "X-Request-ID": {
                    "description": "Request correlation identity",
                    "schema": {"type": "string"},
                },
                "Retry-After": {
                    "description": "Delay in seconds when the error is safely retryable",
                    "schema": {"type": "integer", "minimum": 0, "maximum": 3_600},
                },
            },
        }
        for status_code, codes in sorted(codes_by_status.items())
        if status_code not in exclude_statuses
    }


def _response_from_spec(
    request: Request,
    spec: HTTPErrorSpec,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return error_response(
        request,
        code=spec.code,
        message=spec.message,
        retryable=spec.retryable,
        details=spec.details,
        retry_after_seconds=spec.retry_after_seconds,
        headers=headers,
    )


def _framework_error_spec(status_code: int, message: str) -> HTTPErrorSpec:
    mapping = {
        400: (HTTPErrorCode.INVALID_ARGUMENT, False),
        404: (HTTPErrorCode.NOT_FOUND, False),
        405: (HTTPErrorCode.METHOD_NOT_ALLOWED, False),
        409: (HTTPErrorCode.CONFLICT, False),
        412: (HTTPErrorCode.FAILED_PRECONDITION, False),
        413: (HTTPErrorCode.REQUEST_TOO_LARGE, False),
        422: (HTTPErrorCode.INVALID_ARGUMENT, False),
        429: (HTTPErrorCode.RESOURCE_EXHAUSTED, True),
        502: (HTTPErrorCode.UNAVAILABLE, True),
        503: (HTTPErrorCode.UNAVAILABLE, True),
        504: (HTTPErrorCode.DEADLINE_EXCEEDED, True),
    }
    code, retryable = mapping.get(
        status_code,
        (HTTPErrorCode.INTERNAL, False) if status_code >= 500 else (HTTPErrorCode.INVALID_ARGUMENT, False),
    )
    public_message = "Internal server error" if code is HTTPErrorCode.INTERNAL else message
    return HTTPErrorSpec(code, public_message, retryable)


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "location": [str(part) for part in error.get("loc", ())],
                "message": str(error.get("msg", "Invalid value")),
                "type": str(error.get("type", "value_error")),
            }
            for error in exc.errors()
        ]
        return error_response(
            request,
            code=HTTPErrorCode.INVALID_ARGUMENT,
            message="Request validation failed",
            retryable=False,
            details={"validation_errors": errors},
        )

    @app.exception_handler(ResponseValidationError)
    async def response_validation(request: Request, exc: ResponseValidationError) -> JSONResponse:
        logger.error(
            "HTTP response schema validation failed",
            extra={"request_id": _request_id(request), "error_code": HTTPErrorCode.INTERNAL.value},
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return error_response(
            request,
            code=HTTPErrorCode.INTERNAL,
            message="Internal server error",
            retryable=False,
        )

    @app.exception_handler(StarletteHTTPException)
    async def framework_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "HTTP request failed"
        spec = _framework_error_spec(exc.status_code, message)
        retry_after = None
        if exc.headers is not None:
            retry_after = _bounded_retry_after(exc.headers.get("Retry-After"))
        if retry_after is not None:
            spec = replace(spec, retry_after_seconds=retry_after)
        return _response_from_spec(
            request,
            spec,
            headers=None if exc.headers is None else dict(exc.headers),
        )


async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """供显式 ExceptionMiddleware 使用，确保通用 500 仍经过请求 ID 层。"""

    spec = map_exception(exc)
    log_fields = {
        "request_id": _request_id(request),
        "error_code": spec.code.value,
        "http_status_code": spec.status_code,
        "retryable": spec.retryable,
    }
    if spec.code is HTTPErrorCode.INTERNAL:
        logger.error(
            "Unhandled HTTP API error",
            extra=log_fields,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    else:
        logger.info("Mapped HTTP API error", extra=log_fields)
    return _response_from_spec(request, spec)


__all__ = [
    "ERROR_CODE_TO_HTTP_STATUS",
    "HTTPErrorSpec",
    "documented_error_responses",
    "error_response",
    "install_exception_handlers",
    "map_exception",
    "unhandled_error",
]
