"""只把现有 RuntimeHTTPHandlers 映射为最小 FastAPI 契约。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, cast

from fastapi import APIRouter, FastAPI, Path, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.exceptions import ExceptionMiddleware

from habitus.config import HTTPAPIConfig
from habitus.foundation.observability import (
    NullObserver,
    ObservationEvent,
    ObservationStatus,
    Observer,
    bind_observation_context,
)
from habitus.integrations.http import RuntimeHTTPHandlers
from habitus.integrations.http_api.body_limit import RequestBodyLimitMiddleware
from habitus.integrations.http_api.errors import (
    documented_error_responses,
    error_response,
    install_exception_handlers,
    unhandled_error,
)
from habitus.integrations.http_api.local_security import local_request_violation
from habitus.integrations.http_api.observation import HTTPObservationMiddleware
from habitus.integrations.http_api.request_id import REQUEST_ID_HEADER, RequestIDMiddleware
from habitus.integrations.http_api.schemas import (
    AuditListResult,
    BlockedJobResult,
    CapabilitiesResult,
    ConversationCursorResult,
    FlushRequest,
    FlushResult,
    HealthResult,
    HTTPErrorCode,
    JobListResult,
    JobStatusResult,
    RecallRequest,
    RecallResult,
    RememberJobResult,
    RememberRequest,
    RememberResult,
    RetryJobRequest,
    RetryJobResult,
    SuccessResponse,
)
from habitus.integrations.local_service import ServiceInstanceLock
from habitus.memory.workflow import MemoryJobStatus
from habitus.runtime import Runtime


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError("request ID middleware did not initialize request state")
    return value


def _success(request: Request, result: object) -> SuccessResponse[object]:
    return SuccessResponse(result=result, request_id=_request_id(request))


def _remember_result(payload: dict[str, object]) -> RememberResult:
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise RuntimeError("remember handler returned invalid jobs")
    if any(not isinstance(item, dict) for item in raw_jobs):
        raise RuntimeError("remember handler returned an invalid job entry")
    public_jobs = [
        RememberJobResult.model_validate(
            {
                "memory_sequence": item.get("memory_sequence"),
                "conversation_id": item.get("conversation_id"),
                "started_on": item.get("started_on"),
                "status": item.get("status"),
            }
        )
        for item in raw_jobs
        if isinstance(item, dict)
    ]
    return RememberResult.model_validate(
        {
            "ignored_items": payload.get("ignored_items"),
            "after_turn": payload.get("after_turn"),
            "next_sequence": payload.get("next_sequence"),
            "jobs": public_jobs,
            "consistency": payload.get("consistency"),
        }
    )


def _flush_result(payload: dict[str, object]) -> FlushResult:
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or any(not isinstance(item, dict) for item in raw_jobs):
        raise RuntimeError("flush handler returned invalid jobs")
    return FlushResult.model_validate(
        {
            "jobs": [RememberJobResult.model_validate(item) for item in raw_jobs],
            "consistency": payload.get("consistency"),
        }
    )


def _build_public_router(
    handlers: RuntimeHTTPHandlers,
    observer: Observer,
) -> APIRouter:
    router = APIRouter(
        prefix="/api/v1",
        responses=documented_error_responses(),
    )

    @router.get("/capabilities", response_model=SuccessResponse[CapabilitiesResult])
    async def capabilities(request: Request) -> SuccessResponse[object]:
        result = CapabilitiesResult.model_validate(handlers.capabilities())
        return _success(request, result)

    @router.post("/memory/remember", response_model=SuccessResponse[RememberResult])
    async def remember(
        request: Request,
        body: RememberRequest,
    ) -> SuccessResponse[object]:
        result = await handlers.remember(
            conversation_id=body.conversation_id,
            started_on=body.started_on,
            protocol=body.protocol,
            payload=body.payload,
            start_sequence=body.start_sequence,
            occurred_at=body.occurred_at,
            after_turn=body.after_turn,
            wait_timeout_seconds=body.wait_timeout_seconds,
            delivery_id=body.delivery_id,
        )
        raw_jobs = result.get("jobs")
        if isinstance(raw_jobs, list):
            for item in raw_jobs:
                if not isinstance(item, dict):
                    continue
                sequence = item.get("memory_sequence")
                transaction_id = item.get("transaction_id")
                if not isinstance(sequence, int) or not isinstance(transaction_id, str):
                    continue
                with bind_observation_context(
                    memory_sequence=sequence,
                    transaction_id=transaction_id,
                ):
                    observer.record(
                        ObservationEvent(
                            category="http",
                            operation="job_accepted",
                            status=ObservationStatus.SUCCESS,
                            duration_seconds=0.0,
                            attributes={"job_status": str(item.get("status", "unknown"))},
                        )
                    )
        return _success(request, _remember_result(result))

    @router.get(
        "/memory/conversations/cursor",
        response_model=SuccessResponse[ConversationCursorResult],
    )
    async def conversation_cursor(
        request: Request,
        conversation_id: Annotated[str, Query(min_length=1, max_length=256)],
        started_on: Annotated[date, Query()],
    ) -> SuccessResponse[object]:
        if conversation_id != conversation_id.strip():
            raise ValueError("conversation_id must not contain surrounding whitespace")
        result = await handlers.conversation_cursor(
            conversation_id=conversation_id,
            started_on=started_on,
        )
        return _success(request, ConversationCursorResult.model_validate(result))

    @router.post("/memory/flush", response_model=SuccessResponse[FlushResult])
    async def flush(
        request: Request,
        body: FlushRequest,
    ) -> SuccessResponse[object]:
        result = await handlers.flush(
            conversation_id=body.conversation_id,
            started_on=body.started_on,
            wait_timeout_seconds=body.wait_timeout_seconds,
        )
        return _success(request, _flush_result(result))

    @router.post("/memory/recall", response_model=SuccessResponse[RecallResult])
    async def recall(
        request: Request,
        body: RecallRequest,
    ) -> SuccessResponse[object]:
        result = await handlers.recall(
            body.query,
            conversation_id=body.conversation_id,
            started_on=body.started_on,
            limit=body.limit,
            kinds=body.kinds,
            intention_scope=body.intention_scope,
        )
        return _success(request, RecallResult.model_validate(result))

    @router.get("/memory/jobs", response_model=SuccessResponse[JobListResult])
    async def list_jobs(
        request: Request,
        conversation_id: Annotated[str, Query(min_length=1, max_length=256)],
        started_on: Annotated[date, Query()],
        status: Annotated[MemoryJobStatus | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        before_sequence: Annotated[int | None, Query(gt=0)] = None,
    ) -> SuccessResponse[object]:
        if conversation_id != conversation_id.strip():
            raise ValueError("conversation_id must not contain surrounding whitespace")
        result = await handlers.list_jobs(
            conversation_id=conversation_id,
            started_on=started_on,
            status=status,
            limit=limit,
            before_sequence=before_sequence,
        )
        return _success(request, JobListResult.model_validate(result))

    @router.get(
        "/memory/jobs/blocked",
        response_model=SuccessResponse[BlockedJobResult],
    )
    async def blocked_job(request: Request) -> SuccessResponse[object]:
        result = await handlers.blocked_job()
        return _success(request, BlockedJobResult.model_validate(result))

    @router.get("/memory/jobs/{memory_sequence}", response_model=SuccessResponse[JobStatusResult])
    async def job_status(
        request: Request,
        memory_sequence: Annotated[int, Path(gt=0)],
        conversation_id: Annotated[str, Query(min_length=1, max_length=256)],
        started_on: Annotated[date, Query()],
    ) -> SuccessResponse[object]:
        if conversation_id != conversation_id.strip():
            raise ValueError("conversation_id must not contain surrounding whitespace")
        result = await handlers.job_status(
            memory_sequence,
            conversation_id=conversation_id,
            started_on=started_on,
        )
        return _success(request, JobStatusResult.model_validate(result))

    return router


def _build_operations_router(
    handlers: RuntimeHTTPHandlers,
) -> APIRouter:
    router = APIRouter(
        prefix="/api/v1/operations",
        tags=["operations"],
        responses=documented_error_responses(),
    )

    @router.post(
        "/memory/jobs/{memory_sequence}/retry",
        response_model=SuccessResponse[RetryJobResult],
        status_code=202,
    )
    async def retry_failed_job(
        request: Request,
        body: RetryJobRequest,
        memory_sequence: Annotated[int, Path(gt=0)],
        conversation_id: Annotated[str, Query(min_length=1, max_length=256)],
        started_on: Annotated[date, Query()],
    ) -> SuccessResponse[object]:
        if conversation_id != conversation_id.strip():
            raise ValueError("conversation_id must not contain surrounding whitespace")
        result = await handlers.retry_failed_job(
            memory_sequence,
            conversation_id=conversation_id,
            started_on=started_on,
            expected_version=body.expected_version,
        )
        return _success(request, RetryJobResult.model_validate(result))

    @router.get(
        "/audit",
        response_model=SuccessResponse[AuditListResult],
    )
    async def recent_audit_events(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> SuccessResponse[object]:
        result = await handlers.recent_audit_events(limit=limit)
        return _success(request, AuditListResult.model_validate(result))

    return router


def create_http_app(
    runtime: Runtime,
    *,
    config: HTTPAPIConfig | None = None,
    instance_lock: ServiceInstanceLock | None = None,
) -> FastAPI:
    """创建拥有 Runtime 生命周期、但不复制任何记忆业务逻辑的应用。"""

    if not isinstance(runtime, Runtime):
        raise TypeError("runtime must be Runtime")
    resolved_config = config or HTTPAPIConfig()
    if not isinstance(resolved_config, HTTPAPIConfig):
        raise TypeError("config must be HTTPAPIConfig or None")
    if instance_lock is not None and not isinstance(instance_lock, ServiceInstanceLock):
        raise TypeError("instance_lock must be ServiceInstanceLock or None")
    handlers = RuntimeHTTPHandlers(runtime)
    components = getattr(runtime, "components", None)
    infrastructure = getattr(components, "infrastructure", None)
    candidate_observer = getattr(infrastructure, "observer", None)
    observer: Observer = (
        cast(Observer, candidate_observer)
        if callable(getattr(candidate_observer, "record", None))
        else NullObserver()
    )
    candidate_span_controller = getattr(infrastructure, "managed_observability", None)
    span_controller = (
        candidate_span_controller
        if callable(getattr(candidate_span_controller, "start_span", None))
        else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if instance_lock is not None:
            instance_lock.acquire()
        try:
            await runtime.start()
            yield
        finally:
            try:
                await runtime.close()
            finally:
                if instance_lock is not None:
                    instance_lock.release()

    app = FastAPI(
        title="Habitus Memory API",
        description="Habitus Runtime 的单用户本地记忆服务",
        version="1.0.0",
        lifespan=lifespan,
    )
    install_exception_handlers(app)

    @app.middleware("http")
    async def request_boundary(request: Request, call_next):  # noqa: ANN001, ANN202
        violation = local_request_violation(
            host_header=request.headers.get("host"),
            origin_header=request.headers.get("origin"),
        )
        if violation is not None:
            response = error_response(
                request,
                code=HTTPErrorCode.INVALID_ARGUMENT,
                message=violation,
                retryable=False,
            )
        else:
            content_length = request.headers.get("content-length")
            if not content_length:
                return await call_next(request)
            try:
                declared_size = int(content_length)
            except ValueError:
                response = error_response(
                    request,
                    code=HTTPErrorCode.INVALID_ARGUMENT,
                    message="Content-Length must be an integer",
                    retryable=False,
                )
            else:
                if declared_size < 0:
                    response = error_response(
                        request,
                        code=HTTPErrorCode.INVALID_ARGUMENT,
                        message="Content-Length must not be negative",
                        retryable=False,
                    )
                elif declared_size > resolved_config.max_request_bytes:
                    response = error_response(
                        request,
                        code=HTTPErrorCode.REQUEST_TOO_LARGE,
                        message=(f"Request body exceeds the {resolved_config.max_request_bytes}-byte limit"),
                        retryable=False,
                    )
                else:
                    response = await call_next(request)
        return response

    # 请求身份位于最外层；观测包围异常渲染，从而记录最终 HTTP 状态。
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=resolved_config.max_request_bytes,
    )
    app.add_middleware(ExceptionMiddleware, handlers={Exception: unhandled_error})
    app.add_middleware(
        HTTPObservationMiddleware,
        observer=observer,
        span_controller=span_controller,
    )
    app.add_middleware(RequestIDMiddleware)

    app.include_router(_build_public_router(handlers, observer))
    app.include_router(_build_operations_router(handlers))

    @app.get(
        "/health",
        response_model=SuccessResponse[HealthResult],
        responses=documented_error_responses(),
        tags=["system"],
    )
    async def health(request: Request) -> SuccessResponse[object]:
        return _success(request, HealthResult.model_validate(await handlers.health(deep=False)))

    readiness_responses = documented_error_responses(exclude_statuses=frozenset({503}))
    readiness_responses[503] = {
        "model": SuccessResponse[HealthResult],
        "description": "Runtime is not ready; result contains the current health snapshot",
        "headers": {
            REQUEST_ID_HEADER: {
                "description": "Request correlation identity",
                "schema": {"type": "string"},
            }
        },
    }

    @app.get(
        "/ready",
        response_model=SuccessResponse[HealthResult],
        responses=readiness_responses,
        tags=["system"],
    )
    async def readiness(request: Request):  # noqa: ANN202
        status_code, payload = await handlers.readiness()
        response = SuccessResponse(
            result=HealthResult.model_validate(payload),
            request_id=_request_id(request),
        )
        return JSONResponse(status_code=status_code, content=response.model_dump(mode="json"))

    @app.get(
        "/metrics",
        responses=documented_error_responses(),
        tags=["system"],
    )
    async def metrics() -> PlainTextResponse:
        content, media_type = await handlers.metrics()
        return PlainTextResponse(content, media_type=media_type)

    return app


__all__ = ["REQUEST_ID_HEADER", "create_http_app"]
