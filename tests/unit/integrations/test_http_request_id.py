"""请求 ID 在成功、验证失败、领域失败和未知 500 中保持一致。"""

import re

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.exceptions import ExceptionMiddleware

from integrations.http_api.errors import install_exception_handlers, unhandled_error
from integrations.http_api.request_id import REQUEST_ID_HEADER, RequestIDMiddleware, current_request_id
from memory.conversation import ConversationWriteConflictError
from ModelClient import ModelRateLimitError

_GENERATED_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")


def _app() -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(ExceptionMiddleware, handlers={Exception: unhandled_error})
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ok")
    async def ok():  # noqa: ANN202
        return {"request_id": current_request_id()}

    @app.get("/items/{item_id}")
    async def item(item_id: int):  # noqa: ANN202
        return {"item_id": item_id}

    @app.get("/conflict")
    async def conflict():  # noqa: ANN202
        raise ConversationWriteConflictError("append would create a gap")

    @app.get("/boom")
    async def boom():  # noqa: ANN202
        raise RuntimeError("sensitive internal failure")

    @app.get("/rate-limit")
    async def rate_limit():  # noqa: ANN202
        raise ModelRateLimitError("provider detail", retry_after_seconds=1.2)

    return app


def test_supplied_request_id_is_returned_and_available_inside_request_context() -> None:
    with TestClient(_app(), base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        response = client.get("/ok", headers={REQUEST_ID_HEADER: "agent-turn-42"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "agent-turn-42"
    assert response.json() == {"request_id": "agent-turn-42"}
    assert current_request_id() is None


def test_missing_request_id_generates_one_and_invalid_or_duplicate_values_are_rejected() -> None:
    with TestClient(_app(), base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        generated = client.get("/ok")
        invalid = client.get("/ok", headers={REQUEST_ID_HEADER: "contains spaces"})
        duplicate = client.get(
            "/ok",
            headers=[(REQUEST_ID_HEADER, "first"), (REQUEST_ID_HEADER, "second")],
        )

    assert _GENERATED_REQUEST_ID.fullmatch(generated.headers[REQUEST_ID_HEADER])
    for response in (invalid, duplicate):
        body = response.json()
        assert response.status_code == 400
        assert body["error"]["code"] == "INVALID_ARGUMENT"
        assert body["error"]["retryable"] is False
        assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
        assert _GENERATED_REQUEST_ID.fullmatch(body["request_id"])


def test_validation_framework_and_domain_errors_use_the_stable_envelope() -> None:
    with TestClient(_app(), base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        validation = client.get("/items/not-an-integer", headers={REQUEST_ID_HEADER: "validation-1"})
        method = client.post("/ok", headers={REQUEST_ID_HEADER: "method-1"})
        conflict = client.get("/conflict", headers={REQUEST_ID_HEADER: "conflict-1"})

    assert validation.status_code == 400
    assert validation.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert validation.json()["request_id"] == "validation-1"
    assert method.status_code == 405
    assert method.json()["error"]["code"] == "METHOD_NOT_ALLOWED"
    assert method.json()["request_id"] == "method-1"
    assert conflict.status_code == 409
    assert conflict.json()["error"] == {
        "code": "CONFLICT",
        "message": "append would create a gap",
        "retryable": False,
        "details": {"conflict_type": "conversation_write"},
    }
    assert conflict.json()["request_id"] == "conflict-1"


def test_unhandled_500_keeps_request_id_and_hides_internal_message() -> None:
    with TestClient(_app(), base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        response = client.get("/boom", headers={REQUEST_ID_HEADER: "failure-1"})

    assert response.status_code == 500
    assert response.headers[REQUEST_ID_HEADER] == "failure-1"
    assert response.json() == {
        "status": "error",
        "error": {
            "code": "INTERNAL",
            "message": "Internal server error",
            "retryable": False,
        },
        "request_id": "failure-1",
    }


def test_retryable_failure_returns_body_advice_and_retry_after_header() -> None:
    with TestClient(_app(), base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        response = client.get("/rate-limit", headers={REQUEST_ID_HEADER: "rate-1"})

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "2"
    assert response.json() == {
        "status": "error",
        "error": {
            "code": "RESOURCE_EXHAUSTED",
            "message": "Model provider rate limit exceeded",
            "retryable": True,
            "retry_after_seconds": 2,
        },
        "request_id": "rate-1",
    }
