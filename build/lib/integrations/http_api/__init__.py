"""m2bOS Runtime 的可选 FastAPI 传输适配器。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ERROR_CODE_TO_HTTP_STATUS",
    "HTTPErrorCode",
    "REQUEST_ID_HEADER",
    "create_http_app",
    "current_request_id",
    "documented_error_responses",
]


def __getattr__(name: str) -> Any:
    """只在调用 HTTP API 时加载 FastAPI，保证轻量 Doctor 可独立运行。"""

    if name == "create_http_app":
        from integrations.http_api.app import create_http_app

        return create_http_app
    if name in {"ERROR_CODE_TO_HTTP_STATUS", "documented_error_responses"}:
        from integrations.http_api import errors

        return getattr(errors, name)
    if name in {"REQUEST_ID_HEADER", "current_request_id"}:
        from integrations.http_api import request_id

        return getattr(request_id, name)
    if name == "HTTPErrorCode":
        from integrations.http_api.schemas import HTTPErrorCode

        return HTTPErrorCode
    raise AttributeError(name)
