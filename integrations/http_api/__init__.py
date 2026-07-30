"""m2bOS Runtime 的可选 FastAPI 传输适配器。"""

from integrations.http_api.app import create_http_app
from integrations.http_api.errors import ERROR_CODE_TO_HTTP_STATUS, documented_error_responses
from integrations.http_api.request_id import REQUEST_ID_HEADER, current_request_id
from integrations.http_api.schemas import HTTPErrorCode

__all__ = [
    "ERROR_CODE_TO_HTTP_STATUS",
    "HTTPErrorCode",
    "REQUEST_ID_HEADER",
    "create_http_app",
    "current_request_id",
    "documented_error_responses",
]
