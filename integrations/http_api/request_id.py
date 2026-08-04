"""为本地 HTTP 请求生成由服务端拥有的关联身份。"""

from __future__ import annotations

from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from foundation.observability import bind_observation_context, current_observation_context

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_HEADER_BYTES = b"x-request-id"


def current_request_id() -> str | None:
    """返回当前异步请求的关联身份；请求外返回 None。"""

    return current_observation_context().request_id


class RequestIDMiddleware:
    """为每个本地 HTTP 请求生成、绑定并返回一个新的关联身份。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", ())
                    if name.lower() != _REQUEST_ID_HEADER_BYTES
                ]
                headers.append((_REQUEST_ID_HEADER_BYTES, request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        with bind_observation_context(request_id=request_id):
            await self.app(scope, receive, send_with_request_id)


__all__ = ["REQUEST_ID_HEADER", "RequestIDMiddleware", "current_request_id"]
