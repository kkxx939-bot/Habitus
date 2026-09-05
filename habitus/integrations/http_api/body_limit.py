"""在 JSON 路由解析前限制声明长度和流式请求体的 ASGI 中间件。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]


class RequestBodyLimitExceeded(RuntimeError):
    """请求体在流式读取过程中超过服务配置上限。"""


class RequestBodyLimitMiddleware:
    """包装 ASGI receive，在未知 Content-Length 时仍按实际字节数拒绝。"""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int) or max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be a positive integer")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        received = 0
        exceeded = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal exceeded, received
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if not isinstance(body, bytes):
                    raise RuntimeError("ASGI server returned an invalid HTTP body")
                received += len(body)
                if received > self.max_body_bytes:
                    exceeded = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def limited_send(message: dict[str, Any]) -> None:
            if exceeded and message.get("type") == "http.response.start":
                raise RequestBodyLimitExceeded
            await send(message)

        await self.app(scope, limited_receive, limited_send)


__all__ = ["RequestBodyLimitExceeded", "RequestBodyLimitMiddleware"]
