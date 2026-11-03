"""在 JSON 路由解析前限制声明长度和流式请求体的 ASGI 中间件。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from starlette.datastructures import Headers

from habitus.integrations.http_api.local_security import local_request_violation

ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]


class RequestBodyLimitExceeded(RuntimeError):
    """请求体在流式读取过程中超过服务配置上限。"""


class RequestBodyLimitMiddleware:
    """按实际字节数限制请求体。

    带 Content-Length 的请求由路由前的边界按声明长度先拒绝，这里只按实际读到的字节兜底；
    不带 Content-Length（分块）的请求先把请求体读完，超限就在路由执行前拒绝，
    不超限再把读到的消息原样交给路由。Host / Origin 不合格的请求一个字节也不读，
    直接交给路由前的边界按同一判法拒绝。
    """

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int) or max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be a positive integer")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if not _declares_content_length(scope):
            if _local_request_violation(scope) is not None:
                await self.app(scope, receive, send)
                return
            await self.app(scope, await self._buffered_receive(receive), send)
            return
        received = 0
        exceeded = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal exceeded, received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(_body(message))
                if received > self.max_body_bytes:
                    exceeded = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def limited_send(message: dict[str, Any]) -> None:
            if exceeded and message.get("type") == "http.response.start":
                raise RequestBodyLimitExceeded
            await send(message)

        await self.app(scope, limited_receive, limited_send)

    async def _buffered_receive(self, receive: ASGIReceive) -> ASGIReceive:
        messages: list[dict[str, Any]] = []
        received = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            received += len(_body(message))
            if received > self.max_body_bytes:
                raise RequestBodyLimitExceeded
            if not message.get("more_body", False):
                break

        async def replay() -> dict[str, Any]:
            return messages.pop(0) if messages else await receive()

        return replay


def _declares_content_length(scope: dict[str, Any]) -> bool:
    return any(name.lower() == b"content-length" for name, _value in scope.get("headers", ()))


def _local_request_violation(scope: dict[str, Any]) -> str | None:
    headers = Headers(scope=scope)
    return local_request_violation(host_header=headers.get("host"), origin_header=headers.get("origin"))


def _body(message: dict[str, Any]) -> bytes:
    body = message.get("body", b"")
    if not isinstance(body, bytes):
        raise RuntimeError("ASGI server returned an invalid HTTP body")
    return body


__all__ = ["RequestBodyLimitExceeded", "RequestBodyLimitMiddleware"]
