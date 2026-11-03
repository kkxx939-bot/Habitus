"""本地 HTTP 请求的关联编号：服务端为每个请求新发一个，绑定到观测上下文，并写回响应头。

调用方自带的同名请求头不被采用；下游写过的同名响应头被替换，响应里只留服务端发的这一个。
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from habitus.foundation.observability import bind_observation_context, current_observation_context

REQUEST_ID_HEADER = "X-Request-ID"
_HEADER_KEY = REQUEST_ID_HEADER.lower().encode("ascii")


def current_request_id() -> str | None:
    """返回当前异步请求的关联编号；请求外返回 None。"""

    return current_observation_context().request_id


class RequestIDMiddleware:
    """ASGI 中间件：只处理 HTTP 请求，其余类型原样放行。

    编号同时放进 ``scope["state"]["request_id"]``（路由与错误处理从 ``request.state`` 读）和观测上下文。
    """

    def __init__(self, app: ASGIApp) -> None:
        self._downstream = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._downstream(scope, receive, send)
            return
        identity = uuid4().hex
        scope.setdefault("state", {})["request_id"] = identity
        with bind_observation_context(request_id=identity):
            await self._downstream(scope, receive, _stamp_responses(send, identity))


def _stamp_responses(send: Send, identity: str) -> Send:
    """包一层 send：响应开始那条消息的头里写上关联编号，其余消息原样转发。"""

    async def forward(message: Message) -> None:
        if message["type"] == "http.response.start":
            message = {**message, "headers": _headers_with_identity(message.get("headers", ()), identity)}
        await send(message)

    return forward


def _headers_with_identity(headers: Iterable[tuple[bytes, bytes]], identity: str) -> list[tuple[bytes, bytes]]:
    """去掉同名头（名字不分大小写），在末尾补上服务端发的这一个。"""

    stamped = [pair for pair in headers if pair[0].lower() != _HEADER_KEY]
    stamped.append((_HEADER_KEY, identity.encode("ascii")))
    return stamped


__all__ = ["REQUEST_ID_HEADER", "RequestIDMiddleware", "current_request_id"]
