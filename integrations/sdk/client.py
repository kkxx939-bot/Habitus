"""供外部 Agent 连接本地 m2bOS HTTP 服务的异步客户端。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, Protocol, TypeVar, cast
from urllib.parse import urlparse

import httpx

from integrations.sdk.contracts import (
    AgentFlushResult,
    AgentRecallResult,
    AgentRememberResult,
    ConversationRef,
)
from integrations.sdk.wire import (
    decode_cursor,
    decode_flush,
    decode_protocols,
    decode_recall,
    decode_remember,
    require_mapping,
)

_ResultT = TypeVar("_ResultT")


class AsyncHTTPTransport(Protocol):
    """HTTP Client 依赖的最小异步传输能力。"""

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: object | None,
        params: Mapping[str, str] | None,
    ) -> httpx.Response: ...


class M2BOSServiceError(RuntimeError):
    """m2bOS 服务返回了稳定的公开错误。"""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool,
        request_id: str | None,
        status_code: int,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.request_id = request_id
        self.status_code = status_code


class M2BOSServiceTransportError(ConnectionError):
    """尚未收到 m2bOS 服务响应的网络或协议错误。"""


class M2BOSHTTPClient:
    """实现 AgentMemoryPort，不依赖 Runtime、memory 或具体 Agent SDK。"""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        timeout_seconds: float = 30.0,
        transport: AsyncHTTPTransport | None = None,
    ) -> None:
        if not isinstance(base_url, str) or base_url != base_url.strip():
            raise ValueError("base_url must be normalized text")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url cannot contain a query or fragment")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be non-empty text")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int | float)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        if transport is not None and not callable(getattr(transport, "request", None)):
            raise TypeError("transport must implement AsyncHTTPTransport")
        owned = None if transport is not None else httpx.AsyncClient(timeout=float(timeout_seconds))
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport if transport is not None else cast(AsyncHTTPTransport, owned)
        self._owned_transport = owned

    async def __aenter__(self) -> M2BOSHTTPClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        """只关闭由客户端自身创建的连接池。"""

        if self._owned_transport is not None:
            await self._owned_transport.aclose()

    async def protocols(self) -> tuple[str, ...]:
        """从服务端发现当前真正注册的协议，避免 SDK 复制协议清单。"""

        result = await self._request("GET", "/api/v1/protocols")
        return self._decode(decode_protocols, result)

    async def remember(
        self,
        conversation: ConversationRef,
        *,
        protocol: str,
        payload: object,
        start_sequence: int,
        occurred_at: datetime,
        after_turn: bool | None = None,
        wait_timeout_seconds: float | None = None,
    ) -> AgentRememberResult:
        _require_conversation(conversation)
        body = {
            "conversation_id": conversation.conversation_id,
            "started_on": conversation.started_on.isoformat(),
            "protocol": protocol,
            "payload": payload,
            "start_sequence": start_sequence,
            "occurred_at": occurred_at.isoformat(),
            "after_turn": after_turn,
            "wait_timeout_seconds": wait_timeout_seconds,
        }
        result = await self._request("POST", "/api/v1/memory/remember", json=body)
        return self._decode(decode_remember, result)

    async def recall(
        self,
        query: str,
        *,
        conversation: ConversationRef | None = None,
        limit: int | None = None,
        kinds: tuple[str, ...] = (),
        intention_scope: str = "active",
    ) -> AgentRecallResult:
        if conversation is not None:
            _require_conversation(conversation)
        body: dict[str, object] = {
            "query": query,
            "limit": limit,
            "kinds": list(kinds),
            "intention_scope": intention_scope,
        }
        if conversation is not None:
            body.update(
                conversation_id=conversation.conversation_id,
                started_on=conversation.started_on.isoformat(),
            )
        result = await self._request("POST", "/api/v1/memory/recall", json=body)
        return self._decode(decode_recall, result)

    async def flush(
        self,
        conversation: ConversationRef,
        *,
        wait_timeout_seconds: float | None = None,
    ) -> AgentFlushResult:
        _require_conversation(conversation)
        result = await self._request(
            "POST",
            "/api/v1/memory/flush",
            json={
                "conversation_id": conversation.conversation_id,
                "started_on": conversation.started_on.isoformat(),
                "wait_timeout_seconds": wait_timeout_seconds,
            },
        )
        return self._decode(decode_flush, result)

    async def cursor(self, conversation: ConversationRef) -> int:
        _require_conversation(conversation)
        result = await self._request(
            "GET",
            "/api/v1/memory/conversations/cursor",
            params={
                "conversation_id": conversation.conversation_id,
                "started_on": conversation.started_on.isoformat(),
            },
        )
        return self._decode(decode_cursor, result)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
        params: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        try:
            response = await self._transport.request(
                method,
                f"{self._base_url}{path}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=json,
                params=params,
            )
        except httpx.RequestError as exc:
            raise M2BOSServiceTransportError("m2bOS service request did not receive a response") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise M2BOSServiceTransportError("m2bOS service returned invalid JSON") from exc
        try:
            envelope = require_mapping(payload, "response")
        except ValueError as exc:
            raise M2BOSServiceTransportError("m2bOS service returned an invalid response") from exc
        request_id_value = envelope.get("request_id")
        request_id = request_id_value if isinstance(request_id_value, str) and request_id_value else None
        if response.is_error or envelope.get("status") == "error":
            try:
                error = require_mapping(envelope.get("error"), "error")
            except ValueError as exc:
                raise M2BOSServiceTransportError("m2bOS service returned an invalid error") from exc
            raise M2BOSServiceError(
                self._error_text(error.get("message"), "error.message"),
                code=self._error_text(error.get("code"), "error.code"),
                retryable=self._error_boolean(error.get("retryable"), "error.retryable"),
                request_id=request_id,
                status_code=response.status_code,
            )
        if envelope.get("status") != "ok":
            raise M2BOSServiceTransportError("m2BOS service returned an unknown response envelope")
        try:
            return require_mapping(envelope.get("result"), "result")
        except ValueError as exc:
            raise M2BOSServiceTransportError("m2BOS service returned an invalid result") from exc

    @staticmethod
    def _decode(
        decoder: Callable[[Mapping[str, Any]], _ResultT],
        value: Mapping[str, Any],
    ) -> _ResultT:
        try:
            return decoder(value)
        except (TypeError, ValueError) as exc:
            raise M2BOSServiceTransportError("m2bOS service result violates its wire contract") from exc

    @staticmethod
    def _error_text(value: object, label: str) -> str:
        if not isinstance(value, str) or not value:
            raise M2BOSServiceTransportError(f"m2BOS service returned an invalid {label}")
        return value

    @staticmethod
    def _error_boolean(value: object, label: str) -> bool:
        if not isinstance(value, bool):
            raise M2BOSServiceTransportError(f"m2BOS service returned an invalid {label}")
        return value


def _require_conversation(value: ConversationRef) -> None:
    if not isinstance(value, ConversationRef):
        raise TypeError("conversation must be ConversationRef")


__all__ = [
    "AsyncHTTPTransport",
    "M2BOSHTTPClient",
    "M2BOSServiceError",
    "M2BOSServiceTransportError",
]
