"""不引入用户或租户模型的最小 Bearer Token 认证。"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Header

from foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer


class HTTPAuthenticationError(PermissionError):
    """请求没有提供有效的 HTTP API Bearer Token。"""


def build_bearer_authenticator(
    api_key: str,
    *,
    observer: Observer | None = None,
) -> Callable[..., Awaitable[None]]:
    if not isinstance(api_key, str) or not api_key or api_key != api_key.strip():
        raise ValueError("api_key must be non-empty normalized text")

    resolved_observer = observer or NullObserver()

    async def require_bearer_token(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        scheme, separator, credential = (authorization or "").partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not credential
            or not secrets.compare_digest(credential, api_key)
        ):
            try:
                resolved_observer.record(
                    ObservationEvent(
                        category="security",
                        operation="authentication",
                        status=ObservationStatus.FAILURE,
                        duration_seconds=0.0,
                        attributes={"error_code": "invalid_bearer_token", "retryable": False},
                    )
                )
            except Exception:
                pass
            raise HTTPAuthenticationError("A valid Bearer token is required")

    return require_bearer_token


__all__ = ["HTTPAuthenticationError", "build_bearer_authenticator"]
