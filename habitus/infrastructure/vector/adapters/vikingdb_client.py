"""VikingDB V2 异步 HTTP 客户端与三种认证协议。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Literal, cast
from urllib.parse import urlsplit

import httpx

from habitus.infrastructure.vector.adapters.vikingdb_config import (
    VikingDBVectorStoreConfig,
    bounded_retry_after,
    credential_template_names,
    render_credential_template,
)
from habitus.infrastructure.vector.config import VectorStoreRouteConfig
from habitus.infrastructure.vector.model import (
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreError,
    VectorStoreIntegrityError,
)

_Plane = Literal["data", "private_console", "public_console"]
_RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class VikingDBNotFoundError(VectorStoreError):
    """请求的 VikingDB Collection 或 Index 不存在。"""


class VikingDBRestClient:
    """统一承载并发、重试、响应上限和认证，Adapter 只处理数据语义。"""

    def __init__(
        self,
        route: VectorStoreRouteConfig,
        config: VikingDBVectorStoreConfig,
        *,
        credentials: Mapping[str, str],
    ) -> None:
        if not isinstance(route, VectorStoreRouteConfig):
            raise TypeError("route must be VectorStoreRouteConfig")
        if not isinstance(config, VikingDBVectorStoreConfig):
            raise TypeError("config must be VikingDBVectorStoreConfig")
        if not isinstance(credentials, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in credentials.items()
        ):
            raise TypeError("vikingdb credentials must be a string mapping")
        self.route = route
        self.config = config
        self._credentials = dict(credentials)
        self.data_url = config.data_url(route)
        self.console_url = config.resolved_console_url() if config.auth_mode == "ak_sk" else self.data_url
        if {name.casefold() for name in route.extra_headers} & {
            name.casefold() for name in config.credential_headers
        }:
            raise VectorStoreError("vikingdb credential headers cannot override route.extra_headers")
        self._validate_credentials()
        self._client = httpx.AsyncClient(
            timeout=route.timeout_seconds,
            follow_redirects=False,
        )
        self._semaphore = asyncio.Semaphore(route.max_concurrent)

    async def data(
        self,
        path: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        return await self._request("POST", self.data_url, path, plane="data", body=body)

    async def private_console(
        self,
        path: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        if self.config.auth_mode != "private_headers":
            raise ValueError("private console requests require private_headers mode")
        return await self._request(
            "POST",
            self.console_url,
            path,
            plane="private_console",
            body=body,
        )

    async def public_console(
        self,
        action: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        if self.config.auth_mode != "ak_sk":
            raise ValueError("public console requests require ak_sk mode")
        if not isinstance(action, str) or not action.startswith(("Get", "Create", "Delete")):
            raise ValueError("vikingdb console action is not allowed")
        return await self._request(
            "POST",
            self.console_url,
            "/",
            plane="public_console",
            body=body,
            params={"Action": action, "Version": "2025-06-09"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        origin: str,
        path: str,
        *,
        plane: _Plane,
        body: Mapping[str, object],
        params: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        if not isinstance(path, str) or not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("vikingdb request path must be one absolute API path")
        encoded = json.dumps(dict(body), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        request_params = dict(params or {})
        for attempt in range(self.route.max_retries + 1):
            headers, signed_params, content = self._prepare_request(
                method,
                origin,
                path,
                plane=plane,
                params=request_params,
                encoded_body=encoded,
            )
            try:
                async with self._semaphore:
                    async with self._client.stream(
                        method,
                        f"{origin}{path}",
                        headers=headers,
                        params=signed_params,
                        content=content,
                    ) as response:
                        status = response.status_code
                        retry_after = response.headers.get("retry-after")
                        payload = await self._bounded_payload(response)
            except httpx.TransportError as exc:
                if attempt < self.route.max_retries:
                    await asyncio.sleep(self._retry_delay(attempt, None))
                    continue
                raise VectorStoreBusyError("vikingdb is temporarily unreachable") from exc
            try:
                decoded = self._decode(payload, method, path)
            except VectorStoreIntegrityError:
                if 200 <= status < 300:
                    raise
                if status in _RETRYABLE_STATUSES and attempt < self.route.max_retries:
                    await asyncio.sleep(self._retry_delay(attempt, retry_after))
                    continue
                decoded = {}
            error = _response_error(decoded)
            if 200 <= status < 300 and error is None:
                return decoded
            code, message = error or ("HTTPError", payload[:4000].decode("utf-8", errors="replace"))
            normalized_code = code.casefold()
            if status == 404 or "notfound" in normalized_code or "not_found" in normalized_code:
                raise VikingDBNotFoundError(f"vikingdb resource does not exist: {message}")
            if status == 409 or "alreadyexists" in normalized_code or "conflict" in normalized_code:
                raise VectorStoreConflictError(f"vikingdb rejected a conflicting operation: {message}")
            if status in _RETRYABLE_STATUSES or _retryable_error_code(normalized_code):
                if attempt < self.route.max_retries:
                    await asyncio.sleep(self._retry_delay(attempt, retry_after))
                    continue
                raise VectorStoreBusyError(f"vikingdb remained unavailable: {status} {code}: {message}")
            raise VectorStoreError(f"vikingdb request failed: {status} {code}: {message}")
        raise AssertionError("vikingdb retry loop terminated unexpectedly")

    def _prepare_request(
        self,
        method: str,
        origin: str,
        path: str,
        *,
        plane: _Plane,
        params: Mapping[str, str],
        encoded_body: str,
    ) -> tuple[dict[str, str], dict[str, str], str | bytes]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Habitus/0.1.0",
            **self.route.extra_headers,
        }
        if self.config.auth_mode == "api_key":
            headers["Authorization"] = f"Bearer {self._credentials['api_key']}"
            return headers, dict(params), encoded_body
        if self.config.auth_mode == "private_headers":
            for header_name, template in self.config.credential_headers.items():
                headers[header_name] = render_credential_template(template, self._credentials)
            return headers, dict(params), encoded_body
        return self._signed_request(
            method,
            origin,
            path,
            params=params,
            encoded_body=encoded_body,
            headers=headers,
            plane=plane,
        )

    def _signed_request(
        self,
        method: str,
        origin: str,
        path: str,
        *,
        params: Mapping[str, str],
        encoded_body: str,
        headers: Mapping[str, str],
        plane: _Plane,
    ) -> tuple[dict[str, str], dict[str, str], str | bytes]:
        try:
            from volcengine.auth.SignerV4 import SignerV4  # pyright: ignore[reportMissingImports]
            from volcengine.base.Request import Request  # pyright: ignore[reportMissingImports]
            from volcengine.Credentials import Credentials  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise VectorStoreError(
                "vikingdb ak_sk authentication requires the 'volcengine' package; "
                "install the habitus[vikingdb] extra"
            ) from exc
        parsed = urlsplit(origin)
        request = Request()
        request.set_shema(parsed.scheme)
        request.set_method(method)
        request.set_connection_timeout(self.route.timeout_seconds)
        request.set_socket_timeout(self.route.timeout_seconds)
        request_headers = dict(headers)
        request_headers["Host"] = parsed.netloc
        request.set_headers(request_headers)
        request.set_query(dict(params))
        request.set_host(parsed.netloc)
        request.set_path(path)
        request.set_body(encoded_body)
        credentials = Credentials(
            self._credentials["access_key"],
            self._credentials["secret_key"],
            "vikingdb",
            self.config.region,
            session_token=self._credentials.get("session_token", ""),
        )
        SignerV4.sign(request, credentials)
        if plane not in {"data", "public_console"}:
            raise ValueError("signed vikingdb requests cannot target the private console")
        signed_body = request.body
        if not isinstance(signed_body, str | bytes):
            raise VectorStoreIntegrityError("volcengine signer returned an invalid request body")
        return (
            {str(name): str(value) for name, value in request.headers.items()},
            {str(name): str(value) for name, value in request.query.items()},
            signed_body,
        )

    async def _bounded_payload(self, response: httpx.Response) -> bytes:
        content = bytearray()
        async for chunk in response.aiter_bytes():
            if len(content) + len(chunk) > self.route.max_response_bytes:
                raise VectorStoreIntegrityError("vikingdb response exceeds the configured byte limit")
            content.extend(chunk)
        return bytes(content)

    @staticmethod
    def _decode(payload: bytes, method: str, path: str) -> dict[str, object]:
        if not payload:
            return {}
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VectorStoreIntegrityError(
                f"vikingdb returned invalid JSON for {method.upper()} {path}"
            ) from exc
        if not isinstance(decoded, dict):
            raise VectorStoreIntegrityError("vikingdb response root must be an object")
        return cast(dict[str, object], decoded)

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        parsed = bounded_retry_after(retry_after, self.route.retry_max_delay_seconds)
        if parsed is not None:
            return parsed
        return min(
            self.route.retry_base_delay_seconds * (2**attempt),
            self.route.retry_max_delay_seconds,
        )

    def _validate_credentials(self) -> None:
        invalid = sorted(
            name
            for name, value in self._credentials.items()
            if not name or not value or value != value.strip()
        )
        if invalid:
            raise VectorStoreError(f"vikingdb route contains invalid credential values: {invalid}")
        if self.config.auth_mode == "api_key":
            required = {"api_key"}
            allowed = required
        elif self.config.auth_mode == "ak_sk":
            required = {"access_key", "secret_key"}
            allowed = {*required, "session_token"}
        else:
            required = {
                name
                for template in self.config.credential_headers.values()
                for name in credential_template_names(template)
            }
            allowed = required
        missing = sorted(required - set(self._credentials))
        if missing:
            raise VectorStoreError(f"vikingdb route is missing credentials: {missing}")
        unknown = sorted(set(self._credentials) - allowed)
        if unknown:
            raise VectorStoreError(f"vikingdb route contains unused credentials: {unknown}")


def _response_error(payload: Mapping[str, object]) -> tuple[str, str] | None:
    metadata = payload.get("ResponseMetadata")
    error = metadata.get("Error") if isinstance(metadata, Mapping) else None
    if isinstance(error, Mapping):
        code = str(error.get("Code") or "VikingDBError")
        message = str(error.get("Message") or code)
        return code, message
    raw_error = payload.get("error")
    if isinstance(raw_error, Mapping):
        code = str(raw_error.get("code") or raw_error.get("Code") or "VikingDBError")
        message = str(raw_error.get("message") or raw_error.get("Message") or code)
        return code, message
    code_value = payload.get("code")
    if code_value is not None:
        code = str(code_value)
        if code.casefold() not in {"0", "ok", "success"}:
            return code, str(payload.get("message") or code)
    return None


def _retryable_error_code(value: str) -> bool:
    return any(
        marker in value
        for marker in (
            "busy",
            "internalerror",
            "ratelimit",
            "serviceunavailable",
            "temporar",
            "throttl",
            "timeout",
        )
    )


__all__ = ["VikingDBNotFoundError", "VikingDBRestClient"]
