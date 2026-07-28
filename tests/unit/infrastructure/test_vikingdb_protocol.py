"""VikingDB HTTP 协议、记录编解码和分页完整性测试。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable

import httpx
import pytest

from infrastructure.vector import (
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreError,
    VectorStoreFilter,
    VectorStoreIntegrityError,
    VectorStoreRecord,
    VectorStoreRouteConfig,
)
from infrastructure.vector.adapters import vikingdb as protocol
from infrastructure.vector.adapters.vikingdb import VikingDBBackend
from infrastructure.vector.adapters.vikingdb_client import (
    VikingDBNotFoundError,
    VikingDBRestClient,
)
from infrastructure.vector.adapters.vikingdb_config import VikingDBVectorStoreConfig
from ModelClient import EmbeddingVector


def route(**overrides: object) -> VectorStoreRouteConfig:
    values: dict[str, object] = {
        "base_url": "https://vector.example.com",
        "max_retries": 0,
        "retry_base_delay_seconds": 0.001,
        "retry_max_delay_seconds": 0.001,
        "max_response_bytes": 4096,
    }
    values.update(overrides)
    return VectorStoreRouteConfig(**values)  # type: ignore[arg-type]


async def async_client(
    handler: Callable[[httpx.Request], httpx.Response],
    **route_overrides: object,
) -> VikingDBRestClient:
    instance = VikingDBRestClient(
        route(**route_overrides),
        VikingDBVectorStoreConfig(auth_mode="api_key"),
        credentials={"api_key": "secret"},
    )
    original = instance._client
    instance._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await original.aclose()
    return instance


def client(handler: Callable[[httpx.Request], httpx.Response], **route_overrides: object) -> VikingDBRestClient:
    return asyncio.run(async_client(handler, **route_overrides))


def record(identity: str = "memory://profile.md") -> VectorStoreRecord:
    content = f"content:{identity}"
    return VectorStoreRecord(
        identity=identity,
        vector=EmbeddingVector((1.0, 0.0)),
        content=content,
        content_digest=hashlib.sha256(content.encode()).hexdigest(),
        attributes={
            "uri": identity,
            "level": 2,
            "directory_key": "memory://",
            "parent_key": "memory://",
            "scope_roots": ("memory://",),
            "kind": "profile",
            "revision": 1,
        },
    )


def search_item(value: VectorStoreRecord, *, score: float = 0.9) -> dict[str, object]:
    encoded = protocol._item_from_record(value, scope="default/memory")
    return {
        "id": encoded["id"],
        "fields": {key: item for key, item in encoded.items() if key != "id"},
        "score": score,
    }


def test_api_key_request_uses_bearer_auth_canonical_json_and_absolute_path() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(200, json={"code": 0, "result": {"ok": True}})

    instance = client(handler)
    result = asyncio.run(instance.data("/api/vikingdb/data/search/vector", {"b": 2, "a": "中文"}))
    asyncio.run(instance.close())

    assert result["result"] == {"ok": True}
    assert observed[0].headers["authorization"] == "Bearer secret"
    assert observed[0].content == json.dumps(
        {"b": 2, "a": "中文"}, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()
    invalid_path_client = client(handler)
    with pytest.raises(ValueError, match="absolute API path"):
        asyncio.run(invalid_path_client.data("relative", {}))
    asyncio.run(invalid_path_client.close())


def test_retryable_response_honors_retry_budget_then_succeeds() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"code": "RateLimit", "message": "slow"})
        return httpx.Response(200, json={"code": 0})

    instance = client(handler, max_retries=1)
    assert asyncio.run(instance.data("/api/test", {})) == {"code": 0}
    asyncio.run(instance.close())
    assert calls == 2


@pytest.mark.parametrize(
    ("status", "payload", "error_type"),
    [
        (404, {"code": "NotFound", "message": "missing"}, VikingDBNotFoundError),
        (409, {"code": "Conflict", "message": "exists"}, VectorStoreConflictError),
        (400, {"code": "InvalidArgument", "message": "bad"}, VectorStoreError),
        (503, {"code": "ServiceUnavailable", "message": "down"}, VectorStoreBusyError),
    ],
)
def test_http_and_provider_errors_are_classified_without_silent_fallback(
    status: int,
    payload: dict[str, object],
    error_type: type[Exception],
) -> None:
    instance = client(lambda _request: httpx.Response(status, json=payload))
    with pytest.raises(error_type):
        asyncio.run(instance.data("/api/test", {}))
    asyncio.run(instance.close())


def test_success_response_rejects_invalid_json_root_and_oversized_payload() -> None:
    malformed = client(lambda _request: httpx.Response(200, content=b"not-json"))
    with pytest.raises(VectorStoreIntegrityError, match="invalid JSON"):
        asyncio.run(malformed.data("/api/test", {}))
    asyncio.run(malformed.close())

    oversized = client(
        lambda _request: httpx.Response(200, content=b"{" + b"x" * 5000 + b"}"),
        max_response_bytes=1024,
    )
    with pytest.raises(VectorStoreIntegrityError, match="byte limit"):
        asyncio.run(oversized.data("/api/test", {}))
    asyncio.run(oversized.close())


def test_private_header_credentials_are_exact_and_cannot_overlap_extra_headers() -> None:
    config = VikingDBVectorStoreConfig(
        auth_mode="private_headers",
        region="",
        credential_headers={"X-Tenant-Token": "Bearer {token}"},
    )
    instance = VikingDBRestClient(
        route(),
        config,
        credentials={"token": "secret"},
    )
    headers, _params, _content = instance._prepare_request(
        "POST", instance.data_url, "/api/test", plane="private_console", params={}, encoded_body="{}"
    )
    assert headers["X-Tenant-Token"] == "Bearer secret"
    asyncio.run(instance.close())

    with pytest.raises(VectorStoreError, match="unused credentials"):
        VikingDBRestClient(route(), config, credentials={"token": "secret", "extra": "leak"})
    with pytest.raises(VectorStoreError, match="cannot override"):
        VikingDBRestClient(
            route(extra_headers={"x-tenant-token": "public"}),
            config,
            credentials={"token": "secret"},
        )


def test_record_round_trip_binds_identity_hash_index_fields_and_full_attributes() -> None:
    value = record()
    encoded = search_item(value)
    restored = protocol._record_from_item(encoded, scope="default/memory")
    assert restored == value

    tampered = json.loads(json.dumps(encoded))
    tampered["fields"]["uri_key"] = "wrong"
    with pytest.raises(VectorStoreIntegrityError, match="indexed field"):
        protocol._record_from_item(tampered, scope="default/memory")

    with pytest.raises(ValueError, match="declared scalar index"):
        protocol._compile_filter(VectorStoreFilter(equals={"undeclared": "x"}, one_of={}))


def test_backend_search_rejects_duplicate_pages_and_non_finite_scores() -> None:
    async def scenario(items: list[dict[str, object]], error: str) -> None:
        real_client = await async_client(
            lambda _request: httpx.Response(200, json={"code": 0})
        )
        backend = VikingDBBackend(
            "volcengine",
            "memory",
            VikingDBVectorStoreConfig(auth_mode="api_key", search_page_size=1),
            real_client,
        )
        backend._initialized = True
        calls = 0

        async def data(_path: str, _body: dict[str, object]) -> dict[str, object]:
            nonlocal calls
            item = items[min(calls, len(items) - 1)]
            calls += 1
            return {"result": {"data": [item]}}

        real_client.data = data  # type: ignore[method-assign]
        with pytest.raises(VectorStoreIntegrityError, match=error):
            await backend.search(
                EmbeddingVector((1.0, 0.0)),
                filters=VectorStoreFilter({}, {}),
                limit=2,
            )
        await backend.close()

    item = search_item(record())
    asyncio.run(scenario([item, item], "pagination is unstable"))
    invalid_score = {**item, "score": float("nan")}
    asyncio.run(scenario([invalid_score], "score is not finite"))
