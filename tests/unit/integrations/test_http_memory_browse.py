"""``habitus memory`` 的服务端只读口：``GET /api/v1/memory/tree`` 列目录、``GET /api/v1/memory/document`` 读文档或语义层；
特性 ``memory_tree_v1``；只给 ``memory://`` 地址，不给文件路径；没有的地址 404、正在发布时 503 可重试。"""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from habitus.integrations.http import RuntimeHTTPHandlers
from habitus.integrations.http_api import create_http_app
from habitus.memory.browse import MemoryBrowseNotFoundError, MemoryLayerText, MemoryListing
from habitus.memory.model import MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.tree import MemoryTreeConsistencyError
from habitus.memory.uri import MemoryURI
from habitus.runtime import Runtime
from tests.helpers import document


def _runtime() -> Runtime:
    runtime = object.__new__(Runtime)
    runtime.conversation_protocols = lambda: ("codex_rollout",)  # type: ignore[method-assign]
    runtime.start = AsyncMock()  # type: ignore[method-assign]
    runtime.close = AsyncMock()  # type: ignore[method-assign]
    return runtime


def test_listing_and_reading_return_uris_only_and_the_feature_is_advertised() -> None:
    runtime = _runtime()
    runtime.list_memory = AsyncMock(  # type: ignore[method-assign]
        return_value=MemoryListing(
            MemoryURI("memory://preferences"), (), (MemoryURI("memory://preferences/回答风格.md"),), "偏好：回答简洁。"
        )
    )
    preference = document(MemoryKind.PREFERENCE)
    layer = MemoryLayerText(MemoryURI.from_layer(MemoryDirectory.preferences(), MemoryLevel.OVERVIEW), MemoryLevel.OVERVIEW, "## 偏好")
    runtime.read_memory = AsyncMock(side_effect=[preference, layer])  # type: ignore[method-assign]
    assert "memory_tree_v1" in RuntimeHTTPHandlers(runtime).capabilities()["features"]  # type: ignore[operator]
    with TestClient(create_http_app(runtime), base_url="http://127.0.0.1:8787") as client:
        listed = client.get("/api/v1/memory/tree", params={"uri": "memory://preferences"})
        assert listed.status_code == 200, listed.text
        assert listed.json()["result"] == {
            "uri": "memory://preferences",
            "directories": [],
            "documents": ["memory://preferences/回答风格.md"],
            "abstract": "偏好：回答简洁。",
        }
        read = client.get("/api/v1/memory/document", params={"uri": "memory://preferences/回答风格.md"}).json()["result"]
        assert read["uri"] == "memory://preferences/回答风格.md"
        assert read["node"] == "document" and read["kind"] == "preference" and read["revision"] == 1
        assert read["fields"]["content"] == "- 偏好简洁直接的回答"
        overview = client.get("/api/v1/memory/document", params={"uri": "memory://preferences/.overview.md"}).json()["result"]
        assert overview == {
            "uri": str(layer.uri), "node": "layer", "level": "overview", "text": "## 偏好",
            "kind": None, "revision": None, "created_at": None, "updated_at": None, "last_confirmed_at": None,
            "fields": None, "links": [], "backlinks": [],
        }
    runtime.list_memory.assert_awaited_once_with("memory://preferences")  # type: ignore[attr-defined]


def test_missing_nodes_are_404_and_a_publication_in_flight_is_a_retryable_503() -> None:
    runtime = _runtime()
    runtime.read_memory = AsyncMock(  # type: ignore[method-assign]
        side_effect=[MemoryBrowseNotFoundError("no memory document at memory://tools/x.md"), MemoryTreeConsistencyError("busy")]
    )
    with TestClient(create_http_app(runtime), base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
        missing = client.get("/api/v1/memory/document", params={"uri": "memory://tools/x.md"})
        assert missing.status_code == 404
        busy = client.get("/api/v1/memory/document", params={"uri": "memory://tools/x.md"})
        assert busy.status_code == 503 and busy.json()["error"]["retryable"] is True
