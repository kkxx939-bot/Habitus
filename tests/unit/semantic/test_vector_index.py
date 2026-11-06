"""MemoryTree 真相源到远程 VectorStore 的重建、增量和过滤测试。"""

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from habitus.infrastructure.vector import VectorStoreMatch, VectorStoreState
from habitus.memory.indexing import (
    MemoryVectorIndexError,
    PersistentMemoryVectorIndex,
    memory_embedding_fingerprint,
)
from habitus.memory.intention import MemoryIntentionRecallScope
from habitus.memory.model import MemoryKind, MemoryLevel
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from habitus.model_client import EmbeddingVector
from tests.helpers import document


class Embedder:
    def __init__(self) -> None:
        self.inputs = []

    async def embed_documents(self, values):
        self.inputs.extend(values)
        return tuple(EmbeddingVector((1.0, 0.0)) for _ in values)


class VectorStore:
    def __init__(self) -> None:
        self.current = None
        self.records = {}
        self.filters = []

    async def initialize(self):
        pass

    async def state(self):
        return self.current

    async def read(self, identities):
        return tuple(self.records[item] for item in identities if item in self.records)

    async def replace_all(self, records, **kwargs):
        self.records = {item.identity: item for item in records}
        self.current = VectorStoreState(
            kwargs["schema_version"],
            kwargs["embedding_fingerprint"],
            kwargs["dimension"],
            kwargs["checkpoint"],
            1 if self.current is None else self.current.generation + 1,
            len(records),
        )
        return self.current

    async def apply(self, upserts, deletes, **kwargs):
        for item in upserts:
            self.records[item.identity] = item
        for identity in deletes:
            self.records.pop(identity, None)
        self.current = VectorStoreState(
            self.current.schema_version,
            self.current.embedding_fingerprint,
            self.current.dimension,
            kwargs["checkpoint"],
            self.current.generation + 1,
            len(self.records),
        )
        return self.current

    async def search(self, query_vector, *, filters, limit):
        self.filters.append(filters)
        return tuple(VectorStoreMatch(item, 0.9) for item in self.records.values() if filters.matches(item.attributes))[
            :limit
        ]

    async def scan(self, *, filters, limit):
        return tuple(self.records.values())[:limit]

    async def close(self):
        pass


def index(tmp_path: Path):
    tree = MemoryTree(tmp_path / "memory")
    embedder = Embedder()
    store = VectorStore()
    return (
        tree,
        embedder,
        store,
        PersistentMemoryVectorIndex(tree, embedder, store, dimension=2, embedding_fingerprint="embed-v1"),
    )


def test_memory_embedding_fingerprint_includes_the_effective_document_encoder_route() -> None:
    common = {
        "provider": "provider",
        "model": "embedding-model",
        "dimension": 2,
        "document_prefix": "",
        "document_parameters": {},
    }
    baseline = memory_embedding_fingerprint(
        adapter="adapter-a",
        base_url="https://embedding-a.example.com/v1",
        extra_body={"encoding": "dense"},
        **common,
    )

    assert baseline != memory_embedding_fingerprint(
        adapter="adapter-b",
        base_url="https://embedding-a.example.com/v1",
        extra_body={"encoding": "dense"},
        **common,
    )
    assert baseline != memory_embedding_fingerprint(
        adapter="adapter-a",
        base_url="https://embedding-b.example.com/v1",
        extra_body={"encoding": "dense"},
        **common,
    )
    assert baseline != memory_embedding_fingerprint(
        adapter="adapter-a",
        base_url="https://embedding-a.example.com/v1",
        extra_body={"encoding": "sparse"},
        **common,
    )
    assert baseline != memory_embedding_fingerprint(
        adapter="adapter-a",
        base_url="https://embedding-a.example.com/v1",
        extra_body={"encoding": "dense"},
        **{**common, "document_prefix": "passage: "},
    )


def test_missing_remote_index_is_fully_rebuilt_from_tree_and_consistency_is_clean(tmp_path: Path) -> None:
    tree, embedder, store, vector_index = index(tmp_path)
    tree.write(document(MemoryKind.PROFILE))
    tree.write(document(MemoryKind.PREFERENCE))
    state = asyncio.run(vector_index.ensure_ready())
    report = asyncio.run(vector_index.check_consistency())
    assert state.record_count == 2
    assert len(embedder.inputs) == 2
    assert report.ok
    assert report.expected_count == report.indexed_count == 2


def test_read_only_consistency_audit_reports_stale_record_without_rebuilding(tmp_path: Path) -> None:
    tree, embedder, store, vector_index = index(tmp_path)
    tree.write(document(MemoryKind.PROFILE))
    asyncio.run(vector_index.ensure_ready())
    identity = next(iter(store.records))
    stale_content = store.records[identity].content + "\n陈旧内容"
    store.records[identity] = replace(
        store.records[identity],
        content=stale_content,
        content_digest=hashlib.sha256(stale_content.encode("utf-8")).hexdigest(),
    )
    embedded_before = tuple(embedder.inputs)

    report = asyncio.run(vector_index.audit_consistency())

    assert report.stale_identities == (identity,)
    assert tuple(embedder.inputs) == embedded_before


def test_incremental_sync_updates_changed_l2_and_advances_global_job_checkpoint(tmp_path: Path) -> None:
    tree, embedder, store, vector_index = index(tmp_path)
    original = document(MemoryKind.PREFERENCE)
    tree.write(original)
    asyncio.run(vector_index.ensure_ready())
    changed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 改为极简回答"},
        revision=2,
    )
    tree.write(changed)
    state = asyncio.run(
        vector_index.synchronize(
            changed_uris=(MemoryURI.from_address(changed.address),),
            semantic_results=(),
            checkpoint=2,
        )
    )
    assert state.checkpoint == 2
    assert len(embedder.inputs) == 2
    with pytest.raises(MemoryVectorIndexError, match="backwards"):
        asyncio.run(vector_index.synchronize(changed_uris=(), semantic_results=(), checkpoint=1))


def test_l2_search_applies_kind_and_scope_filters_before_top_k(tmp_path: Path) -> None:
    tree, _embedder, store, vector_index = index(tmp_path)
    tree.write(document(MemoryKind.PROFILE))
    tree.write(document(MemoryKind.PREFERENCE))
    asyncio.run(vector_index.ensure_ready())
    matches = asyncio.run(
        vector_index.search(
            EmbeddingVector((1.0, 0.0)),
            roots=(MemoryURI.root(),),
            levels=(MemoryLevel.DETAIL,),
            kinds=(MemoryKind.PREFERENCE,),
            intention_scope=MemoryIntentionRecallScope.ACTIVE,
            limit=5,
        )
    )
    assert tuple(match.uri.to_address().kind for match in matches) == (MemoryKind.PREFERENCE,)
    assert store.filters[-1].one_of["kind"] == ("preference",)


def test_l2_kind_filter_cannot_be_mixed_with_directory_levels(tmp_path: Path) -> None:
    tree, _embedder, _store, vector_index = index(tmp_path)
    tree.write(document(MemoryKind.PROFILE))
    with pytest.raises(ValueError, match="cannot be mixed"):
        asyncio.run(
            vector_index.search(
                EmbeddingVector((1.0, 0.0)),
                roots=(MemoryURI.root(),),
                levels=(MemoryLevel.DETAIL, MemoryLevel.ABSTRACT),
                kinds=(),
                intention_scope=MemoryIntentionRecallScope.ACTIVE,
                limit=5,
            )
        )


def test_write_and_recall_through_an_openai_compatible_service_use_their_own_prefix_and_parameters(
    tmp_path: Path,
) -> None:
    """W7 全链路：L2 写进树 → 建索引走真实 EmbeddingClient + OpenAI 兼容适配器（文档前缀与文档参数），
    召回时查询走查询前缀与查询参数；假服务按 OpenAI Embeddings 响应格式回包。"""

    import json

    import httpx

    from habitus.model_client import EmbeddingClient, EmbeddingModelConfig, ProviderConfig
    from habitus.model_client.adapters import OpenAICompatibleEmbeddingProvider
    from habitus.runtime.embedding_fingerprints import embedding_fingerprints

    config = EmbeddingModelConfig(
        ProviderConfig(
            provider="custom",
            adapter="openai_compatible_embedding",
            model="embedding-model",
            base_url="http://127.0.0.1:11434/v1",
            max_retries=0,
        ),
        dimension=2,
        query_parameters={"task": "query"},
        document_parameters={"task": "document"},
        query_prefix="query: ",
        document_prefix="passage: ",
    )
    requests: list[dict[str, object]] = []

    def service(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    provider = OpenAICompatibleEmbeddingProvider(config)
    transport = httpx.MockTransport(service)
    provider._client = lambda: httpx.AsyncClient(transport=transport)
    embedder = EmbeddingClient(config, provider)
    tree = MemoryTree(tmp_path / "memory")
    store = VectorStore()
    vector_index = PersistentMemoryVectorIndex(
        tree, embedder, store, dimension=2, embedding_fingerprint=embedding_fingerprints(config)[0]
    )
    tree.write(document(MemoryKind.PREFERENCE))

    async def write_and_recall():
        await vector_index.ensure_ready()
        written = len(requests)
        query = await embedder.embed_query("what does the user prefer")
        matches = await vector_index.search(
            query,
            roots=(MemoryURI.root(),),
            levels=(MemoryLevel.DETAIL,),
            kinds=(MemoryKind.PREFERENCE,),
            intention_scope=MemoryIntentionRecallScope.ACTIVE,
            limit=5,
        )
        return written, matches

    written, matches = asyncio.run(write_and_recall())

    assert written >= 1
    documents, (query_request,) = requests[:written], requests[written:]
    assert all(item["task"] == "document" and str(item["input"]).startswith("passage: ") for item in documents)
    assert query_request == {
        "task": "query",
        "model": "embedding-model",
        "input": "query: what does the user prefer",
        "encoding_format": "float",
    }
    assert tuple(match.uri.to_address().kind for match in matches) == (MemoryKind.PREFERENCE,)
    assert store.current.embedding_fingerprint == embedding_fingerprints(config)[0]
