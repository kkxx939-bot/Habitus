"""向量通用模型、路由配置和显式 Adapter 工厂测试。"""

import asyncio
import hashlib

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector import (
    PublishedVectorStore,
    VectorStoreConfig,
    VectorStoreError,
    VectorStoreFactory,
    VectorStoreFilter,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreRequirements,
    VectorStoreRouteConfig,
)
from habitus.infrastructure.vector.adapters import register_builtin_vector_adapters
from habitus.model_client import EmbeddingVector
from tests.lock_store import ProcessLocalLockStore


def record(identity: str = "memory://profile", *, kind: str = "profile") -> VectorStoreRecord:
    content = f"content:{identity}"
    return VectorStoreRecord(
        identity=identity,
        vector=EmbeddingVector((1.0, 0.0)),
        content=content,
        content_digest=hashlib.sha256(content.encode()).hexdigest(),
        attributes={"kind": kind, "scope_roots": ("memory://", "memory://profile")},
    )


def test_vector_record_filter_and_match_have_backend_independent_semantics() -> None:
    item = record()
    selected = VectorStoreFilter(
        equals={"kind": "profile"},
        one_of={"scope_roots": ("memory://profile", "memory://events")},
    )
    rejected = VectorStoreFilter(equals={"kind": "event"}, one_of={})

    assert selected.matches(item.attributes)
    assert not rejected.matches(item.attributes)
    assert VectorStoreMatch(item, 0.75).score == 0.75
    with pytest.raises(ValueError, match="content_digest"):
        VectorStoreRecord(item.identity, item.vector, item.content, "0" * 64, item.attributes)
    with pytest.raises(ValueError, match="multiple operators"):
        VectorStoreFilter(equals={"kind": "profile"}, one_of={"kind": ("profile",)})
    with pytest.raises(ValueError, match="finite cosine"):
        VectorStoreMatch(item, 2.0)


@pytest.mark.parametrize(
    "route",
    [
        {"base_url": "http://remote.example.com"},
        {"base_url": "https://user:pass@example.com"},
        {"extra_headers": {"Authorization": "secret"}},
        {"credential_ref": "not/valid"},
    ],
)
def test_vector_route_rejects_insecure_address_and_credential_leaks(route: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        VectorStoreRouteConfig(**route)


def test_vector_config_is_strict_and_requirements_cannot_exceed_search_capacity() -> None:
    config = VectorStoreConfig.from_mapping(
        {
            "route": {"provider": "Custom", "adapter": "Fake_Vector"},
            "collection": "memory-prod",
            "options": {"schema_mode": "precreated"},
        }
    )
    assert config.provider == "custom"
    assert config.adapter == "fake_vector"
    with pytest.raises(ValueError, match="unknown"):
        VectorStoreConfig.from_mapping({"unknown": True})
    with pytest.raises(ValueError, match="cannot exceed"):
        VectorStoreRequirements(2, max_records=10, max_search_hits=11, max_record_chars=100)


def test_factory_resolves_credentials_and_rejects_unregistered_or_mismatched_backend() -> None:
    captured = {}
    factory = VectorStoreFactory()

    def builder(context):
        captured.update(context.credentials)

        class Backend:
            adapter_name = "fake_vector"
            provider_name = "custom"
            collection = "memory"
            max_records = 100
            max_search_hits = 10

            async def initialize(self):
                pass

            async def read_metadata(self, names):
                return {}

            async def write_metadata(self, values, *, dimension):
                pass

            async def ensure_schema(self, dimension, *, published_dimension):
                pass

            async def read(self, identities):
                return ()

            async def delete_all(self):
                pass

            async def upsert(self, records):
                pass

            async def delete(self, identities):
                pass

            async def validate_records(self, records, *, replacing):
                pass

            async def wait_visible(self, upserts, deletes, *, complete):
                pass

            async def search(self, query_vector, *, filters, limit):
                return ()

            async def scan(self, *, filters, limit):
                return ()

            async def close(self):
                pass

        return Backend()

    factory.register_adapter("fake_vector", builder)
    config = VectorStoreConfig(
        route=VectorStoreRouteConfig(provider="custom", adapter="fake_vector", credential_ref="fake_vector")
    )
    requirements = VectorStoreRequirements(2, 100, 10, 100)
    publication_lock = PathLock(ProcessLocalLockStore())
    store = factory.create(
        config,
        requirements=requirements,
        credentials={"api_key": " secret "},
        path_lock=publication_lock,
    )
    assert isinstance(store, PublishedVectorStore)
    assert captured == {"api_key": "secret"}
    with pytest.raises(ValueError, match="already registered"):
        factory.register_adapter("fake_vector", builder)
    with pytest.raises(VectorStoreError, match="not registered"):
        VectorStoreFactory().create(
            config,
            requirements=requirements,
            credentials={"api_key": "x"},
            path_lock=publication_lock,
        )
    with pytest.raises(VectorStoreError, match="missing"):
        factory.create(
            config,
            requirements=requirements,
            credentials={},
            path_lock=publication_lock,
        )
    with pytest.raises(VectorStoreError, match="PathLock"):
        factory.create(
            config,
            requirements=requirements,
            credentials={"api_key": "x"},
        )


def test_builtin_local_vector_store_is_the_default_and_needs_a_data_root(tmp_path) -> None:
    factory = register_builtin_vector_adapters()
    config = VectorStoreConfig()
    assert (config.provider, config.adapter, config.route.credential_ref) == ("local", "local_sqlite", "")
    requirements = VectorStoreRequirements(2, 100, 10, 100)
    lock = PathLock(ProcessLocalLockStore())
    with pytest.raises(VectorStoreError, match="data_root"):
        factory.create(config, requirements=requirements, path_lock=lock)
    store = factory.create(config, requirements=requirements, path_lock=lock, data_root=tmp_path / "vector")
    assert isinstance(store, PublishedVectorStore)
    assert (store.adapter_name, store.provider_name, store.collection) == ("local_sqlite", "local", "memory")
    assert store.backend.max_records == 100
    assert store.backend.max_search_hits == 10
    assert not (tmp_path / "vector").exists(), "构造不得落盘"
    asyncio.run(store.close())
