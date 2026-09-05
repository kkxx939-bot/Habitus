"""向量通用模型、路由配置和显式 Adapter 工厂测试。"""

import asyncio
import hashlib

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.infrastructure.store.sqlite import SQLiteLockStore
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
    VectorStoreUnsupportedTopologyError,
)
from habitus.infrastructure.vector.adapters import register_builtin_vector_adapters
from habitus.infrastructure.vector.adapters.vikingdb_config import (
    VikingDBVectorStoreConfig,
    bounded_retry_after,
    render_credential_template,
)
from habitus.model_client import EmbeddingVector


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
            "route": {"provider": "VolcEngine", "adapter": "VikingDB"},
            "collection": "memory-prod",
            "options": {"schema_mode": "precreated"},
        }
    )
    assert config.provider == "volcengine"
    assert config.adapter == "vikingdb"
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
            adapter_name = "vikingdb"
            requires_cross_process_publication_fencing = False
            provider_name = "volcengine"
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

    factory.register_adapter(
        "vikingdb",
        builder,
        requires_cross_process_publication_fencing=False,
    )
    config = VectorStoreConfig(route=VectorStoreRouteConfig(credential_ref="vikingdb"))
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
        factory.register_adapter(
            "vikingdb",
            builder,
            requires_cross_process_publication_fencing=False,
        )
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


def test_factory_rejects_process_local_fencing_before_invoking_remote_backend_builder() -> None:
    builder_calls: list[str] = []

    def builder(_context):
        builder_calls.append("builder")
        raise AssertionError("unsupported remote builder must not be invoked")

    factory = VectorStoreFactory()
    factory.register_adapter(
        "vikingdb",
        builder,
        requires_cross_process_publication_fencing=True,
    )
    with pytest.raises(VectorStoreUnsupportedTopologyError, match="host-scoped"):
        factory.create(
            VectorStoreConfig(),
            requirements=VectorStoreRequirements(2, 100, 10, 100),
            path_lock=PathLock(ProcessLocalLockStore()),
        )
    assert builder_calls == []


def test_vikingdb_options_cover_public_private_and_capacity_combinations() -> None:
    route = VectorStoreRouteConfig(max_response_bytes=1_000_000)
    public = VikingDBVectorStoreConfig(auth_mode="api_key", region="cn-beijing")
    assert public.data_url(route).startswith("https://api-vikingdb")

    private = VikingDBVectorStoreConfig(
        auth_mode="private_headers",
        region="",
        credential_headers={"X-Token": "Bearer {token}"},
    )
    assert private.data_url(VectorStoreRouteConfig(base_url="https://vector.internal")) == "https://vector.internal"
    assert render_credential_template("Bearer {token}", {"token": "secret"}) == "Bearer secret"
    with pytest.raises(ValueError, match="requires route.base_url"):
        private.data_url(VectorStoreRouteConfig())
    with pytest.raises(ValueError, match="managed schema"):
        VikingDBVectorStoreConfig(auth_mode="api_key", schema_mode="managed")
    with pytest.raises(ValueError, match="max_records"):
        public.validate_requirements(VectorStoreRequirements(2, public.max_records + 1, 10, 100), route)
    assert bounded_retry_after("999", 10) == 10
    assert bounded_retry_after("invalid", 10) is None


def test_builtin_vikingdb_requires_host_scoped_publication_fencing(tmp_path) -> None:
    factory = register_builtin_vector_adapters()
    config = VectorStoreConfig(
        route=VectorStoreRouteConfig(credential_ref="vikingdb"),
    )
    requirements = VectorStoreRequirements(2, 100, 10, 100)
    with pytest.raises(VectorStoreUnsupportedTopologyError, match="host-scoped"):
        factory.create(
            config,
            requirements=requirements,
            credentials={"api_key": "secret"},
            path_lock=PathLock(ProcessLocalLockStore()),
        )
    store = factory.create(
        config,
        requirements=requirements,
        credentials={"api_key": "secret"},
        path_lock=PathLock(SQLiteLockStore(tmp_path / "publication.sqlite3", initialize=False)),
    )
    assert isinstance(store, PublishedVectorStore)
    assert store.adapter_name == "vikingdb"
    assert store.provider_name == config.provider
    assert store.collection == config.collection
    asyncio.run(store.close())
