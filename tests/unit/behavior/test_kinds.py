"""行为类型词表：登记约束、单文件耐久与模型归属。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import datetime, timezone
from typing import Any

import pytest

from behavior.kinds import (
    BehaviorKindConfig,
    BehaviorKindConflictError,
    BehaviorKindError,
    BehaviorKindLimitError,
    BehaviorKindRegistry,
    BehaviorKindResolver,
    BehaviorKindStore,
    BehaviorKindStoreError,
)
from ModelClient import (
    ChatClient,
    ChatModelConfig,
    ChatRequest,
    ModelResponse,
    ModelStructuredOutputError,
    PreparedChatRequest,
    ProviderCapabilities,
    ProviderConfig,
    StructuredChatClient,
)

NOW = datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc)


class ScriptedProvider:
    """按脚本回放结构化输出的假 Provider；记录调用次数与提示词。"""

    provider_name = "fake"
    model = "fake-1"
    is_remote = False
    capabilities = ProviderCapabilities(
        async_completion=True,
        streaming=False,
        tools=False,
        structured_output_mode="json_schema",
        reasoning=False,
    )

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        self.bodies = bodies
        self.calls = 0
        self.prompts: list[str] = []

    def prepare(self, request: ChatRequest, *, stream: bool) -> PreparedChatRequest:
        return PreparedChatRequest(
            request=request,
            body=b"{}",
            model_visible_body=b"{}",
            reserved_output_tokens=1_000,
            stream=stream,
        )

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
        self.prompts.append(request.request.messages[-1].content or "")
        body = self.bodies[min(self.calls, len(self.bodies) - 1)]
        self.calls += 1
        return ModelResponse(
            content=json.dumps(body, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            finish_reason="stop",
        )

    def complete(self, request: PreparedChatRequest) -> ModelResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: PreparedChatRequest) -> Iterator[Any]:  # pragma: no cover
        raise NotImplementedError

    def stream_async(self, request: PreparedChatRequest) -> AsyncIterator[Any]:  # pragma: no cover
        raise NotImplementedError

    def health_check(self) -> Mapping[str, object]:  # pragma: no cover
        return {}

    async def aclose(self) -> None:  # pragma: no cover
        return None


def build_resolver(
    bodies: list[dict[str, Any]], *, config: BehaviorKindConfig | None = None
) -> tuple[BehaviorKindResolver, ScriptedProvider]:
    provider = ScriptedProvider(bodies)
    model_config = ChatModelConfig(
        route=ProviderConfig(
            provider="fake",
            adapter="openai_compatible_chat",
            model="fake-1",
            base_url="https://example.invalid",
            credential_ref="FAKE_KEY",
        ),
        context_window_tokens=128_000,
        max_output_tokens=8_000,
        structured_output_mode="json_schema",
    )
    client = StructuredChatClient(ChatClient(model_config, provider), validation_retries=1)
    return BehaviorKindResolver(client, config=config), provider


# --- 登记约束 -------------------------------------------------------------------------


def test_registry_orders_and_resolves_by_canonical_identity() -> None:
    registry = BehaviorKindRegistry({"洗手": ("洗了手", "清洁双手"), "Wash Dishes": ()})
    assert registry.tokens == ("Wash Dishes", "洗手")
    assert registry.token_for("清洁双手") == "洗手"
    assert registry.token_for("洗手") == "洗手"
    assert registry.token_for("wash dishes") == "Wash Dishes"
    assert registry.token_for("做饭") is None


def test_registry_rejects_identity_collisions() -> None:
    with pytest.raises(BehaviorKindError, match="collides"):
        BehaviorKindRegistry({"洗手": ("洗了手",), "做饭": ("洗了手",)})
    with pytest.raises(BehaviorKindError, match="collides"):
        BehaviorKindRegistry({"洗手": ("做饭",), "做饭": ()})
    with pytest.raises(BehaviorKindError, match="collides"):
        BehaviorKindRegistry({"Wash": (), "wash": ()})


def test_registry_mutations_return_new_instances() -> None:
    base = BehaviorKindRegistry({"洗手": ()})
    grown = base.with_new_kind("做饭").with_alias("洗手", "洗了手")
    assert base.kind_count == 1 and base.aliases_of("洗手") == ()
    assert grown.kind_count == 2 and grown.aliases_of("洗手") == ("洗了手",)
    with pytest.raises(BehaviorKindError, match="not registered"):
        base.with_alias("做饭", "备菜")
    with pytest.raises(BehaviorKindError, match="collides"):
        grown.with_alias("做饭", "洗手")


# --- 单文件耐久 ------------------------------------------------------------------------


def test_store_roundtrip_and_missing_file(tmp_path) -> None:
    store = BehaviorKindStore(tmp_path)
    empty = store.read()
    assert empty.revision == 0 and empty.registry.kind_count == 0
    registry = BehaviorKindRegistry({"洗手": ("洗了手",)})
    written = store.replace(registry, expected_revision=0, timestamp=NOW)
    assert written.revision == 1
    loaded = store.read()
    assert loaded.revision == 1
    assert loaded.registry.kinds == registry.kinds
    assert loaded.updated_at == NOW


def test_store_replace_requires_matching_revision(tmp_path) -> None:
    store = BehaviorKindStore(tmp_path)
    store.replace(BehaviorKindRegistry({"洗手": ()}), expected_revision=0, timestamp=NOW)
    with pytest.raises(BehaviorKindConflictError, match="revision"):
        store.replace(BehaviorKindRegistry({"洗手": ()}), expected_revision=0, timestamp=NOW)


def test_store_rejects_tampered_content(tmp_path) -> None:
    store = BehaviorKindStore(tmp_path)
    store.replace(BehaviorKindRegistry({"洗手": ()}), expected_revision=0, timestamp=NOW)
    path = tmp_path / "kinds.md"
    path.write_text(path.read_text().replace("# 行为类型词表", "# 行为词表"), encoding="utf-8")
    with pytest.raises(BehaviorKindStoreError, match="canonically"):
        store.read()


def test_store_enforces_kind_capacity(tmp_path) -> None:
    store = BehaviorKindStore(tmp_path, config=BehaviorKindConfig(max_kinds=1))
    oversized = BehaviorKindRegistry({"洗手": (), "做饭": ()})
    with pytest.raises(BehaviorKindLimitError, match="kind bound"):
        store.replace(oversized, expected_revision=0, timestamp=NOW)


def test_store_rejects_naive_timestamp(tmp_path) -> None:
    store = BehaviorKindStore(tmp_path)
    with pytest.raises(BehaviorKindError, match="timezone-aware"):
        store.replace(
            BehaviorKindRegistry({"洗手": ()}),
            expected_revision=0,
            timestamp=NOW.replace(tzinfo=None),
        )


# --- 模型归属 -------------------------------------------------------------------------


def test_resolver_known_name_skips_the_model() -> None:
    resolver, provider = build_resolver([])
    registry = BehaviorKindRegistry({"洗手": ("洗了手",)})
    resolution = asyncio.run(resolver.resolve("洗了手", registry))
    assert resolution.token == "洗手"
    assert not resolution.created and not resolution.model_called
    assert resolution.validation_attempts == 0
    assert provider.calls == 0
    assert resolution.registry is registry


def test_resolver_bootstraps_empty_registry_without_model() -> None:
    resolver, provider = build_resolver([])
    resolution = asyncio.run(resolver.resolve("洗手", BehaviorKindRegistry()))
    assert resolution.token == "洗手" and resolution.created
    assert not resolution.model_called and provider.calls == 0
    assert resolution.registry.tokens == ("洗手",)


def test_resolver_reuses_token_verbatim_as_alias() -> None:
    resolver, provider = build_resolver([{"match": "洗手"}])
    registry = BehaviorKindRegistry({"洗手": (), "做饭": ()})
    resolution = asyncio.run(resolver.resolve("清洁双手", registry))
    assert resolution.token == "洗手" and not resolution.created
    assert resolution.model_called and provider.calls == 1
    assert resolution.registry.aliases_of("洗手") == ("清洁双手",)
    assert "【已有类型清单】" in provider.prompts[0]
    assert "清洁双手" in provider.prompts[0]


def test_resolver_null_match_creates_new_kind() -> None:
    resolver, provider = build_resolver([{"match": None}])
    registry = BehaviorKindRegistry({"洗手": ()})
    resolution = asyncio.run(resolver.resolve("做饭", registry))
    assert resolution.token == "做饭" and resolution.created and resolution.model_called
    assert resolution.registry.tokens == ("做饭", "洗手")
    assert provider.calls == 1


def test_resolver_retries_invalid_match_then_succeeds() -> None:
    resolver, provider = build_resolver([{"match": "没登记过"}, {"match": "洗手"}])
    registry = BehaviorKindRegistry({"洗手": ()})
    resolution = asyncio.run(resolver.resolve("清洁双手", registry))
    assert resolution.token == "洗手"
    assert resolution.validation_attempts == 2
    assert provider.calls == 2


def test_resolver_fails_after_exhausting_retries() -> None:
    resolver, provider = build_resolver([{"match": "没登记过"}])
    registry = BehaviorKindRegistry({"洗手": ()})
    with pytest.raises(ModelStructuredOutputError):
        asyncio.run(resolver.resolve("清洁双手", registry))
    assert provider.calls == 2


def test_resolver_enforces_kind_capacity_on_create() -> None:
    resolver, _provider = build_resolver(
        [{"match": None}], config=BehaviorKindConfig(max_kinds=1)
    )
    registry = BehaviorKindRegistry({"洗手": ()})
    with pytest.raises(BehaviorKindLimitError, match="capacity"):
        asyncio.run(resolver.resolve("做饭", registry))


def test_registry_coexists_with_the_tree_at_the_same_root(tmp_path) -> None:
    """kinds.md 并入地址空间：与树同根、互不干扰——枚举不见它、词表照常 CAS。"""

    from datetime import datetime, timedelta, timezone

    from behavior import BehaviorDocumentWriter, BehaviorKind, BehaviorTree
    from infrastructure.store.locks import ProcessLocalLockStore
    from tests.unit.behavior.tree_payloads import local, occurrence_payload

    tree = BehaviorTree(tmp_path / "behavior-tree")
    store = BehaviorKindStore(tmp_path / "behavior-tree")
    assert store.path == tree.root / "kinds.md"

    now = datetime(2026, 8, 16, 23, 0, tzinfo=timezone(timedelta(hours=8)))
    store.replace(BehaviorKindRegistry({"洗手": ()}), expected_revision=0, timestamp=now)

    writer = BehaviorDocumentWriter(tree, ProcessLocalLockStore(), clock=lambda: local(23, 0))
    writer.publish(BehaviorKind.OCCURRENCE, occurrence_payload())

    # 树的枚举只认 occurrences/gaps，词表文件不冒充节点
    assert len(tree.list_addresses(BehaviorKind.OCCURRENCE)) == 1
    assert tree.list_addresses(BehaviorKind.GAP) == ()
    # 词表在树写入之后照常读回与 CAS 演进
    snapshot = store.read()
    assert snapshot.revision == 1 and snapshot.registry.token_for("洗手") == "洗手"
    store.replace(snapshot.registry.with_new_kind("做饭"), expected_revision=1, timestamp=now)
    assert store.read().registry.token_for("做饭") == "做饭"
