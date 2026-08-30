"""行为类型词表：登记约束、命中账与存活期、单文件耐久、向量旁册、批量模型归属。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any

import pytest

from behavior.kinds import (
    HIT_DAYS_KEPT,
    BehaviorKindConfig,
    BehaviorKindConflictError,
    BehaviorKindEntry,
    BehaviorKindError,
    BehaviorKindLimitError,
    BehaviorKindRegistry,
    BehaviorKindRequest,
    BehaviorKindResolver,
    BehaviorKindStore,
    BehaviorKindStoreError,
    BehaviorKindVectorIndex,
    BehaviorKindVectorStore,
)
from behavior.kinds.vectors import literal_kinds, nearest_kinds
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
from ModelClient.embedding import EmbeddingVector

NOW = datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc)
D1 = date(2026, 8, 20)
D2 = date(2026, 8, 22)


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


class FakeEmbedder:
    """按字表给确定性向量：共享字越多越近（够做候选检索的形状测试）。"""

    provider_name = "fake"
    model = "fake-embed"
    is_remote = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    ALPHABET = "洗手碗做饭吃药看电视打电话与医生通话交谈"
    DIMENSION = len(ALPHABET) + 1

    @classmethod
    def _vector(cls, text: str) -> EmbeddingVector:
        values = [float(text.count(ch)) for ch in cls.ALPHABET] + [0.01]
        return EmbeddingVector(values=tuple(values))

    async def embed_query(self, text: str) -> EmbeddingVector:
        return self._vector(text)

    async def embed_documents(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        self.calls.append(tuple(texts))
        return tuple(self._vector(text) for text in texts)

    async def aclose(self) -> None:  # pragma: no cover
        return None


def build_resolver(
    bodies: list[dict[str, Any]],
    *,
    config: BehaviorKindConfig | None = None,
    embedder: FakeEmbedder | None = None,
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
    return BehaviorKindResolver(client, config=config, embedder=embedder), provider


def items(*rows: tuple[str, str | None, str | None]) -> dict[str, Any]:
    return {"items": [{"name": n, "match": m, "label": label} for n, m, label in rows]}


# --- 登记约束 -------------------------------------------------------------------------


def test_registry_orders_and_resolves_by_canonical_identity() -> None:
    registry = BehaviorKindRegistry({"洗手": ("洗了手", "清洁双手"), "Wash Dishes": ()})
    assert registry.tokens == ("Wash Dishes", "洗手")
    assert registry.token_for("清洁双手") == "洗手"
    assert registry.token_for("洗手") == "洗手"
    assert registry.token_for("wash dishes") == "Wash Dishes"
    assert registry.token_for("做饭") is None
    # 别名序列表示从未命中的登记条目：label 默认等于 token，不参与过期
    entry = registry.entry_of("洗手")
    assert entry.label == "洗手" and entry.hit_days == () and entry.last_hit_day is None


def test_registry_rejects_identity_collisions() -> None:
    with pytest.raises(BehaviorKindError, match="collides"):
        BehaviorKindRegistry({"洗手": ("洗了手",), "做饭": ("洗了手",)})
    with pytest.raises(BehaviorKindError, match="collides"):
        BehaviorKindRegistry({"洗手": ("做饭",), "做饭": ()})
    with pytest.raises(BehaviorKindError, match="collides"):
        BehaviorKindRegistry({"Wash": (), "wash": ()})


def test_registry_mutations_return_new_instances() -> None:
    base = BehaviorKindRegistry({"洗手": ()})
    grown = base.with_new_kind("做饭", label="做饭").with_alias("洗手", "洗了手")
    assert base.kind_count == 1 and base.aliases_of("洗手") == ()
    assert grown.kind_count == 2 and grown.aliases_of("洗手") == ("洗了手",)
    with pytest.raises(BehaviorKindError, match="not registered"):
        base.with_alias("做饭", "备菜")
    with pytest.raises(BehaviorKindError, match="collides"):
        grown.with_alias("做饭", "洗手")


# --- 命中账与存活期 ---------------------------------------------------------------------


def test_hit_account_counts_distinct_days_and_accepts_late_days() -> None:
    entry = BehaviorKindEntry(token="洗手", label="洗手")
    entry = entry.with_hit(D2).with_hit(D2)  # 同一天两次：一天、两次
    assert entry.hit_days == (D2,) and entry.hit_days_total == 1 and entry.hit_count == 2
    assert entry.created_on == D2
    late = entry.with_hit(D1)  # 晚到的更早一天插入重排
    assert late.hit_days == (D1, D2) and late.max_gap_days == 2 and late.created_on == D2


def test_hit_days_are_bounded_to_the_most_recent() -> None:
    entry = BehaviorKindEntry(token="洗手", label="洗手")
    for offset in range(HIT_DAYS_KEPT + 5):
        entry = entry.with_hit(date.fromordinal(D1.toordinal() + offset))
    assert len(entry.hit_days) == HIT_DAYS_KEPT
    assert entry.hit_days_total == HIT_DAYS_KEPT + 5
    assert entry.hit_days[0] == date.fromordinal(D1.toordinal() + 5)


def test_expiry_follows_the_kind_own_rhythm() -> None:
    weekly = BehaviorKindEntry(token="打球", label="打球")
    for offset in (0, 7, 14):
        weekly = weekly.with_hit(date.fromordinal(D1.toordinal() + offset))
    once = BehaviorKindEntry(token="卷账单", label="卷账单").with_hit(D1)
    registry = BehaviorKindRegistry({"打球": weekly, "卷账单": once, "预置": ()})
    # 一次性名字：基础期（30 天）后删；周频行为：max(30, 3×7=21)=30 天，从最近命中算
    on = date.fromordinal(D1.toordinal() + 31)
    assert registry.expired(on=on, base_days=30, gap_multiplier=3) == ("卷账单",)
    on = date.fromordinal(D1.toordinal() + 14 + 31)
    assert set(registry.expired(on=on, base_days=30, gap_multiplier=3)) == {"卷账单", "打球"}
    # 月频行为按自己的间隔续命：间隔 30 天 → 90 天内不删
    monthly = BehaviorKindEntry(token="交房租", label="交房租").with_hit(D1).with_hit(
        date.fromordinal(D1.toordinal() + 30)
    )
    deadline = monthly.expires_after(base_days=30, gap_multiplier=3)
    assert deadline == date.fromordinal(D1.toordinal() + 30 + 90)
    # 从未命中的预置条目不过期
    assert "预置" not in registry.expired(on=date(2030, 1, 1), base_days=30, gap_multiplier=3)


def test_merge_folds_aliases_and_account_into_the_target() -> None:
    a = BehaviorKindEntry(token="与医生通话", label="与医生通话", aliases=("挂号电话",)).with_hit(D1)
    b = BehaviorKindEntry(token="打电话", label="打电话").with_hit(D2)
    registry = BehaviorKindRegistry({"与医生通话": a, "打电话": b}).merged("与医生通话", "打电话")
    assert registry.tokens == ("打电话",)
    merged = registry.entry_of("打电话")
    assert set(merged.aliases) == {"与医生通话", "挂号电话"}
    assert merged.hit_days == (D1, D2) and merged.hit_count == 2
    assert registry.token_for("挂号电话") == "打电话"


# --- 单文件耐久 ------------------------------------------------------------------------


def test_store_roundtrip_and_missing_file(tmp_path) -> None:
    store = BehaviorKindStore(tmp_path)
    empty = store.read()
    assert empty.revision == 0 and empty.registry.kind_count == 0
    registry = BehaviorKindRegistry({"洗手": ("洗了手",)}).with_hit("洗手", D1).with_label("洗手", "清洁双手")
    written = store.replace(registry, expected_revision=0, timestamp=NOW)
    assert written.revision == 1
    loaded = store.read()
    assert loaded.revision == 1 and loaded.registry == registry
    assert loaded.registry.entry_of("洗手").label == "清洁双手"
    assert loaded.registry.entry_of("洗手").hit_days == (D1,)
    assert loaded.updated_at == NOW
    assert "清洁双手〔洗手〕 ×1 最近 2026-08-20：洗了手" in (tmp_path / "kinds.md").read_text()


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


def test_store_rejects_the_v1_format(tmp_path) -> None:
    """v1（只有正名+别名）不做读取兼容：旧根走 rebuild 补齐。"""

    path = tmp_path / "kinds.md"
    v1 = {
        "schema_version": "behavior_kinds_v1",
        "revision": 1,
        "updated_at": "2026-08-22T12:00:00.000000Z",
        "kinds": [{"token": "洗手", "aliases": []}],
    }
    path.write_text(
        "# 行为类型词表\n\n- 洗手\n<!-- HABITUS_BEHAVIOR_KINDS\n" + json.dumps(v1) + "\n-->\n",
        encoding="utf-8",
    )
    with pytest.raises(BehaviorKindStoreError, match="unsupported"):
        BehaviorKindStore(tmp_path).read()


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


# --- 向量旁册与候选 ------------------------------------------------------------------


def test_vector_store_roundtrip_and_model_mismatch(tmp_path) -> None:
    store = BehaviorKindVectorStore(tmp_path, model="m1", dimension=3)
    index = store.empty().with_vectors({"洗手": (3.0, 0.0, 4.0)})
    store.replace(index)
    loaded = store.read()
    assert loaded.has("洗手")
    unit = loaded.vectors["洗手"]
    assert abs(unit[0] - 0.6) < 1e-3 and abs(unit[2] - 0.8) < 1e-3  # 单位化后 float16 往返
    # 换了 embedding 模型：旁册作废，按空索引处理（派生物，可重算）
    assert not BehaviorKindVectorStore(tmp_path, model="m2", dimension=3).read().has("洗手")


def test_nearest_kinds_ranks_by_the_best_of_token_and_label() -> None:
    registry = BehaviorKindRegistry({"洗手": ("洗了手",), "做饭": ()}).with_label("做饭", "烹饪")
    index = BehaviorKindVectorIndex("m", 2, {"洗手": (1.0, 0.0), "烹饪": (0.0, 1.0)})
    assert nearest_kinds((0.9, 0.1), registry, index, limit=2) == ("洗手", "做饭")
    assert nearest_kinds((0.1, 0.9), registry, index, limit=1) == ("做饭",)
    # 没有向量的 kind 不参与
    assert nearest_kinds((1.0, 0.0), registry, BehaviorKindIndexEmpty(), limit=5) == ()


def BehaviorKindIndexEmpty() -> BehaviorKindVectorIndex:
    return BehaviorKindVectorIndex("m", 2)


def test_literal_kinds_is_the_fallback_ranking() -> None:
    registry = BehaviorKindRegistry({"洗手": ("洗了手",), "做饭": (), "洗碗": ()})
    assert literal_kinds("洗了手", registry, limit=1) == ("洗手",)
    assert set(literal_kinds("洗碗筷", registry, limit=2)) == {"洗碗", "洗手"}


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


def test_resolver_records_a_hit_on_the_fast_path() -> None:
    resolver, provider = build_resolver([])
    registry = BehaviorKindRegistry({"洗手": ("洗了手",)})
    batch = asyncio.run(
        resolver.resolve_many((BehaviorKindRequest("洗了手", day=D1), BehaviorKindRequest("洗手", day=D2)), registry)
    )
    assert provider.calls == 0 and batch.tokens == {"洗了手": "洗手", "洗手": "洗手"}
    assert batch.registry.entry_of("洗手").hit_days == (D1, D2)


def test_resolver_bootstraps_empty_registry_without_model() -> None:
    resolver, provider = build_resolver([])
    resolution = asyncio.run(resolver.resolve("洗手", BehaviorKindRegistry()))
    assert resolution.token == "洗手" and resolution.created
    assert not resolution.model_called and provider.calls == 0
    assert resolution.registry.tokens == ("洗手",)


def test_resolver_reuses_candidate_verbatim_as_alias_and_records_hit() -> None:
    resolver, provider = build_resolver([items(("N1", "K1", None))])
    registry = BehaviorKindRegistry({"洗手": (), "做饭": ()})
    batch = asyncio.run(
        resolver.resolve_many((BehaviorKindRequest("清洁双手", day=D1, evidence="在水池边搓手"),), registry)
    )
    assert batch.tokens == {"清洁双手": "洗手"} and not batch.created
    assert batch.model_calls == 1 and provider.calls == 1
    assert batch.registry.aliases_of("洗手") == ("清洁双手",)
    assert batch.registry.entry_of("洗手").hit_days == (D1,)
    prompt = provider.prompts[0]
    assert "【候选类型】" in prompt and "K1  洗手" in prompt and "K2  做饭" in prompt
    assert "N1  清洁双手  候选：K1、K2  证据：在水池边搓手" in prompt


def test_resolver_null_match_creates_new_kind_with_the_model_label() -> None:
    resolver, provider = build_resolver([items(("N1", None, "做饭"))])
    registry = BehaviorKindRegistry({"洗手": ()})
    batch = asyncio.run(resolver.resolve_many((BehaviorKindRequest("做晚饭", day=D1),), registry))
    assert batch.tokens == {"做晚饭": "做晚饭"} and batch.created == ("做晚饭",)
    assert batch.registry.tokens == ("做晚饭", "洗手")
    assert batch.registry.label_of("做晚饭") == "做饭"  # token 是原话，label 是模型给的可读名
    assert provider.calls == 1


def test_resolver_lets_batch_names_point_at_each_other() -> None:
    """同批里「与Katrina交流」指向「与Alice交流」：只建一个 kind，后者成别名。"""

    resolver, provider = build_resolver([items(("N1", None, "交谈"), ("N2", "N1", None))])
    registry = BehaviorKindRegistry({"洗手": ()})
    batch = asyncio.run(
        resolver.resolve_many(
            (BehaviorKindRequest("与Alice交流", day=D1), BehaviorKindRequest("与Katrina交流", day=D1)),
            registry,
        )
    )
    assert batch.tokens == {"与Alice交流": "与Alice交流", "与Katrina交流": "与Alice交流"}
    assert batch.created == ("与Alice交流",)
    assert batch.registry.aliases_of("与Alice交流") == ("与Katrina交流",)
    assert provider.calls == 1


def test_resolver_reasks_only_the_offending_names() -> None:
    """第一次 N2 越界（指向不是它自己候选的 K）、N1 合法：只重问 N2。"""

    resolver, provider = build_resolver(
        [
            items(("N1", "K1", None), ("N2", "K9", None)),  # K9 不存在 → 结构违约，整批打回一次
            items(("N1", "K1", None), ("N2", None, None)),  # N2 null 却没 label → 只重问 N2
            items(("N1", None, "散步")),  # 重问时 N2 变成了唯一的 N1
        ],
        config=BehaviorKindConfig(validation_rounds=2),
    )
    registry = BehaviorKindRegistry({"洗手": ()})
    batch = asyncio.run(
        resolver.resolve_many(
            (BehaviorKindRequest("清洁双手", day=D1), BehaviorKindRequest("出门走走", day=D1)), registry
        )
    )
    assert batch.tokens == {"清洁双手": "洗手", "出门走走": "出门走走"}
    assert batch.registry.label_of("出门走走") == "散步"
    assert provider.calls == 3
    assert any("kind_validation_rejected" in note for note in batch.signals)


def test_resolver_exhausted_reasks_fall_back_to_a_new_kind() -> None:
    resolver, provider = build_resolver(
        [items(("N1", None, None))], config=BehaviorKindConfig(validation_rounds=1)
    )
    registry = BehaviorKindRegistry({"洗手": ()})
    batch = asyncio.run(resolver.resolve_many((BehaviorKindRequest("擦桌子", day=D1),), registry))
    assert batch.tokens == {"擦桌子": "擦桌子"} and batch.created == ("擦桌子",)
    assert batch.registry.label_of("擦桌子") == "擦桌子"
    assert provider.calls == 2
    assert any("kind_validation_exhausted" in note for note in batch.signals)


def test_resolver_fails_when_the_model_never_returns_the_structure() -> None:
    resolver, provider = build_resolver([{"match": "洗手"}])  # 旧形状：结构层拒绝
    registry = BehaviorKindRegistry({"洗手": ()})
    with pytest.raises(ModelStructuredOutputError):
        asyncio.run(resolver.resolve("清洁双手", registry))
    assert provider.calls == 2


def test_resolver_enforces_kind_capacity_on_create() -> None:
    resolver, _provider = build_resolver(
        [items(("N1", None, "做饭"))], config=BehaviorKindConfig(max_kinds=1)
    )
    registry = BehaviorKindRegistry({"洗手": ()})
    with pytest.raises(BehaviorKindLimitError, match="capacity"):
        asyncio.run(resolver.resolve("做饭", registry))


def test_resolver_uses_embeddings_for_candidates_and_fills_the_index() -> None:
    embedder = FakeEmbedder()
    resolver, provider = build_resolver(
        [items(("N1", "K1", None))],
        config=BehaviorKindConfig(vector_candidates=1, frequent_candidates=0),
        embedder=embedder,
    )
    registry = BehaviorKindRegistry({"洗手": (), "看电视": ()})
    batch = asyncio.run(
        resolver.resolve_many(
            (BehaviorKindRequest("洗了手", day=D1),), registry, vectors=BehaviorKindVectorIndex("fake-embed", FakeEmbedder.DIMENSION)
        )
    )
    assert batch.tokens == {"洗了手": "洗手"}
    # 候选只给了向量最近的一个：洗手（共享"洗""手"），看电视没进候选
    assert "K1  洗手" in provider.prompts[0] and "看电视" not in provider.prompts[0]
    # 词表里缺向量的 token 顺带补齐进旁册
    assert batch.vectors is not None and batch.vectors.has("洗手") and batch.vectors.has("看电视")
    assert embedder.calls and "洗了手" in embedder.calls[0]


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


class _FlakyThenScriptedProvider(ScriptedProvider):
    """第一次调用断连，之后按脚本回放：resolver 自己吃掉瞬态错误，归约整轮不因此作废。"""

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        super().__init__(bodies)
        self.failed_once = False

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
        from ModelClient.contracts import ModelTransportError

        if not self.failed_once:
            self.failed_once = True
            raise ModelTransportError("ReadTimeout")
        return await super().complete_async(request)


def test_resolver_retries_a_transient_error_then_resolves() -> None:
    provider = _FlakyThenScriptedProvider([items(("N1", None, "擦桌子"))])
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
    resolver = BehaviorKindResolver(
        StructuredChatClient(ChatClient(model_config, provider), validation_retries=1),
        config=BehaviorKindConfig(transient_retries=2, transient_retry_delay_seconds=0.0),
    )
    registry = BehaviorKindRegistry({"洗手": ()})
    resolution = asyncio.run(resolver.resolve("擦桌子", registry))
    assert resolution.created is True and resolution.token == "擦桌子"
    assert provider.calls == 1 and provider.failed_once is True
