"""记忆提取上下文的受控检索、读取和关系扩展场景矩阵。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from infrastructure.editor.snapshot import SnapshotBatch, SnapshotReadConfig, VersionedSnapshot
from memory.document import MemoryLinkType, MemoryStoredLink
from memory.editor import (
    MemoryExtractionConfig,
    MemoryExtractionContext,
    MemoryExtractionError,
    MemoryRelatedContext,
    MemoryRetrievalAction,
    MemoryRetrievalDecision,
    MemoryRetrievalStatus,
)
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind
from memory.retrieval import MemorySearchHit
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI
from tests.helpers import codec, document


class RecordingSearch:
    """返回排队结果，并记录每次附加语义检索的完整约束。"""

    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def search(self, query: str, **kwargs: object) -> object:
        self.calls.append((query, kwargs))
        current = self.results.pop(0) if self.results else ()
        if isinstance(current, BaseException):
            raise current
        return current


def retrieval_decision(
    action: MemoryRetrievalAction,
    *,
    value: str = "补充查询",
) -> MemoryRetrievalDecision:
    """构造一个满足动作互斥规则的检索决策。"""

    if action is MemoryRetrievalAction.FINISH:
        return MemoryRetrievalDecision(
            MemoryRetrievalStatus.SUFFICIENT,
            action,
            None,
            None,
            "已有上下文足够。",
        )
    if action is MemoryRetrievalAction.SEARCH:
        return MemoryRetrievalDecision(
            MemoryRetrievalStatus.INSUFFICIENT,
            action,
            value,
            None,
            "需要补充搜索。",
        )
    return MemoryRetrievalDecision(
        MemoryRetrievalStatus.INSUFFICIENT,
        action,
        None,
        value,
        "需要读取完整节点。",
    )


def related_documents(*, consistent: bool = True):
    """构造带有一条双向持久关系的偏好和实体文档。"""

    source_base = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 偏好简洁回答"},
    )
    target_base = document(
        MemoryKind.ENTITY,
        fields={"category": "项目", "name": "Habitus", "summary": "用户正在重构的记忆系统。"},
    )
    relation = MemoryStoredLink(
        MemoryURI.from_address(source_base.address),
        MemoryURI.from_address(target_base.address),
        MemoryLinkType.DERIVED_FROM,
    )
    source = codec().build(
        source_base.kind,
        source_base.fields,
        metadata=source_base.metadata,
        links=(relation,),
    )
    target = codec().build(
        target_base.kind,
        target_base.fields,
        metadata=target_base.metadata,
        backlinks=(relation,) if consistent else (),
    )
    return source, target, relation


def extraction_context(
    tmp_path: Path,
    *,
    documents=(),
    selected=(),
    hits: tuple[MemorySearchHit, ...] = (),
    roots: tuple[MemoryURI, ...] | None = None,
    query: str = "初始查询",
    search: RecordingSearch | None = None,
    config: MemoryExtractionConfig | None = None,
    reader_config: SnapshotReadConfig | None = None,
) -> tuple[MemoryExtractionContext, MemoryTree, MemorySnapshotReader, RecordingSearch]:
    """使用真实 MemoryTree 和 SnapshotReader 构造提取上下文。"""

    tree = MemoryTree(tmp_path / "memory")
    for current in documents:
        tree.write(current)
    reader = MemorySnapshotReader(tree, config=reader_config)
    snapshots = reader.read_many(selected)
    initial = MemoryRelatedContext(
        conversation_id="conversation-1",
        segment_id="segment-1",
        source_segment_digest="0" * 64,
        query=query,
        search_roots=(MemoryURI.from_directory(MemoryDirectory.preferences()),)
        if roots is None
        else roots,
        search_hits=hits,
        snapshots=snapshots,
    )
    semantic_search = search or RecordingSearch()
    context = MemoryExtractionContext(
        initial,
        snapshot_reader=reader,
        semantic_search=semantic_search,
        config=config or MemoryExtractionConfig(),
    )
    return context, tree, reader, semantic_search


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"initial": object()}, "initial"),
        ({"snapshot_reader": object()}, "snapshot_reader"),
        ({"semantic_search": object()}, "semantic_search"),
        ({"config": object()}, "config"),
    ],
)
def test_context_rejects_each_invalid_collaborator(
    tmp_path: Path,
    replacement: dict[str, object],
    message: str,
) -> None:
    tree = MemoryTree(tmp_path / "memory")
    reader = MemorySnapshotReader(tree)
    initial = MemoryRelatedContext(
        "conversation-1",
        "segment-1",
        "0" * 64,
        "初始查询",
        (MemoryURI.from_directory(MemoryDirectory.preferences()),),
        (),
        reader.read_many(()),
    )
    values = {
        "initial": initial,
        "snapshot_reader": reader,
        "semantic_search": RecordingSearch(),
        "config": MemoryExtractionConfig(),
        **replacement,
    }
    with pytest.raises(TypeError, match=message):
        MemoryExtractionContext(**values)


@pytest.mark.parametrize(
    ("field", "reader_config", "expected"),
    [
        (
            "max_old_memory_items",
            SnapshotReadConfig(max_items=1, max_item_bytes=100, max_total_bytes=100),
            "item limit",
        ),
        (
            "max_old_memory_bytes",
            SnapshotReadConfig(max_items=64, max_item_bytes=100, max_total_bytes=100),
            "byte limit",
        ),
    ],
)
def test_context_limit_cannot_exceed_snapshot_reader_limit(
    tmp_path: Path,
    field: str,
    reader_config: SnapshotReadConfig,
    expected: str,
) -> None:
    config = MemoryExtractionConfig(**{field: 2 if field == "max_old_memory_items" else 101})
    with pytest.raises(ValueError, match=expected):
        extraction_context(tmp_path, config=config, reader_config=reader_config)


def test_initial_hit_requires_a_complete_existing_snapshot(tmp_path: Path) -> None:
    uri = MemoryURI.from_address(MemoryAddress.preference("不存在"))
    hit = MemorySearchHit(uri, 0.8)
    with pytest.raises(MemoryExtractionError, match="no complete existing snapshot"):
        extraction_context(tmp_path, selected=(uri,), hits=(hit,))


def test_initial_documents_receive_stable_page_ids_and_sorted_properties(tmp_path: Path) -> None:
    first = document(MemoryKind.PROFILE)
    second = document(MemoryKind.PREFERENCE)
    second_uri = MemoryURI.from_address(second.address)
    hit = MemorySearchHit(second_uri, 0.8)
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        documents=(first, second),
        selected=(second_uri, MemoryURI.from_address(first.address)),
        hits=(hit,),
    )

    identities = tuple(snapshot.identity for snapshot in context.snapshots.snapshots)
    assert identities == tuple(sorted(identities))
    assert context.page_ids.existing_items() == tuple(enumerate(identities, start=1))
    assert context.search_hits == (hit,)
    assert set(context.allowed_read_uris) == set(identities)


def test_initial_query_is_cached_without_calling_semantic_search(tmp_path: Path) -> None:
    search = RecordingSearch(AssertionError("缓存查询不应调用后端"))
    context, _tree, _reader, _search = extraction_context(tmp_path, search=search, query="  初始   查询  ")
    observation = asyncio.run(
        context.execute(retrieval_decision(MemoryRetrievalAction.SEARCH, value="初始 查询"), iteration=1)
    )

    assert observation.cached is True
    assert observation.result_uris == ()
    assert search.calls == []


@pytest.mark.parametrize("invalid", [None, object(), "search", 1, True])
def test_execute_requires_a_retrieval_decision(tmp_path: Path, invalid: object) -> None:
    context, _tree, _reader, _search = extraction_context(tmp_path)
    with pytest.raises(TypeError, match="decision"):
        asyncio.run(context.execute(invalid, iteration=1))  # type: ignore[arg-type]


def test_execute_rejects_finish_because_finish_has_no_tool_action(tmp_path: Path) -> None:
    context, _tree, _reader, _search = extraction_context(tmp_path)
    with pytest.raises(ValueError, match="finish does not execute"):
        asyncio.run(context.execute(retrieval_decision(MemoryRetrievalAction.FINISH), iteration=1))


def test_additional_search_uses_all_intentions_and_configured_scope(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    hit = MemorySearchHit(uri, 0.75)
    search = RecordingSearch((hit,))
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        documents=(target,),
        search=search,
        config=MemoryExtractionConfig(additional_search_limit=7),
    )
    observation = asyncio.run(
        context.execute(retrieval_decision(MemoryRetrievalAction.SEARCH, value="  回答   风格 "), iteration=2)
    )

    assert observation.result_uris == (str(uri),)
    assert observation.added_uris == (str(uri),)
    assert context.page_ids.page_id_for(uri) == 1
    assert search.calls == [
        (
            "回答 风格",
            {
                "roots": (MemoryURI.from_directory(MemoryDirectory.preferences()),),
                "kinds": (),
                "intention_scope": MemoryIntentionRecallScope.ALL,
                "limit": 7,
            },
        )
    ]


def test_repeated_additional_query_is_cached_and_keeps_best_global_hit(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    search = RecordingSearch((MemorySearchHit(uri, 0.6),))
    context, _tree, _reader, _search = extraction_context(tmp_path, documents=(target,), search=search)
    decision = retrieval_decision(MemoryRetrievalAction.SEARCH, value="回答风格")

    first = asyncio.run(context.execute(decision, iteration=1))
    second = asyncio.run(context.execute(decision, iteration=2))

    assert first.cached is False
    assert second.cached is True
    assert len(search.calls) == 1
    assert context.search_hits == (MemorySearchHit(uri, 0.6),)


def test_additional_search_deduplicates_uri_using_highest_score(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    search = RecordingSearch((MemorySearchHit(uri, 0.2), MemorySearchHit(uri, 0.9)))
    context, _tree, _reader, _search = extraction_context(tmp_path, documents=(target,), search=search)
    observation = asyncio.run(
        context.execute(retrieval_decision(MemoryRetrievalAction.SEARCH), iteration=1)
    )

    assert observation.result_uris == (str(uri),)
    assert context.search_hits == (MemorySearchHit(uri, 0.9),)


def test_search_fits_new_hits_to_remaining_item_slots_by_relevance(tmp_path: Path) -> None:
    first = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "A", "content": "- A"},
    )
    second = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "B", "content": "- B"},
    )
    first_hit = MemorySearchHit(MemoryURI.from_address(first.address), 0.9)
    second_hit = MemorySearchHit(MemoryURI.from_address(second.address), 0.8)
    search = RecordingSearch((first_hit, second_hit))
    config = MemoryExtractionConfig(max_old_memory_items=1)
    reader_config = SnapshotReadConfig(max_items=1, max_item_bytes=256_000, max_total_bytes=4_000_000)
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        documents=(first, second),
        search=search,
        config=config,
        reader_config=reader_config,
    )
    observation = asyncio.run(
        context.execute(retrieval_decision(MemoryRetrievalAction.SEARCH), iteration=1)
    )

    assert observation.result_uris == (str(first_hit.uri),)
    assert context.search_hits == (first_hit,)


@pytest.mark.parametrize(
    ("raw_result", "message"),
    [
        (RuntimeError("provider down"), "semantic search failed"),
        ("invalid", "sequence of hits"),
        ((object(),), "invalid hit"),
    ],
)
def test_additional_search_wraps_backend_and_result_contract_failures(
    tmp_path: Path,
    raw_result: object,
    message: str,
) -> None:
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        search=RecordingSearch(raw_result),
    )
    with pytest.raises(MemoryExtractionError, match=message):
        asyncio.run(
            context.execute(retrieval_decision(MemoryRetrievalAction.SEARCH), iteration=1)
        )


def test_additional_search_rejects_more_hits_than_requested(tmp_path: Path) -> None:
    documents = tuple(
        document(MemoryKind.PREFERENCE, fields={"topic": f"主题-{index}", "content": f"- {index}"})
        for index in range(3)
    )
    hits = tuple(MemorySearchHit(MemoryURI.from_address(item.address), 0.9 - index / 10) for index, item in enumerate(documents))
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        documents=documents,
        search=RecordingSearch(hits),
        config=MemoryExtractionConfig(additional_search_limit=2),
    )
    with pytest.raises(MemoryExtractionError, match="requested limit"):
        asyncio.run(context.execute(retrieval_decision(MemoryRetrievalAction.SEARCH), iteration=1))


def test_additional_search_rejects_out_of_scope_hit(tmp_path: Path) -> None:
    target = document(MemoryKind.ENTITY)
    hit = MemorySearchHit(MemoryURI.from_address(target.address), 0.8)
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        documents=(target,),
        search=RecordingSearch((hit,)),
    )
    with pytest.raises(MemoryExtractionError, match="out-of-scope"):
        asyncio.run(context.execute(retrieval_decision(MemoryRetrievalAction.SEARCH), iteration=1))


def test_additional_search_rejects_hit_that_disappeared_before_snapshot_read(tmp_path: Path) -> None:
    uri = MemoryURI.from_address(MemoryAddress.preference("已消失"))
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        search=RecordingSearch((MemorySearchHit(uri, 0.8),)),
    )
    with pytest.raises(MemoryExtractionError, match="disappeared before read"):
        asyncio.run(context.execute(retrieval_decision(MemoryRetrievalAction.SEARCH), iteration=1))


def test_search_without_allowed_roots_fails_before_backend_call(tmp_path: Path) -> None:
    search = RecordingSearch(AssertionError("不应访问后端"))
    context, _tree, _reader, _search = extraction_context(tmp_path, roots=(), search=search)
    with pytest.raises(MemoryExtractionError, match="no allowed memory roots"):
        asyncio.run(context.execute(retrieval_decision(MemoryRetrievalAction.SEARCH), iteration=1))
    assert search.calls == []


@pytest.mark.parametrize("uri", ["invalid", "memory://preferences", "memory://preferences/.overview.md"])
def test_read_requires_a_valid_l2_memory_uri(tmp_path: Path, uri: str) -> None:
    context, _tree, _reader, _search = extraction_context(tmp_path)
    with pytest.raises(MemoryExtractionError, match="valid L2"):
        asyncio.run(
            context.execute(retrieval_decision(MemoryRetrievalAction.READ, value=uri), iteration=1)
        )


def test_read_rejects_valid_but_unexposed_uri(tmp_path: Path) -> None:
    context, _tree, _reader, _search = extraction_context(tmp_path)
    uri = str(MemoryURI.from_address(MemoryAddress.preference("未暴露")))
    with pytest.raises(MemoryExtractionError, match="not exposed"):
        asyncio.run(
            context.execute(retrieval_decision(MemoryRetrievalAction.READ, value=uri), iteration=1)
        )


def test_read_of_prefetched_document_is_cached(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        documents=(target,),
        selected=(uri,),
    )
    observation = asyncio.run(
        context.execute(retrieval_decision(MemoryRetrievalAction.READ, value=str(uri)), iteration=3)
    )

    assert observation.cached is True
    assert observation.added_uris == ()
    assert observation.result_uris == (str(uri),)


def test_consistent_one_hop_relation_is_eagerly_read_and_exposed(tmp_path: Path) -> None:
    source, target, _relation = related_documents()
    source_uri = MemoryURI.from_address(source.address)
    target_uri = MemoryURI.from_address(target.address)
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        documents=(source, target),
        selected=(source_uri,),
    )

    assert tuple(snapshot.identity for snapshot in context.snapshots.snapshots) == tuple(
        sorted((str(source_uri), str(target_uri)))
    )
    assert set(context.allowed_read_uris) == {str(source_uri), str(target_uri)}
    assert context.page_ids.page_ids() == frozenset({1, 2})


def test_inconsistent_one_hop_relation_aborts_context_construction(tmp_path: Path) -> None:
    source, target, _relation = related_documents(consistent=False)
    with pytest.raises(MemoryExtractionError, match="not bidirectionally consistent"):
        extraction_context(
            tmp_path,
            documents=(source, target),
            selected=(MemoryURI.from_address(source.address),),
        )


def test_missing_one_hop_relation_target_aborts_context_construction(tmp_path: Path) -> None:
    source, _target, _relation = related_documents()
    with pytest.raises(MemoryExtractionError, match="does not exist"):
        extraction_context(
            tmp_path,
            documents=(source,),
            selected=(MemoryURI.from_address(source.address),),
        )


def test_relation_expansion_respects_per_seed_limit(tmp_path: Path) -> None:
    source_base = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "风格", "content": "- 简洁"},
    )
    targets = tuple(
        document(
            MemoryKind.ENTITY,
            fields={"category": "项目", "name": f"项目-{index}", "summary": f"项目 {index}"},
        )
        for index in range(2)
    )
    source_uri = MemoryURI.from_address(source_base.address)
    links = tuple(
        MemoryStoredLink(source_uri, MemoryURI.from_address(item.address), MemoryLinkType.DERIVED_FROM)
        for item in targets
    )
    source = codec().build(
        source_base.kind,
        source_base.fields,
        metadata=source_base.metadata,
        links=links,
    )
    linked_targets = tuple(
        codec().build(
            item.kind,
            item.fields,
            metadata=item.metadata,
            backlinks=(link,),
        )
        for item, link in zip(targets, links, strict=True)
    )
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        documents=(source, *linked_targets),
        selected=(source_uri,),
        config=MemoryExtractionConfig(
            max_relation_neighbors_per_seed=1,
            max_relation_neighbors_total=2,
        ),
    )

    identities = tuple(snapshot.identity for snapshot in context.snapshots.snapshots)
    assert len(identities) == 2
    assert str(source_uri) in identities


def test_context_detects_memory_revision_change_during_search(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    initial_hit = MemorySearchHit(uri, 0.5)
    search = RecordingSearch((MemorySearchHit(uri, 0.9),))
    context, tree, _reader, _search = extraction_context(
        tmp_path,
        documents=(target,),
        selected=(uri,),
        hits=(initial_hit,),
        search=search,
    )
    updated = codec().build(
        target.kind,
        target.fields,
        metadata=target.metadata.next_revision(target.metadata.updated_at),
    )
    tree.write(updated)

    with pytest.raises(MemoryExtractionError, match="changed during extraction"):
        asyncio.run(
            context.execute(
                retrieval_decision(MemoryRetrievalAction.SEARCH, value="不同查询"),
                iteration=1,
            )
        )


@pytest.mark.parametrize("query", ["", "   ", "x" * 6])
def test_search_query_is_non_empty_and_bounded(tmp_path: Path, query: str) -> None:
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        config=MemoryExtractionConfig(max_query_chars=5),
    )
    with pytest.raises(MemoryExtractionError, match="query"):
        context._query(query)


def test_observation_size_limit_is_enforced_after_action(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        documents=(target,),
        search=RecordingSearch((MemorySearchHit(uri, 0.8),)),
        config=MemoryExtractionConfig(max_observation_chars=1),
    )
    with pytest.raises(MemoryExtractionError, match="observation exceeds"):
        asyncio.run(context.execute(retrieval_decision(MemoryRetrievalAction.SEARCH), iteration=1))


def test_snapshots_and_page_ids_are_defensive_views(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    context, _tree, _reader, _search = extraction_context(
        tmp_path,
        documents=(target,),
        selected=(uri,),
    )
    copied = context.page_ids
    copied.register_new(MemoryURI.from_address(MemoryAddress.preference("新节点")), 100)

    assert context.page_ids.resolve(100) is None
    assert context.snapshots.total_bytes == sum(
        item.size_bytes for item in context.snapshots.snapshots
    )


def test_existing_snapshot_with_mismatched_document_identity_is_rejected(tmp_path: Path) -> None:
    context, _tree, _reader, _search = extraction_context(tmp_path)
    target = document(MemoryKind.PREFERENCE)
    wrong = VersionedSnapshot(
        identity=str(MemoryURI.from_address(MemoryAddress.preference("错误身份"))),
        state="found",
        value=target,
        revision=target.metadata.revision,
        source_digest="0" * 64,
        size_bytes=1,
    )
    with pytest.raises(MemoryExtractionError, match="identity does not match"):
        context._ingest(SnapshotBatch((wrong,), 1))


def test_reingesting_same_snapshot_is_idempotent_but_changed_snapshot_is_rejected(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    context, _tree, reader, _search = extraction_context(
        tmp_path,
        documents=(target,),
        selected=(uri,),
    )
    original = reader.read(uri)
    assert context._ingest(SnapshotBatch((original,), original.size_bytes)) == ()

    changed = replace(original, source_digest="f" * 64)
    with pytest.raises(MemoryExtractionError, match="changed during extraction"):
        context._ingest(SnapshotBatch((changed,), changed.size_bytes))
