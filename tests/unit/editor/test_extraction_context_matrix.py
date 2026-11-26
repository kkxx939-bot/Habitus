"""记忆提取上下文的列目录、读取和关系扩展场景矩阵。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from habitus.infrastructure.editor.snapshot import SnapshotBatch, SnapshotReadConfig, VersionedSnapshot
from habitus.memory.document import MemoryLinkType, MemoryStoredLink
from habitus.memory.editor import (
    MemoryExtractionCapacityError,
    MemoryExtractionConfig,
    MemoryExtractionContext,
    MemoryExtractionError,
    MemoryRelatedContext,
    MemoryRetrievalAction,
    MemoryRetrievalDecision,
    MemoryRetrievalStatus,
)
from habitus.memory.editor.retrieval import MemoryNameEntry
from habitus.memory.model import MemoryAddress, MemoryKind
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from tests.helpers import codec, document


def finish_decision() -> MemoryRetrievalDecision:
    return MemoryRetrievalDecision(
        MemoryRetrievalStatus.SUFFICIENT, MemoryRetrievalAction.FINISH, None, (), "已有上下文足够。"
    )


def list_decision(directory: str) -> MemoryRetrievalDecision:
    return MemoryRetrievalDecision(
        MemoryRetrievalStatus.INSUFFICIENT, MemoryRetrievalAction.LIST, directory, (), "需要列目录。"
    )


def read_decision(*uris: str) -> MemoryRetrievalDecision:
    return MemoryRetrievalDecision(
        MemoryRetrievalStatus.INSUFFICIENT, MemoryRetrievalAction.READ, None, uris, "需要读取完整节点。"
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
    listing: tuple[MemoryURI, ...] = (),
    config: MemoryExtractionConfig | None = None,
    reader_config: SnapshotReadConfig | None = None,
) -> tuple[MemoryExtractionContext, MemoryTree, MemorySnapshotReader]:
    """使用真实 MemoryTree 和 SnapshotReader 构造提取上下文；``listing`` 是第一轮给出的名字清单。"""

    tree = MemoryTree(tmp_path / "memory")
    for current in documents:
        tree.write(current)
    reader = MemorySnapshotReader(tree, config=reader_config)
    snapshots = reader.read_many(selected)
    initial = MemoryRelatedContext(
        conversation_id="conversation-1",
        segment_id="segment-1",
        source_segment_digest="0" * 64,
        listing=tuple(MemoryNameEntry(uri) for uri in listing),
        snapshots=snapshots,
    )
    context = MemoryExtractionContext(
        initial,
        snapshot_reader=reader,
        config=config or MemoryExtractionConfig(),
    )
    return context, tree, reader


def preference(topic: str):
    return document(MemoryKind.PREFERENCE, fields={"topic": topic, "content": f"- {topic}"})


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


def test_initial_documents_receive_stable_page_ids_and_sorted_properties(tmp_path: Path) -> None:
    first = document(MemoryKind.PROFILE)
    second = document(MemoryKind.PREFERENCE)
    second_uri = MemoryURI.from_address(second.address)
    context, _tree, _reader = extraction_context(
        tmp_path,
        documents=(first, second),
        selected=(second_uri, MemoryURI.from_address(first.address)),
    )

    identities = tuple(snapshot.identity for snapshot in context.snapshots.snapshots)
    assert identities == tuple(sorted(identities))
    assert context.page_ids.existing_items() == tuple(enumerate(identities, start=1))
    assert set(context.allowed_read_uris) == set(identities)


def test_name_listing_exposes_its_uris_for_reading_without_reading_them(tmp_path: Path) -> None:
    target = preference("回答风格")
    uri = MemoryURI.from_address(target.address)
    context, _tree, _reader = extraction_context(tmp_path, documents=(target,), listing=(uri,))

    assert context.snapshots.snapshots == ()
    assert context.allowed_read_uris == (str(uri),)


def test_execute_rejects_finish_because_finish_has_no_tool_action(tmp_path: Path) -> None:
    context, _tree, _reader = extraction_context(tmp_path)
    with pytest.raises(ValueError, match="finish does not execute"):
        asyncio.run(context.execute(finish_decision(), iteration=1))


def test_list_returns_direct_cards_and_child_directories_and_exposes_cards(tmp_path: Path) -> None:
    top = preference("回答风格")
    entity = document(MemoryKind.ENTITY, fields={"category": "项目", "name": "Habitus", "summary": "记忆系统。"})
    context, _tree, _reader = extraction_context(tmp_path, documents=(top, entity))
    top_uri = str(MemoryURI.from_address(top.address))
    entity_uri = str(MemoryURI.from_address(entity.address))

    preferences = asyncio.run(context.execute(list_decision("habitus://preferences"), iteration=1))
    entities = asyncio.run(context.execute(list_decision("habitus://entities"), iteration=2))
    project = asyncio.run(context.execute(list_decision("habitus://entities/项目"), iteration=3))

    assert preferences.action is MemoryRetrievalAction.LIST
    assert preferences.inputs == ("habitus://preferences",)
    assert preferences.result_uris == (top_uri,)
    assert entities.result_uris == ("habitus://entities/项目",)
    assert project.result_uris == (entity_uri,)
    assert preferences.added_uris == () and project.added_uris == ()
    assert set(context.allowed_read_uris) == {top_uri, entity_uri}
    assert set(context.listed_directories) == {"habitus://preferences", "habitus://entities", "habitus://entities/项目"}
    assert context.snapshots.snapshots == ()


def test_repeated_list_is_cached(tmp_path: Path) -> None:
    target = preference("回答风格")
    context, _tree, _reader = extraction_context(tmp_path, documents=(target,))
    first = asyncio.run(context.execute(list_decision("habitus://preferences"), iteration=1))
    second = asyncio.run(context.execute(list_decision("habitus://preferences"), iteration=2))

    assert first.cached is False
    assert second.cached is True
    assert second.result_uris == first.result_uris


def test_list_of_missing_directory_is_empty(tmp_path: Path) -> None:
    context, _tree, _reader = extraction_context(tmp_path)
    observation = asyncio.run(context.execute(list_decision("habitus://tools"), iteration=1))
    assert observation.result_uris == ()


@pytest.mark.parametrize(
    ("directory", "message"),
    [
        ("invalid", "valid habitus:// directory"),
        ("habitus://profile.md", "requires a directory"),
        ("habitus://", "only lists"),
    ],
)
def test_list_rejects_invalid_non_directory_and_unlistable_targets(
    tmp_path: Path, directory: str, message: str
) -> None:
    context, _tree, _reader = extraction_context(tmp_path)
    with pytest.raises(MemoryExtractionError, match=message):
        asyncio.run(context.execute(list_decision(directory), iteration=1))


@pytest.mark.parametrize("uri", ["invalid", "habitus://preferences", "habitus://preferences/.overview.md"])
def test_read_requires_a_valid_l2_memory_uri(tmp_path: Path, uri: str) -> None:
    context, _tree, _reader = extraction_context(tmp_path)
    with pytest.raises(MemoryExtractionError, match="valid L2"):
        asyncio.run(context.execute(read_decision(uri), iteration=1))


def test_read_rejects_valid_uri_outside_listing_and_listed_directories(tmp_path: Path) -> None:
    listed = preference("已列出")
    hidden = preference("未暴露")
    context, _tree, _reader = extraction_context(
        tmp_path,
        documents=(listed, hidden),
        listing=(MemoryURI.from_address(listed.address),),
    )
    with pytest.raises(MemoryExtractionError, match="not exposed"):
        asyncio.run(context.execute(read_decision(str(MemoryURI.from_address(hidden.address))), iteration=1))
    assert context.snapshots.snapshots == ()


def test_read_of_listed_uris_adds_complete_snapshots_and_page_ids(tmp_path: Path) -> None:
    first = preference("A")
    second = preference("B")
    uris = tuple(MemoryURI.from_address(item.address) for item in (first, second))
    context, _tree, _reader = extraction_context(tmp_path, documents=(first, second), listing=uris)
    observation = asyncio.run(context.execute(read_decision(*(str(uri) for uri in uris)), iteration=1))

    assert observation.action is MemoryRetrievalAction.READ
    assert observation.inputs == tuple(str(uri) for uri in uris)
    assert set(observation.added_uris) == {str(uri) for uri in uris}
    assert {context.page_ids.page_id_for(uri) for uri in uris} == {1, 2}


def test_read_after_list_is_allowed(tmp_path: Path) -> None:
    target = preference("回答风格")
    uri = str(MemoryURI.from_address(target.address))
    context, _tree, _reader = extraction_context(tmp_path, documents=(target,))
    with pytest.raises(MemoryExtractionError, match="not exposed"):
        asyncio.run(context.execute(read_decision(uri), iteration=1))
    asyncio.run(context.execute(list_decision("habitus://preferences"), iteration=1))
    observation = asyncio.run(context.execute(read_decision(uri), iteration=2))
    assert observation.added_uris == (uri,)


def test_read_is_truncated_to_configured_reads_per_action(tmp_path: Path) -> None:
    items = tuple(preference(f"主题-{index}") for index in range(3))
    uris = tuple(str(MemoryURI.from_address(item.address)) for item in items)
    context, _tree, _reader = extraction_context(
        tmp_path,
        documents=items,
        listing=tuple(MemoryURI.parse(uri) for uri in uris),
        config=MemoryExtractionConfig(max_reads_per_action=2),
    )
    observation = asyncio.run(context.execute(read_decision(*uris), iteration=1))
    assert observation.inputs == uris[:2]
    assert len(context.snapshots.snapshots) == 2


def test_listed_uri_that_disappeared_reads_as_missing_without_page_id(tmp_path: Path) -> None:
    uri = MemoryURI.from_address(MemoryAddress.preference("已消失"))
    context, _tree, _reader = extraction_context(tmp_path, listing=(uri,))
    observation = asyncio.run(context.execute(read_decision(str(uri)), iteration=1))
    assert observation.added_uris == (str(uri),)
    assert context.page_ids.page_id_for(uri) is None


def test_read_of_prefetched_document_is_cached(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    context, _tree, _reader = extraction_context(
        tmp_path,
        documents=(target,),
        selected=(uri,),
    )
    observation = asyncio.run(context.execute(read_decision(str(uri)), iteration=3))

    assert observation.cached is True
    assert observation.added_uris == ()
    assert observation.result_uris == (str(uri),)


def test_read_beyond_item_limit_fails_as_capacity(tmp_path: Path) -> None:
    items = tuple(preference(f"主题-{index}") for index in range(2))
    uris = tuple(MemoryURI.from_address(item.address) for item in items)
    context, _tree, _reader = extraction_context(
        tmp_path,
        documents=items,
        listing=uris,
        config=MemoryExtractionConfig(max_old_memory_items=1),
        reader_config=SnapshotReadConfig(max_items=2, max_item_bytes=256_000, max_total_bytes=4_000_000),
    )
    with pytest.raises(MemoryExtractionCapacityError, match="item limit"):
        asyncio.run(context.execute(read_decision(*(str(uri) for uri in uris)), iteration=1))


def test_consistent_one_hop_relation_is_eagerly_read_and_exposed(tmp_path: Path) -> None:
    source, target, _relation = related_documents()
    source_uri = MemoryURI.from_address(source.address)
    target_uri = MemoryURI.from_address(target.address)
    context, _tree, _reader = extraction_context(
        tmp_path,
        documents=(source, target),
        selected=(source_uri,),
    )

    assert tuple(snapshot.identity for snapshot in context.snapshots.snapshots) == tuple(
        sorted((str(source_uri), str(target_uri)))
    )
    assert set(context.allowed_read_uris) == {str(source_uri), str(target_uri)}
    assert context.page_ids.page_ids() == frozenset({1, 2})


def test_read_expands_one_hop_relations_of_newly_read_document(tmp_path: Path) -> None:
    source, target, _relation = related_documents()
    source_uri = MemoryURI.from_address(source.address)
    target_uri = MemoryURI.from_address(target.address)
    context, _tree, _reader = extraction_context(tmp_path, documents=(source, target), listing=(source_uri,))
    observation = asyncio.run(context.execute(read_decision(str(source_uri)), iteration=1))

    assert observation.relation_expanded_uris == (str(target_uri),)
    assert {snapshot.identity for snapshot in context.snapshots.snapshots} == {str(source_uri), str(target_uri)}


def test_inconsistent_one_hop_relation_is_skipped_with_a_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """提取补一跳这条读路径：对方没记另一边的关联不再中止，记一条日志跳过（阶段 3，两边分开写）。"""

    source, target, _relation = related_documents(consistent=False)
    with caplog.at_level(logging.WARNING):
        context, _tree, _reader = extraction_context(
            tmp_path,
            documents=(source, target),
            selected=(MemoryURI.from_address(source.address),),
        )
    assert str(MemoryURI.from_address(source.address)) in context._snapshots
    assert "skipped" in caplog.text and str(MemoryURI.from_address(target.address)) in caplog.text


def test_missing_one_hop_relation_target_is_skipped_with_a_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source, target, _relation = related_documents()
    with caplog.at_level(logging.WARNING):
        context, _tree, _reader = extraction_context(
            tmp_path,
            documents=(source,),
            selected=(MemoryURI.from_address(source.address),),
        )
    missing = context._snapshots.get(str(MemoryURI.from_address(target.address)))
    assert missing is None or not missing.exists
    assert "skipped" in caplog.text and str(MemoryURI.from_address(target.address)) in caplog.text


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
    context, _tree, _reader = extraction_context(
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


def test_observation_size_limit_is_enforced_after_action(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    context, _tree, _reader = extraction_context(
        tmp_path,
        documents=(target,),
        config=MemoryExtractionConfig(max_observation_chars=1),
    )
    with pytest.raises(MemoryExtractionError, match="observation exceeds"):
        asyncio.run(context.execute(list_decision("habitus://preferences"), iteration=1))


def test_snapshots_and_page_ids_are_defensive_views(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    context, _tree, _reader = extraction_context(
        tmp_path,
        documents=(target,),
        selected=(uri,),
    )
    copied = context.page_ids
    copied.register_resolved(MemoryURI.from_address(MemoryAddress.preference("新节点")), 100)

    assert context.page_ids.resolve(100) is None
    assert context.snapshots.total_bytes == sum(item.size_bytes for item in context.snapshots.snapshots)


def test_existing_snapshot_with_mismatched_document_identity_is_rejected(tmp_path: Path) -> None:
    context, _tree, _reader = extraction_context(tmp_path)
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
        context.ingest(SnapshotBatch((wrong,), 1))


def test_reingesting_same_snapshot_is_idempotent_but_changed_snapshot_is_rejected(tmp_path: Path) -> None:
    target = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(target.address)
    context, _tree, reader = extraction_context(
        tmp_path,
        documents=(target,),
        selected=(uri,),
    )
    original = reader.read(uri)
    assert context.ingest(SnapshotBatch((original,), original.size_bytes)) == ()

    changed = replace(original, source_digest="f" * 64)
    with pytest.raises(MemoryExtractionError, match="changed during extraction"):
        context.ingest(SnapshotBatch((changed,), changed.size_bytes))


def test_names_cut_from_the_listing_by_its_token_budget_are_not_readable_until_listed(tmp_path: Path) -> None:
    """名字清单按 ``max_listing_tokens`` 截断：没给模型看到的地址不算暴露过，列过目录之后才能读。"""

    shown, cut = preference("看得到"), preference("被截掉的偏好主题名字很长" * 4)
    uris = tuple(MemoryURI.from_address(item.address) for item in (shown, cut))
    context, _tree, _reader = extraction_context(
        tmp_path,
        documents=(shown, cut),
        listing=uris,
        config=MemoryExtractionConfig(max_listing_tokens=len(str(uris[0])) // 2 + 4),
    )

    assert [entry.uri for entry in context.shown_listing] == [uris[0]]
    assert context.omitted_listing == 1
    with pytest.raises(MemoryExtractionError, match="not exposed"):
        asyncio.run(context.execute(read_decision(str(uris[1])), iteration=1))
    asyncio.run(context.execute(list_decision("habitus://preferences"), iteration=1))
    observation = asyncio.run(context.execute(read_decision(str(uris[1])), iteration=2))
    assert observation.added_uris == (str(uris[1]),)
