"""多查询、直接类型过滤、Links 一跳扩展和关系完整性的组合检索测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from memory.document import MemoryLinkType, MemoryStoredLink
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryKind
from memory.retrieval import (
    MemoryQueryPlan,
    MemorySearchError,
    MemorySearchHit,
    MemoryTypedQuery,
)
from memory.uri import MemoryURI
from tests.helpers import codec, document
from tests.unit.retrieval.test_search_service import SemanticSearch, SummarySearch, service


def linked_documents(*, second_kind: MemoryKind = MemoryKind.PREFERENCE, completed: bool = False):
    first = document(MemoryKind.PROFILE)
    if second_kind is MemoryKind.INTENTION:
        fields = {
            "intent_name": "补齐记忆测试",
            "status": "completed" if completed else "open",
            "next_step": "复核覆盖",
        }
        second = document(MemoryKind.INTENTION, fields=fields)
    else:
        second = document(second_kind)
    first_uri = MemoryURI.from_address(first.address)
    second_uri = MemoryURI.from_address(second.address)
    relation = MemoryStoredLink(first_uri, second_uri, MemoryLinkType.BELONGS_TO)
    document_codec = codec()
    linked_first = document_codec.build(
        first.kind,
        first.fields,
        metadata=first.metadata,
        links=(relation,),
    )
    linked_second = document_codec.build(
        second.kind,
        second.fields,
        metadata=second.metadata,
        backlinks=(relation,),
    )
    return linked_first, linked_second, relation


def test_kind_filter_applies_to_direct_top_k_but_does_not_cut_valid_one_hop_neighbor(tmp_path: Path) -> None:
    profile, preference, relation = linked_documents()
    profile_uri = MemoryURI.from_address(profile.address)
    semantic = SemanticSearch((MemorySearchHit(profile_uri, 0.95),))
    instance = service(tmp_path, semantic=semantic, summaries=SummarySearch())
    instance.tree.write(profile)
    instance.tree.write(preference)

    result = asyncio.run(instance.find("用户背景", kinds=(MemoryKind.PROFILE,), limit=1))

    assert len(result.memories) == 1
    assert result.memories[0].document.kind is MemoryKind.PROFILE
    assert result.memories[0].related[0].document.kind is MemoryKind.PREFERENCE
    assert result.memories[0].related[0].relation == relation
    assert semantic.calls[0][1]["kinds"] == (MemoryKind.PROFILE,)


def test_one_hop_expansion_does_not_recursively_expand_neighbor_relations(tmp_path: Path) -> None:
    profile, preference, first_link = linked_documents()
    entity = document(MemoryKind.ENTITY)
    preference_uri = MemoryURI.from_address(preference.address)
    entity_uri = MemoryURI.from_address(entity.address)
    second_link = MemoryStoredLink(preference_uri, entity_uri, MemoryLinkType.DERIVED_FROM)
    document_codec = codec()
    preference = document_codec.build(
        preference.kind,
        preference.fields,
        metadata=preference.metadata,
        links=(second_link,),
        backlinks=(first_link,),
    )
    entity = document_codec.build(
        entity.kind,
        entity.fields,
        metadata=entity.metadata,
        backlinks=(second_link,),
    )
    profile_uri = MemoryURI.from_address(profile.address)
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(profile_uri, 0.9),)),
        summaries=SummarySearch(),
    )
    for value in (profile, preference, entity):
        instance.tree.write(value)

    result = asyncio.run(instance.find("用户是谁"))

    assert tuple(item.document.kind for item in result.memories[0].related) == (MemoryKind.PREFERENCE,)
    assert entity_uri not in {
        MemoryURI.from_address(item.document.address) for item in result.memories[0].related
    }


def test_inconsistent_or_missing_backlink_fails_closed_instead_of_returning_partial_relation(
    tmp_path: Path,
) -> None:
    profile, preference, _relation = linked_documents()
    broken_preference = codec().build(
        preference.kind,
        preference.fields,
        metadata=preference.metadata,
    )
    profile_uri = MemoryURI.from_address(profile.address)
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(profile_uri, 0.9),)),
        summaries=SummarySearch(),
    )
    instance.tree.write(profile)
    instance.tree.write(broken_preference)

    with pytest.raises(MemorySearchError, match="Links/Backlinks are inconsistent"):
        asyncio.run(instance.find("用户是谁"))


def test_completed_intention_is_excluded_only_as_active_relation_neighbor(tmp_path: Path) -> None:
    profile, intention, _relation = linked_documents(
        second_kind=MemoryKind.INTENTION,
        completed=True,
    )
    profile_uri = MemoryURI.from_address(profile.address)
    class ScopeSearch(SemanticSearch):
        async def search(self, query, **kwargs):
            self.calls.append((query, kwargs))
            if kwargs["intention_scope"] is MemoryIntentionRecallScope.COMPLETED:
                return ()
            return (MemorySearchHit(profile_uri, 0.9),)

    instance = service(tmp_path, semantic=ScopeSearch(), summaries=SummarySearch())
    instance.tree.write(profile)
    instance.tree.write(intention)

    active = asyncio.run(instance.find("用户是谁"))
    completed = asyncio.run(
        instance.find(
            "历史完成事项",
            intention_scope=MemoryIntentionRecallScope.COMPLETED,
        )
    )

    assert active.memories[0].related == ()
    assert completed.intention_scope is MemoryIntentionRecallScope.COMPLETED


def test_multi_query_failure_aborts_whole_result_instead_of_returning_partial_memory(tmp_path: Path) -> None:
    profile = document(MemoryKind.PROFILE)
    uri = MemoryURI.from_address(profile.address)

    class PerQuerySearch:
        async def search(self, query, **_kwargs):
            if query == "第二查询":
                raise TimeoutError("vector timeout")
            return (MemorySearchHit(uri, 0.9),)

    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    instance.semantic_search = PerQuerySearch()
    instance.tree.write(profile)
    plan = MemoryQueryPlan(
        "原始问题",
        (
            MemoryTypedQuery("第一查询", "身份", 1),
            MemoryTypedQuery("第二查询", "背景", 2),
        ),
    )

    with pytest.raises(MemorySearchError, match="semantic search failed"):
        asyncio.run(
            instance._execute(
                plan,
                (MemoryURI.root(),),
                2,
                None,
                (),
                MemoryIntentionRecallScope.ACTIVE,
                assess_for_summary=False,
            )
        )


def test_semantic_backend_cannot_ignore_root_kind_or_requested_result_bound(tmp_path: Path) -> None:
    preference = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(preference.address)
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)),
        summaries=SummarySearch(),
    )
    instance.tree.write(preference)

    with pytest.raises(MemorySearchError, match="kind filter"):
        asyncio.run(instance.find("用户是谁", kinds=(MemoryKind.PROFILE,)))

    class TooMany:
        async def search(self, _query, **_kwargs):
            return tuple(MemorySearchHit(uri, 0.9) for _ in range(100))

    instance.semantic_search = TooMany()
    with pytest.raises(MemorySearchError, match="requested result limit"):
        asyncio.run(instance.find("用户偏好", limit=1))
