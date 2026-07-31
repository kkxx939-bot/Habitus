"""Dense、目录分层、分数传播与 Reranker 组合召回场景矩阵。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

import pytest

from memory.indexing import MemoryVectorMatch
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from memory.retrieval.search import (
    MemorySearchMode,
    MemorySemanticSearchConfig,
    MemorySemanticSearchEngine,
)
from memory.uri import MemoryURI
from ModelClient import EmbeddingVector


class RecordingEmbedder:
    def __init__(self, result: object = None) -> None:
        self.result = EmbeddingVector((1.0, 0.0)) if result is None else result
        self.queries: list[str] = []

    async def embed_query(self, query: str):
        self.queries.append(query)
        return self.result


class ScriptedIndex:
    def __init__(
        self,
        *,
        searches: list[object] | None = None,
        children: dict[str, object] | None = None,
    ) -> None:
        self.searches = list(searches or [()])
        self.children = dict(children or {})
        self.search_calls: list[dict[str, object]] = []
        self.child_calls: list[dict[str, object]] = []

    async def search(self, _vector, **kwargs: object):
        self.search_calls.append(kwargs)
        current = self.searches.pop(0)
        if isinstance(current, BaseException):
            raise current
        return current

    async def search_children(self, _vector, **kwargs: object):
        self.child_calls.append(kwargs)
        parent = str(kwargs["parent"])
        current = self.children.get(parent, ())
        if isinstance(current, BaseException):
            raise current
        return current


class ScriptedReranker:
    provider_name = "test"
    model = "test"
    is_remote = False

    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def rerank(self, query: str, documents: Sequence[str]):
        values = tuple(documents)
        self.calls.append((query, values))
        current = self.results.pop(0)
        if isinstance(current, BaseException):
            raise current
        return current


def detail_match(
    name: str,
    score: float,
    *,
    kind: MemoryKind = MemoryKind.PREFERENCE,
    content: str | None = None,
) -> MemoryVectorMatch:
    if kind is MemoryKind.PROFILE:
        address = MemoryAddress.profile()
    elif kind is MemoryKind.ENTITY:
        address = MemoryAddress.entity("项目", name)
    elif kind is MemoryKind.TOOL:
        address = MemoryAddress.tool(name)
    elif kind is MemoryKind.INTENTION:
        address = MemoryAddress.intention(name)
    else:
        address = MemoryAddress.preference(name)
    return MemoryVectorMatch(
        MemoryURI.from_address(address),
        MemoryLevel.DETAIL,
        MemoryDirectory.for_address(address),
        content or f"{name} 的完整记忆",
        score,
    )


def layer_match(
    directory: MemoryDirectory,
    level: MemoryLevel,
    score: float,
    *,
    content: str = "目录语义",
) -> MemoryVectorMatch:
    return MemoryVectorMatch(
        MemoryURI.from_layer(directory, level),
        level,
        directory,
        content,
        score,
    )


def engine(
    *,
    index: ScriptedIndex | None = None,
    embedder: RecordingEmbedder | None = None,
    reranker: ScriptedReranker | None = None,
    config: MemorySemanticSearchConfig | None = None,
) -> MemorySemanticSearchEngine:
    return MemorySemanticSearchEngine(
        embedder=embedder or RecordingEmbedder(),
        index=index or ScriptedIndex(),
        reranker=reranker,
        config=config,
    )


def search(current: MemorySemanticSearchEngine, **overrides: object):
    values = {
        "query": "回答风格",
        "roots": (MemoryURI.root(),),
        "kinds": (),
        "intention_scope": MemoryIntentionRecallScope.ACTIVE,
        "limit": 5,
        **overrides,
    }
    return asyncio.run(current.search(**values))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"embedder": object()}, "embedder"),
        ({"index": object()}, "index"),
        ({"reranker": object()}, "reranker"),
    ],
)
def test_engine_rejects_each_invalid_collaborator(kwargs: dict[str, object], message: str) -> None:
    values = {
        "embedder": RecordingEmbedder(),
        "index": ScriptedIndex(),
        "reranker": None,
        **kwargs,
    }
    with pytest.raises(TypeError, match=message):
        MemorySemanticSearchEngine(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_multiplier", 0),
        ("candidate_multiplier", 11),
        ("min_vector_candidates", True),
        ("directory_candidates", 0),
        ("child_candidates", 10_001),
        ("max_directory_expansions", 0),
        ("max_rerank_candidates", 10_001),
        ("max_rerank_document_chars", 0),
        ("vector_score_threshold", math.nan),
        ("vector_score_threshold", 1.1),
        ("rerank_score_threshold", True),
        ("rerank_score_threshold", math.inf),
        ("score_propagation_alpha", -0.1),
        ("score_propagation_alpha", 1.1),
        ("rerank_hierarchy", 1),
    ],
)
def test_semantic_search_config_rejects_each_invalid_resource_boundary(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemorySemanticSearchConfig(**{field: value})


@pytest.mark.parametrize("query", [None, object(), "", "   ", 1, True])
def test_search_requires_non_empty_text_query(query: object) -> None:
    with pytest.raises(ValueError, match="query"):
        search(engine(), query=query)


@pytest.mark.parametrize(
    "roots",
    [
        None,
        [],
        (),
        (MemoryURI.from_address(MemoryAddress.profile()),),
        (MemoryURI.root(), MemoryURI.root()),
    ],
)
def test_search_requires_unique_directory_root_tuple(roots: object) -> None:
    with pytest.raises((TypeError, ValueError), match="root"):
        search(engine(), roots=roots)


@pytest.mark.parametrize("kinds", [None, [], [MemoryKind.PROFILE], (MemoryKind.PROFILE, MemoryKind.PROFILE)])
def test_search_requires_unique_kind_tuple(kinds: object) -> None:
    with pytest.raises((TypeError, ValueError), match="kinds"):
        search(engine(), kinds=kinds)


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5, "5", None])
def test_search_limit_is_strict_integer_within_public_bound(limit: object) -> None:
    with pytest.raises(ValueError, match="limit"):
        search(engine(), limit=limit)


def test_query_embedding_is_generated_once_for_direct_and_hierarchical_search() -> None:
    embedder = RecordingEmbedder()
    index = ScriptedIndex(searches=[(), ()])
    current = engine(index=index, embedder=embedder)

    assert search(current, query="  回答风格  ") == ()
    assert embedder.queries == ["回答风格"]
    assert len(index.search_calls) == 2


def test_invalid_query_embedding_aborts_before_index_search() -> None:
    index = ScriptedIndex()
    current = engine(index=index, embedder=RecordingEmbedder((1.0, 0.0)))

    with pytest.raises(TypeError, match="EmbeddingVector"):
        search(current)
    assert index.search_calls == []


def test_vector_mode_returns_sorted_thresholded_direct_l2_hits() -> None:
    low = detail_match("低分", -0.6)
    middle = detail_match("中分", 0.5)
    high = detail_match("高分", 0.9)
    index = ScriptedIndex(searches=[(low, high, middle)])
    current = engine(
        index=index,
        config=MemorySemanticSearchConfig(
            mode=MemorySearchMode.VECTOR,
            vector_score_threshold=0.0,
        ),
    )

    result = search(current, limit=2, kinds=(MemoryKind.PREFERENCE,))

    assert tuple(hit.uri for hit in result) == (high.uri, middle.uri)
    assert tuple(hit.score for hit in result) == (0.9, 0.5)
    call = index.search_calls[0]
    assert call["levels"] == (MemoryLevel.DETAIL,)
    assert call["kinds"] == (MemoryKind.PREFERENCE,)
    assert call["limit"] == 20


def test_default_vector_admission_rejects_negative_candidates_without_filling_limit() -> None:
    rejected = detail_match("无关", -0.01)
    admitted = detail_match("相关", 0.7)
    current = engine(
        index=ScriptedIndex(searches=[(rejected, admitted)]),
        config=MemorySemanticSearchConfig(mode=MemorySearchMode.VECTOR),
    )

    result = search(current, limit=5)

    assert tuple(hit.uri for hit in result) == (admitted.uri,)


def test_direct_candidates_deduplicate_by_uri_and_keep_highest_score() -> None:
    low = detail_match("相同", 0.2)
    high = detail_match("相同", 0.8, content="更相关版本")
    current = engine(
        index=ScriptedIndex(searches=[(low, high)]),
        config=MemorySemanticSearchConfig(mode=MemorySearchMode.VECTOR),
    )

    result = search(current)
    assert len(result) == 1
    assert result[0].score == 0.8


def test_direct_search_ignores_non_detail_matches_defensively() -> None:
    directory = MemoryDirectory.preferences()
    unexpected = layer_match(directory, MemoryLevel.ABSTRACT, 0.9)
    current = engine(
        index=ScriptedIndex(searches=[(unexpected,)]),
        config=MemorySemanticSearchConfig(mode=MemorySearchMode.VECTOR),
    )
    assert search(current) == ()


@pytest.mark.parametrize("invalid", ["bad", object(), (object(),)])
def test_index_result_must_be_sequence_of_vector_matches(invalid: object) -> None:
    current = engine(
        index=ScriptedIndex(searches=[invalid]),
        config=MemorySemanticSearchConfig(mode=MemorySearchMode.VECTOR),
    )
    with pytest.raises(TypeError, match="vector index"):
        search(current)


def test_hierarchical_search_starts_from_each_requested_root() -> None:
    preferences = MemoryURI.from_directory(MemoryDirectory.preferences())
    entities = MemoryURI.from_directory(MemoryDirectory.entities())
    preference = detail_match("回答风格", 0.8)
    entity = detail_match("m2bOS", 0.7, kind=MemoryKind.ENTITY)
    index = ScriptedIndex(
        searches=[(), ()],
        children={str(preferences): (preference,), str(entities): (entity,)},
    )
    current = engine(index=index)

    result = search(current, roots=(preferences, entities))
    assert tuple(hit.uri for hit in result) == (preference.uri, entity.uri)
    assert {str(call["parent"]) for call in index.child_calls} == {str(preferences), str(entities)}


def test_directory_match_queues_nested_directory_and_propagates_parent_score() -> None:
    root = MemoryURI.from_directory(MemoryDirectory.entities())
    category = MemoryDirectory.entities("项目")
    category_layer = layer_match(category, MemoryLevel.ABSTRACT, 0.9)
    target = detail_match("m2bOS", 0.5, kind=MemoryKind.ENTITY)
    index = ScriptedIndex(
        searches=[(), (category_layer,)],
        children={str(root): (), str(MemoryURI.from_directory(category)): (target,)},
    )
    current = engine(
        index=index,
        config=MemorySemanticSearchConfig(score_propagation_alpha=0.8),
    )

    result = search(current, roots=(root,))
    assert result[0].score == pytest.approx(0.58)
    assert result[0].vector_score == 0.5


def test_hierarchy_candidate_replaces_lower_scored_direct_candidate() -> None:
    root = MemoryURI.from_directory(MemoryDirectory.preferences())
    direct = detail_match("回答风格", 0.2)
    hierarchical = detail_match("回答风格", 0.9)
    index = ScriptedIndex(
        searches=[(direct,), ()],
        children={str(root): (hierarchical,)},
    )
    result = search(engine(index=index), roots=(root,))

    assert len(result) == 1
    assert result[0].score == 0.9
    assert result[0].vector_score == 0.9


def test_directory_expansion_bound_stops_deeper_traversal() -> None:
    root = MemoryURI.root()
    preferences = MemoryDirectory.preferences()
    child_layer = layer_match(preferences, MemoryLevel.ABSTRACT, 0.9)
    target = detail_match("回答风格", 0.9)
    index = ScriptedIndex(
        searches=[(), ()],
        children={
            str(root): (child_layer,),
            str(MemoryURI.from_directory(preferences)): (target,),
        },
    )
    current = engine(
        index=index,
        config=MemorySemanticSearchConfig(max_directory_expansions=1),
    )

    assert search(current) == ()
    assert len(index.child_calls) == 1


def test_final_reranker_receives_bounded_documents_and_controls_order() -> None:
    first = detail_match("A", 0.9, content="A" * 20)
    second = detail_match("B", 0.8, content="B" * 20)
    third = detail_match("C", 0.7, content="C" * 20)
    reranker = ScriptedReranker((0.1, 0.95))
    current = engine(
        index=ScriptedIndex(searches=[(first, second, third)]),
        reranker=reranker,
        config=MemorySemanticSearchConfig(
            mode=MemorySearchMode.VECTOR,
            max_rerank_candidates=2,
            max_rerank_document_chars=8,
            rerank_score_threshold=0.2,
        ),
    )

    result = search(current, limit=3)
    assert tuple(hit.uri for hit in result) == (second.uri,)
    assert result[0].score == result[0].rerank_score == 0.95
    assert result[0].vector_score == 0.8
    assert reranker.calls == [("回答风格", ("AAAAA...", "BBBBB..."))]


def test_default_rerank_admission_rejects_low_relevance_without_filling_limit() -> None:
    relevant = detail_match("相关", 0.7)
    hitchhiker = detail_match("搭便车", 0.9)
    reranker = ScriptedReranker((0.19, 0.91))
    current = engine(
        index=ScriptedIndex(searches=[(relevant, hitchhiker)]),
        reranker=reranker,
        config=MemorySemanticSearchConfig(mode=MemorySearchMode.VECTOR),
    )

    result = search(current, limit=5)

    assert tuple(hit.uri for hit in result) == (relevant.uri,)
    assert result[0].rerank_score == 0.91


def test_default_rerank_admission_can_return_zero_results() -> None:
    first = detail_match("A", 0.9)
    second = detail_match("B", 0.8)
    current = engine(
        index=ScriptedIndex(searches=[(first, second)]),
        reranker=ScriptedReranker((0.1, 0.19)),
        config=MemorySemanticSearchConfig(mode=MemorySearchMode.VECTOR),
    )

    assert search(current, limit=5) == ()


@pytest.mark.parametrize("scores", [[], [0.5], (0.5,), (0.5, math.nan)])
def test_final_reranker_invalid_output_falls_back_to_vector_scores(scores: object) -> None:
    matches = (detail_match("A", 0.9), detail_match("B", 0.8))
    current = engine(
        index=ScriptedIndex(searches=[matches]),
        reranker=ScriptedReranker(scores),
        config=MemorySemanticSearchConfig(mode=MemorySearchMode.VECTOR),
    )
    result = search(current)
    assert tuple(hit.vector_score for hit in result) == (0.9, 0.8)
    assert all(hit.rerank_score is None for hit in result)


def test_reranker_failure_uses_vector_admission_instead_of_accepting_every_candidate() -> None:
    admitted = detail_match("相关", 0.8)
    rejected = detail_match("无关", -0.1)
    current = engine(
        index=ScriptedIndex(searches=[(admitted, rejected)]),
        reranker=ScriptedReranker(RuntimeError("reranker unavailable")),
        config=MemorySemanticSearchConfig(mode=MemorySearchMode.VECTOR),
    )

    result = search(current, limit=5)

    assert tuple(hit.uri for hit in result) == (admitted.uri,)
    assert result[0].rerank_score is None


def test_hierarchy_reranker_scores_children_before_parent_propagation() -> None:
    root = MemoryURI.from_directory(MemoryDirectory.preferences())
    first = detail_match("A", 0.8)
    second = detail_match("B", 0.7)
    reranker = ScriptedReranker((0.1, 0.9), (0.7, 0.6))
    index = ScriptedIndex(
        searches=[(), ()],
        children={str(root): (first, second)},
    )
    current = engine(
        index=index,
        reranker=reranker,
        config=MemorySemanticSearchConfig(rerank_hierarchy=True),
    )

    result = search(current, roots=(root,))
    assert tuple(hit.uri for hit in result) == (second.uri, first.uri)
    assert len(reranker.calls) == 2


@pytest.mark.parametrize("scores", [[0.5], (0.5,), (math.inf, 0.5)])
def test_hierarchy_reranker_invalid_output_falls_back_to_vector_scores(scores: object) -> None:
    root = MemoryURI.from_directory(MemoryDirectory.preferences())
    children = (detail_match("A", 0.8), detail_match("B", 0.7))
    current = engine(
        index=ScriptedIndex(searches=[(), ()], children={str(root): children}),
        reranker=ScriptedReranker(scores),
        config=MemorySemanticSearchConfig(rerank_hierarchy=True),
    )
    result = search(current, roots=(root,))
    assert tuple(hit.vector_score for hit in result) == (0.8, 0.7)


@pytest.mark.parametrize(
    ("maximum", "expected"),
    [(1, "a"), (2, "ab"), (3, "abc"), (4, "a..."), (10, "abcdefghij")],
)
def test_rerank_text_truncation_is_deterministic(maximum: int, expected: str) -> None:
    current = engine(
        config=MemorySemanticSearchConfig(max_rerank_document_chars=maximum)
    )
    assert current._rerank_text("abcdefghij") == expected


@pytest.mark.parametrize(
    ("local", "parent", "alpha", "expected"),
    [(0.5, 0.0, 0.8, 0.5), (0.5, 1.0, 0.8, 0.6), (0.2, 0.8, 0.5, 0.5)],
)
def test_score_propagation_combines_local_and_parent_signal(
    local: float,
    parent: float,
    alpha: float,
    expected: float,
) -> None:
    current = engine(config=MemorySemanticSearchConfig(score_propagation_alpha=alpha))
    assert current._propagated(local, parent) == pytest.approx(expected)
