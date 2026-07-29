"""SearchService 服务边界、后备检索和关系容量的风险矩阵。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memory.document import MemoryLinkType, MemoryStoredLink
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryDirectory, MemoryKind
from memory.retrieval import (
    MemoryContextAssembler,
    MemoryRetrievalGrader,
    MemorySearchError,
    MemorySearchHit,
    MemorySearchQueryPlanner,
    MemorySearchServiceConfig,
    SearchService,
)
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI
from tests.helpers import codec, document
from tests.unit.retrieval.test_search_service import (
    SemanticSearch,
    SummarySearch,
    conversation_context_reader,
    service,
    structured,
    summary_match,
)


def _constructor_arguments(instance: SearchService) -> dict[str, object]:
    return {
        "tree": instance.tree,
        "snapshot_reader": instance.snapshot_reader,
        "semantic_search": instance.semantic_search,
        "summary_search": instance.summary_search,
        "query_planner": instance.query_planner,
        "retrieval_grader": instance.retrieval_grader,
        "conversation_context": instance.conversation_context,
        "assembler": instance.assembler,
        "config": instance.config,
        "intention_reviewer": instance.intention_reviewer,
        "clock": instance.clock,
    }


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("tree", object(), "tree must be MemoryTree"),
        ("snapshot_reader", object(), "snapshot_reader must be MemorySnapshotReader"),
        ("semantic_search", object(), "semantic_search must implement"),
        ("summary_search", object(), "summary_search must implement"),
        ("query_planner", object(), "query_planner must be"),
        ("retrieval_grader", object(), "retrieval_grader must be"),
        ("conversation_context", object(), "conversation_context must be"),
        ("assembler", object(), "assembler must be"),
        ("config", object(), "config must be"),
        ("intention_reviewer", object(), "intention_reviewer must be"),
        ("clock", 1, "clock must be callable"),
    ],
)
def test_constructor_rejects_invalid_collaborator(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    arguments = _constructor_arguments(instance)
    arguments[field] = invalid
    with pytest.raises((TypeError, ValueError), match=message):
        SearchService(**arguments)  # type: ignore[arg-type]


def test_constructor_rejects_snapshot_reader_from_another_tree(tmp_path: Path) -> None:
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    arguments = _constructor_arguments(instance)
    arguments["snapshot_reader"] = MemorySnapshotReader(MemoryTree(tmp_path / "other"))
    with pytest.raises(ValueError, match="share one memory tree"):
        SearchService(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("component", ["query_planner", "retrieval_grader", "conversation_context", "assembler"])
def test_constructor_rejects_component_with_different_config(tmp_path: Path, component: str) -> None:
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    arguments = _constructor_arguments(instance)
    other = replace(instance.config, default_limit=instance.config.default_limit + 1)
    if component == "query_planner":
        arguments[component] = MemorySearchQueryPlanner(structured([]), config=other)
    elif component == "retrieval_grader":
        arguments[component] = MemoryRetrievalGrader(structured([]), config=other)
    elif component == "conversation_context":
        arguments[component] = conversation_context_reader(tmp_path / "other-context", other)
    else:
        arguments[component] = MemoryContextAssembler(config=other)
    with pytest.raises(ValueError, match="share one search config"):
        SearchService(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["", " ", "\n\t", 1, None])
def test_find_rejects_empty_or_non_text_query(tmp_path: Path, query: object) -> None:
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    with pytest.raises(ValueError, match="query must be non-empty"):
        asyncio.run(instance.find(query))  # type: ignore[arg-type]


def test_find_strips_query_and_applies_default_root_limit_and_scope(tmp_path: Path) -> None:
    semantic = SemanticSearch()
    instance = service(tmp_path, semantic=semantic, summaries=SummarySearch())
    result = asyncio.run(instance.find("  用户是谁  "))
    assert result.query == "用户是谁"
    assert result.target_roots == (MemoryURI.root(),)
    assert semantic.calls == [
        (
            "用户是谁",
            {
                "roots": (MemoryURI.root(),),
                "kinds": (),
                "intention_scope": MemoryIntentionRecallScope.ACTIVE,
                "limit": instance.config.default_limit * instance.config.candidate_multiplier,
            },
        )
    ]


@pytest.mark.parametrize("limit", [0, -1, 21, True, 1.5, "1"])
def test_find_rejects_invalid_limit(tmp_path: Path, limit: object) -> None:
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    with pytest.raises(ValueError, match="limit is outside"):
        asyncio.run(instance.find("查询", limit=limit))  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [True, "0.5", float("nan"), float("inf"), -float("inf")])
def test_find_rejects_invalid_score_threshold(tmp_path: Path, threshold: object) -> None:
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    with pytest.raises(ValueError, match="score_threshold must be finite"):
        asyncio.run(instance.find("查询", score_threshold=threshold))  # type: ignore[arg-type]


def test_score_threshold_filters_before_reading_l2(tmp_path: Path) -> None:
    preference = document()
    semantic = SemanticSearch((MemorySearchHit(MemoryURI.from_address(preference.address), 0.49),))
    instance = service(tmp_path, semantic=semantic, summaries=SummarySearch())
    instance.tree.write(preference)
    result = asyncio.run(instance.find("偏好", score_threshold=0.5))
    assert result.memories == ()
    assert result.query_results[0].hits == ()


@pytest.mark.parametrize("target", [(), [], 1, (MemoryURI.from_address(document().address),)])
def test_find_rejects_invalid_target_roots(tmp_path: Path, target: object) -> None:
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(instance.find("查询", target_uris=target))  # type: ignore[arg-type]


def test_roots_are_deduplicated_sorted_and_remove_nested_children(tmp_path: Path) -> None:
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    entities = MemoryURI.from_directory(MemoryDirectory.entities())
    category = MemoryURI("memory://entities/项目")
    result = asyncio.run(instance.find("项目", target_uris=(category, entities, category)))
    assert result.target_roots == (entities,)


def test_root_count_is_checked_before_deduplication(tmp_path: Path) -> None:
    config = replace(MemorySearchServiceConfig(), max_target_roots=2)
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch(), config=config)
    repeated = (MemoryURI.root(), MemoryURI.root(), MemoryURI.root())
    with pytest.raises(ValueError, match="root count"):
        asyncio.run(instance.find("查询", target_uris=repeated))


@pytest.mark.parametrize("kinds", [[], (MemoryKind.PROFILE, MemoryKind.PROFILE)])
def test_find_rejects_invalid_or_duplicate_kind_filter(tmp_path: Path, kinds: object) -> None:
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(instance.find("查询", kinds=kinds))  # type: ignore[arg-type]


@pytest.mark.parametrize("raw", ["hits", 1, object()])
def test_semantic_backend_must_return_a_sequence(tmp_path: Path, raw: object) -> None:
    class InvalidSemantic:
        async def search(self, _query, **_kwargs):
            return raw

    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    instance.semantic_search = InvalidSemantic()
    with pytest.raises(MemorySearchError, match="must return a sequence"):
        asyncio.run(instance.find("查询"))


def test_semantic_backend_hit_items_are_strict(tmp_path: Path) -> None:
    instance = service(tmp_path, semantic=SemanticSearch((object(),)), summaries=SummarySearch())
    with pytest.raises(MemorySearchError, match="invalid hit"):
        asyncio.run(instance.find("查询"))


def test_semantic_backend_cannot_return_out_of_scope_uri(tmp_path: Path) -> None:
    profile = document(MemoryKind.PROFILE)
    uri = MemoryURI.from_address(profile.address)
    instance = service(tmp_path, semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)), summaries=SummarySearch())
    instance.tree.write(profile)
    with pytest.raises(MemorySearchError, match="out-of-scope"):
        asyncio.run(instance.find("身份", target_uris="memory://preferences"))


def test_duplicate_semantic_hits_keep_highest_score_and_stable_order(tmp_path: Path) -> None:
    first = document(MemoryKind.PROFILE)
    second = document(MemoryKind.PREFERENCE)
    first_uri = MemoryURI.from_address(first.address)
    second_uri = MemoryURI.from_address(second.address)
    semantic = SemanticSearch(
        (
            MemorySearchHit(first_uri, 0.3),
            MemorySearchHit(second_uri, 0.8),
            MemorySearchHit(first_uri, 0.9),
        )
    )
    instance = service(tmp_path, semantic=semantic, summaries=SummarySearch())
    instance.tree.write(first)
    instance.tree.write(second)
    result = asyncio.run(instance.find("记忆"))
    assert tuple(item.uri for item in result.memories) == (first_uri, second_uri)
    assert tuple(item.hit.score for item in result.memories) == (0.9, 0.8)


def test_missing_l2_document_from_index_fails_closed(tmp_path: Path) -> None:
    uri = MemoryURI.from_address(document().address)
    instance = service(tmp_path, semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)), summaries=SummarySearch())
    with pytest.raises(MemorySearchError, match="missing or invalid L2"):
        asyncio.run(instance.find("偏好"))


def test_snapshot_read_failure_is_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uri = MemoryURI.from_address(document().address)
    instance = service(tmp_path, semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)), summaries=SummarySearch())

    def fail(_uris):
        raise OSError("disk unavailable")

    monkeypatch.setattr(instance.snapshot_reader, "read_many", fail)
    with pytest.raises(MemorySearchError, match="failed to read complete"):
        asyncio.run(instance.find("偏好"))


@pytest.mark.parametrize("clock_value", ["now", datetime(2026, 7, 1)])
def test_search_clock_must_return_timezone_aware_datetime(tmp_path: Path, clock_value: object) -> None:
    instance = service(
        tmp_path,
        semantic=SemanticSearch(),
        summaries=SummarySearch(),
        clock=lambda: clock_value,  # type: ignore[return-value]
    )
    with pytest.raises((TypeError, ValueError), match="clock must return"):
        asyncio.run(instance.find("查询"))


def test_timezone_clock_is_normalized_for_active_intention_review(tmp_path: Path) -> None:
    intention = document(MemoryKind.INTENTION)
    uri = MemoryURI.from_address(intention.address)
    now = datetime(2026, 7, 10, 16, tzinfo=timezone.utc)
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)),
        summaries=SummarySearch(),
        clock=lambda: now,
    )
    instance.tree.write(intention)
    result = asyncio.run(instance.find("待办"))
    assert result.memories[0].intention_review is not None
    assert result.memories[0].intention_review.unconfirmed_days == 9


@pytest.mark.parametrize("raw", ["matches", 1, object()])
def test_invalid_summary_backend_degrades_to_memory_only(tmp_path: Path, raw: object) -> None:
    class InvalidSummary:
        async def search(self, _query, *, limit):
            return raw

    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    instance.summary_search = InvalidSummary()
    result = asyncio.run(instance.search("历史"))
    assert result.summary_fallbacks == ()
    assert result.degradations[0].stage.value == "summary_fallback"


def test_summary_backend_failure_degrades_to_memory_only(tmp_path: Path) -> None:
    class FailingSummary:
        async def search(self, _query, *, limit):
            raise TimeoutError("summary timeout")

    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    instance.summary_search = FailingSummary()
    result = asyncio.run(instance.search("历史"))
    assert result.summary_fallbacks == ()
    assert result.degradations[0].stage.value == "summary_fallback"


def test_summary_backend_over_limit_degrades_to_memory_only(tmp_path: Path) -> None:
    config = replace(MemorySearchServiceConfig(), summary_fallback_limit=1)
    summaries = SummarySearch((summary_match(), summary_match()))
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=summaries, config=config)
    result = asyncio.run(instance.search("历史"))
    assert result.summary_fallbacks == ()
    assert result.degradations[0].stage.value == "summary_fallback"


def test_summary_backend_invalid_matches_degrade_to_memory_only(tmp_path: Path) -> None:
    match = summary_match()
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch((match, match)))
    result = asyncio.run(instance.search("历史"))
    assert result.degradations[0].stage.value == "summary_fallback"
    instance.summary_search = SummarySearch((object(),))
    result = asyncio.run(instance.search("历史"))
    assert result.degradations[0].stage.value == "summary_fallback"


def test_disabled_summary_fallback_still_grades_but_never_queries_summary(tmp_path: Path) -> None:
    config = replace(MemorySearchServiceConfig(), summary_fallback_enabled=False)
    summaries = SummarySearch((summary_match(),))
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=summaries, config=config)
    result = asyncio.run(instance.search("历史"))
    assert result.retrieval_assessment is not None
    assert not result.summary_fallback_attempted
    assert result.summary_fallbacks == ()
    assert summaries.calls == []


def _linked_preferences(count: int):
    seed = document(MemoryKind.PROFILE)
    seed_uri = MemoryURI.from_address(seed.address)
    neighbors = [
        document(MemoryKind.PREFERENCE, fields={"topic": f"主题-{index}", "content": f"- 偏好-{index}"})
        for index in range(count)
    ]
    links = tuple(
        MemoryStoredLink(seed_uri, MemoryURI.from_address(item.address), MemoryLinkType.BELONGS_TO)
        for item in neighbors
    )
    built_seed = codec().build(seed.kind, seed.fields, metadata=seed.metadata, links=links)
    built_neighbors = tuple(
        codec().build(item.kind, item.fields, metadata=item.metadata, backlinks=(link,))
        for item, link in zip(neighbors, links, strict=True)
    )
    return built_seed, built_neighbors


@pytest.mark.parametrize(
    ("per_match", "total", "expected"),
    [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)],
)
def test_relation_expansion_respects_per_match_and_total_capacity(
    tmp_path: Path,
    per_match: int,
    total: int,
    expected: int,
) -> None:
    config = replace(
        MemorySearchServiceConfig(),
        max_relation_neighbors_per_match=per_match,
        max_relation_neighbors_total=total,
    )
    seed, neighbors = _linked_preferences(3)
    uri = MemoryURI.from_address(seed.address)
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)),
        summaries=SummarySearch(),
        config=config,
    )
    instance.tree.write(seed)
    for neighbor in neighbors:
        instance.tree.write(neighbor)
    result = asyncio.run(instance.find("身份"))
    assert len(result.memories[0].related) == expected


def test_relation_neighbor_outside_target_root_is_not_loaded(tmp_path: Path) -> None:
    seed = document(MemoryKind.PREFERENCE)
    neighbor = document(MemoryKind.ENTITY)
    seed_uri = MemoryURI.from_address(seed.address)
    neighbor_uri = MemoryURI.from_address(neighbor.address)
    relation = MemoryStoredLink(seed_uri, neighbor_uri, MemoryLinkType.RELATED_TO)
    if relation.from_uri == seed_uri:
        seed = codec().build(seed.kind, seed.fields, metadata=seed.metadata, links=(relation,))
        neighbor = codec().build(neighbor.kind, neighbor.fields, metadata=neighbor.metadata, backlinks=(relation,))
    else:
        seed = codec().build(seed.kind, seed.fields, metadata=seed.metadata, backlinks=(relation,))
        neighbor = codec().build(neighbor.kind, neighbor.fields, metadata=neighbor.metadata, links=(relation,))
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(seed_uri, 0.9),)),
        summaries=SummarySearch(),
    )
    instance.tree.write(seed)
    instance.tree.write(neighbor)

    result = asyncio.run(
        instance.find(
            "回答偏好",
            target_uris=MemoryURI.from_directory(MemoryDirectory.preferences()),
        )
    )

    assert len(result.memories) == 1
    assert result.memories[0].uri == seed_uri
    assert result.memories[0].related == ()


def test_relation_to_missing_neighbor_fails_closed(tmp_path: Path) -> None:
    seed, _neighbors = _linked_preferences(1)
    uri = MemoryURI.from_address(seed.address)
    instance = service(tmp_path, semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)), summaries=SummarySearch())
    instance.tree.write(seed)
    with pytest.raises(MemorySearchError, match="missing L2"):
        asyncio.run(instance.find("身份"))


def test_relation_snapshot_failure_is_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed, neighbors = _linked_preferences(1)
    uri = MemoryURI.from_address(seed.address)
    instance = service(tmp_path, semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)), summaries=SummarySearch())
    instance.tree.write(seed)
    instance.tree.write(neighbors[0])
    original = instance.snapshot_reader.read
    neighbor_uri = MemoryURI.from_address(neighbors[0].address)

    def fail_neighbor(target):
        if MemoryURI.parse(target) == neighbor_uri:
            raise OSError("disk unavailable")
        return original(target)

    monkeypatch.setattr(instance.snapshot_reader, "read", fail_neighbor)
    with pytest.raises(MemorySearchError, match="one-hop"):
        asyncio.run(instance.find("身份"))


def test_directly_matched_link_neighbor_is_rendered_as_relation_only(tmp_path: Path) -> None:
    seed, neighbors = _linked_preferences(1)
    neighbor = neighbors[0]
    seed_uri = MemoryURI.from_address(seed.address)
    neighbor_uri = MemoryURI.from_address(neighbor.address)
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(seed_uri, 0.9), MemorySearchHit(neighbor_uri, 0.8))),
        summaries=SummarySearch(),
    )
    instance.tree.write(seed)
    instance.tree.write(neighbor)
    result = asyncio.run(instance.find("身份和偏好", limit=2))
    assert len(result.memories) == 2
    assert "<memory_relation" in result.context
    assert "<related_memory" not in result.context


def test_execute_requires_boolean_summary_assessment_flag(tmp_path: Path) -> None:
    instance = service(tmp_path, semantic=SemanticSearch(), summaries=SummarySearch())
    plan = instance.query_planner.direct("查询")
    with pytest.raises(TypeError, match="must be boolean"):
        asyncio.run(
            instance._execute(
                plan,
                (MemoryURI.root(),),
                1,
                None,
                (),
                MemoryIntentionRecallScope.ACTIVE,
                assess_for_summary=1,  # type: ignore[arg-type]
            )
        )
