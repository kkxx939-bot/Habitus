"""记忆检索配置、查询、命中、关系节点、充分性与结果不变量矩阵。"""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import timedelta

import pytest

from memory.document import MemoryLinkType, MemoryStoredLink
from memory.intention import (
    MemoryIntentionRecallScope,
    MemoryIntentionReview,
    MemoryIntentionReviewer,
)
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind
from memory.retrieval import (
    MemoryMatchedMemory,
    MemoryQueryPlan,
    MemoryQueryPlanContent,
    MemoryQueryResult,
    MemoryRelatedMemory,
    MemoryRetrievalAssessment,
    MemoryRetrievalSufficiency,
    MemorySearchHit,
    MemorySearchResult,
    MemorySearchServiceConfig,
    MemoryTypedQuery,
)
from memory.uri import MemoryURI
from tests.helpers import BASE_TIME, document

INTEGER_CONFIG_FIELDS = tuple(
    field.name
    for field in fields(MemorySearchServiceConfig)
    if field.name != "summary_fallback_enabled"
)


def _hit(kind: MemoryKind = MemoryKind.PREFERENCE, score: float = 0.9) -> tuple[MemorySearchHit, object]:
    item = document(kind)
    return MemorySearchHit(MemoryURI.from_address(item.address), score), item


def _matched(kind: MemoryKind = MemoryKind.PREFERENCE, score: float = 0.9) -> MemoryMatchedMemory:
    hit, item = _hit(kind, score)
    review = None
    if kind is MemoryKind.INTENTION:
        review = MemoryIntentionReviewer().review(item, now=BASE_TIME)
    return MemoryMatchedMemory(hit, item, ("查询",), intention_review=review)


@pytest.mark.parametrize("field", INTEGER_CONFIG_FIELDS)
@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, [], {}])
def test_search_config_rejects_non_integer_for_every_integer_field(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(MemorySearchServiceConfig(), **{field: value})


@pytest.mark.parametrize("field", INTEGER_CONFIG_FIELDS)
@pytest.mark.parametrize("value", [-1, 10**9])
def test_search_config_rejects_out_of_range_for_every_integer_field(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        replace(MemorySearchServiceConfig(), **{field: value})


@pytest.mark.parametrize("value", [True, False])
def test_search_config_accepts_explicit_summary_fallback_switch(value: bool) -> None:
    assert replace(MemorySearchServiceConfig(), summary_fallback_enabled=value).summary_fallback_enabled is value


@pytest.mark.parametrize("value", [0, 1, "true", None, [], {}])
def test_search_config_rejects_non_boolean_summary_fallback_switch(value: object) -> None:
    with pytest.raises(TypeError):
        replace(MemorySearchServiceConfig(), summary_fallback_enabled=value)


@pytest.mark.parametrize(
    "changes",
    [
        {"default_limit": 21},
        {"max_planned_query_chars": 5_000, "max_query_chars": 4_999},
        {"max_relation_neighbors_per_match": 6, "max_relation_neighbors_total": 5},
        {"summary_fallback_max_query_chars": 5_000, "max_query_chars": 4_999},
        {"summary_fallback_max_context_chars": 120_001},
        {"max_context_chars": 120_001, "retrieval_grader_max_context_chars": 120_000},
    ],
)
def test_search_config_rejects_cross_field_capacity_inversions(changes: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        replace(MemorySearchServiceConfig(), **changes)


@pytest.mark.parametrize("decision", tuple(MemoryRetrievalSufficiency))
def test_retrieval_assessment_round_trips_each_decision(decision: MemoryRetrievalSufficiency) -> None:
    if decision is MemoryRetrievalSufficiency.SUFFICIENT:
        value = MemoryRetrievalAssessment(decision, "长期记忆足够。", (), None)
    else:
        value = MemoryRetrievalAssessment(decision, "缺少过程细节。", ("历史过程",), "查找历史过程")
    restored = MemoryRetrievalAssessment.model_validate(
        {
            "decision": value.decision.value,
            "reason": value.reason,
            "missing_information": list(value.missing_information),
            "summary_query": value.summary_query,
        }
    )
    assert restored == value


@pytest.mark.parametrize("reason", ["", " ", " reason", "reason ", None, 1, "x" * 10_001])
def test_retrieval_assessment_rejects_invalid_reason(reason: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryRetrievalAssessment(MemoryRetrievalSufficiency.SUFFICIENT, reason, (), None)  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", [[], "missing", None, ("",), (" ",), (" x",), ("x ",), (1,), ("x", "x")])
def test_retrieval_assessment_rejects_invalid_missing_information_container_or_items(missing: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryRetrievalAssessment(
            MemoryRetrievalSufficiency.INSUFFICIENT,
            "缺少信息。",
            missing,  # type: ignore[arg-type]
            "查询历史",
        )


@pytest.mark.parametrize(
    ("missing", "summary_query"),
    [
        (("缺口",), "查询"),
        ((), None),
        (("缺口",), None),
        ((), "查询"),
    ],
)
def test_sufficient_assessment_forbids_and_insufficient_requires_fallback_fields(
    missing: tuple[str, ...],
    summary_query: str | None,
) -> None:
    if missing == () and summary_query is None:
        MemoryRetrievalAssessment(MemoryRetrievalSufficiency.SUFFICIENT, "足够。", missing, summary_query)
    else:
        with pytest.raises(ValueError):
            MemoryRetrievalAssessment(MemoryRetrievalSufficiency.SUFFICIENT, "足够。", missing, summary_query)
    if missing and summary_query:
        MemoryRetrievalAssessment(MemoryRetrievalSufficiency.INSUFFICIENT, "不足。", missing, summary_query)
    else:
        with pytest.raises(ValueError):
            MemoryRetrievalAssessment(MemoryRetrievalSufficiency.INSUFFICIENT, "不足。", missing, summary_query)


@pytest.mark.parametrize("value", [None, [], (), "assessment", 1, True])
def test_retrieval_assessment_parser_requires_exact_object(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryRetrievalAssessment.model_validate(value)


@pytest.mark.parametrize("field", ["decision", "reason", "missing_information", "summary_query"])
def test_retrieval_assessment_parser_rejects_each_missing_field(field: str) -> None:
    value = {"decision": "sufficient", "reason": "足够。", "missing_information": [], "summary_query": None}
    value.pop(field)
    with pytest.raises(ValueError):
        MemoryRetrievalAssessment.model_validate(value)


@pytest.mark.parametrize("score", [-100.0, -1.0, -0.0, 0, 0.5, 1.0, 100.0])
def test_search_hit_accepts_finite_final_scores(score: float) -> None:
    uri = MemoryURI.from_address(MemoryAddress.profile())
    hit = MemorySearchHit(uri, score, vector_score=max(-1.0, min(1.0, score)))
    assert hit.score == float(score)


@pytest.mark.parametrize("score", [True, False, "0.5", None, [], {}])
@pytest.mark.parametrize("field", ["score", "vector_score", "rerank_score"])
def test_search_hit_rejects_non_numeric_score_fields(score: object, field: str) -> None:
    kwargs = {"score": 0.5, "vector_score": 0.5, "rerank_score": 0.5}
    kwargs[field] = score
    if score is None and field in {"vector_score", "rerank_score"}:
        hit = MemorySearchHit(MemoryURI.from_address(MemoryAddress.profile()), **kwargs)  # type: ignore[arg-type]
        assert getattr(hit, field) == (0.5 if field == "vector_score" else None)
        return
    with pytest.raises(TypeError):
        MemorySearchHit(MemoryURI.from_address(MemoryAddress.profile()), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["score", "vector_score", "rerank_score"])
def test_search_hit_rejects_non_finite_score_fields(score: float, field: str) -> None:
    kwargs = {"score": 0.5, "vector_score": 0.5, "rerank_score": 0.5}
    kwargs[field] = score
    with pytest.raises(ValueError, match="finite"):
        MemorySearchHit(MemoryURI.from_address(MemoryAddress.profile()), **kwargs)


@pytest.mark.parametrize("vector_score", [-1.01, 1.01, -100.0, 100.0])
def test_search_hit_rejects_vector_score_outside_cosine_range(vector_score: float) -> None:
    with pytest.raises(ValueError, match="between -1 and 1"):
        MemorySearchHit(MemoryURI.from_address(MemoryAddress.profile()), 0.5, vector_score=vector_score)


@pytest.mark.parametrize("uri", [MemoryURI.root(), MemoryURI.from_directory(MemoryDirectory.preferences()), MemoryURI("memory://preferences/.abstract.md")])
def test_search_hit_requires_l2_document_uri(uri: MemoryURI) -> None:
    with pytest.raises(ValueError, match="L2"):
        MemorySearchHit(uri, 0.5)


@pytest.mark.parametrize("priority", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(("query", "intent"), [("用户偏好", "查找偏好"), ("x", "y"), ("中文 查询", "时间线")])
def test_typed_query_round_trips_valid_priorities(query: str, intent: str, priority: int) -> None:
    value = MemoryTypedQuery(query, intent, priority)
    assert MemoryTypedQuery.from_dict(value.to_dict()) == value


@pytest.mark.parametrize("priority", [0, 6, -1, True, False, 1.0, "1", None])
def test_typed_query_rejects_invalid_priority(priority: object) -> None:
    with pytest.raises(ValueError):
        MemoryTypedQuery("查询", "意图", priority)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["query", "intent"])
@pytest.mark.parametrize("value", ["", " ", " x", "x ", None, 1, "x" * 5_001])
def test_typed_query_rejects_invalid_text_fields(field: str, value: object) -> None:
    kwargs = {"query": "查询", "intent": "意图", "priority": 1}
    kwargs[field] = value
    if field == "intent" and isinstance(value, str) and len(value) == 5_001:
        value = "x" * 1_001
        kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        MemoryTypedQuery(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, [], "query", {}, {"query": "q", "intent": "i"}, {"query": "q", "intent": "i", "priority": 1, "uri": "x"}])
def test_typed_query_parser_requires_exact_shape(value: object) -> None:
    with pytest.raises(ValueError):
        MemoryTypedQuery.from_dict(value)


@pytest.mark.parametrize("count", range(1, 9))
def test_query_plan_content_accepts_one_to_eight_unique_queries(count: int) -> None:
    queries = tuple(MemoryTypedQuery(f"查询-{index}", f"意图-{index}", index % 5 + 1) for index in range(count))
    content = MemoryQueryPlanContent(queries)
    assert MemoryQueryPlanContent.model_validate({"queries": [item.to_dict() for item in queries]}) == content


@pytest.mark.parametrize("count", [0, 9, 10, 100])
def test_query_plan_content_rejects_count_outside_bound(count: int) -> None:
    queries = tuple(MemoryTypedQuery(f"查询-{index}", f"意图-{index}", 1) for index in range(count))
    with pytest.raises(ValueError):
        MemoryQueryPlanContent(queries)


@pytest.mark.parametrize("variant", ["same", "case"])
def test_query_plan_content_rejects_duplicate_semantics(variant: str) -> None:
    second_query = {"same": "Query", "case": "query"}[variant]
    second_intent = {"same": "Intent", "case": "intent"}[variant]
    first_query = "Query"
    first_intent = "Intent"
    with pytest.raises(ValueError, match="duplicate"):
        MemoryQueryPlanContent(
            (
                MemoryTypedQuery(first_query, first_intent, 1),
                MemoryTypedQuery(second_query, second_intent, 2),
            )
        )


@pytest.mark.parametrize("conversation_id", [None, "conversation-1", "中文会话"])
def test_query_plan_records_contextual_identity_only_when_present(conversation_id: str | None) -> None:
    plan = MemoryQueryPlan("原始问题", (MemoryTypedQuery("查询", "意图", 1),), conversation_id)
    assert plan.contextual is (conversation_id is not None)


@pytest.mark.parametrize("hits", [[], {}, "hits", None, (object(),)])
def test_query_result_requires_tuple_of_hits(hits: object) -> None:
    with pytest.raises(TypeError):
        MemoryQueryResult(MemoryTypedQuery("查询", "意图", 1), hits)  # type: ignore[arg-type]


def test_query_result_requires_unique_score_sorted_hits() -> None:
    profile = MemorySearchHit(MemoryURI.from_address(MemoryAddress.profile()), 0.9)
    preference = MemorySearchHit(MemoryURI.from_address(MemoryAddress.preference("主题")), 0.8)
    assert MemoryQueryResult(MemoryTypedQuery("查询", "意图", 1), (profile, preference)).hits == (profile, preference)
    with pytest.raises(ValueError, match="sorted"):
        MemoryQueryResult(MemoryTypedQuery("查询", "意图", 1), (preference, profile))
    with pytest.raises(ValueError, match="unique"):
        MemoryQueryResult(MemoryTypedQuery("查询", "意图", 1), (profile, profile))


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_matched_memory_binds_hit_document_query_and_intention_review(kind: MemoryKind) -> None:
    value = _matched(kind)
    assert value.uri == value.hit.uri
    assert value.document.kind is kind
    assert (value.intention_review is not None) is (kind is MemoryKind.INTENTION)


@pytest.mark.parametrize("queries", [(), [], ("",), (" ",), (1,), ("q", "q")])
def test_matched_memory_requires_nonempty_unique_query_tuple(queries: object) -> None:
    hit, item = _hit()
    with pytest.raises((TypeError, ValueError)):
        MemoryMatchedMemory(hit, item, queries)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", [kind for kind in MemoryKind if kind is not MemoryKind.INTENTION])
def test_non_intention_matched_memory_rejects_review(kind: MemoryKind) -> None:
    hit, item = _hit(kind)
    review = MemoryIntentionReview("current", BASE_TIME, 0)
    with pytest.raises(ValueError, match="only active"):
        MemoryMatchedMemory(hit, item, ("查询",), intention_review=review)


def test_active_intention_matched_memory_requires_matching_review() -> None:
    hit, item = _hit(MemoryKind.INTENTION)
    with pytest.raises(ValueError, match="requires"):
        MemoryMatchedMemory(hit, item, ("查询",))
    wrong = MemoryIntentionReview("current", BASE_TIME + timedelta(days=1), 0)
    with pytest.raises(ValueError, match="does not match"):
        MemoryMatchedMemory(hit, item, ("查询",), intention_review=wrong)


@pytest.mark.parametrize("link_type", tuple(MemoryLinkType))
def test_related_memory_accepts_either_relation_endpoint(link_type: MemoryLinkType) -> None:
    left = document(MemoryKind.PROFILE)
    right = document(MemoryKind.PREFERENCE)
    relation = MemoryStoredLink(
        MemoryURI.from_address(left.address),
        MemoryURI.from_address(right.address),
        link_type,
    )
    assert MemoryRelatedMemory(relation, left).document == left
    assert MemoryRelatedMemory(relation, right).document == right


def test_related_memory_rejects_document_outside_relation() -> None:
    left = document(MemoryKind.PROFILE)
    right = document(MemoryKind.PREFERENCE)
    outside = document(MemoryKind.ENTITY)
    relation = MemoryStoredLink(
        MemoryURI.from_address(left.address),
        MemoryURI.from_address(right.address),
        MemoryLinkType.DERIVED_FROM,
    )
    with pytest.raises(ValueError, match="endpoint"):
        MemoryRelatedMemory(relation, outside)


def test_matched_memory_relations_must_be_unique_and_sorted() -> None:
    seed = document(MemoryKind.PROFILE)
    left = document(MemoryKind.PREFERENCE)
    right = document(MemoryKind.ENTITY)
    first_link = MemoryStoredLink(MemoryURI.from_address(seed.address), MemoryURI.from_address(left.address), MemoryLinkType.DERIVED_FROM)
    second_link = MemoryStoredLink(MemoryURI.from_address(seed.address), MemoryURI.from_address(right.address), MemoryLinkType.DERIVED_FROM)
    first = MemoryRelatedMemory(first_link, left)
    second = MemoryRelatedMemory(second_link, right)
    ordered = tuple(sorted((first, second), key=lambda item: item.relation.identity))
    hit = MemorySearchHit(MemoryURI.from_address(seed.address), 0.9)
    MemoryMatchedMemory(hit, seed, ("查询",), related=ordered)
    with pytest.raises(ValueError, match="sorted"):
        MemoryMatchedMemory(hit, seed, ("查询",), related=tuple(reversed(ordered)))
    with pytest.raises(ValueError, match="unique"):
        MemoryMatchedMemory(hit, seed, ("查询",), related=(ordered[0], ordered[0]))


def _search_result(**overrides: object) -> MemorySearchResult:
    query = "用户偏好"
    typed = MemoryTypedQuery(query, "偏好", 1)
    plan = MemoryQueryPlan(query, (typed,))
    memory = _matched()
    values: dict[str, object] = {
        "query": query,
        "target_roots": (MemoryURI.root(),),
        "kinds": (MemoryKind.PREFERENCE,),
        "intention_scope": MemoryIntentionRecallScope.ACTIVE,
        "plan": plan,
        "query_results": (MemoryQueryResult(typed, (memory.hit,)),),
        "memories": (memory,),
        "retrieval_assessment": None,
        "summary_fallback_attempted": False,
        "summary_fallbacks": (),
        "context": "memory context",
        "budget_exhausted": False,
    }
    values.update(overrides)
    return MemorySearchResult(**values)  # type: ignore[arg-type]


def test_search_result_accepts_consistent_primary_memory_result() -> None:
    result = _search_result()
    assert result.total == 1
    assert result.context


@pytest.mark.parametrize("roots", [(), [], (MemoryURI.from_address(MemoryAddress.profile()),), (MemoryURI("memory://preferences/.abstract.md"),)])
def test_search_result_requires_nonempty_directory_roots(roots: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _search_result(target_roots=roots)


@pytest.mark.parametrize("kinds", [[], (MemoryKind.PREFERENCE, MemoryKind.PREFERENCE), ("unknown",)])
def test_search_result_requires_unique_kind_tuple(kinds: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _search_result(kinds=kinds)


def test_search_result_requires_plan_and_query_results_to_match_original_query() -> None:
    other = MemoryQueryPlan("其他问题", (MemoryTypedQuery("其他问题", "其他", 1),))
    with pytest.raises(ValueError, match="plan"):
        _search_result(plan=other)
    with pytest.raises(ValueError, match="query_results"):
        _search_result(query_results=())


def test_search_result_rejects_direct_memory_outside_kind_filter() -> None:
    with pytest.raises(ValueError, match="outside"):
        _search_result(kinds=(MemoryKind.PROFILE,))


@pytest.mark.parametrize("context", ["", "context"])
@pytest.mark.parametrize("has_memory", [True, False])
def test_search_result_context_presence_exactly_tracks_returned_content(context: str, has_memory: bool) -> None:
    if has_memory == bool(context):
        _search_result(
            memories=(_matched(),) if has_memory else (),
            context=context,
            kinds=(MemoryKind.PREFERENCE,) if has_memory else (),
        )
    else:
        with pytest.raises(ValueError, match="context presence"):
            _search_result(
                memories=(_matched(),) if has_memory else (),
                context=context,
                kinds=(MemoryKind.PREFERENCE,) if has_memory else (),
            )


@pytest.mark.parametrize("field", ["summary_fallback_attempted", "budget_exhausted"])
@pytest.mark.parametrize("value", [0, 1, "false", None, [], {}])
def test_search_result_requires_boolean_control_flags(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        _search_result(**{field: value})
