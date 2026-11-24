"""受控 ReAct、Retrieval Grader 和候选解析的严格契约测试。"""

import pytest

from habitus.memory.editor import (
    MemoryExtractionConfig,
    MemoryRetrievalAction,
    MemoryRetrievalDecision,
    MemoryRetrievalFinalDecision,
    MemoryRetrievalObservation,
    MemoryRetrievalStatus,
)


def test_retrieval_decision_allows_exactly_one_action_shape() -> None:
    finish = MemoryRetrievalDecision.model_validate(
        {
            "status": "sufficient",
            "action": "finish",
            "directory": None,
            "uris": [],
            "keywords": [],
            "reason": "旧记忆已经足够。",
        }
    )
    listing = MemoryRetrievalDecision.model_validate(
        {
            "status": "insufficient",
            "action": "memory_list",
            "directory": "habitus://entities/项目",
            "uris": [],
            "keywords": [],
            "reason": "需要看看这个目录。",
        }
    )
    read = MemoryRetrievalDecision.model_validate(
        {
            "status": "insufficient",
            "action": "memory_read",
            "directory": None,
            "uris": ["habitus://profile.md", "habitus://preferences/回答风格.md"],
            "keywords": [],
            "reason": "需要阅读全文。",
        }
    )
    search = MemoryRetrievalDecision.model_validate(
        {
            "status": "insufficient",
            "action": "memory_search",
            "directory": "habitus://entities",
            "uris": [],
            "keywords": ["拉新方案"],
            "reason": "名字看不出，按内容找。",
        }
    )
    assert (finish.action, listing.action, search.action, read.action) == tuple(MemoryRetrievalAction)
    assert read.uris == ("habitus://profile.md", "habitus://preferences/回答风格.md")
    assert search.keywords == ("拉新方案",)

    with pytest.raises(ValueError, match="requires keywords"):
        MemoryRetrievalDecision(MemoryRetrievalStatus.INSUFFICIENT, MemoryRetrievalAction.SEARCH, None, (), "缺词")
    with pytest.raises(ValueError, match="only memory_search carries keywords"):
        MemoryRetrievalDecision(
            MemoryRetrievalStatus.INSUFFICIENT,
            MemoryRetrievalAction.READ,
            None,
            ("habitus://profile.md",),
            "读时带词",
            keywords=("词",),
        )

    with pytest.raises(ValueError, match="requires a directory"):
        MemoryRetrievalDecision(
            MemoryRetrievalStatus.INSUFFICIENT,
            MemoryRetrievalAction.LIST,
            None,
            (),
            "缺少目录",
        )
    with pytest.raises(ValueError, match="requires uris"):
        MemoryRetrievalDecision(
            MemoryRetrievalStatus.INSUFFICIENT,
            MemoryRetrievalAction.READ,
            None,
            (),
            "缺少地址",
        )
    with pytest.raises(ValueError, match="unique"):
        MemoryRetrievalDecision(
            MemoryRetrievalStatus.INSUFFICIENT,
            MemoryRetrievalAction.READ,
            None,
            ("habitus://profile.md", "habitus://profile.md"),
            "重复地址",
        )
    with pytest.raises(ValueError, match="must finish"):
        MemoryRetrievalDecision(
            MemoryRetrievalStatus.SUFFICIENT,
            MemoryRetrievalAction.READ,
            None,
            ("habitus://profile.md",),
            "错误组合",
        )
    with pytest.raises(ValueError, match="unknown fields"):
        MemoryRetrievalDecision.model_validate(
            {
                "status": "sufficient",
                "action": "finish",
                "directory": None,
                "uris": [],
                "keywords": [],
                "reason": "完成",
                "confidence": 1,
            }
        )


def test_final_retrieval_round_schema_and_decision_only_allow_finish() -> None:
    schema = MemoryRetrievalFinalDecision.model_json_schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert properties["action"]["enum"] == ["finish"]
    assert properties["directory"]["type"] == "null"
    assert properties["uris"]["maxItems"] == 0

    ended = MemoryRetrievalFinalDecision.model_validate(
        {
            "status": "insufficient",
            "action": "finish",
            "directory": None,
            "uris": [],
            "keywords": [],
            "reason": "列目录没有找到旧记忆。",
        }
    )
    assert ended.status is MemoryRetrievalStatus.INSUFFICIENT
    assert isinstance(ended, MemoryRetrievalDecision)
    with pytest.raises(ValueError, match="can only finish"):
        MemoryRetrievalFinalDecision(
            MemoryRetrievalStatus.INSUFFICIENT,
            MemoryRetrievalAction.LIST,
            "habitus://preferences",
            (),
            "上下文不足",
        )
    with pytest.raises(ValueError, match="cannot carry"):
        MemoryRetrievalFinalDecision(
            MemoryRetrievalStatus.SUFFICIENT,
            MemoryRetrievalAction.FINISH,
            "habitus://preferences",
            (),
            "完成",
        )
    with pytest.raises(ValueError, match="insufficient retrieval cannot finish"):
        MemoryRetrievalDecision(
            MemoryRetrievalStatus.INSUFFICIENT,
            MemoryRetrievalAction.FINISH,
            None,
            (),
            "动作轮的不足仍须继续检索",
        )


def test_observation_is_a_bounded_audit_reference_not_a_document_copy() -> None:
    observation = MemoryRetrievalObservation(
        1,
        MemoryRetrievalAction.LIST,
        ("habitus://preferences",),
        ("habitus://preferences/回答风格.md",),
        (),
        (),
    )
    assert set(observation.to_dict()) == {
        "iteration",
        "action",
        "inputs",
        "result_uris",
        "added_uris",
        "relation_expanded_uris",
        "cached",
    }
    with pytest.raises(ValueError, match="duplicate"):
        MemoryRetrievalObservation(
            1,
            MemoryRetrievalAction.READ,
            ("habitus://profile.md",),
            ("habitus://profile.md", "habitus://profile.md"),
            (),
            (),
        )
    with pytest.raises(ValueError, match="requires its action inputs"):
        MemoryRetrievalObservation(1, MemoryRetrievalAction.READ, (), (), (), ())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_retrieval_iterations", 0),
        ("max_retrieval_iterations", 9),
        ("max_old_memory_items", 100),
        ("max_reads_per_action", 0),
        ("max_listing_tokens", 0),
    ],
)
def test_extraction_config_enforces_cost_and_page_id_bounds(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        MemoryExtractionConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    ["max_candidate_regenerations", "reviewer_max_output_tokens", "additional_search_limit", "max_query_chars"],
)
def test_extraction_config_does_not_keep_removed_review_or_search_controls(field: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        MemoryExtractionConfig(**{field: 1})
