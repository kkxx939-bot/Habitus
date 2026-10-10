"""白天归类：整条内容进提示词、三种结局、漏答只重问漏的、用尽当待定、lane 互不串。"""

from __future__ import annotations

import asyncio

import pytest

from habitus.behavior.kinds.classify import DaytimeClassifier, OccurrenceContent, Outcome
from habitus.behavior.kinds.config import BehaviorKindConfig
from habitus.behavior.kinds.ids import Lane
from habitus.behavior.kinds.model import BehaviorKindError, Vocabulary
from habitus.model_client.contracts import ModelTransportError
from tests.unit.behavior.kinds_fixtures import (
    EDIT,
    RESEARCH,
    answer,
    edit_class,
    request,
    research_class,
    scripted_caller,
)


def vocabulary() -> Vocabulary:
    return Vocabulary(version=1, classes={EDIT: edit_class(), RESEARCH: research_class()})


def classify(bodies: list[object], requests: list, *, config: BehaviorKindConfig | None = None):  # type: ignore[no-untyped-def]
    caller, provider = scripted_caller(bodies, config=config)
    result = asyncio.run(DaytimeClassifier(caller).classify(requests, vocabulary()))
    return result, provider


def test_three_outcomes_write_id_or_lane_markers() -> None:
    result, provider = classify(
        [answer(("R1", "C1", None), ("R2", "都不是", "统计代码规模"), ("R3", "不是一件事", None))],
        [request("a", "调整记忆召回代码"), request("b", "数一下仓库行数"), request("c", "git status")],
    )
    assert result.verdicts["a"].token == "s-k0001" and result.verdicts["a"].outcome is Outcome.CLASS
    assert result.verdicts["b"].token == "s-待定" and result.verdicts["b"].proposed == "统计代码规模"
    assert result.verdicts["c"].token == "s-非事件" and result.verdicts["c"].outcome is Outcome.NOT_EVENT
    assert result.model_calls == 1 and provider.calls == 1


def test_prompt_carries_criteria_reminders_exclusions_and_whole_chain() -> None:
    content = OccurrenceContent(
        name="研究算子", summaries=("读 vLLM 源码", "改了本地实现"), goal="弄清关系", steps=("读", "改")
    )
    from habitus.behavior.kinds.classify import ClassifyRequest

    _, provider = classify([answer(("R1", "C2", None))], [ClassifyRequest("a", Lane.SESSION, content)])
    prompt = provider.prompts[0]
    for expected in (
        "判据：动手改代码",
        "提醒句：该调研一下了",
        "不含：动手改代码（→修改代码）",
        "例：为 Tagent",
        "改了本地实现",
        "目标：弄清关系",
        "步骤：读／改",
    ):
        assert expected in prompt


def test_unanswered_records_are_reasked_alone_then_kept_pending() -> None:
    config = BehaviorKindConfig(validation_rounds=1)
    result, provider = classify(
        [answer(("R1", "C1", None)), answer(("R1", "都不是", None))],
        [request("a", "改代码"), request("b", "某件新事")],
        config=config,
    )
    assert result.verdicts["a"].token == "s-k0001"
    # 第二轮只重问 b（提示词里只剩一条）；「都不是」却没给提议名仍算没答好，轮数用尽 → 待定，提议名退回原话
    assert provider.calls == 2 and "改代码" not in provider.prompts[1].split("## 待归类的记录")[1]
    assert result.verdicts["b"].token == "s-待定" and result.verdicts["b"].proposed == "某件新事"
    assert any("kind_classify_exhausted 'b'" in signal for signal in result.signals)


def test_shape_errors_are_rejected_and_exhaust_to_pending() -> None:
    result, _ = classify(
        [{"items": [{"record": "R9", "reason": "x", "choice": "C1", "proposed": None}]}],
        [request("a", "改代码")],
        config=BehaviorKindConfig(validation_rounds=0),
    )
    assert result.verdicts["a"].outcome is Outcome.PENDING
    assert any(signal.startswith("kind_classify_rejected") for signal in result.signals)


def test_transient_errors_are_retried() -> None:
    config = BehaviorKindConfig(transient_retries=1, transient_retry_delay_seconds=0)
    result, provider = classify(
        [ModelTransportError("断连"), answer(("R1", "C1", None))], [request("a", "改代码")], config=config
    )
    assert result.verdicts["a"].token == "s-k0001" and provider.calls == 2


def test_a_lane_without_classes_still_asks_whether_each_record_is_one_thing() -> None:
    """词表从空开始（裁定 29）：这条 lane 还没有类也照样问——判出不是一件事的单条命令，给真事一个提议名，再进待定池。"""

    result, provider = classify(
        [answer(("R1", "都不是", "吃饭"), ("R2", "不是一件事", None))],
        [request("a", "吃饭", lane=Lane.PHYSICAL), request("b", "抬手", lane=Lane.PHYSICAL)],
    )
    assert provider.calls == 1 and "还没有类" in provider.prompts[0]
    assert result.verdicts["a"].token == "p-待定" and result.verdicts["a"].proposed == "吃饭"
    assert result.verdicts["b"].outcome is Outcome.NOT_EVENT


def test_lanes_never_see_each_others_classes() -> None:
    result, provider = classify(
        [answer(("R1", "C1", None)), answer(("R1", "都不是", "吃饭"))],
        [request("a", "改代码"), request("b", "吃饭", lane=Lane.PHYSICAL)],
    )
    # 两条 lane 各问一次；物理 lane 那次看不到会话 lane 的类
    assert provider.calls == 2 and result.verdicts["b"].token == "p-待定"
    assert "修改代码" not in provider.prompts[1]


def test_batches_follow_the_configured_size() -> None:
    config = BehaviorKindConfig(batch_size=1)
    _, provider = classify(
        [answer(("R1", "C1", None))], [request("a", "改代码"), request("b", "改代码二")], config=config
    )
    assert provider.calls == 2


def test_candidate_cap_only_raises_a_signal() -> None:
    config = BehaviorKindConfig(lane_candidate_cap=1)
    result, _ = classify([answer(("R1", "C1", None))], [request("a", "改代码")], config=config)
    assert any(signal.startswith("kind_lane_over_candidate_cap") for signal in result.signals)


def test_duplicate_keys_are_refused() -> None:
    caller, _ = scripted_caller([answer()])
    with pytest.raises(BehaviorKindError):
        asyncio.run(DaytimeClassifier(caller).classify([request("a", "x"), request("a", "y")], vocabulary()))


def test_one_repeated_record_is_reasked_alone_and_does_not_void_the_batch() -> None:
    config = BehaviorKindConfig(validation_rounds=1)
    twice = {
        "items": [
            {"record": "R1", "reason": "理由", "choice": "C1", "proposed": None},
            {"record": "R2", "reason": "理由", "choice": "C2", "proposed": None},
            {"record": "R2", "reason": "理由", "choice": "C1", "proposed": None},
        ]
    }
    result, provider = classify(
        [twice, answer(("R2", "C2", None))], [request("a", "改代码"), request("b", "读源码")], config=config
    )
    assert result.verdicts["a"].token == "s-k0001" and result.verdicts["b"].token == "s-k0002"
    assert provider.calls == 2 and "改代码" not in provider.prompts[1].split("## 待归类的记录")[1]


def test_input_too_large_or_safety_blocks_degrade_the_batch_to_pending() -> None:
    from habitus.model_client.contracts import ModelContentSafetyError, ModelInputTooLargeError

    for error in (ModelInputTooLargeError("太长"), ModelContentSafetyError("拦截")):
        result, _ = classify([error], [request("a", "改代码")], config=BehaviorKindConfig(validation_rounds=0))
        assert result.verdicts["a"].outcome is Outcome.PENDING
        assert any(type(error).__name__ in signal for signal in result.signals)
