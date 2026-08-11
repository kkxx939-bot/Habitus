"""三档行动决策(沉默/建议/直接执行)的合同测试。"""

from __future__ import annotations

import asyncio

import pytest

from ModelClient import (
    ChatClient,
    ChatModelConfig,
    ModelTransportError,
    ProviderConfig,
    StructuredChatClient,
)
from prediction import (
    PredictionContext,
    PredictionPatternGenerationPublisher,
    PredictionPatternGraph,
    PredictionPublisher,
    PredictionRunSourceBinding,
    PredictionTree,
)
from prediction.learning import PredictionPatternLearner
from prediction.predictor import PatternGraphPredictor, PredictionDecisionConfig
from Runtime.prediction_execution import (
    PredictionActionPlan,
    PredictionTierConfig,
    resolve_action,
)
from Runtime.prediction_llm import PatternGraphLLMAdvisor, PredictionConstraintChecker
from tests.unit.prediction.test_pattern_learning import _WATER, LEARNED_AT, _transition
from tests.unit.runtime.test_prediction_llm import QueueProvider, _client

_BINDINGS = (
    PredictionRunSourceBinding("behavior://behaviors/events/2026/08/08/runtime.md", 1, "b" * 64),
)


def _broken_checker() -> PredictionConstraintChecker:
    provider = QueueProvider([])

    def _raise(request):
        raise ModelTransportError("provider down")

    provider.complete = _raise  # type: ignore[method-assign]
    config = ChatModelConfig(
        ProviderConfig(
            provider="test-provider",
            adapter="test-adapter",
            model="test-model",
            max_retries=0,
        ),
        structured_output_mode="json_schema",
    )
    return PredictionConstraintChecker(StructuredChatClient(ChatClient(config, provider)))


def _published_graph(tmp_path, documents) -> PredictionPatternGraph:
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(documents)
    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    return PredictionPatternGraph(tree)


def _resolve(predictor, advisor, context, **kwargs) -> PredictionActionPlan:
    return asyncio.run(
        resolve_action(
            predictor,
            advisor,
            context,
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
            **kwargs,
        )
    )


def test_unqualified_executions_and_grey_zone_map_to_suggestions(tmp_path) -> None:
    documents = (
        *(_transition(f"tier-{index}") for index in range(6)),
        _transition("tier-w", label=_WATER),
    )
    graph = _published_graph(tmp_path, documents)
    context = PredictionContext.from_document(_transition("tier-query"))
    advisor = PatternGraphLLMAdvisor(_client())

    confident = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(
            execute_probability_threshold=0.4, execute_margin=0.1, min_execute_support=2
        ),
        advisor_digest=advisor.advisor_digest,
    )
    plan = _resolve(confident, advisor, context)
    assert plan.tier == "suggest"
    assert plan.reasons == ("not_yet_stable_routine",)
    assert plan.constraint_status == "not_applicable"

    qualified_plan = _resolve(confident, advisor, context, qualifier=lambda run: True)
    assert qualified_plan.tier == "execute"
    assert qualified_plan.reasons == ()
    assert qualified_plan.constraint_status == "bypassed_no_checker"
    assert qualified_plan.checker_digest is None

    strict = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(execute_probability_threshold=0.9, execute_margin=0.5),
        advisor_digest=advisor.advisor_digest,
    )
    grey = _resolve(strict, PatternGraphLLMAdvisor(_client('{"preferences": []}')), context)
    assert grey.tier == "suggest"
    assert "probability_below_threshold" in grey.reasons

    silent_config = PredictionTierConfig(suggest_probability=0.99, suggest_margin=0.9)
    silent = _resolve(
        strict,
        PatternGraphLLMAdvisor(_client('{"preferences": []}')),
        context,
        config=silent_config,
    )
    assert silent.tier == "silent"
    assert silent.reasons


def test_constraint_check_vetoes_execute_and_promotes_compliant_runner_up(tmp_path) -> None:
    documents = (
        *(_transition(f"veto-{index}") for index in range(6)),
        *(_transition(f"veto-w{index}", label=_WATER) for index in range(4)),
    )
    graph = _published_graph(tmp_path, documents)
    context = PredictionContext.from_document(_transition("veto-query"))
    advisor = PatternGraphLLMAdvisor(_client())
    predictor = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(
            execute_probability_threshold=0.2, execute_margin=0.05, min_execute_support=2
        ),
        advisor_digest=advisor.advisor_digest,
    )

    baseline = predictor.predict(
        context, predicted_at=LEARNED_AT, horizon_seconds=3600, source_bindings=_BINDINGS
    ).run
    assert baseline.candidates[0].semantics == "打开空调"
    ac_key = str(baseline.candidates[0].payload["branch_key"])
    water_key = next(
        str(item.payload["branch_key"])
        for item in baseline.candidates
        if item.semantics == "倒一杯水"
    )
    verdicts = (
        f'{{"verdicts": [{{"branch_key": "{ac_key}", "allowed": false,'
        f' "reason": "晚间不开空调"}},'
        f' {{"branch_key": "{water_key}", "allowed": true, "reason": "无冲突"}}]}}'
    )

    plan = _resolve(
        predictor,
        advisor,
        context,
        qualifier=lambda run: True,
        checker=PredictionConstraintChecker(_client(verdicts)),
        memory_context=("晚上尽量不开空调",),
    )

    assert plan.tier == "execute"
    assert plan.run.candidates[0].semantics == "倒一杯水"
    assert all(item.semantics != "打开空调" for item in plan.run.candidates)
    assert plan.constraint_status == "passed"
    assert plan.checker_digest is not None
    assert plan.verdicts_digest is not None
    assert plan.run.excluded_branch_keys == (ac_key,)


def test_constraint_check_failure_downgrades_to_suggestion(tmp_path) -> None:
    documents = tuple(_transition(f"fc-{index}") for index in range(5))
    graph = _published_graph(tmp_path, documents)
    context = PredictionContext.from_document(_transition("fc-query"))
    advisor = PatternGraphLLMAdvisor(_client())
    predictor = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(
            execute_probability_threshold=0.4, execute_margin=0.1, min_execute_support=2
        ),
        advisor_digest=advisor.advisor_digest,
    )
    broken_checker = _broken_checker()

    plan = _resolve(
        predictor,
        advisor,
        context,
        qualifier=lambda run: True,
        checker=broken_checker,
        memory_context=("晚上尽量不开空调",),
    )

    assert plan.tier == "suggest"
    assert plan.reasons == ("constraint_check_unavailable",)
    assert plan.constraint_status == "unavailable"
    assert plan.verdicts_digest is None

    no_context_plan = _resolve(
        predictor,
        advisor,
        context,
        qualifier=lambda run: True,
        checker=broken_checker,
        memory_context=(),
    )
    assert no_context_plan.tier == "execute"
    assert no_context_plan.constraint_status == "bypassed_no_memory_context"
    assert no_context_plan.checker_digest == broken_checker.checker_digest

    with pytest.raises(TypeError, match="PredictionTierConfig"):
        _resolve(predictor, advisor, context, config="bad")  # type: ignore[arg-type]


def test_incomplete_verdicts_fail_closed_instead_of_fail_open(tmp_path) -> None:
    documents = tuple(_transition(f"cov-{index}") for index in range(5))
    graph = _published_graph(tmp_path, documents)
    context = PredictionContext.from_document(_transition("cov-query"))
    advisor = PatternGraphLLMAdvisor(_client())
    predictor = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(
            execute_probability_threshold=0.4, execute_margin=0.1, min_execute_support=2
        ),
        advisor_digest=advisor.advisor_digest,
    )

    plan = _resolve(
        predictor,
        advisor,
        context,
        qualifier=lambda run: True,
        checker=PredictionConstraintChecker(
            _client('{"verdicts": []}', '{"verdicts": []}')
        ),
        memory_context=("晚上尽量不开空调",),
    )

    assert plan.tier == "suggest"
    assert plan.reasons == ("constraint_check_unavailable",)
    assert plan.constraint_status == "unavailable"


def test_unreviewed_retry_winner_needs_a_fresh_review_before_executing(tmp_path) -> None:
    curtain = ("拉上窗帘", "device_control", ("窗帘",))
    documents = (
        *(_transition(f"rr-{index}") for index in range(6)),
        *(_transition(f"rr-w{index}", label=_WATER) for index in range(5)),
        *(
            _transition(f"rr-t{index}", label=("打开电视", "device_control", ("电视",)))
            for index in range(4)
        ),
        *(_transition(f"rr-c{index}", label=curtain) for index in range(3)),
    )
    graph = _published_graph(tmp_path, documents)
    context = PredictionContext.from_document(_transition("rr-query"))
    advisor_payload = '{"preferences": []}'
    config = PredictionDecisionConfig(
        execute_probability_threshold=0.05, execute_margin=0.01, min_execute_support=2
    )
    probe = PatternGraphPredictor(graph, config=config)
    first = probe.predict(
        context, predicted_at=LEARNED_AT, horizon_seconds=3600, source_bindings=_BINDINGS
    ).run
    first_keys = [str(item.payload["branch_key"]) for item in first.candidates]
    assert "拉上窗帘" not in {item.semantics for item in first.candidates}
    retry_probe = probe.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
        excluded_branch_keys=tuple(sorted(first_keys)),
    ).run
    assert retry_probe.candidates[0].semantics == "拉上窗帘"

    def _verdicts(keys, allowed) -> str:
        items = ", ".join(
            f'{{"branch_key": "{key}", "allowed": {"true" if allowed else "false"},'
            f' "reason": "记忆边界"}}'
            for key in keys
        )
        return f'{{"verdicts": [{items}]}}'

    blocked_all = _verdicts(first_keys, allowed=False)
    allow_retry = _verdicts(
        [str(item.payload["branch_key"]) for item in retry_probe.candidates], allowed=True
    )
    advisor = PatternGraphLLMAdvisor(_client(advisor_payload))
    predictor = PatternGraphPredictor(
        graph, config=config, advisor_digest=advisor.advisor_digest
    )

    plan = _resolve(
        predictor,
        advisor,
        context,
        qualifier=lambda run: True,
        checker=PredictionConstraintChecker(_client(blocked_all, allow_retry)),
        memory_context=("晚上尽量不开空调",),
    )

    assert plan.tier == "execute"
    assert plan.run.candidates[0].semantics == "拉上窗帘"
    assert plan.constraint_status == "passed"
    assert plan.verdicts_digest is not None

    denied = _resolve(
        predictor,
        PatternGraphLLMAdvisor(_client(advisor_payload)),
        context,
        qualifier=lambda run: True,
        checker=PredictionConstraintChecker(
            _client(
                blocked_all,
                _verdicts(
                    [str(item.payload["branch_key"]) for item in retry_probe.candidates],
                    allowed=False,
                ),
            )
        ),
        memory_context=("晚上尽量不开空调",),
    )
    assert denied.tier == "suggest"
    assert denied.constraint_status == "violated"
    assert denied.reasons[0] == "memory_constraint_violation"


def test_memory_context_is_validated_at_the_entry_point(tmp_path) -> None:
    documents = tuple(_transition(f"val-{index}") for index in range(4))
    graph = _published_graph(tmp_path, documents)
    context = PredictionContext.from_document(_transition("val-query"))
    advisor = PatternGraphLLMAdvisor(_client())
    predictor = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(execute_probability_threshold=0.9),
        advisor_digest=advisor.advisor_digest,
    )

    with pytest.raises(ValueError, match="non-empty text"):
        _resolve(predictor, advisor, context, memory_context=("",))
    with pytest.raises(TypeError, match="sequence of text entries"):
        _resolve(predictor, advisor, context, memory_context="晚上尽量不开空调")
    with pytest.raises(ValueError, match="bounded entry count"):
        _resolve(predictor, advisor, context, memory_context=tuple(f"条目{i}" for i in range(25)))
