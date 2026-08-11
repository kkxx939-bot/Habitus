"""行为预测 LLM 蒸馏与在线顾问的假 Provider 合同测试。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from memory.model import MemoryKind
from ModelClient import (
    ChatClient,
    ChatModelConfig,
    ModelResponse,
    ModelTransportError,
    PreparedChatRequest,
    ProviderCapabilities,
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
from prediction.learning import PredictionPatternLearner, behavior_branch_catalog
from prediction.learning.keys import logical_state_key, sequence_state_identity
from prediction.predictor import PatternGraphPredictor, PredictionDecisionConfig
from Runtime.prediction_llm import (
    PatternGraphLLMAdvisor,
    PredictionConstraintChecker,
    distill_behavior_prior,
    predict_with_advice,
)
from tests.helpers import document
from tests.model_helpers import prepare_chat_request
from tests.unit.prediction.test_pattern_learning import _WATER, LEARNED_AT, _transition

_BINDINGS = (
    PredictionRunSourceBinding("behavior://behaviors/events/2026/08/08/runtime.md", 1, "b" * 64),
)


@dataclass
class QueueProvider:
    responses: list[ModelResponse]
    provider_name: str = "test-provider"
    model: str = "test-model"
    is_remote: bool = False
    capabilities: ProviderCapabilities = ProviderCapabilities(structured_output_mode="json_schema")

    def __post_init__(self) -> None:
        self.requests: list = []

    prepare = staticmethod(prepare_chat_request)

    def complete(self, request: PreparedChatRequest) -> ModelResponse:
        self.requests.append(request.request)
        return self.responses.pop(0)

    async def complete_async(self, request: PreparedChatRequest) -> ModelResponse:
        return self.complete(request)

    def stream(self, request: PreparedChatRequest):
        return iter(())

    async def stream_async(self, request: PreparedChatRequest):
        if False:
            yield None

    def health_check(self) -> dict[str, object]:
        return {"ok": True}

    async def aclose(self) -> None:
        return None


def _client(*payloads: str, retries: int = 1, model: str = "test-model") -> StructuredChatClient:
    provider = QueueProvider([
        ModelResponse(payload, model, "test-provider", finish_reason="stop")
        for payload in payloads
    ])
    config = ChatModelConfig(
        ProviderConfig(
            provider="test-provider",
            adapter="test-adapter",
            model=model,
            max_retries=0,
        ),
        structured_output_mode="json_schema",
    )
    return StructuredChatClient(ChatClient(config, provider), validation_retries=retries)


def test_prior_distillation_maps_persona_to_known_behaviors_only() -> None:
    samples = (_transition("llm-a"), _transition("llm-b", label=_WATER))
    catalog = behavior_branch_catalog(samples)
    persona = document(
        MemoryKind.PROFILE,
        fields={"content": "- 怕热\n- 独居"},
    )
    client = _client('{"shares": [{"behavior": "打开空调", "share": 0.6}]}')

    prior = asyncio.run(
        distill_behavior_prior(
            client,
            persona_documents=(persona,),
            branch_catalog=catalog,
            domain="device_control",
        )
    )

    root_key = logical_state_key(sequence_state_identity("action", "device_control", None))
    pseudo = prior.pseudo_counts(root_key)
    assert len(pseudo) == 1
    assert next(iter(pseudo.values())) == pytest.approx(0.6 * 2.0)


def test_prior_distillation_retries_when_shares_exceed_one() -> None:
    samples = (_transition("llm-c"), _transition("llm-d", label=_WATER))
    catalog = behavior_branch_catalog(samples)
    client = _client(
        '{"shares": [{"behavior": "打开空调", "share": 0.8}, {"behavior": "倒一杯水", "share": 0.7}]}',
        '{"shares": [{"behavior": "打开空调", "share": 0.5}]}',
    )

    prior = asyncio.run(
        distill_behavior_prior(
            client,
            persona_documents=(),
            branch_catalog=catalog,
            domain="device_control",
        )
    )

    root_key = logical_state_key(sequence_state_identity("action", "device_control", None))
    assert sum(prior.pseudo_counts(root_key).values()) == pytest.approx(1.0)


def test_prior_distillation_normalizes_domain_through_the_vocabulary() -> None:
    samples = (_transition("llm-norm-a"), _transition("llm-norm-b", label=_WATER))
    catalog = behavior_branch_catalog(samples)
    client = _client('{"shares": [{"behavior": "打开空调", "share": 0.4}]}')

    prior = asyncio.run(
        distill_behavior_prior(
            client,
            persona_documents=(),
            branch_catalog=catalog,
            domain="Device_Control",
        )
    )

    root_key = logical_state_key(sequence_state_identity("action", "device_control", None))
    assert prior.pseudo_counts(root_key)


def test_advisor_digest_binds_the_model_route() -> None:
    first = PatternGraphLLMAdvisor(_client(model="deepseek-chat"))
    second = PatternGraphLLMAdvisor(_client(model="gpt-99-turbo"))

    assert first.advisor_digest != second.advisor_digest
    assert first.advisor_digest == PatternGraphLLMAdvisor(_client(model="deepseek-chat")).advisor_digest



def test_advisor_receives_memory_context_and_checker_reviews_candidates(tmp_path) -> None:
    documents = (
        _transition("mc-a"),
        _transition("mc-b"),
        _transition("mc-c", label=_WATER),
        _transition("mc-d", label=_WATER),
    )
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(documents)
    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    graph = PredictionPatternGraph(tree)
    context = PredictionContext.from_document(_transition("mc-query"))
    config = PredictionDecisionConfig(
        execute_probability_threshold=0.25,
        execute_margin=0.05,
        min_execute_support=2,
    )
    probe = PatternGraphPredictor(graph, config=config)
    baseline = probe.predict(
        context, predicted_at=LEARNED_AT, horizon_seconds=3600, source_bindings=_BINDINGS
    )
    water_key = next(
        str(item.payload["branch_key"])
        for item in baseline.run.candidates
        if item.semantics == "倒一杯水"
    )

    advisor = PatternGraphLLMAdvisor(
        _client(f'{{"preferences": [{{"branch_key": "{water_key}", "weight": 1.0}}]}}')
    )
    predictor = PatternGraphPredictor(graph, config=config, advisor_digest=advisor.advisor_digest)
    decision = asyncio.run(
        predict_with_advice(
            predictor,
            advisor,
            context,
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
            memory_context=("未完成事项「给家人倒水」(状态 open)",),
        )
    )
    assert decision.run.candidates[0].semantics == "倒一杯水"
    provider = advisor.client.client.provider
    assert isinstance(provider, QueueProvider)
    provider_request = provider.requests[0]
    assert any(
        "未完成事项「给家人倒水」" in message.content for message in provider_request.messages
    )

    ac_key = next(
        str(item.payload["branch_key"])
        for item in baseline.run.candidates
        if item.semantics == "打开空调"
    )
    checker = PredictionConstraintChecker(
        _client(
            f'{{"verdicts": [{{"branch_key": "{ac_key}", "allowed": false,'
            f' "reason": "晚间不开空调"}},'
            f' {{"branch_key": "{water_key}", "allowed": true, "reason": "无冲突"}}]}}'
        )
    )
    verdicts = asyncio.run(
        checker.review(baseline.run, memory_context=("晚上尽量不开空调",))
    )
    assert verdicts[ac_key]["allowed"] is False
    assert verdicts[water_key]["allowed"] is True
    assert asyncio.run(
        checker.review(baseline.run, memory_context=())
    ) == {}


def test_risk_blocked_decisions_never_consult_the_advisor(tmp_path) -> None:
    from tests.unit.prediction.test_pattern_learning import _consequence

    transitions = (
        *(_transition(f"riskadv-{index}") for index in range(7)),
        *(_transition(f"riskadv-w{index}", label=_WATER) for index in range(6)),
    )
    consequences = tuple(
        _consequence(
            f"riskadv-{index}",
            outcome_type="correction",
            valence="negative",
            outcome_semantics="用户随即关闭了空调",
            delay_seconds=30.0,
        )
        for index in range(3)
    )
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(transitions)
    patterns = PredictionPatternLearner().learn(
        transitions, learned_at=LEARNED_AT, consequences=consequences
    )
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    graph = PredictionPatternGraph(tree)
    context = PredictionContext.from_document(_transition("riskadv-query"))
    silent_advisor = PatternGraphLLMAdvisor(_client())
    predictor = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(
            execute_probability_threshold=0.3,
            execute_margin=0.08,
            min_execute_support=2,
        ),
        advisor_digest=silent_advisor.advisor_digest,
    )

    decision = asyncio.run(
        predict_with_advice(
            predictor,
            silent_advisor,
            context,
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
        )
    )

    assert not decision.execute
    assert "negative_outcome_risk" in decision.blocked_gates
    assert "margin_below_threshold" in decision.blocked_gates


def test_predict_with_advice_gates_the_llm_and_reorders_uncertain_ties(tmp_path) -> None:
    documents = (
        _transition("adv-a"),
        _transition("adv-b"),
        _transition("adv-c", label=_WATER),
        _transition("adv-d", label=_WATER),
    )
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(documents)
    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    graph = PredictionPatternGraph(tree)
    context = PredictionContext.from_document(_transition("adv-query"))
    config = PredictionDecisionConfig(
        execute_probability_threshold=0.25,
        execute_margin=0.05,
        min_execute_support=2,
    )

    probe = PatternGraphPredictor(graph, config=config)
    baseline = probe.predict(
        context, predicted_at=LEARNED_AT, horizon_seconds=3600, source_bindings=_BINDINGS
    )
    assert not baseline.execute
    assert "margin_below_threshold" in baseline.blocked_gates
    water_key = next(
        str(item.payload["branch_key"])
        for item in baseline.run.candidates
        if item.semantics == "倒一杯水"
    )

    advisor = PatternGraphLLMAdvisor(
        _client(f'{{"preferences": [{{"branch_key": "{water_key}", "weight": 1.0}}]}}')
    )
    predictor = PatternGraphPredictor(
        graph, config=config, advisor_digest=advisor.advisor_digest
    )
    decision = asyncio.run(
        predict_with_advice(
            predictor,
            advisor,
            context,
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
        )
    )

    assert decision.run.candidates[0].semantics == "倒一杯水"
    assert decision.run.predictor_digest == predictor.predictor_digest

    routine_documents = tuple(_transition(f"adv-sure-{index}") for index in range(4))
    routine_tree = PredictionTree(tmp_path / "prediction-sure")
    PredictionPublisher(routine_tree).publish(routine_documents)
    routine_patterns = PredictionPatternLearner().learn(routine_documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(routine_tree).publish(routine_patterns)
    silent_advisor = PatternGraphLLMAdvisor(_client())
    confident_predictor = PatternGraphPredictor(
        PredictionPatternGraph(routine_tree),
        config=config,
        advisor_digest=silent_advisor.advisor_digest,
    )
    confident = asyncio.run(
        predict_with_advice(
            confident_predictor,
            silent_advisor,
            PredictionContext.from_document(_transition("adv-sure-query")),
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
        )
    )
    assert confident.execute


def test_advisor_unavailability_falls_back_to_the_unadvised_decision(tmp_path) -> None:
    documents = (
        _transition("adv-down-a"),
        _transition("adv-down-b"),
        _transition("adv-down-c", label=_WATER),
        _transition("adv-down-d", label=_WATER),
    )
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(documents)
    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    graph = PredictionPatternGraph(tree)
    context = PredictionContext.from_document(_transition("adv-down-query"))
    config = PredictionDecisionConfig(
        execute_probability_threshold=0.25,
        execute_margin=0.05,
        min_execute_support=2,
    )

    provider = QueueProvider([])

    def _raise(request: PreparedChatRequest) -> ModelResponse:
        raise ModelTransportError("provider down")

    provider.complete = _raise  # type: ignore[method-assign]
    broken = StructuredChatClient(
        ChatClient(
            ChatModelConfig(
                ProviderConfig(
                    provider="test-provider",
                    adapter="test-adapter",
                    model="test-model",
                    max_retries=0,
                ),
                structured_output_mode="json_schema",
            ),
            provider,
        )
    )
    advisor = PatternGraphLLMAdvisor(broken)
    predictor = PatternGraphPredictor(
        graph, config=config, advisor_digest=advisor.advisor_digest
    )

    decision = asyncio.run(
        predict_with_advice(
            predictor,
            advisor,
            context,
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
        )
    )

    assert not decision.execute
    assert "margin_below_threshold" in decision.blocked_gates
    assert decision.run.advisor_adjustment is None
