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
    derive_stance_table,
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


def test_stance_table_derivation_and_bridge_consumption() -> None:
    client = _client(
        '{"stances": [{"text": "回家先开空调", "stance": "support"},'
        ' {"text": "除非下雨否则每天浇水", "stance": "support"}]}'
    )

    table = asyncio.run(
        derive_stance_table(client, ("回家先开空调", "除非下雨否则每天浇水"))
    )

    assert table.lookup("回家先开空调") == "support"
    assert table.lookup("除非下雨否则每天浇水") == "support"
    assert table.lookup("没解析过的条目") is None
    assert table.version != type(table)().version


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
