"""Segment/Range Summary 的真实结构化生成、来源绑定和幂等边界测试。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from memory.conversation import (
    ConversationAddress,
    ConversationRangeSummaryGenerator,
    ConversationSegmentSummaryCompactionConfig,
    ConversationSummaryCompactionConfig,
    ConversationSummaryCompactionError,
    ConversationSummaryCompactionPlan,
    ConversationSummaryConfig,
    ConversationSummaryError,
    ConversationSummaryGenerator,
    ConversationSummaryService,
    ConversationSummaryStore,
)
from memory.conversation.layout import ConversationLayout
from ModelClient import (
    ChatClient,
    ChatModelConfig,
    ModelResponse,
    ProviderCapabilities,
    ProviderConfig,
    StructuredChatClient,
)
from pre.conversation import ConversationMessageRole, ConversationRangeSummaryStage
from tests.helpers import BASE_TIME, closed_turn, message, segment, segment_summary, summary_content

SUMMARY_TIME = BASE_TIME + timedelta(minutes=1)


@dataclass
class RecordingProvider:
    responses: list[dict[str, object]]
    requests: list[object] = field(default_factory=list)
    provider_name: str = "test"
    model: str = "summary-test"
    is_remote: bool = False
    capabilities: ProviderCapabilities = ProviderCapabilities()

    def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            json.dumps(self.responses.pop(0), ensure_ascii=False),
            self.model,
            self.provider_name,
        )

    async def complete_async(self, request):
        return self.complete(request)

    def stream(self, _request):
        return iter(())

    async def stream_async(self, _request):
        if False:
            yield None

    def health_check(self):
        return {"ok": True}


def structured(provider: RecordingProvider) -> StructuredChatClient:
    config = ChatModelConfig(
        ProviderConfig(
            provider="test",
            adapter="test",
            model="summary-test",
            base_url="https://example.com",
            max_retries=0,
        )
    )
    return StructuredChatClient(ChatClient(config, provider), validation_retries=0)


def content_dict() -> dict[str, object]:
    return summary_content().to_dict()


def test_segment_summary_binds_all_source_identity_in_code_and_preserves_process_semantics() -> None:
    source = segment(
        conversation_id="conversation-1",
        segment_id="000000000000-000000000003",
        messages=closed_turn() + closed_turn(start_sequence=2, prompt="改为更详细回答"),
    )
    provider = RecordingProvider([content_dict()])
    generator = ConversationSummaryGenerator(structured(provider), clock=lambda: SUMMARY_TIME)

    result = asyncio.run(generator.generate(source))

    result.require_matches_source(source)
    assert result.conversation_id == source.conversation_id
    assert result.segment_id == source.segment_id
    assert result.source_message_digest == source.digest
    assert result.generated_at == SUMMARY_TIME
    request = provider.requests[0]
    assert request.prompt_version == "conversation_segment_summary_v2"
    assert "不要把内容分类成长记忆" in request.messages[-2].content
    assert source.messages[-2].content in request.messages[-1].content


def test_partial_segment_summary_preserves_system_owned_turn_boundaries() -> None:
    source = segment(
        segment_id="000000000000-000000000000",
        messages=(message(0, ConversationMessageRole.PROMPT, "继续分析这个大任务"),),
    )
    provider = RecordingProvider([content_dict()])
    generator = ConversationSummaryGenerator(structured(provider), clock=lambda: SUMMARY_TIME)

    result = asyncio.run(generator.generate(source))

    assert not result.starts_mid_turn
    assert result.ends_mid_turn
    result.require_matches_source(source)
    request = provider.requests[0]
    assert "ends_mid_turn=true" in request.messages[-1].content


def test_segment_summary_rejects_oversized_complete_source_without_truncating_or_calling_model() -> None:
    source = segment(messages=closed_turn(prompt="x" * 10_000))
    provider = RecordingProvider([content_dict()])
    generator = ConversationSummaryGenerator(
        structured(provider),
        config=ConversationSummaryConfig(max_input_chars=1024),
    )

    with pytest.raises(ConversationSummaryError, match="input bound"):
        asyncio.run(generator.generate(source))
    assert provider.requests == []


def test_summary_service_reuses_bound_immutable_summary_without_second_model_call(tmp_path: Path) -> None:
    source = segment(segment_id="000000000000-000000000001")
    address = ConversationAddress(source.conversation_id, date(2026, 7, 1))
    store = ConversationSummaryStore(ConversationLayout(tmp_path / "conversation"))
    provider = RecordingProvider([content_dict(), content_dict()])
    service = ConversationSummaryService(
        store,
        ConversationSummaryGenerator(structured(provider), clock=lambda: SUMMARY_TIME),
    )

    first = asyncio.run(service.get_or_create(address, source))
    second = asyncio.run(service.get_or_create(address, source))

    assert first == second
    assert len(provider.requests) == 1


def test_range_summary_uses_contiguous_sources_and_system_owned_range_identity() -> None:
    first_source = segment(
        segment_id="000000000000-000000000001",
        messages=closed_turn(start_sequence=0),
    )
    second_source = segment(
        segment_id="000000000002-000000000003",
        messages=closed_turn(start_sequence=2),
    )
    first = segment_summary(first_source)
    second = segment_summary(second_source)
    plan = ConversationSummaryCompactionPlan(
        ConversationRangeSummaryStage.RANGE,
        (first, second),
        "source_count",
    )
    provider = RecordingProvider([content_dict()])
    generator = ConversationRangeSummaryGenerator(
        structured(provider),
        clock=lambda: SUMMARY_TIME,
    )

    result = asyncio.run(generator.generate(plan))

    result.require_matches_sources((first, second))
    assert result.range_id == "000000000000-000000000003"
    assert result.source_refs == plan.source_refs
    assert provider.requests[0].prompt_version == "conversation_range_summary_v2"


def test_range_summary_enforces_stage_specific_source_bound_before_model_call() -> None:
    first = replace(
        segment_summary(
            segment(segment_id="000000000000-000000000001", messages=closed_turn(start_sequence=0))
        ),
        overview="x" * 200,
    )
    second = segment_summary(
        segment(segment_id="000000000002-000000000003", messages=closed_turn(start_sequence=2))
    )
    plan = ConversationSummaryCompactionPlan(
        ConversationRangeSummaryStage.RANGE,
        (first, second),
        "source_count",
    )
    provider = RecordingProvider([content_dict()])
    compaction = ConversationSummaryCompactionConfig(
        segment_to_range=ConversationSegmentSummaryCompactionConfig(
            min_age_days=0,
            min_source_count=2,
            max_wait_days=180,
            max_source_count=20,
            max_source_chars=1024,
        )
    )
    generator = ConversationRangeSummaryGenerator(
        structured(provider),
        compaction_config=compaction,
    )

    with pytest.raises(ConversationSummaryCompactionError, match="source exceeds"):
        asyncio.run(generator.generate(plan))
    assert provider.requests == []
