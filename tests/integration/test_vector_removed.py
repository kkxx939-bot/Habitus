"""向量与重排整个删掉之后：不配向量也能起来；旧配置、旧盘面明确报错；摘要模型看当地时间。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from habitus.config import ConfigError, HabitusConfig
from habitus.conversation.source import ConversationSourceError
from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import ConversationSummaryGenerator
from habitus.memory.workflow import MemoryConversationOutput
from habitus.pre.conversation import ConversationMessageRole
from habitus.runtime import build_runtime
from habitus.runtime.runtime_lifecycle import VectorLayoutError
from tests.helpers import message, segment
from tests.integration.test_runtime_assembly import REPOSITORY_ROOT, runtime_config, runtime_dependencies
from tests.lock_store import ProcessLocalLockStore
from tests.unit.conversation.test_summary_generation import RecordingProvider, content_dict, structured


def _example() -> dict:  # type: ignore[type-arg]
    return yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))


def test_a_config_without_any_vector_or_rerank_model_starts_and_reports_healthy(tmp_path: Path) -> None:
    runtime = build_runtime(
        runtime_config(tmp_path), providers=runtime_dependencies(), path_lock=PathLock(ProcessLocalLockStore())
    )

    async def scenario() -> None:
        runtime.initialize()
        report = await runtime.health()
        names = {check.name for check in report.checks}
        assert "model_chat" in names
        assert not names & {"memory_vector", "summary_vector", "model_embedding", "model_rerank"}
        assert report.ready
        await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("group", "name"),
    [
        ("models", "embedding"),
        ("models", "rerank"),
        ("memory", "vector_store"),
        ("memory", "vector_index"),
        ("memory", "semantic_search"),
        ("conversation", "summary_vector_store"),
        ("conversation", "summary_vector_index"),
    ],
)
def test_a_config_that_still_has_a_vector_or_rerank_section_names_it_and_says_to_delete_it(
    group: str, name: str
) -> None:
    payload = _example()
    payload.setdefault(group, {})[name] = {"route": {"adapter": "x"}}

    with pytest.raises(ConfigError, match=rf"已经从 Habitus 移除.*config\.{group}\.{name}"):
        HabitusConfig.from_mapping(payload)


def test_a_leftover_vector_store_directory_is_refused_at_startup(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    (config.storage_root / "vector").mkdir(parents=True)
    runtime = build_runtime(config, providers=runtime_dependencies(), path_lock=PathLock(ProcessLocalLockStore()))

    with pytest.raises(VectorLayoutError, match="vector search has been removed"):
        runtime.initialize()


def test_an_old_output_file_carrying_the_vector_fingerprint_is_refused_by_name() -> None:
    with pytest.raises(ConversationSourceError, match="embedding_fingerprint"):
        MemoryConversationOutput.from_dict(
            {
                "schema_version": "memory_conversation_output_v1",
                "output_id": "0" * 64,
                "source_id": "0" * 64,
                "source_payload_digest": "0" * 64,
                "consumer": "memory",
                "processor_fingerprint": "0" * 64,
                "ingest_result_snapshot": {},
                "recorded_at": "2026-01-01T00:00:00Z",
                "output_record_digest": "0" * 64,
            }
        )


def test_the_summary_model_sees_the_local_time_of_a_morning_conversation() -> None:
    """当地 03-24 早上 07:31 的对话：给摘要模型的时刻是 07:31+08:00，不是 UTC 的前一天 23:31。"""

    morning = datetime(2026, 3, 23, 23, 31, tzinfo=UTC)
    source = segment(
        messages=(
            replace(message(0, ConversationMessageRole.PROMPT, "记一下体重 62.7"), occurred_at=morning),
            replace(
                message(1, ConversationMessageRole.COMPLETION, "记好了"), occurred_at=morning + timedelta(seconds=5)
            ),
        )
    )
    provider = RecordingProvider([content_dict()])
    generator = ConversationSummaryGenerator(structured(provider), zone=timezone(timedelta(hours=8)))

    asyncio.run(generator.generate(source))

    prompt = provider.requests[0].messages[-1].content
    assert "2026-03-24T07:31:00+08:00" in prompt
    assert "2026-03-23T23:31" not in prompt
    assert "当地时间" in provider.requests[0].messages[-2].content
