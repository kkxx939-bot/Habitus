"""一轮很长时按段抽取：每次模型请求都在预算内，整轮放得下时照旧整轮抽取（裁定 80 问题 1）。

chat 走默认路由的真实 ``OpenAICompatibleChatProvider.prepare``（token 预算与生产同一估法），只把网络发送换成按请求名字
回的脚本答复；不调真实模型。
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

import yaml

from habitus.config import HabitusConfig
from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import ConversationAddress
from habitus.memory.workflow import MemoryJobStatus
from habitus.model_client import ModelResponse
from habitus.model_client.adapters.openai_compatible_chat import OpenAICompatibleChatProvider
from habitus.pre.conversation import (
    ConversationBatch,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from habitus.runtime import build_runtime
from tests.helpers import message, summary_content
from tests.integration.test_memory_job_full_chain import (
    DurableVectorBackend,
    EmbeddingProvider,
    RerankProvider,
)
from tests.lock_store import ProcessLocalLockStore

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONVERSATION = "conversation-long-turn"
OPENING = "把仓库里每个模块的源码都读一遍，然后告诉我哪里写得不好"


class ScriptedChat(OpenAICompatibleChatProvider):
    """``prepare`` 是仓库原样（继承）；只把网络发送换成脚本答复，并记下每次真正发出的请求估出的 token。"""

    def __init__(self, config) -> None:
        super().__init__(config, api_key="test-no-network")
        self.estimates: list[tuple[str, int]] = []
        self.candidate_payloads: list[dict[str, object]] = []

    async def complete_async(self, request):
        return self.complete(request)

    def complete(self, request):
        chat_request = request.request
        name = chat_request.response_format.name
        self.estimates.append((name, request.estimated_input_tokens))
        summary = summary_content()
        payloads: dict[str, object] = {
            "conversation_segment_summary_field_operations": {
                "operations": [
                    {"field": "overview", "operation": "update", "content": summary.overview},
                    {"field": "chronology", "operation": "append", "items": list(summary.chronology)},
                    {"field": "ending_state", "operation": "update", "content": summary.ending_state},
                ]
            },
            "memory_retrieval_decision": {
                "status": "irrelevant",
                "action": "finish",
                "query": None,
                "uri": None,
                "reason": "当前没有相关旧记忆。",
            },
            "memory_candidate_batch": {
                "profile": [],
                "preferences": [],
                "entities": [],
                "tools": [],
                "events": [],
                "intentions": [],
                "identity_proposals": [],
                "relations": [],
            },
        }
        if name == "memory_candidate_batch":
            self.candidate_payloads.append(json.loads(str(chat_request.messages[-1].content).split("\n", 1)[1]))
        if name == "memory_directory_overview":
            snapshot = json.loads(str(chat_request.messages[-1].content).split("目录快照：", 1)[1])
            payloads[name] = {
                "directory_summary": "目录保存有来源支持的长期记忆。",
                "entries": [
                    {"name": item["name"], "kind": item["kind"], "summary": f"{item['name']} 的稳定摘要。"}
                    for item in snapshot["entries"]
                ],
            }
        return ModelResponse(json.dumps(payloads[name], ensure_ascii=False), self.model, self.provider_name)


def runtime_and_chat(tmp_path: Path, *, max_segment_tokens: int | None = None):
    from habitus.infrastructure.vector import VectorStoreFactory
    from habitus.model_client import ProviderFactory

    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    # chat 保持默认路由（deepseek / openai_compatible_chat / 64000 / json_object），只去掉凭据引用（不发网络）
    payload["models"]["chat"]["route"]["credential_ref"] = ""
    for route in (
        payload["models"]["embedding"]["route"],
        payload["models"]["rerank"]["route"],
        payload["memory"]["vector_store"]["route"],
        payload["conversation"]["summary_vector_store"]["route"],
    ):
        route.update(provider="fake", credential_ref="")
    payload["models"]["embedding"]["route"]["adapter"] = "fake_embedding"
    payload["models"]["rerank"]["route"]["adapter"] = "fake_rerank"
    payload["memory"]["vector_store"]["route"]["adapter"] = "fake_vector"
    payload["conversation"]["summary_vector_store"]["route"]["adapter"] = "fake_vector"
    if max_segment_tokens is not None:
        payload["conversation"]["segmentation"].update(
            max_segment_tokens=max_segment_tokens, retained_message_token_budget=max_segment_tokens
        )
    config = HabitusConfig.from_mapping(payload)

    chats: list[ScriptedChat] = []

    def chat_builder(context):
        chat = ScriptedChat(context.config)
        chats.append(chat)
        return chat

    providers = ProviderFactory()
    providers.register_adapter("chat", "openai_compatible_chat", chat_builder)
    providers.register_adapter(
        "embedding",
        "fake_embedding",
        lambda context: EmbeddingProvider(context.route.provider, context.route.model, context.config.dimension),
    )
    providers.register_adapter(
        "rerank", "fake_rerank", lambda context: RerankProvider(context.route.provider, context.route.model)
    )
    vectors = VectorStoreFactory()
    vectors.register_adapter(
        "fake_vector",
        lambda context: DurableVectorBackend(context.config.provider, context.config.collection),
    )
    runtime = build_runtime(
        config, providers=providers, vector_stores=vectors, path_lock=PathLock(ProcessLocalLockStore())
    )
    runtime.initialize()
    return runtime, chats[0]


def reading_turn(pairs: int, *, lines: int = 330, start: int = 0):
    """一句话之后连读 ``pairs`` 个文件（默认每个约 20 KB 文字），最后一条答复收尾。"""

    messages = [message(start, ConversationMessageRole.PROMPT, OPENING)]
    sequence = start + 1
    for index in range(pairs):
        call_id = f"call-{start}-{index}"
        body = "\n".join(
            f"module_{index}.py line {line}: def handler_{line}(value): return value * {line}" for line in range(330)
        )
        messages.append(
            message(
                sequence,
                ConversationMessageRole.TOOL_CALL,
                {"path": f"module_{index}.py"},
                tool_call_id=call_id,
                tool_name="Read",
            )
        )
        messages.append(
            message(
                sequence + 1,
                ConversationMessageRole.TOOL_RESULT,
                body,
                tool_call_id=call_id,
                tool_name="Read",
                tool_status=ConversationToolResultStatus.COMPLETED,
                content_mode=ConversationToolResultContentMode.INLINE,
            )
        )
        sequence += 2
    messages.append(message(sequence, ConversationMessageRole.COMPLETION, "读完了，问题集中在错误处理。"))
    return tuple(messages)


def drain(runtime) -> None:
    while True:
        oldest = runtime.components.workflow.jobs.oldest_uncommitted()
        if oldest is None or oldest.status is not MemoryJobStatus.QUEUED:
            return
        result = asyncio.run(runtime.run_next())
        assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED


def test_an_oversized_turn_is_extracted_segment_by_segment_within_the_request_budget(tmp_path: Path) -> None:
    runtime, chat = runtime_and_chat(tmp_path)
    address = ConversationAddress(CONVERSATION, date(2026, 7, 1))
    turn = reading_turn(14)
    enqueuer = runtime.components.workflow.enqueuer
    # afterTurn 只封住过大轮次的前缀，轮尾留在 live：先跑这些轮中间段的 Job，它们要从 live 读到整轮再判
    appended = enqueuer.append_and_maybe_enqueue(address, ConversationBatch(CONVERSATION, turn), after_turn=True)
    assert appended.jobs and runtime.components.conversation.journal.read_live(address) is not None
    drain(runtime)
    enqueuer.flush(address)
    drain(runtime)

    journal = runtime.components.conversation.journal
    history = journal.list_history(address)
    whole = journal.read_editor_turn(address, history[0]).whole
    assert len(history) > 1 and whole.messages == turn
    assert not runtime.components.memory.editor.turn_fits_whole(whole)

    limit = runtime.config.memory.extraction.max_input_tokens
    assert chat.estimates and max(tokens for _name, tokens in chat.estimates) <= limit
    # 每段各抽各的：候选请求一段一次，各自只带本段原文；第一段从开轮处起，后面各段带上开头那句话与前面各段的摘要
    assert len(chat.candidate_payloads) == len(history)
    for index, (payload, segment) in enumerate(zip(chat.candidate_payloads, history, strict=True)):
        assert payload["conversation_segment"]["segment_id"] == segment.segment_id
        context = payload["turn_context"]
        assert len(context["earlier_segment_summaries"]) + context["omitted_earlier_segment_count"] == index
        if index == 0:
            assert context["opening_prompts"] == []
        else:
            assert [item["content"] for item in context["opening_prompts"]] == [OPENING]
            assert context["earlier_segment_summaries"]


def test_a_turn_that_fits_is_still_extracted_once_as_a_whole_without_turn_context(tmp_path: Path) -> None:
    # 段上限调小，让一轮放得下整轮抽取、又跨了几段
    runtime, chat = runtime_and_chat(tmp_path, max_segment_tokens=2_000)
    address = ConversationAddress(CONVERSATION, date(2026, 7, 1))
    turn = reading_turn(4, lines=60)
    runtime.components.workflow.enqueuer.append_and_maybe_enqueue(
        address, ConversationBatch(CONVERSATION, turn), after_turn=True
    )
    runtime.components.workflow.enqueuer.flush(address)
    drain(runtime)

    history = runtime.components.conversation.journal.list_history(address)
    assert len(history) > 1
    assert len(chat.candidate_payloads) == 1
    (payload,) = chat.candidate_payloads
    assert "turn_context" not in payload
    assert payload["conversation_segment"]["segment_id"] == f"{0:012d}-{len(turn) - 1:012d}"
