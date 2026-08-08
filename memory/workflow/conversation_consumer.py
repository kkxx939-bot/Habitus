"""把现有 ConversationMemoryEnqueuer 包装为耐久 Source Consumer。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone

from conversation.source import (
    ConversationConsumerRunDisposition,
    ConversationConsumerRunResult,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
)
from conversation.source.fence import ConversationConsumerExecutionLease
from foundation.integrity import canonical_digest
from memory.conversation import (
    ConversationAddress,
    ConversationIngressRequest,
    ConversationMessageJournal,
    ConversationSemanticBoundaryScorer,
)
from memory.workflow.conversation_output import (
    MEMORY_CONVERSATION_OUTPUT_SCHEMA_VERSION,
    MemoryConversationOutput,
    MemoryConversationOutputStore,
)
from memory.workflow.ingest import ConversationMemoryEnqueuer, ConversationMemoryIngestResult

# TODO(conversation-source): Reducer、Chunker、Retention 或边界评分语义变化时，必须同步提升本版本
# 以及 fingerprint 中对应的组件版本；这些版本当前由人工维护。
MEMORY_CONVERSATION_PROCESSOR_SCHEMA_VERSION = "memory_conversation_processor_v1"


class MemoryConversationConsumer:
    """只在本地派生 Batch 上调用既有 Reducer/Chunker/Journal/Retention 主链。"""

    consumer = ConversationSourceConsumer.MEMORY

    def __init__(
        self,
        enqueuer: ConversationMemoryEnqueuer,
        journal: ConversationMessageJournal,
        boundary_scorer: ConversationSemanticBoundaryScorer,
        output_store: MemoryConversationOutputStore,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if enqueuer.conversations is not journal:
            raise ValueError("Memory Conversation Consumer must share the enqueuer journal")
        if not isinstance(output_store, MemoryConversationOutputStore):
            raise TypeError("output_store must be MemoryConversationOutputStore")
        self.enqueuer = enqueuer
        self.journal = journal
        self.boundary_scorer = boundary_scorer
        self.output_store = output_store
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        # 显式版本与确定性配置共同形成契约；不依赖 callable repr 或进程地址。
        # TODO(conversation-source): 下列组件版本必须与各算法入口处的版本提示成对更新。
        self.processor_fingerprint = canonical_digest(
            {
                "schema_version": MEMORY_CONVERSATION_PROCESSOR_SCHEMA_VERSION,
                "output_schema_version": MEMORY_CONVERSATION_OUTPUT_SCHEMA_VERSION,
                "tool_result_reducer_version": "conversation_tool_result_reducer_v1",
                "text_chunker_version": "conversation_message_chunker_v1",
                "retention_planner_version": "conversation_retention_planner_v1",
                "segmentation_config": asdict(enqueuer.retention_planner.config),
                "message_chunker_max_tokens": enqueuer.message_chunker.max_message_tokens,
                "semantic_boundary": {
                    "version": "conversation_semantic_boundary_scorer_v1",
                    "embedding_fingerprint": boundary_scorer.embedding_fingerprint,
                    "max_unit_chars": boundary_scorer.max_unit_chars,
                },
            }
        )

    async def execute(
        self,
        envelope: ConversationSourceEnvelope,
        lease: ConversationConsumerExecutionLease,
    ) -> ConversationConsumerRunResult:
        if not isinstance(envelope, ConversationSourceEnvelope):
            raise TypeError("envelope must be ConversationSourceEnvelope")
        address = ConversationAddress(envelope.conversation_id, envelope.started_on)
        ingress = ConversationIngressRequest(envelope.delivery_id, envelope.request_digest)
        appended = await asyncio.to_thread(
            self.enqueuer.append,
            address,
            envelope.batch,
            omit_tool_call_ids=envelope.omit_tool_call_ids,
            ingress=ingress,
        )
        boundary_hints = None
        preview = await asyncio.to_thread(
            self.enqueuer.preview_retention,
            address,
            after_turn=envelope.after_turn,
        )
        if preview.should_seal:
            live = await asyncio.to_thread(self.journal.read_live, address)
            boundary_hints = await self.boundary_scorer.score(live)
        jobs, retention = await asyncio.to_thread(
            self.enqueuer.enqueue_ready_segments,
            address,
            after_turn=envelope.after_turn,
            flush=False,
            boundary_hints=boundary_hints,
        )
        ingest = ConversationMemoryIngestResult(appended, jobs, retention)
        output = MemoryConversationOutput.create(
            source=envelope,
            processor_fingerprint=self.processor_fingerprint,
            ingest_result=ingest,
            recorded_at=self.clock(),
        )
        stored = await lease.run_fenced(lambda: self.output_store.put(envelope, output))
        return ConversationConsumerRunResult(
            disposition=ConversationConsumerRunDisposition.OUTPUT_WRITTEN,
            output_ref=self.output_store.ref(stored),
            skip_reason=None,
            runtime_result=ingest,
        )


__all__ = ["MEMORY_CONVERSATION_PROCESSOR_SCHEMA_VERSION", "MemoryConversationConsumer"]
