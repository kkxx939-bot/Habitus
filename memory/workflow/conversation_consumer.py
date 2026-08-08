"""把现有 ConversationMemoryEnqueuer 包装为 SourceEnvelope Consumer。"""

from __future__ import annotations

import asyncio

from conversation.source import (
    ConversationConsumerExecution,
    ConversationConsumerReceipt,
    ConversationConsumerReceiptState,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    ConversationSourceError,
    conversation_source_request_digest,
)
from foundation.integrity import canonical_digest
from memory.conversation import (
    ConversationAddress,
    ConversationAppendResult,
    ConversationAppendStatus,
    ConversationIngressError,
    ConversationIngressRequest,
    ConversationIngressState,
    ConversationMessageJournal,
    ConversationSemanticBoundaryScorer,
)
from memory.workflow.ingest import ConversationMemoryEnqueuer, ConversationMemoryIngestResult


class MemoryConversationConsumer:
    """只在本地派生 Batch 上调用既有 Reducer/Chunker/Journal/Retention 主链。"""

    consumer = ConversationSourceConsumer.MEMORY

    def __init__(
        self,
        enqueuer: ConversationMemoryEnqueuer,
        journal: ConversationMessageJournal,
        boundary_scorer: ConversationSemanticBoundaryScorer,
    ) -> None:
        if enqueuer.conversations is not journal:
            raise ValueError("Memory Conversation Consumer must share the enqueuer journal")
        self.enqueuer = enqueuer
        self.journal = journal
        self.boundary_scorer = boundary_scorer

    async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
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
        result = ConversationMemoryIngestResult(appended, jobs, retention)
        return ConversationConsumerExecution(
            state=ConversationConsumerReceiptState.SUCCEEDED,
            result=result,
            result_id=self.result_id(envelope.source_id),
            result_digest=self.result_digest(envelope),
        )

    async def completed(
        self,
        envelope: ConversationSourceEnvelope,
        receipt: ConversationConsumerReceipt,
    ) -> ConversationMemoryIngestResult:
        """不重跑终态 Consumer，只从当前 Journal 恢复兼容的只读返回值。"""

        if receipt.state is not ConversationConsumerReceiptState.SUCCEEDED:
            raise ConversationSourceError("Memory Consumer cannot have a SKIPPED receipt")
        if receipt.result_id != self.result_id(envelope.source_id):
            raise ConversationSourceError("Memory Consumer receipt has the wrong result identity")
        if receipt.result_digest != self.result_digest(envelope):
            raise ConversationSourceError("Memory Consumer receipt has the wrong terminal digest")
        address = ConversationAddress(envelope.conversation_id, envelope.started_on)
        live, next_sequence, retention = await asyncio.gather(
            asyncio.to_thread(self.journal.read_live, address),
            asyncio.to_thread(self._replay_next_sequence, envelope, address),
            asyncio.to_thread(
                self.enqueuer.preview_retention,
                address,
                # 终态 Source 不得把旧 after_turn 边界重新施加到后来追加的 live 状态。
                after_turn=False,
            ),
        )
        append = ConversationAppendResult(
            status=ConversationAppendStatus.UNCHANGED,
            appended_count=0,
            live=live,
            next_sequence=next_sequence,
        )
        return ConversationMemoryIngestResult(append=append, jobs=(), retention=retention)

    @staticmethod
    def result_id(source_id: str) -> str:
        return canonical_digest(
            {
                "schema_version": "memory_conversation_consumer_result_v1",
                "source_id": source_id,
            }
        )

    @staticmethod
    def result_digest(envelope: ConversationSourceEnvelope) -> str:
        """摘要绑定确定性的 Source 终态，不绑定 CREATED/UNCHANGED 观察结果。"""

        return canonical_digest(
            {
                "schema_version": "memory_conversation_consumer_terminal_v1",
                "source_id": envelope.source_id,
                "source_digest": envelope.content_digest,
            }
        )

    def _replay_next_sequence(
        self,
        envelope: ConversationSourceEnvelope,
        address: ConversationAddress,
    ) -> int:
        implicit_digest = conversation_source_request_digest(
            conversation_id=envelope.conversation_id,
            started_on=envelope.started_on,
            protocol=envelope.protocol,
            batch=envelope.batch,
            after_turn=envelope.after_turn,
            omit_tool_call_ids=envelope.omit_tool_call_ids,
        )
        if envelope.delivery_id == implicit_digest and envelope.request_digest == implicit_digest:
            return self.journal.next_sequence(address)
        try:
            ingress = self.journal.ingress_receipts.read(address, envelope.delivery_id)
        except ConversationIngressError as exc:
            raise ConversationSourceError("Memory Consumer ingress receipt cannot be read") from exc
        if (
            ingress is None
            or ingress.state is not ConversationIngressState.COMMITTED
            or ingress.request_digest != envelope.request_digest
            or ingress.next_sequence is None
        ):
            raise ConversationSourceError("Memory Consumer terminal receipt has no matching committed ingress")
        return ingress.next_sequence


__all__ = ["MemoryConversationConsumer"]
