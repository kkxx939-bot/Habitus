"""Runtime 对 Conversation Source 恢复与双 Consumer 装配的集成验证。"""

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path

from conversation import (
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    conversation_source_request_digest,
)
from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import ConversationAddress
from pre.conversation import ConversationBatch
from Runtime import build_runtime
from tests.helpers import closed_turn
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies


def test_runtime_start_recovers_durable_source_with_only_missing_receipts(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path),
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )
        runtime.initialize()
        started_on = date(2026, 8, 7)
        batch = ConversationBatch("source-recovery", closed_turn())
        request_digest = conversation_source_request_digest(
            conversation_id=batch.conversation_id,
            started_on=started_on,
            protocol="normalized",
            batch=batch,
            after_turn=False,
            omit_tool_call_ids=frozenset(),
        )
        envelope = ConversationSourceEnvelope.create(
            conversation_id=batch.conversation_id,
            started_on=started_on,
            protocol="normalized",
            batch=batch,
            after_turn=False,
            omit_tool_call_ids=frozenset(),
            delivery_id=request_digest,
            request_digest=request_digest,
            created_at=datetime(2026, 8, 7, 1, 0, tzinfo=timezone.utc),
        )
        runtime.components.conversation.sources.put(envelope)
        assert runtime.components.conversation.source_recovery.pending()[0].missing_consumers == (
            ConversationSourceConsumer.MEMORY,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        )

        await runtime.start()

        assert runtime.components.conversation.source_recovery.pending() == ()
        assert await runtime.read_live_conversation(ConversationAddress(batch.conversation_id, started_on)) == batch
        memory_receipt = runtime.components.conversation.source_receipts.read(
            envelope.source_id,
            ConversationSourceConsumer.MEMORY,
        )
        projection_receipt = runtime.components.conversation.source_receipts.read(
            envelope.source_id,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        )
        assert memory_receipt is not None
        assert projection_receipt is not None
        assert projection_receipt.result_id is not None
        projection = runtime.components.conversation.behavior_projections.read(projection_receipt.result_id)
        assert projection is not None and projection.source_id == envelope.source_id
        await runtime.close()

    asyncio.run(scenario())
