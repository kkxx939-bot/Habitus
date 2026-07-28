"""Agent 框架可直接复用的记忆写入、等待和召回门面。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from memory.conversation import ConversationAddress
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryKind
from memory.retrieval import MemorySearchResult
from Runtime import (
    MemoryConsistencySnapshot,
    Runtime,
    RuntimeConversationProtocolIngestResult,
)


@dataclass(frozen=True)
class AgentRememberResult:
    """协议写入结果以及调用方要求等待时得到的逐 Job 一致性终态。"""

    ingest: RuntimeConversationProtocolIngestResult
    consistency: tuple[MemoryConsistencySnapshot, ...]


class AgentMemoryGateway:
    """不绑定 LangChain、OpenAI Agents 或其他 Agent SDK 的公共边界。"""

    def __init__(self, runtime: Runtime) -> None:
        if not isinstance(runtime, Runtime):
            raise TypeError("runtime must be Runtime")
        self.runtime = runtime

    async def remember(
        self,
        address: ConversationAddress,
        *,
        protocol: str,
        payload: object,
        start_sequence: int,
        occurred_at: datetime,
        after_turn: bool | None = None,
        wait_timeout_seconds: float | None = None,
    ) -> AgentRememberResult:
        ingest = await self.runtime.append_protocol_conversation(
            address,
            protocol=protocol,
            payload=payload,
            start_sequence=start_sequence,
            occurred_at=occurred_at,
            after_turn=after_turn,
        )
        if wait_timeout_seconds is None:
            return AgentRememberResult(ingest, ())
        settled = tuple(
            await asyncio.gather(
                *(
                    self.runtime.wait_memory_consistency(
                    job,
                    timeout_seconds=wait_timeout_seconds,
                )
                    for job in ingest.ingest.jobs
                )
            )
        )
        return AgentRememberResult(ingest, settled)

    async def recall(
        self,
        query: str,
        *,
        conversation: ConversationAddress | None = None,
        limit: int | None = None,
        kinds: tuple[MemoryKind, ...] = (),
        intention_scope: MemoryIntentionRecallScope = MemoryIntentionRecallScope.ACTIVE,
    ) -> MemorySearchResult:
        return await self.runtime.search_memory(
            query,
            conversation=conversation,
            limit=limit,
            kinds=kinds,
            intention_scope=intention_scope,
        )


__all__ = ["AgentMemoryGateway", "AgentRememberResult"]
