"""把 Runtime 适配为 Agent 生命周期可依赖的稳定记忆能力口。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from habitus.integrations.sdk.contracts import (
    AgentFlushResult,
    AgentMemoryConsistency,
    AgentMemoryJob,
    AgentRecallDegradation,
    AgentRecallMemory,
    AgentRecallResult,
    AgentRecallSummary,
    AgentRememberResult,
    ConversationRef,
)
from habitus.memory.conversation import ConversationAddress, ConversationSummaryReference
from habitus.memory.intention import MemoryIntentionRecallScope
from habitus.memory.model import MemoryKind
from habitus.memory.workflow import MemoryJob
from habitus.runtime import MemoryConsistencySnapshot, Runtime


@dataclass(frozen=True)
class AgentGatewayRememberDetails:
    """供同进程传输适配器关联内部 Job 观测的结果。"""

    public: AgentRememberResult
    runtime_jobs: tuple[MemoryJob, ...]


class AgentMemoryGateway:
    """不绑定 LangChain、OpenAI Agents 或其他 Agent SDK 的公共边界。"""

    def __init__(self, runtime: Runtime) -> None:
        if not isinstance(runtime, Runtime):
            raise TypeError("runtime must be Runtime")
        self.runtime = runtime

    async def remember(
        self,
        conversation: ConversationRef,
        *,
        protocol: str,
        payload: object,
        start_sequence: int,
        occurred_at: datetime,
        after_turn: bool | None = None,
        wait_timeout_seconds: float | None = None,
        delivery_id: str | None = None,
    ) -> AgentRememberResult:
        return (
            await self.remember_with_runtime_jobs(
                conversation,
                protocol=protocol,
                payload=payload,
                start_sequence=start_sequence,
                occurred_at=occurred_at,
                after_turn=after_turn,
                wait_timeout_seconds=wait_timeout_seconds,
                delivery_id=delivery_id,
            )
        ).public

    async def remember_with_runtime_jobs(
        self,
        conversation: ConversationRef,
        *,
        protocol: str,
        payload: object,
        start_sequence: int,
        occurred_at: datetime,
        after_turn: bool | None = None,
        wait_timeout_seconds: float | None = None,
        delivery_id: str | None = None,
    ) -> AgentGatewayRememberDetails:
        """写入并额外保留仅供同进程 HTTP 观测使用的 Runtime Job。"""

        address = self._address(conversation)
        ingest = await self.runtime.append_protocol_conversation(
            address,
            protocol=protocol,
            payload=payload,
            start_sequence=start_sequence,
            occurred_at=occurred_at,
            after_turn=after_turn,
            delivery_id=delivery_id,
        )
        settled = await self._wait(ingest.ingest.jobs, wait_timeout_seconds)
        return AgentGatewayRememberDetails(
            public=AgentRememberResult(
                ignored_items=ingest.adaptation.ignored_items,
                after_turn=ingest.effective_after_turn,
                next_sequence=ingest.next_sequence,
                jobs=tuple(self._job(job) for job in ingest.ingest.jobs),
                consistency=tuple(self._consistency(snapshot) for snapshot in settled),
            ),
            runtime_jobs=ingest.ingest.jobs,
        )

    async def recall(
        self,
        query: str,
        *,
        conversation: ConversationRef | None = None,
        limit: int | None = None,
        kinds: tuple[str, ...] = (),
        intention_scope: str = "active",
    ) -> AgentRecallResult:
        result = await self.runtime.search_memory(
            query,
            conversation=None if conversation is None else self._address(conversation),
            limit=limit,
            kinds=tuple(MemoryKind(kind) for kind in kinds),
            intention_scope=MemoryIntentionRecallScope(intention_scope),
        )
        return AgentRecallResult(
            query=result.query,
            queries=tuple(item.query for item in result.plan.queries),
            context=result.context,
            memories=tuple(
                AgentRecallMemory(
                    uri=str(memory.uri),
                    score=memory.hit.score,
                    matched_queries=memory.matched_queries,
                )
                for memory in result.memories
            ),
            summaries=tuple(
                AgentRecallSummary(reference=match.reference.identity, score=match.score)
                for match in result.summary_fallbacks
            ),
            degradations=tuple(
                AgentRecallDegradation(stage=item.stage.value, error_type=item.error_type)
                for item in result.degradations
            ),
            budget_exhausted=result.budget_exhausted,
        )

    async def flush(
        self,
        conversation: ConversationRef,
        *,
        wait_timeout_seconds: float | None = None,
    ) -> AgentFlushResult:
        """在会话关闭边界提交剩余完整轮次，并可选择等待长期记忆终态。"""

        flushed = await self.runtime.flush_conversation(self._address(conversation))
        settled = await self._wait(flushed.jobs, wait_timeout_seconds)
        return AgentFlushResult(
            jobs=tuple(self._job(job) for job in flushed.jobs),
            consistency=tuple(self._consistency(snapshot) for snapshot in settled),
        )

    async def record_use(
        self,
        *,
        memory_uris: tuple[str, ...] = (),
        summary_references: tuple[str, ...] = (),
        used_at: datetime | None = None,
    ) -> None:
        """手动兼容入口；标准 recall 已按最终模型可见 Context 自动计入，勿重复上报。"""

        await self.runtime.record_memory_use(
            memory_uris=memory_uris,
            summary_references=tuple(
                ConversationSummaryReference.parse(value) for value in summary_references
            ),
            used_at=used_at,
        )

    async def cursor(self, conversation: ConversationRef) -> int:
        """读取服务端耐久游标，供 Agent 进程重启后继续会话。"""

        return await self.runtime.conversation_cursor(self._address(conversation))

    async def _wait(
        self,
        jobs: tuple[MemoryJob, ...],
        timeout_seconds: float | None,
    ) -> tuple[MemoryConsistencySnapshot, ...]:
        if timeout_seconds is None or not jobs:
            return ()
        return tuple(
            await asyncio.gather(
                *(
                    self.runtime.wait_memory_consistency(job, timeout_seconds=timeout_seconds)
                    for job in jobs
                )
            )
        )

    @staticmethod
    def _job(job: MemoryJob) -> AgentMemoryJob:
        return AgentMemoryJob(
            memory_sequence=job.memory_sequence,
            conversation_id=job.conversation_id,
            started_on=job.started_on,
            status=job.status.value,
        )

    @staticmethod
    def _consistency(snapshot: MemoryConsistencySnapshot) -> AgentMemoryConsistency:
        return AgentMemoryConsistency(
            memory_sequence=snapshot.requested_job.memory_sequence,
            state=snapshot.state.value,
        )

    @staticmethod
    def _address(conversation: ConversationRef) -> ConversationAddress:
        if not isinstance(conversation, ConversationRef):
            raise TypeError("conversation must be ConversationRef")
        return ConversationAddress(conversation.conversation_id, conversation.started_on)


__all__ = ["AgentGatewayRememberDetails", "AgentMemoryGateway"]
