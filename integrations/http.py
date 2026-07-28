"""供 FastAPI、Starlette 或原生 ASGI 路由复用的无框架 HTTP Handler。"""

from __future__ import annotations

from datetime import date, datetime

from integrations.agent import AgentMemoryGateway
from memory.conversation import ConversationAddress
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryKind
from Runtime import Runtime, RuntimeHealthReport


class RuntimeHTTPHandlers:
    """只提供稳定 handler，不擅自决定端口、认证、URL 或 Web 框架。"""

    def __init__(self, runtime: Runtime) -> None:
        if not isinstance(runtime, Runtime):
            raise TypeError("runtime must be Runtime")
        self.runtime = runtime
        self.agent = AgentMemoryGateway(runtime)

    def protocols(self) -> dict[str, object]:
        return {"protocols": list(self.runtime.conversation_protocols())}

    async def remember(
        self,
        *,
        conversation_id: str,
        started_on: date,
        protocol: str,
        payload: object,
        start_sequence: int,
        occurred_at: datetime,
        after_turn: bool | None = None,
        wait_timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        """供 HTTP 路由层调用的协议写入 handler；URL、认证和 owner 由上层决定。"""

        result = await self.agent.remember(
            ConversationAddress(conversation_id, started_on),
            protocol=protocol,
            payload=payload,
            start_sequence=start_sequence,
            occurred_at=occurred_at,
            after_turn=after_turn,
            wait_timeout_seconds=wait_timeout_seconds,
        )
        return {
            "ignored_items": result.ingest.adaptation.ignored_items,
            "after_turn": result.ingest.adaptation.after_turn,
            "jobs": [
                {
                    "memory_sequence": job.memory_sequence,
                    "conversation_id": job.conversation_id,
                    "segment_id": job.segment_id,
                    "source_segment_digest": job.source_segment_digest,
                    "transaction_id": job.transaction_id,
                    "status": job.status.value,
                }
                for job in result.ingest.ingest.jobs
            ],
            "consistency": [
                {
                    "memory_sequence": snapshot.requested_job.memory_sequence,
                    "state": snapshot.state.value,
                }
                for snapshot in result.consistency
            ],
        }

    async def recall(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        started_on: date | None = None,
        limit: int | None = None,
        kinds: tuple[MemoryKind, ...] = (),
        intention_scope: MemoryIntentionRecallScope = MemoryIntentionRecallScope.ACTIVE,
    ) -> dict[str, object]:
        """返回 Agent 可用上下文和可溯源命中，不暴露记忆树内部文件布局。"""

        if (conversation_id is None) != (started_on is None):
            raise ValueError("conversation_id and started_on must be provided together")
        address = (
            None
            if conversation_id is None or started_on is None
            else ConversationAddress(conversation_id, started_on)
        )
        result = await self.agent.recall(
            query,
            conversation=address,
            limit=limit,
            kinds=kinds,
            intention_scope=intention_scope,
        )
        return {
            "query": result.query,
            "queries": [item.query for item in result.plan.queries],
            "context": result.context,
            "memories": [
                {
                    "uri": str(memory.uri),
                    "score": memory.hit.score,
                    "matched_queries": list(memory.matched_queries),
                }
                for memory in result.memories
            ],
            "summaries": [
                {
                    "reference": match.reference.identity,
                    "score": match.score,
                }
                for match in result.summary_fallbacks
            ],
            "degradations": [
                {"stage": item.stage.value, "error_type": item.error_type}
                for item in result.degradations
            ],
            "budget_exhausted": result.budget_exhausted,
        }

    async def health(self, *, deep: bool = False) -> dict[str, object]:
        report = await self.runtime.health(deep=deep)
        return self._payload(report)

    async def readiness(self) -> tuple[int, dict[str, object]]:
        report = await self.runtime.health(deep=False)
        return (200 if report.ready else 503), self._payload(report)

    def metrics(self) -> tuple[str, str]:
        return self.runtime.prometheus_metrics(), "text/plain; version=0.0.4; charset=utf-8"

    @staticmethod
    def _payload(report: RuntimeHealthReport) -> dict[str, object]:
        return {
            "status": report.status.value,
            "ready": report.ready,
            "checked_at": report.checked_at.isoformat(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "detail": check.detail,
                    "critical": check.critical,
                }
                for check in report.checks
            ],
        }


__all__ = ["RuntimeHTTPHandlers"]
