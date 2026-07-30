"""供 FastAPI、Starlette 或原生 ASGI 路由复用的无框架 HTTP Handler。"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime

from integrations.agent import AgentMemoryGateway
from memory.conversation import ConversationAddress
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryKind
from memory.workflow import MemoryJob, MemoryJobBlockedError, MemoryJobStatus
from Runtime import Runtime, RuntimeHealthReport

_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_NAMED_SECRET = re.compile(r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b\s*[:=]\s*[^\s,;]+")
_PREFIXED_SECRET = re.compile(r"(?i)\b(?:ghp|github_pat|sk|xox[baprs])[-_][A-Za-z0-9._-]+")
_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])(?:/[A-Za-z0-9._-]+){2,}")
_PUBLIC_JOB_ERROR_MAX_CHARS = 500


class HTTPMemoryJobNotFoundError(LookupError):
    """HTTP 调用方给出的公开 Job 身份在保留窗口内不存在。"""


class HTTPMemoryJobConflictError(RuntimeError):
    """HTTP 运维操作基于陈旧 Job 版本或不满足当前状态前置条件。"""


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
                    "started_on": job.started_on.isoformat(),
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
            None if conversation_id is None or started_on is None else ConversationAddress(conversation_id, started_on)
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
                {"stage": item.stage.value, "error_type": item.error_type} for item in result.degradations
            ],
            "budget_exhausted": result.budget_exhausted,
        }

    async def job_status(
        self,
        memory_sequence: int,
        *,
        conversation_id: str,
        started_on: date,
    ) -> dict[str, object]:
        """只经 Runtime 公共门面查询仍在耐久保留窗口内的 Job 状态。"""

        if isinstance(memory_sequence, bool) or not isinstance(memory_sequence, int) or memory_sequence <= 0:
            raise ValueError("memory_sequence must be a positive integer")
        job = await self._find_job(
            memory_sequence,
            conversation_id=conversation_id,
            started_on=started_on,
        )
        blocked = await self.runtime.failed_memory_job()
        return await self._job_payload(
            job,
            blocking_sequence=None if blocked is None else blocked.memory_sequence,
        )

    async def list_jobs(
        self,
        *,
        conversation_id: str,
        started_on: date,
        status: MemoryJobStatus | None = None,
        limit: int = 50,
        before_sequence: int | None = None,
    ) -> dict[str, object]:
        """倒序分页列出一个 Conversation 仍在耐久保留窗口内的 Job。"""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if before_sequence is not None and (
            isinstance(before_sequence, bool) or not isinstance(before_sequence, int) or before_sequence <= 0
        ):
            raise ValueError("before_sequence must be a positive integer or None")
        resolved_status = None if status is None else MemoryJobStatus(status)
        address = ConversationAddress(conversation_id, started_on)
        jobs = await self.runtime.list_memory_jobs(address)
        filtered = [
            job
            for job in reversed(jobs)
            if (before_sequence is None or job.memory_sequence < before_sequence)
            and (resolved_status is None or job.status is resolved_status)
        ]
        page = filtered[:limit]
        blocked = await self.runtime.failed_memory_job()
        blocking_sequence = None if blocked is None else blocked.memory_sequence
        return {
            "jobs": [await self._job_payload(job, blocking_sequence=blocking_sequence) for job in page],
            "next_before_sequence": (page[-1].memory_sequence if len(filtered) > limit and page else None),
        }

    async def blocked_job(self) -> dict[str, object]:
        """返回当前阻塞全局有序队列的最早 FAILED Job，不枚举其他会话。"""

        failed = await self.runtime.failed_memory_job()
        if failed is None:
            return {"job": None}
        return {
            "job": await self._job_payload(
                failed,
                blocking_sequence=failed.memory_sequence,
            )
        }

    async def retry_failed_job(
        self,
        memory_sequence: int,
        *,
        conversation_id: str,
        started_on: date,
        expected_version: str,
    ) -> dict[str, object]:
        """仅经 Runtime 重试当前阻塞队列且版本完全匹配的 FAILED Job。"""

        if not isinstance(expected_version, str) or re.fullmatch(r"[0-9a-f]{64}", expected_version) is None:
            raise ValueError("expected_version must be lowercase SHA-256 text")
        job = await self._find_job(
            memory_sequence,
            conversation_id=conversation_id,
            started_on=started_on,
        )
        if self._job_version(job) != expected_version:
            raise HTTPMemoryJobConflictError("memory job changed after it was inspected")
        failed = await self.runtime.failed_memory_job()
        if failed is None or failed.memory_sequence != job.memory_sequence:
            raise HTTPMemoryJobConflictError("memory job is not the current blocking failed job")
        if self._job_version(failed) != expected_version:
            raise HTTPMemoryJobConflictError("memory job changed after it was inspected")

        previous = await self._job_payload(
            failed,
            blocking_sequence=failed.memory_sequence,
        )
        try:
            retried = await self.runtime.retry_failed_memory_job(failed)
        except MemoryJobBlockedError as exc:
            raise HTTPMemoryJobConflictError(str(exc)) from exc
        reopened = await self._job_payload(
            retried.reopened_job,
            blocking_sequence=None,
        )
        return {
            "previous": previous,
            "job": reopened,
            "worker_restarted": retried.worker_restarted,
        }

    async def health(self, *, deep: bool = False) -> dict[str, object]:
        report = await self.runtime.health(deep=deep)
        return self._payload(report)

    async def readiness(self) -> tuple[int, dict[str, object]]:
        report = await self.runtime.health(deep=False)
        return (200 if report.ready else 503), self._payload(report)

    async def recent_audit_events(self, *, limit: int = 100) -> dict[str, object]:
        """返回不含请求正文、密钥和内部路径的最小审计记录。"""

        records = await self.runtime.recent_audit_events(limit=limit)
        return {
            "events": [
                {
                    "audit_id": item.audit_id,
                    "occurred_at": item.occurred_at.isoformat(),
                    "category": item.category,
                    "operation": item.operation,
                    "status": item.status,
                    "request_id": item.request_id,
                    "memory_sequence": item.memory_sequence,
                    "attributes": item.attributes,
                }
                for item in records
            ]
        }

    async def metrics(self) -> tuple[str, str]:
        await self.runtime.refresh_observability()
        return self.runtime.prometheus_metrics(), "text/plain; version=0.0.4; charset=utf-8"

    async def _find_job(
        self,
        memory_sequence: int,
        *,
        conversation_id: str,
        started_on: date,
    ) -> MemoryJob:
        address = ConversationAddress(conversation_id, started_on)
        jobs = await self.runtime.list_memory_jobs(address)
        job = next((item for item in jobs if item.memory_sequence == memory_sequence), None)
        if job is None:
            raise HTTPMemoryJobNotFoundError(
                f"memory job {memory_sequence} was not found in its configured retention window"
            )
        return job

    async def _job_payload(
        self,
        job: MemoryJob,
        *,
        blocking_sequence: int | None,
    ) -> dict[str, object]:
        snapshot = await self.runtime.memory_consistency(job)
        current = snapshot.current_job or job
        blocking = blocking_sequence == current.memory_sequence
        last_failure = None if current.last_error is None else {"message": self._sanitize_failure(current.last_error)}
        return {
            "memory_sequence": current.memory_sequence,
            "conversation_id": current.conversation_id,
            "started_on": current.started_on.isoformat(),
            "state": snapshot.state.value,
            "job_status": current.status.value,
            "terminal": snapshot.terminal,
            "attempts": current.attempts,
            "next_attempt_at": (None if current.next_attempt_at is None else current.next_attempt_at.isoformat()),
            "last_failure": last_failure,
            "blocking": blocking,
            "manual_action_required": blocking and snapshot.state.value == "failed",
            "version": self._job_version(current),
            "created_at": current.created_at.isoformat(),
            "updated_at": current.updated_at.isoformat(),
        }

    @staticmethod
    def _job_version(job: MemoryJob) -> str:
        fields = (
            str(job.memory_sequence),
            job.conversation_id,
            job.started_on.isoformat(),
            job.segment_id,
            job.source_segment_digest,
            job.transaction_id,
            job.status.value,
            str(job.attempts),
            str(job.claim_generation),
            "" if job.next_attempt_at is None else job.next_attempt_at.isoformat(),
            "" if job.last_error is None else job.last_error,
            job.updated_at.isoformat(),
        )
        return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()

    @staticmethod
    def _sanitize_failure(value: str) -> str:
        sanitized = " ".join(value.split())
        sanitized = _BEARER_SECRET.sub("Bearer [REDACTED]", sanitized)
        sanitized = _NAMED_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", sanitized)
        sanitized = _PREFIXED_SECRET.sub("[REDACTED]", sanitized)
        sanitized = _ABSOLUTE_PATH.sub("[PATH]", sanitized)
        if len(sanitized) > _PUBLIC_JOB_ERROR_MAX_CHARS:
            return sanitized[:_PUBLIC_JOB_ERROR_MAX_CHARS] + "...[truncated]"
        return sanitized or "memory job failed"

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


__all__ = [
    "HTTPMemoryJobConflictError",
    "HTTPMemoryJobNotFoundError",
    "RuntimeHTTPHandlers",
]
