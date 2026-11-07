"""HTTP 路由背后的 handler：只经 Runtime 公共门面取结果，直接产出 ``schemas`` 里的响应模型。

## TODO(AGENT-CONTEXT-001)：把记忆接进 Agent 上下文管理（借鉴 OpenViking；用户 2026-09-23 要求登记，实现时机由他定）

**问题**：memory 现在只做"存 + 召回注入"，不防 Agent 上下文爆炸。插件 hooks
（``plugins/memory-plugin-shared/lib/hook-runner.mjs``）里：``UserPromptSubmit`` 调 ``recall`` 往上下文里
**加**内容（``MemorySearchServiceConfig.max_context_chars=120000`` 字符上限）；``Stop``/``SessionEnd`` 只抄走
transcript；``PreCompact`` 只是先 ``flush`` 封段，压缩仍由宿主自己做；``SessionStart`` 不注入任何东西。
``ConversationRetentionPlanner``（保留 3 轮 / 12000 token）与段摘要 → Range → Archive 压的都是 Habitus
自己的 ``live.jsonl`` 副本，不是 Agent 的 prompt；压缩产物只在 ``SearchService.search()`` 判定长期记忆
不足时当 Summary 兜底用。HTTP 的记忆出口只有 remember / recall / flush / cursor，没有"按预算给出会话上下文"
的能力。

**参考（OpenViking ``volcengine/OpenViking`` main，2026-09-23 核对）**：存储这一半 Habitus 已有自己的实现，只借鉴了
OpenViking 的设计（retention 默认值取同一组数、归档摘要、L0/L1/L2），缺的是"把压缩结果送回上下文"的另一半。OpenViking 有三种接法：

1. **服务端组装 + 宿主替换**（OpenClaw ContextEngine）：``Session.get_session_context(token_budget=128000)``
   返回"最新归档 overview + 按预算裁剪的活跃消息"（``GET /sessions/{id}/context``）；
   ``examples/openclaw-plugin/context-engine.ts`` 的 ``assemble`` 用它**替换**宿主 messages 并记
   ``tokensSaved``，``afterTurn`` 触发 commit，``compact`` 接管压缩。
2. **Agent 自管上下文窗口**（``examples/pi-experimental-context-management``，提交 ``bf8c5e9d``，仿 Codex
   ``context_management`` 实验模式，不做 LLM 摘要式压缩）：给模型 ``new_context(reason, notes, next_steps?)`` /
   ``get_context_remaining`` / ``history``（list_windows / list_items / read_item / search_contents）三个工具；
   ``new_context`` 同步分支 → 写交接笔记 → ``commit(keepRecentCount=0)`` 归档本窗口 → 服务端生成 7 节
   Working Memory；之后宿主的 context hook 把每次请求改写成"窗口头 + 新窗口消息"，旧内容按 id 回查原文；
   剩余不足时软/硬提醒各一次，兜底退回宿主压缩。设计说明见该目录 ``CONTEXT-WINDOW.md``。
3. **只能用 hooks 的宿主**（``examples/claude-code-memory-plugin``）：``PreCompact`` 只 commit，
   ``session-start.mjs`` 在 ``source`` 为 ``compact`` / ``resume`` 时把归档经 ``additionalContext`` 注入——
   是宿主压缩后的补全，不是压缩本身。

**Habitus 已有的数据**：``ConversationSummaryCompactor.frontier(address).active``（按序号不重叠的 Segment /
Range / Archive 摘要，字段 overview / chronology / corrections / ending_state / open_threads）、
``ConversationMessageJournal.read_live``（未封段消息）、``history/*.jsonl``（原文，直到 Archive 退役释放）、
``PersistentConversationSummaryVectorIndex``（摘要检索）。

**候选方案（未裁定）**：

- A. 对应 1：Runtime 加"按 token 预算组装会话上下文"——最新摘要 + live 消息，超预算按回合裁剪；本模块与
  路由各加一个出口；插件在宿主提供消息改写能力时用它替换 messages。
- B. 对应 2：加窗口工具（切窗口 = 强制封段 + 生成交接摘要；history 工具按 segment_id + sequence 回查
  ``history/*.jsonl``，退役后回落到摘要）；插件在宿主 context hook 里改写请求。
- C. 对应 3（改动最小、Claude Code 可立即用）：``SessionStart`` 在 ``source`` 为 compact / resume 时注入最近
  frontier 摘要，``PreCompact`` 维持现状。

**前置核对**：A、B 都要求宿主能改写发给模型的消息列表。Claude Code 的 hooks 不能改写历史，只能走 C；
Codex 插件（``plugins/habitus-memory``）的宿主能力未核对。先确认目标宿主，再选方案。

**具体场景**：长编码会话写到 150k token，宿主自动压缩后丢了早先的纠正（"不要用 SQLite"）——C 能在压缩后
把 corrections 补回来；A/B 能在压缩发生前就只给模型"摘要 + 最近几轮"，并在需要时按 id 取回原话。

**影响面**：不动记忆树、六类 schema、Editor 与检索链；新增 Runtime/本模块/路由的出口与插件 hook 逻辑；B 还要
为 history 回查定义与 Archive 退役（``ConversationLifecycleManager._retire_archive_chain`` 释放原文）的关系。
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import re
from dataclasses import dataclass
from datetime import date, datetime

from habitus.foundation.redaction import redact_sensitive_text
from habitus.integrations.http_api.schemas import (
    AuditEventResult,
    AuditListResult,
    BlockedJobResult,
    CapabilitiesResult,
    ConsistencyResult,
    ConversationCursorResult,
    FlushResult,
    HealthCheckResult,
    HealthResult,
    JobFailureResult,
    JobListResult,
    JobStatusResult,
    MemoryLinkResult,
    MemoryListResult,
    MemoryReadResult,
    RecallDegradationResult,
    RecallMemoryResult,
    RecallResult,
    RecallSummaryResult,
    RememberJobResult,
    RememberResult,
    ReminderClaimResult,
    ReminderDayResult,
    ReminderHandoffResult,
    ReminderProgressResult,
    ReminderResult,
    RetryJobResult,
)
from habitus.memory.browse import MemoryLayerText
from habitus.memory.conversation import ConversationAddress
from habitus.memory.intention import MemoryIntentionRecallScope
from habitus.memory.model import MemoryKind, MemoryLevel
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow import MemoryJob, MemoryJobBlockedError, MemoryJobStatus
from habitus.runtime import MemoryConsistencySnapshot, Runtime, RuntimeHealthReport

_PUBLIC_JOB_ERROR_MAX_CHARS = 500
_SERVICE_API_VERSION = "1.0"
_SERVICE_FEATURES = (
    "conversation_cursor",
    "flush",
    "memory_jobs",
    "memory_tree_v1",
    "recall",
    "remember",
    "remember_idempotency_v1",
    "remember_parts_v1",
    "reminders_report_v1",
)
#: 提醒出口的取口：只在预测层开着时公布（没开时取口恒为空，插件与 ``habitus doctor`` 按它判提醒开没开）。
_REMINDERS_FEATURE = "reminders_v1"


class HTTPMemoryJobNotFoundError(LookupError):
    """HTTP 调用方给出的公开 Job 身份在保留窗口内不存在。"""


class HTTPMemoryJobConflictError(RuntimeError):
    """HTTP 运维操作基于陈旧 Job 版本或不满足当前状态前置条件。"""


@dataclass(frozen=True)
class RememberOutcome:
    """一次写入的公开结果，加上这次受理的内部 Job——后者只供同进程观测关联事务身份，不进响应。"""

    result: RememberResult
    accepted_jobs: tuple[MemoryJob, ...]


class RuntimeHTTPHandlers:
    """只提供稳定 handler，不擅自决定端口、认证、URL 或 Web 框架。"""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def capabilities(self) -> CapabilitiesResult:
        """返回插件建立连接前可验证的稳定服务能力；``reminders_v1`` 只在提醒出口开着时带上。"""

        try:
            service_version = importlib.metadata.version("habitus")
        except importlib.metadata.PackageNotFoundError:
            service_version = "0.1.0"
        return CapabilitiesResult(
            api_version=_SERVICE_API_VERSION,
            service_version=service_version,
            protocols=list(self.runtime.conversation_protocols()),
            features=[*_SERVICE_FEATURES, *((_REMINDERS_FEATURE,) if self.runtime.reminders_enabled() else ())],
        )

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
        delivery_id: str | None = None,
    ) -> RememberOutcome:
        """供 HTTP 路由层调用的协议写入 handler；URL、认证和 owner 由上层决定。"""

        ingest = await self.runtime.append_protocol_conversation(
            ConversationAddress(conversation_id, started_on),
            protocol=protocol,
            payload=payload,
            start_sequence=start_sequence,
            occurred_at=occurred_at,
            after_turn=after_turn,
            delivery_id=delivery_id,
        )
        jobs = ingest.ingest.jobs
        settled = await self._wait(jobs, wait_timeout_seconds)
        return RememberOutcome(
            result=RememberResult(
                ignored_items=ingest.adaptation.ignored_items,
                after_turn=ingest.effective_after_turn,
                next_sequence=ingest.next_sequence,
                jobs=[self._accepted_job(job) for job in jobs],
                consistency=[self._consistency(snapshot) for snapshot in settled],
            ),
            accepted_jobs=jobs,
        )

    async def conversation_cursor(
        self,
        *,
        conversation_id: str,
        started_on: date,
    ) -> ConversationCursorResult:
        """返回 Agent 恢复同一 Conversation 时应继续使用的耐久序号。"""

        next_sequence = await self.runtime.conversation_cursor(ConversationAddress(conversation_id, started_on))
        return ConversationCursorResult(next_sequence=next_sequence)

    async def flush(
        self,
        *,
        conversation_id: str,
        started_on: date,
        wait_timeout_seconds: float | None = None,
    ) -> FlushResult:
        """在 Agent 会话关闭边界显式提交剩余完整轮次，并可选择等待长期记忆终态。"""

        flushed = await self.runtime.flush_conversation(ConversationAddress(conversation_id, started_on))
        settled = await self._wait(flushed.jobs, wait_timeout_seconds)
        return FlushResult(
            jobs=[self._accepted_job(job) for job in flushed.jobs],
            consistency=[self._consistency(snapshot) for snapshot in settled],
        )

    async def recall(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        started_on: date | None = None,
        limit: int | None = None,
        kinds: tuple[MemoryKind, ...] = (),
        intention_scope: MemoryIntentionRecallScope = MemoryIntentionRecallScope.ACTIVE,
    ) -> RecallResult:
        """返回 Agent 可用上下文和可溯源命中，不暴露记忆树内部文件布局。"""

        if (conversation_id is None) != (started_on is None):
            raise ValueError("conversation_id and started_on must be provided together")
        conversation = (
            None if conversation_id is None or started_on is None else ConversationAddress(conversation_id, started_on)
        )
        result = await self.runtime.search_memory(
            query,
            conversation=conversation,
            limit=limit,
            kinds=kinds,
            intention_scope=intention_scope,
        )
        return RecallResult(
            query=result.query,
            queries=[item.query for item in result.plan.queries],
            context=result.context,
            memories=[
                RecallMemoryResult(
                    uri=str(memory.uri),
                    score=memory.hit.score,
                    matched_queries=list(memory.matched_queries),
                )
                for memory in result.memories
            ],
            summaries=[
                RecallSummaryResult(reference=match.reference.identity, score=match.score)
                for match in result.summary_fallbacks
            ],
            degradations=[
                RecallDegradationResult(stage=item.stage.value, error_type=item.error_type)
                for item in result.degradations
            ],
            budget_exhausted=result.budget_exhausted,
        )

    async def job_status(
        self,
        memory_sequence: int,
        *,
        conversation_id: str,
        started_on: date,
    ) -> JobStatusResult:
        """只经 Runtime 公共门面查询仍在耐久保留窗口内的 Job 状态。"""

        if memory_sequence <= 0:
            raise ValueError("memory_sequence must be a positive integer")
        job = await self._find_job(
            memory_sequence,
            conversation_id=conversation_id,
            started_on=started_on,
        )
        blocked = await self.runtime.failed_memory_job()
        return await self._job_status(
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
    ) -> JobListResult:
        """倒序分页列出一个 Conversation 仍在耐久保留窗口内的 Job。"""

        if not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if before_sequence is not None and before_sequence <= 0:
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
        return JobListResult(
            jobs=[await self._job_status(job, blocking_sequence=blocking_sequence) for job in page],
            next_before_sequence=(page[-1].memory_sequence if len(filtered) > limit and page else None),
        )

    async def blocked_job(self) -> BlockedJobResult:
        """返回当前阻塞全局有序队列的最早 FAILED Job，不枚举其他会话。"""

        failed = await self.runtime.failed_memory_job()
        if failed is None:
            return BlockedJobResult(job=None)
        return BlockedJobResult(
            job=await self._job_status(
                failed,
                blocking_sequence=failed.memory_sequence,
            )
        )

    async def retry_failed_job(
        self,
        memory_sequence: int,
        *,
        conversation_id: str,
        started_on: date,
        expected_version: str,
    ) -> RetryJobResult:
        """仅经 Runtime 重试当前阻塞队列且版本完全匹配的 FAILED Job。"""

        if re.fullmatch(r"[0-9a-f]{64}", expected_version) is None:
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

        previous = await self._job_status(
            failed,
            blocking_sequence=failed.memory_sequence,
        )
        try:
            retried = await self.runtime.retry_failed_memory_job(failed)
        except MemoryJobBlockedError as exc:
            raise HTTPMemoryJobConflictError(str(exc)) from exc
        reopened = await self._job_status(
            retried.reopened_job,
            blocking_sequence=None,
        )
        return RetryJobResult(
            previous=previous,
            job=reopened,
            worker_restarted=retried.worker_restarted,
        )

    async def list_memory(self, uri: str | None = None) -> MemoryListResult:
        """列记忆树的一个目录（省略是根）：子目录、文档的 ``memory://`` 地址与目录摘要，不给文件路径。"""

        listing = await self.runtime.list_memory(uri)
        return MemoryListResult(
            uri=str(listing.uri),
            directories=[str(item) for item in listing.directories],
            documents=[str(item) for item in listing.documents],
            abstract=listing.abstract,
        )

    async def read_memory(self, uri: str) -> MemoryReadResult:
        """读一个记忆文档（结构化字段 + 正文），或一个目录的摘要 / 概览层原文。"""

        value = await self.runtime.read_memory(uri)
        if isinstance(value, MemoryLayerText):
            return MemoryReadResult(
                uri=str(value.uri),
                node="layer",
                level="abstract" if value.level is MemoryLevel.ABSTRACT else "overview",
                text=value.text,
            )
        return MemoryReadResult(
            uri=str(MemoryURI.from_address(value.address)),
            node="document",
            kind=value.kind.value,
            revision=value.metadata.revision,
            created_at=value.metadata.created_at,
            updated_at=value.metadata.updated_at,
            last_confirmed_at=value.metadata.last_confirmed_at,
            fields=dict(value.fields),
            text=value.markdown_body,
            links=[MemoryLinkResult.model_validate(link.to_dict()) for link in value.links],
            backlinks=[MemoryLinkResult.model_validate(link.to_dict()) for link in value.backlinks],
        )

    async def reminders(self, day: date | None = None) -> ReminderDayResult:
        """某天（省略是服务所在时区的今天）每条提醒走到了哪一步：待取 / 取走了 / 说了 / 宿主没取，回话与结算结果。"""

        report = await self.runtime.reminders(day)
        return ReminderDayResult(
            day=report.day,
            reminders=[
                # 阶段与回话由 Runtime 按字符串给，取值是否在约定的几个之内交给 Schema 判
                ReminderProgressResult.model_validate(
                    {
                        "request_id": item.expectation_id,
                        "kind": item.kind,
                        "text": item.text,
                        "deadline": item.deadline,
                        "stage": item.stage,
                        "awaiting_at": item.awaiting_at,
                        "spoken_at": item.spoken_at,
                        "spoken_via": item.spoken_via,
                        "response": item.response,
                        "responded_at": item.responded_at,
                        "outcome": item.outcome,
                        "attempts": item.attempts,
                        "handoffs": [
                            ReminderHandoffResult(
                                host=handoff.host,
                                session=handoff.session,
                                claimed_at=handoff.claimed_at,
                                ended=handoff.ended,
                            )
                            for handoff in item.handoffs
                        ],
                    }
                )
                for item in report.reminders
            ],
            awaiting=report.count("待取"),
            taken=report.count("取走了"),
            spoken=report.count("说了"),
            never_claimed=report.count("宿主没取"),
            responded=report.responded,
        )

    async def claim_reminders(self, *, host: str, session: str, at: datetime, limit: int) -> ReminderClaimResult:
        """宿主来取此刻可取的提醒（方案 32 第 4.1 条）；``session`` 是宿主侧会话的身份，与它送转写时的 conversation_id 同一个。"""

        claimed = await self.runtime.claim_reminders(host=host, session=session, at=at, limit=limit)
        return ReminderClaimResult(
            reminders=[
                ReminderResult(
                    request_id=item.request_id,
                    kind=item.kind,
                    text=item.text,
                    options=list(item.options),
                    deadline=item.deadline,
                    lease_until=item.lease_until,
                    claimed_at=item.claimed_at,
                )
                for item in claimed
            ]
        )

    async def health(self, *, deep: bool = False) -> HealthResult:
        report = await self.runtime.health(deep=deep)
        return self._health(report)

    async def readiness(self) -> tuple[int, HealthResult]:
        report = await self.runtime.health(deep=False)
        return (200 if report.ready else 503), self._health(report)

    async def recent_audit_events(self, *, limit: int = 100) -> AuditListResult:
        """返回不含请求正文、密钥和内部路径的最小审计记录。"""

        records = await self.runtime.recent_audit_events(limit=limit)
        return AuditListResult(
            events=[
                AuditEventResult(
                    audit_id=item.audit_id,
                    occurred_at=item.occurred_at,
                    category=item.category,
                    operation=item.operation,
                    status=item.status,
                    request_id=item.request_id,
                    memory_sequence=item.memory_sequence,
                    attributes=item.attributes,
                )
                for item in records
            ]
        )

    async def metrics(self) -> tuple[str, str]:
        await self.runtime.refresh_observability()
        return self.runtime.prometheus_metrics(), "text/plain; version=0.0.4; charset=utf-8"

    async def _wait(
        self,
        jobs: tuple[MemoryJob, ...],
        timeout_seconds: float | None,
    ) -> tuple[MemoryConsistencySnapshot, ...]:
        if timeout_seconds is None or not jobs:
            return ()
        return tuple(
            await asyncio.gather(
                *(self.runtime.wait_memory_consistency(job, timeout_seconds=timeout_seconds) for job in jobs)
            )
        )

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

    async def _job_status(
        self,
        job: MemoryJob,
        *,
        blocking_sequence: int | None,
    ) -> JobStatusResult:
        snapshot = await self.runtime.memory_consistency(job)
        current = snapshot.current_job or job
        blocking = blocking_sequence == current.memory_sequence
        last_failure = (
            None if current.last_error is None else JobFailureResult(message=self._sanitize_failure(current.last_error))
        )
        return JobStatusResult(
            memory_sequence=current.memory_sequence,
            conversation_id=current.conversation_id,
            started_on=current.started_on,
            state=snapshot.state.value,
            job_status=current.status.value,
            terminal=snapshot.terminal,
            attempts=current.attempts,
            next_attempt_at=current.next_attempt_at,
            last_failure=last_failure,
            blocking=blocking,
            manual_action_required=blocking and snapshot.state.value == "failed",
            version=self._job_version(current),
            created_at=current.created_at,
            updated_at=current.updated_at,
        )

    @staticmethod
    def _accepted_job(job: MemoryJob) -> RememberJobResult:
        return RememberJobResult(
            memory_sequence=job.memory_sequence,
            conversation_id=job.conversation_id,
            started_on=job.started_on,
            status=job.status.value,
        )

    @staticmethod
    def _consistency(snapshot: MemoryConsistencySnapshot) -> ConsistencyResult:
        return ConsistencyResult(
            memory_sequence=snapshot.requested_job.memory_sequence,
            state=snapshot.state.value,
        )

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
        sanitized = redact_sensitive_text(value)
        if len(sanitized) > _PUBLIC_JOB_ERROR_MAX_CHARS:
            return sanitized[:_PUBLIC_JOB_ERROR_MAX_CHARS] + "...[truncated]"
        return sanitized or "memory job failed"

    @staticmethod
    def _health(report: RuntimeHealthReport) -> HealthResult:
        return HealthResult(
            status=report.status.value,
            ready=report.ready,
            recall_ready=report.recall_ready,
            checked_at=report.checked_at,
            checks=[
                HealthCheckResult(
                    name=check.name,
                    status=check.status.value,
                    detail=check.detail,
                    critical=check.critical,
                )
                for check in report.checks
            ],
        )


__all__ = [
    "HTTPMemoryJobConflictError",
    "HTTPMemoryJobNotFoundError",
    "RememberOutcome",
    "RuntimeHTTPHandlers",
]
