"""把 Runtime 适配为 Agent 生命周期可依赖的稳定记忆能力口。

## TODO(AGENT-CONTEXT-001)：把记忆接进 Agent 上下文管理（借鉴 OpenViking；用户 2026-09-23 要求登记，实现时机由他定）

**问题**：memory 现在只做"存 + 召回注入"，不防 Agent 上下文爆炸。插件 hooks
（``plugins/memory-plugin-shared/lib/hook-runner.mjs``）里：``UserPromptSubmit`` 调 ``recall`` 往上下文里
**加**内容（``MemorySearchServiceConfig.max_context_chars=120000`` 字符上限）；``Stop``/``SessionEnd`` 只抄走
transcript；``PreCompact`` 只是先 ``flush`` 封段，压缩仍由宿主自己做；``SessionStart`` 不注入任何东西。
``ConversationRetentionPlanner``（保留 3 轮 / 12000 token）与段摘要 → Range → Archive 压的都是 Habitus
自己的 ``live.jsonl`` 副本，不是 Agent 的 prompt；压缩产物只在 ``SearchService.search()`` 判定长期记忆
不足时当 Summary 兜底用。本类对外只有 remember / recall / flush / record_use，没有"按预算给出会话上下文"
的能力。

**参考（OpenViking ``volcengine/OpenViking`` main，2026-09-23 核对）**：存储这一半 Habitus 已移植（retention
默认值一致、归档摘要、L0/L1/L2），缺的是"把压缩结果送回上下文"的另一半。OpenViking 有三种接法：

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

- A. 对应 1：Runtime 加"按 token 预算组装会话上下文"——最新摘要 + live 消息，超预算按回合裁剪；本类与
  HTTP 各加一个出口；插件在宿主提供消息改写能力时用它替换 messages。
- B. 对应 2：加窗口工具（切窗口 = 强制封段 + 生成交接摘要；history 工具按 segment_id + sequence 回查
  ``history/*.jsonl``，退役后回落到摘要）；插件在宿主 context hook 里改写请求。
- C. 对应 3（改动最小、Claude Code 可立即用）：``SessionStart`` 在 ``source`` 为 compact / resume 时注入最近
  frontier 摘要，``PreCompact`` 维持现状。

**前置核对**：A、B 都要求宿主能改写发给模型的消息列表。Claude Code 的 hooks 不能改写历史，只能走 C；
Codex 插件（``plugins/habitus-memory``）的宿主能力未核对。先确认目标宿主，再选方案。

**具体场景**：长编码会话写到 150k token，宿主自动压缩后丢了早先的纠正（"不要用 SQLite"）——C 能在压缩后
把 corrections 补回来；A/B 能在压缩发生前就只给模型"摘要 + 最近几轮"，并在需要时按 id 取回原话。

**影响面**：不动记忆树、六类 schema、Editor 与检索链；新增 Runtime/本类/HTTP 的出口与插件 hook 逻辑；B 还要
为 history 回查定义与 Archive 退役（``ConversationLifecycleManager._retire_archive_chain`` 释放原文）的关系。
"""

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
