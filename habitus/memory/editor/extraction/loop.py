"""Conversation 到严格候选和初步字段计划的受控解析流程。"""

from __future__ import annotations

from dataclasses import replace
from datetime import tzinfo
from typing import TypeVar

from habitus.infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from habitus.memory.document import MemoryDocument
from habitus.memory.editor.candidate import MemoryCandidateBatch, MemoryCandidateError, MemoryCandidateViolation
from habitus.memory.editor.extraction.config import MemoryExtractionConfig
from habitus.memory.editor.extraction.context import MemoryExtractionContext
from habitus.memory.editor.extraction.model import (
    MemoryExtractionCapacityError,
    MemoryExtractionPermanentError,
    MemoryExtractionResult,
    MemoryRetrievalAction,
    MemoryRetrievalDecision,
    MemoryRetrievalFinalDecision,
    MemoryRetrievalObservation,
    MemoryRetrievalStatus,
)
from habitus.memory.editor.extraction.prompt import MemoryExtractionPromptBuilder
from habitus.memory.editor.extraction.turn import MemoryTurnContext
from habitus.memory.editor.mutation import (
    MemoryFieldMergeError,
    MemoryMutationPlanner,
    MemoryMutationPlanningError,
    MemoryMutationReadSetLoader,
    MemoryNodeMatchError,
)
from habitus.memory.editor.retrieval import MemoryRelatedContext, MemoryRelatedRetriever
from habitus.memory.model import MemoryAddress
from habitus.memory.uri import MemoryURI
from habitus.model_client import (
    ChatCallContext,
    ChatMessage,
    ChatRequest,
    ModelInputTooLargeError,
    ModelStructuredOutputError,
    StructuredChatClient,
    StructuredResponse,
    estimate_utf8_bytes_tokens,
)
from habitus.pre.conversation import ConversationMessageRole, ConversationSegment

T = TypeVar("T")
_RETRIEVAL_PROMPT_VERSION = "memory_retrieval_grader_v3"
_CANDIDATE_PROMPT_VERSION = "memory_candidate_extraction_v3"


class MemoryExtractionLoop:
    """串联受控检索、模型候选生成和确定性初步字段计划。"""

    def __init__(
        self,
        *,
        client: StructuredChatClient,
        retriever: MemoryRelatedRetriever,
        zone: tzinfo,
        config: MemoryExtractionConfig | None = None,
        mutation_reader: MemoryMutationReadSetLoader | None = None,
        mutation_planner: MemoryMutationPlanner | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        if not isinstance(retriever, MemoryRelatedRetriever):
            raise TypeError("retriever must be a MemoryRelatedRetriever")
        if not isinstance(zone, tzinfo):
            raise TypeError("zone must be a tzinfo")
        if config is not None and not isinstance(config, MemoryExtractionConfig):
            raise TypeError("config must be a MemoryExtractionConfig")
        if mutation_reader is not None and not isinstance(mutation_reader, MemoryMutationReadSetLoader):
            raise TypeError("mutation_reader must be a MemoryMutationReadSetLoader")
        if mutation_planner is not None and not isinstance(mutation_planner, MemoryMutationPlanner):
            raise TypeError("mutation_planner must be a MemoryMutationPlanner")
        self.client = client
        self.retriever = retriever
        self.zone = zone
        self.config = config or MemoryExtractionConfig()
        self.mutation_reader = mutation_reader or MemoryMutationReadSetLoader(retriever.snapshot_reader)
        self.mutation_planner = mutation_planner or MemoryMutationPlanner(retriever.schema_registry)
        self.prompt_builder = MemoryExtractionPromptBuilder(self.config, retriever.schema_registry, zone=zone)

    def turn_fits_whole(self, whole: ConversationSegment) -> bool:
        """整轮拼起来能不能一次抽取：对话材料（不带旧记忆的检索判定与候选两种请求，按真实请求投影估）放得进
        ``max_input_tokens`` 减去旧记忆预算 ``max_old_memory_tokens`` 剩下的那部分。

        旧记忆按自己的预算另行装入（``_fit_initial_context``），所以只有这样估，读到多少旧记忆都不会让请求超限；
        配置里这两个数的关系本来就是"旧记忆之外留给对话与提示词"（``config/root.py`` 校验）。只看这一轮的消息与配置，
        同一轮里每一段的 Job 判出来都一样。
        """

        if not isinstance(whole, ConversationSegment):
            raise TypeError("whole must be a ConversationSegment")
        return self._material_fits(whole, None)

    async def extract(
        self,
        segment: ConversationSegment,
        *,
        turn_context: MemoryTurnContext | None = None,
    ) -> MemoryExtractionResult:
        """解析完整片段；检索、规划失败明确终止，候选通不过来源校验时修补一轮、仍不过只剔除那几条。

        ``turn_context`` 只在一轮太长、按段分开抽取时给：带上这一轮开头的人话和前面各段的段摘要，段摘要放不下时
        从最早的一份起舍掉。
        """

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        if turn_context is not None:
            if not isinstance(turn_context, MemoryTurnContext):
                raise TypeError("turn_context must be a MemoryTurnContext or None")
            while turn_context.earlier_summaries and not self._material_fits(segment, turn_context):
                turn_context = turn_context.without_oldest_summary()
        initial = self._fit_initial_context(segment, await self.retriever.retrieve(segment))
        context = MemoryExtractionContext(
            initial,
            snapshot_reader=self.retriever.snapshot_reader,
            semantic_search=self.retriever.semantic_search,
            config=self.config,
        )
        decisions: list[MemoryRetrievalDecision] = []
        observations: list[MemoryRetrievalObservation] = []
        ended_at_limit = False

        for iteration in range(1, self.config.max_retrieval_iterations):
            response = await self._complete_model(
                self._retrieval_request(
                    segment, context, decisions, observations, allow_action=True, turn_context=turn_context
                ),
                model_class=MemoryRetrievalDecision,
                name="memory_retrieval_decision",
                prompt_version=_RETRIEVAL_PROMPT_VERSION,
            )
            decision = response.value
            decisions.append(decision)
            if decision.action is MemoryRetrievalAction.FINISH:
                break
            observations.append(await context.execute(decision, iteration=iteration))
        else:
            ended_at_limit = await self._final_retrieval_round(
                segment, context, decisions, observations, turn_context=turn_context
            )

        old_memories = context.snapshots
        page_ids = context.page_ids
        candidate_request = self.prompt_builder.candidate_request(segment, context, turn_context=turn_context)
        candidate_response = await self._complete_model(
            candidate_request,
            model_class=MemoryCandidateBatch,
            name="memory_candidate_batch",
            prompt_version=_CANDIDATE_PROMPT_VERSION,
        )
        try:
            candidates = candidate_response.value
            violations = candidates.context_violations(segment, old_memories, page_ids, zone=self.zone)
            if violations:
                candidates = await self._repaired_candidates(candidate_request, candidate_response, violations)
            candidates, rejections = candidates.without_context_violations(
                segment, old_memories, page_ids, zone=self.zone
            )
            candidates.validate_context(segment, old_memories, page_ids, zone=self.zone)
        except MemoryCandidateError as exc:
            raise MemoryExtractionPermanentError("memory candidate batch failed source-context validation") from exc

        try:
            mutation_read_set = self.mutation_reader.load(candidates, old_memories)
            mutations = self.mutation_planner.plan(candidates, mutation_read_set, page_ids)
        except (
            MemoryFieldMergeError,
            MemoryMutationPlanningError,
            MemoryNodeMatchError,
        ) as exc:
            raise MemoryExtractionPermanentError("memory candidate batch failed preliminary mutation planning") from exc

        return MemoryExtractionResult(
            conversation_id=segment.conversation_id,
            segment_id=segment.segment_id,
            source_segment_digest=segment.digest,
            candidates=candidates,
            mutations=mutations,
            old_memories=old_memories,
            page_ids=page_ids.copy(),
            retrieval_decisions=tuple(decisions),
            retrieval_observations=tuple(observations),
            retrieval_ended_at_limit=ended_at_limit,
            source_rejections=rejections,
        )

    async def _repaired_candidates(
        self,
        request: ChatRequest,
        response: StructuredResponse[MemoryCandidateBatch],
        violations: tuple[MemoryCandidateViolation, ...],
    ) -> MemoryCandidateBatch:
        """把通不过来源校验的条目与原因发回模型改一次（照 OpenViking ``extract_loop.py`` 的修补轮）。

        修补轮走同一个结构化客户端与同一份输入预算；请求放不下、或答复用完结构纠错仍不合格时，退回第一次的
        答复，由调用方只剔除不合格的条目。
        """

        listing = "\n".join(f"- {violation.describe()}" for violation in violations)
        repair = replace(
            request,
            messages=(
                *request.messages,
                ChatMessage(role="assistant", content=response.raw_text),
                ChatMessage(
                    role="user",
                    content=(
                        "上一次输出的 MemoryCandidateBatch 里，下列条目没有通过来源校验（冒号后是违反的规则）：\n"
                        f"{listing}\n"
                        "重新输出一个完整的 MemoryCandidateBatch：能依据完整事实与旧记忆改对的条目改对，"
                        "缺少来源、改不对的条目删掉；其余条目原样保留。不得为通过校验编造事实。"
                    ),
                ),
            ),
        )
        try:
            repaired = await self._complete_model(
                repair,
                model_class=MemoryCandidateBatch,
                name="memory_candidate_batch",
                prompt_version=_CANDIDATE_PROMPT_VERSION,
            )
        except (MemoryExtractionCapacityError, ModelStructuredOutputError):
            return response.value
        return repaired.value

    async def _final_retrieval_round(
        self,
        segment: ConversationSegment,
        context: MemoryExtractionContext,
        decisions: list[MemoryRetrievalDecision],
        observations: list[MemoryRetrievalObservation],
        *,
        turn_context: MemoryTurnContext | None,
    ) -> bool:
        """轮次用完后只问一次结束决策，返回检索是否在轮次上限处结束。

        这一轮不执行任何工具，候选生成只用已读到的旧记忆：判定不足、或答复用完结构纠错
        仍不合格，都不让 Job 失败，只记为在上限处结束。搜索与读取本身的失败发生在前面的
        动作轮，照常明确失败。
        """

        try:
            response = await self._complete_model(
                self._retrieval_request(
                    segment, context, decisions, observations, allow_action=False, turn_context=turn_context
                ),
                model_class=MemoryRetrievalFinalDecision,
                name="memory_retrieval_decision",
                prompt_version=_RETRIEVAL_PROMPT_VERSION,
            )
        except ModelStructuredOutputError:
            return True
        decision = response.value
        decisions.append(decision)
        return decision.status is MemoryRetrievalStatus.INSUFFICIENT

    def _retrieval_request(
        self,
        segment: ConversationSegment,
        context: MemoryExtractionContext,
        decisions: list[MemoryRetrievalDecision],
        observations: list[MemoryRetrievalObservation],
        *,
        allow_action: bool,
        turn_context: MemoryTurnContext | None,
    ) -> ChatRequest:
        return self.prompt_builder.retrieval_request(
            segment,
            context,
            decisions=tuple(decisions),
            observations=tuple(observations),
            allow_action=allow_action,
            turn_context=turn_context,
        )

    def _material_fits(self, segment: ConversationSegment, turn_context: MemoryTurnContext | None) -> bool:
        """不带旧记忆的两种请求是否都放得进留给对话与提示词的那部分预算。"""

        budget = self.config.max_input_tokens - self.config.max_old_memory_tokens
        try:
            retrieval = self.prompt_builder.retrieval_request(
                segment, None, decisions=(), observations=(), allow_action=True, turn_context=turn_context
            )
            candidate = self.prompt_builder.candidate_request(segment, None, turn_context=turn_context)
        except MemoryExtractionCapacityError:
            return False
        return (
            self.client.estimate_model_input_tokens(
                retrieval, model_class=MemoryRetrievalDecision, name="memory_retrieval_decision"
            )
            <= budget
            and self.client.estimate_model_input_tokens(
                candidate, model_class=MemoryCandidateBatch, name="memory_candidate_batch"
            )
            <= budget
        )

    async def _complete_model(
        self,
        request: ChatRequest,
        *,
        model_class: type[T],
        name: str,
        prompt_version: str,
    ) -> StructuredResponse[T]:
        try:
            return await self.client.complete_model_async(
                request,
                model_class=model_class,
                name=name,
                context=ChatCallContext(
                    prompt_version=prompt_version,
                    input_token_limit=self.config.max_input_tokens,
                ),
            )
        except ModelInputTooLargeError as exc:
            raise MemoryExtractionCapacityError(
                "model-visible extraction request exceeds its configured token limit"
            ) from exc

    def _fit_initial_context(
        self,
        segment: ConversationSegment,
        initial: MemoryRelatedContext,
    ) -> MemoryRelatedContext:
        """完整保留高优先级旧节点，并按相关性整节点准入其余快照。"""

        by_identity = {snapshot.identity: snapshot for snapshot in initial.snapshots.snapshots}
        essential = {str(MemoryURI.from_address(MemoryAddress.profile()))}
        for message in segment.messages:
            if message.role is ConversationMessageRole.TOOL_CALL:
                assert message.tool_name is not None
                essential.add(str(MemoryURI.from_address(MemoryAddress.tool(message.tool_name))))

        priority: list[str] = []
        seen: set[str] = set()
        for identity in (*sorted(essential), *(str(hit.uri) for hit in initial.search_hits)):
            if identity in by_identity and identity not in seen:
                priority.append(identity)
                seen.add(identity)
        priority.extend(identity for identity in sorted(by_identity) if identity not in seen)

        selected: list[VersionedSnapshot[MemoryDocument]] = []
        total_bytes = 0
        total_tokens = 0
        for identity in priority:
            snapshot = by_identity[identity]
            next_bytes = total_bytes + snapshot.size_bytes
            next_tokens = total_tokens + estimate_utf8_bytes_tokens(snapshot.size_bytes)
            fits = (
                len(selected) < self.config.max_old_memory_items
                and next_bytes <= self.config.max_old_memory_bytes
                and next_tokens <= self.config.max_old_memory_tokens
            )
            if not fits:
                if identity in essential and snapshot.exists:
                    raise MemoryExtractionCapacityError(
                        "required profile or tool memory cannot fit the extraction input budget"
                    )
                continue
            selected.append(snapshot)
            total_bytes = next_bytes
            total_tokens = next_tokens

        selected_ids = {snapshot.identity for snapshot in selected}
        return MemoryRelatedContext(
            conversation_id=initial.conversation_id,
            segment_id=initial.segment_id,
            source_segment_digest=initial.source_segment_digest,
            query=initial.query,
            search_roots=initial.search_roots,
            search_hits=tuple(hit for hit in initial.search_hits if str(hit.uri) in selected_ids),
            snapshots=SnapshotBatch(
                snapshots=tuple(sorted(selected, key=lambda snapshot: snapshot.identity)),
                total_bytes=total_bytes,
            ),
        )


__all__ = ["MemoryExtractionLoop"]
