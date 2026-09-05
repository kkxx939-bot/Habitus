"""Conversation 到严格候选和初步字段计划的受控解析流程。"""

from __future__ import annotations

from typing import TypeVar

from habitus.infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from habitus.memory.document import MemoryDocument
from habitus.memory.editor.candidate import MemoryCandidateBatch, MemoryCandidateError
from habitus.memory.editor.extraction.config import MemoryExtractionConfig
from habitus.memory.editor.extraction.context import MemoryExtractionContext
from habitus.memory.editor.extraction.model import (
    MemoryExtractionCapacityError,
    MemoryExtractionPermanentError,
    MemoryExtractionResult,
    MemoryRetrievalAction,
    MemoryRetrievalDecision,
    MemoryRetrievalObservation,
)
from habitus.memory.editor.extraction.prompt import MemoryExtractionPromptBuilder
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
    ChatRequest,
    ModelInputTooLargeError,
    StructuredChatClient,
    StructuredResponse,
    estimate_utf8_bytes_tokens,
)
from habitus.pre.conversation import ConversationMessageRole, ConversationSegment

T = TypeVar("T")


class MemoryExtractionLoop:
    """串联受控检索、模型候选生成和确定性初步字段计划。"""

    def __init__(
        self,
        *,
        client: StructuredChatClient,
        retriever: MemoryRelatedRetriever,
        config: MemoryExtractionConfig | None = None,
        mutation_reader: MemoryMutationReadSetLoader | None = None,
        mutation_planner: MemoryMutationPlanner | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be a StructuredChatClient")
        if not isinstance(retriever, MemoryRelatedRetriever):
            raise TypeError("retriever must be a MemoryRelatedRetriever")
        if config is not None and not isinstance(config, MemoryExtractionConfig):
            raise TypeError("config must be a MemoryExtractionConfig")
        if mutation_reader is not None and not isinstance(mutation_reader, MemoryMutationReadSetLoader):
            raise TypeError("mutation_reader must be a MemoryMutationReadSetLoader")
        if mutation_planner is not None and not isinstance(mutation_planner, MemoryMutationPlanner):
            raise TypeError("mutation_planner must be a MemoryMutationPlanner")
        self.client = client
        self.retriever = retriever
        self.config = config or MemoryExtractionConfig()
        self.mutation_reader = mutation_reader or MemoryMutationReadSetLoader(retriever.snapshot_reader)
        self.mutation_planner = mutation_planner or MemoryMutationPlanner(retriever.schema_registry)
        self.prompt_builder = MemoryExtractionPromptBuilder(self.config)

    async def extract(self, segment: ConversationSegment) -> MemoryExtractionResult:
        """解析完整片段；任何检索、候选或规划失败都明确终止。"""

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        initial = self._fit_initial_context(segment, await self.retriever.retrieve(segment))
        context = MemoryExtractionContext(
            initial,
            snapshot_reader=self.retriever.snapshot_reader,
            semantic_search=self.retriever.semantic_search,
            config=self.config,
        )
        decisions: list[MemoryRetrievalDecision] = []
        observations: list[MemoryRetrievalObservation] = []

        for iteration in range(1, self.config.max_retrieval_iterations + 1):
            allow_action = iteration < self.config.max_retrieval_iterations
            request = self.prompt_builder.retrieval_request(
                segment,
                context,
                decisions=tuple(decisions),
                observations=tuple(observations),
                allow_action=allow_action,
            )
            response = await self._complete_model(
                request,
                model_class=MemoryRetrievalDecision,
                name="memory_retrieval_decision",
                prompt_version="memory_retrieval_grader_v1",
            )
            decision = response.value
            decisions.append(decision)
            decision.require_action_allowed(allow_action=allow_action)
            if decision.action is MemoryRetrievalAction.FINISH:
                break
            observations.append(await context.execute(decision, iteration=iteration))
        else:  # pragma: no cover - 最终轮动作会在上方明确失败
            raise MemoryExtractionPermanentError("retrieval loop ended without a final decision")

        old_memories = context.snapshots
        page_ids = context.page_ids
        candidate_response = await self._complete_model(
            self.prompt_builder.candidate_request(segment, context),
            model_class=MemoryCandidateBatch,
            name="memory_candidate_batch",
            prompt_version="memory_candidate_extraction_v2",
        )
        candidates = candidate_response.value
        try:
            candidates.validate_context(segment, old_memories, page_ids)
        except MemoryCandidateError as exc:
            raise MemoryExtractionPermanentError(
                "memory candidate batch failed source-context validation"
            ) from exc

        try:
            mutation_read_set = self.mutation_reader.load(candidates, old_memories)
            mutations = self.mutation_planner.plan(candidates, mutation_read_set, page_ids)
        except (
            MemoryFieldMergeError,
            MemoryMutationPlanningError,
            MemoryNodeMatchError,
        ) as exc:
            raise MemoryExtractionPermanentError(
                "memory candidate batch failed preliminary mutation planning"
            ) from exc

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
            search_hits=tuple(
                hit for hit in initial.search_hits if str(hit.uri) in selected_ids
            ),
            snapshots=SnapshotBatch(
                snapshots=tuple(sorted(selected, key=lambda snapshot: snapshot.identity)),
                total_bytes=total_bytes,
            ),
        )



__all__ = ["MemoryExtractionLoop"]
