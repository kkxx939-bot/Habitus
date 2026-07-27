"""Conversation 到初步字段计划和严格候选审查的受控解析闭环。"""

from __future__ import annotations

from memory.editor.candidate import MemoryCandidateBatch, MemoryCandidateError
from memory.editor.extraction.config import MemoryExtractionConfig
from memory.editor.extraction.context import MemoryExtractionContext
from memory.editor.extraction.model import (
    MemoryCandidateRejectedError,
    MemoryCandidateReview,
    MemoryCandidateReviewIssue,
    MemoryExtractionError,
    MemoryExtractionResult,
    MemoryRetrievalAction,
    MemoryRetrievalDecision,
    MemoryRetrievalObservation,
    MemoryReviewDecision,
    MemoryReviewIssueCode,
)
from memory.editor.extraction.prompt import MemoryExtractionPromptBuilder
from memory.editor.identity import MemoryIdentityPlanner, MemoryIdentityPlanningError
from memory.editor.mutation import (
    MemoryFieldMergeError,
    MemoryMutationPlan,
    MemoryMutationPlanner,
    MemoryMutationPlanningError,
    MemoryMutationReadSetLoader,
    MemoryNodeMatchError,
)
from memory.editor.retrieval import MemoryRelatedRetriever
from ModelClient import StructuredChatClient
from pre.conversation import ConversationSegment


class MemoryExtractionLoop:
    """串联受控检索、候选生成、初步字段计划和第二遍审查。"""

    def __init__(
        self,
        *,
        client: StructuredChatClient,
        retriever: MemoryRelatedRetriever,
        config: MemoryExtractionConfig | None = None,
        mutation_reader: MemoryMutationReadSetLoader | None = None,
        mutation_planner: MemoryMutationPlanner | None = None,
        identity_planner: MemoryIdentityPlanner | None = None,
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
        if identity_planner is not None and not isinstance(identity_planner, MemoryIdentityPlanner):
            raise TypeError("identity_planner must be a MemoryIdentityPlanner")
        self.client = client
        self.retriever = retriever
        self.config = config or MemoryExtractionConfig()
        self.mutation_reader = mutation_reader or MemoryMutationReadSetLoader(retriever.snapshot_reader)
        self.mutation_planner = mutation_planner or MemoryMutationPlanner(retriever.schema_registry)
        self.identity_planner = identity_planner or MemoryIdentityPlanner(retriever.schema_registry)
        self.prompt_builder = MemoryExtractionPromptBuilder(self.config)

    async def extract(self, segment: ConversationSegment) -> MemoryExtractionResult:
        """解析完整片段；任何不足或复核失败都明确终止，不生成降级候选。"""

        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be a ConversationSegment")
        initial = await self.retriever.retrieve(segment)
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
            response = await self.client.complete_model_async(
                request,
                model_class=MemoryRetrievalDecision,
                name="memory_retrieval_decision",
            )
            decision = response.value
            decisions.append(decision)
            decision.require_action_allowed(allow_action=allow_action)
            if decision.action is MemoryRetrievalAction.FINISH:
                break
            observations.append(await context.execute(decision, iteration=iteration))
        else:  # pragma: no cover - 最终轮动作会在上方明确失败
            raise MemoryExtractionError("retrieval loop ended without a final decision")

        old_memories = context.snapshots
        page_ids = context.page_ids
        feedback: tuple[MemoryCandidateReviewIssue, ...] = ()
        previous_candidates: MemoryCandidateBatch | None = None
        mutations: MemoryMutationPlan | None = None
        maximum_attempts = self.config.max_candidate_regenerations + 1
        for attempt in range(1, maximum_attempts + 1):
            candidate_response = await self.client.complete_model_async(
                self.prompt_builder.candidate_request(
                    segment,
                    context,
                    feedback=feedback,
                    previous_candidates=previous_candidates,
                ),
                model_class=MemoryCandidateBatch,
                name="memory_candidate_batch",
            )
            candidates = candidate_response.value
            try:
                candidates.validate_context(segment, old_memories, page_ids)
            except MemoryCandidateError as exc:
                feedback = (self._context_issue(exc),)
                previous_candidates = candidates
                if attempt >= maximum_attempts:
                    raise MemoryExtractionError("memory candidate batch failed source-context validation") from exc
                continue

            try:
                mutation_read_set = self.mutation_reader.load(candidates, old_memories)
                mutations = self.mutation_planner.plan(candidates, mutation_read_set, page_ids)
            except (
                MemoryFieldMergeError,
                MemoryMutationPlanningError,
                MemoryNodeMatchError,
            ) as exc:
                feedback = (self._context_issue(exc),)
                previous_candidates = candidates
                if attempt >= maximum_attempts:
                    raise MemoryExtractionError("memory candidate batch failed preliminary mutation planning") from exc
                continue

            review_response = await self.client.complete_model_async(
                self.prompt_builder.review_request(segment, context, candidates, mutations),
                model_class=MemoryCandidateReview,
                name="memory_candidate_review",
            )
            review = review_response.value
            if review.decision is MemoryReviewDecision.ACCEPT:
                extraction = MemoryExtractionResult(
                    conversation_id=segment.conversation_id,
                    segment_id=segment.segment_id,
                    source_segment_digest=segment.digest,
                    candidates=candidates,
                    mutations=mutations,
                    old_memories=old_memories,
                    page_ids=page_ids.copy(),
                    retrieval_decisions=tuple(decisions),
                    retrieval_observations=tuple(observations),
                    review=review,
                    candidate_attempts=attempt,
                )
                try:
                    self.identity_planner.plan(extraction)
                except MemoryIdentityPlanningError as exc:
                    feedback = (self._context_issue(exc),)
                    previous_candidates = candidates
                    if attempt >= maximum_attempts:
                        raise MemoryExtractionError(
                            "reviewed identity proposals failed deterministic planning"
                        ) from exc
                    continue
                return extraction
            feedback = review.issues
            previous_candidates = candidates
            if attempt >= maximum_attempts:
                details = "; ".join(f"{issue.code.value}: {issue.detail}" for issue in review.issues)
                raise MemoryCandidateRejectedError(
                    f"memory candidate batch remained rejected after {attempt} attempt(s): {details}"
                )
        raise AssertionError("candidate review loop exhausted without a result")  # pragma: no cover

    @staticmethod
    def _context_issue(error: ValueError) -> MemoryCandidateReviewIssue:
        detail = str(error).replace("\n", " ")[:800]
        lowered = detail.casefold()
        if "same_memory" in lowered or "merge" in lowered or "duplicate identity" in lowered:
            code = MemoryReviewIssueCode.UNJUSTIFIED_IDENTITY_MERGE
        elif "remove_memory" in lowered or "delete" in lowered:
            code = MemoryReviewIssueCode.UNJUSTIFIED_MEMORY_DELETE
        elif "tool" in lowered:
            code = MemoryReviewIssueCode.INVALID_TOOL_GENERALIZATION
        elif "intention" in lowered:
            code = MemoryReviewIssueCode.EVENT_INTENTION_CONFUSION
        elif "relation remove" in lowered:
            code = MemoryReviewIssueCode.INVALID_RELATION_REMOVE
        elif "relation" in lowered:
            code = MemoryReviewIssueCode.UNJUSTIFIED_RELATION
        else:
            code = MemoryReviewIssueCode.INVALID_PAGE_IDENTITY
        return MemoryCandidateReviewIssue(code=code, detail=detail)


__all__ = ["MemoryExtractionLoop"]
