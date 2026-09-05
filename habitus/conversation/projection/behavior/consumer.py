"""把 Behavior 投影接入统一 Source Consumer 交付契约的适配层。"""

from __future__ import annotations

from habitus.conversation.projection.behavior.projector import ConversationBehaviorProjector
from habitus.conversation.projection.behavior.store import ConversationBehaviorProjectionStore
from habitus.conversation.source.fence import ConversationConsumerExecutionLease
from habitus.conversation.source.model import ConversationSourceEnvelope
from habitus.conversation.source.receipt import (
    ConversationConsumerRunDisposition,
    ConversationConsumerRunResult,
    ConversationSourceConsumer,
)


class ConversationBehaviorProjectionConsumer:
    consumer = ConversationSourceConsumer.BEHAVIOR_PROJECTION
    # 投影只依赖单个不可变 Source，同一 Conversation 内没有跨 Source 的先后要求。
    ordered_within_conversation = False

    def __init__(
        self,
        projector: ConversationBehaviorProjector,
        store: ConversationBehaviorProjectionStore,
    ) -> None:
        self.projector = projector
        self.store = store
        self.output_store = store
        self.processor_fingerprint = projector.processor_fingerprint

    async def execute(
        self,
        envelope: ConversationSourceEnvelope,
        lease: ConversationConsumerExecutionLease,
    ) -> ConversationConsumerRunResult:
        projected = self.projector.project(envelope)
        if projected is None:
            return ConversationConsumerRunResult(
                disposition=ConversationConsumerRunDisposition.SKIPPED,
                output_ref=None,
                skip_reason="NO_ELIGIBLE_MESSAGES",
                runtime_result=None,
            )
        stored = await lease.run_fenced(lambda: self.store.put(envelope, projected))
        return ConversationConsumerRunResult(
            disposition=ConversationConsumerRunDisposition.OUTPUT_WRITTEN,
            output_ref=self.store.ref(stored),
            skip_reason=None,
            runtime_result=stored,
        )


__all__ = ["ConversationBehaviorProjectionConsumer"]
