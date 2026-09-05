"""不可变 Conversation Source、Output、Outcome、执行栅栏与恢复。"""

from habitus.conversation.source.coordinator import (
    ConversationSourceCoordinator,
    ConversationSourceDispatchHandle,
    ConversationSourcePendingDelivery,
)
from habitus.conversation.source.delivery import (
    ConversationConsumerDelivery,
    ConversationConsumerEnsureResult,
    ConversationOrderedPredecessorBrokenError,
    ConversationOrderedPredecessorPendingError,
)
from habitus.conversation.source.fence import (
    ConversationConsumerExecutionFence,
    ConversationConsumerExecutionLease,
    ConversationConsumerLeaseLostError,
)
from habitus.conversation.source.model import (
    ConversationSourceEnvelope,
    ConversationSourceError,
    conversation_source_request_digest,
)
from habitus.conversation.source.receipt import (
    ConsumerOutputRef,
    ConversationConsumerOutcome,
    ConversationConsumerOutcomeState,
    ConversationConsumerOutcomeStore,
    ConversationConsumerRunDisposition,
    ConversationConsumerRunResult,
    ConversationSourceConsumer,
    conversation_consumer_output_id,
)
from habitus.conversation.source.recovery import (
    ConversationSourceRecovery,
    ConversationSourceRecoveryEntry,
    ConversationSourceRecoveryResult,
)
from habitus.conversation.source.repair import (
    ConversationSourceOutputRepair,
    ConversationSourceRepairError,
    ConversationSourceRepairResult,
)
from habitus.conversation.source.state import (
    ConversationConsumerBrokenOutcomeError,
    ConversationConsumerCorruptionError,
    ConversationConsumerDeliveryState,
    ConversationConsumerState,
    ConversationConsumerStateInspector,
)
from habitus.conversation.source.store import ConversationSourceStore

__all__ = [
    "ConsumerOutputRef",
    "ConversationConsumerBrokenOutcomeError",
    "ConversationConsumerCorruptionError",
    "ConversationConsumerDelivery",
    "ConversationConsumerDeliveryState",
    "ConversationConsumerEnsureResult",
    "ConversationConsumerExecutionFence",
    "ConversationConsumerExecutionLease",
    "ConversationConsumerLeaseLostError",
    "ConversationConsumerOutcome",
    "ConversationConsumerOutcomeState",
    "ConversationConsumerOutcomeStore",
    "ConversationConsumerRunDisposition",
    "ConversationConsumerRunResult",
    "ConversationConsumerState",
    "ConversationConsumerStateInspector",
    "ConversationOrderedPredecessorBrokenError",
    "ConversationOrderedPredecessorPendingError",
    "ConversationSourceConsumer",
    "ConversationSourceCoordinator",
    "ConversationSourceDispatchHandle",
    "ConversationSourceEnvelope",
    "ConversationSourceOutputRepair",
    "ConversationSourceRepairError",
    "ConversationSourceRepairResult",
    "ConversationSourceError",
    "ConversationSourcePendingDelivery",
    "ConversationSourceRecovery",
    "ConversationSourceRecoveryEntry",
    "ConversationSourceRecoveryResult",
    "ConversationSourceStore",
    "conversation_consumer_output_id",
    "conversation_source_request_digest",
]
