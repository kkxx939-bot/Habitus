"""按领域分组暴露 Runtime 已经完成的依赖组装结果。"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import cast

from habitus.conversation import (
    ConversationBehaviorProjectionConsumer,
    ConversationBehaviorProjectionStore,
    ConversationConsumerDelivery,
    ConversationConsumerExecutionFence,
    ConversationConsumerOutcomeStore,
    ConversationConsumerStateInspector,
    ConversationSourceConsumer,
    ConversationSourceCoordinator,
    ConversationSourceRecovery,
    ConversationSourceStore,
)
from habitus.foundation.observability import MetricRegistry, Observer
from habitus.infrastructure.observability import ManagedObservability
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector import VectorStoreFactory
from habitus.memory.compaction import MemoryLifecycleManager
from habitus.memory.conversation import (
    ConversationMessageJournal,
    ConversationRetentionPlanner,
    ConversationSemanticBoundaryScorer,
    ConversationSummaryCompactor,
    ConversationSummaryExpander,
    ConversationSummaryService,
    PersistentConversationSummaryVectorIndex,
    SQLiteConversationSummaryUseStore,
)
from habitus.memory.editor import MemoryEditor
from habitus.memory.indexing import PersistentMemoryVectorIndex
from habitus.memory.retrieval import MemorySemanticSearchEngine, SearchService
from habitus.memory.semantic import MemorySemanticRefresher
from habitus.memory.tree import MemoryTree
from habitus.memory.workflow import (
    ConversationLifecycleManager,
    ConversationMemoryEnqueuer,
    MemoryChangeReceiptStore,
    MemoryConversationConsumer,
    MemoryConversationOutputStore,
    MemoryJobRunner,
    MemoryJobStore,
)
from habitus.model_client import (
    ChatClient,
    Embedder,
    ProviderFactory,
    Reranker,
    StructuredChatClient,
)
from habitus.runtime.behavior import BehaviorRuntimeComponents
from habitus.runtime.lifecycle import LifecycleWorker
from habitus.runtime.prediction import PredictionRuntimeComponents
from habitus.runtime.worker import MemoryWorker


@dataclass(frozen=True)
class RuntimeInfrastructure:
    """Runtime 共享的锁基础设施及其显式初始化边界。"""

    path_lock: PathLock
    vector_stores: VectorStoreFactory
    observability: MetricRegistry = field(default_factory=MetricRegistry)
    observer: Observer | None = None
    managed_observability: ManagedObservability | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path_lock, PathLock):
            raise TypeError("path_lock must be PathLock")
        if not isinstance(self.vector_stores, VectorStoreFactory):
            raise TypeError("vector_stores must be VectorStoreFactory")
        if not isinstance(self.observability, MetricRegistry):
            raise TypeError("observability must be MetricRegistry")
        if self.observer is None:
            object.__setattr__(self, "observer", self.observability)
        elif not callable(getattr(self.observer, "record", None)):
            raise TypeError("observer must implement record")
        if self.managed_observability is not None and not isinstance(
            self.managed_observability,
            ManagedObservability,
        ):
            raise TypeError("managed_observability must be ManagedObservability or None")

    def initialize(self) -> None:
        if self.managed_observability is not None:
            self.managed_observability.initialize()
        initializer = getattr(self.path_lock.lock_store, "initialize", None)
        if initializer is not None:
            if not callable(initializer):
                raise TypeError("lock store initialize attribute must be callable")
            initializer()


@dataclass(frozen=True)
class RuntimeModels:
    """Runtime 内共享的一组模型运行对象。"""

    providers: ProviderFactory
    chat: ChatClient
    structured_chat: StructuredChatClient
    embedder: Embedder
    reranker: Reranker | None

    def __post_init__(self) -> None:
        if not isinstance(self.providers, ProviderFactory):
            raise TypeError("providers must be ProviderFactory")
        if not isinstance(self.chat, ChatClient):
            raise TypeError("chat must be ChatClient")
        if not isinstance(self.structured_chat, StructuredChatClient):
            raise TypeError("structured_chat must be StructuredChatClient")
        if self.structured_chat.client is not self.chat:
            raise ValueError("structured_chat must wrap the shared chat client")
        if not callable(getattr(self.embedder, "embed_query", None)) or not callable(
            getattr(self.embedder, "embed_documents", None)
        ):
            raise TypeError("embedder must implement the Embedder contract")
        if self.reranker is not None and not callable(getattr(self.reranker, "rerank", None)):
            raise TypeError("reranker must implement the Reranker contract")

    async def aclose(self) -> None:
        """关闭所有模型能力；即使某一项失败也继续释放剩余资源。"""

        resources = (self.chat, self.embedder, self.reranker)
        first_error: BaseException | None = None
        seen: set[int] = set()
        for resource in resources:
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            close = getattr(resource, "aclose", None)
            if not callable(close):
                if first_error is None:
                    first_error = TypeError("runtime model resource does not implement aclose")
                continue
            try:
                await cast(Awaitable[None], close())
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


@dataclass(frozen=True)
class RuntimeConversation:
    """Conversation Source、原文、独立投影、切段与摘要服务。"""

    sources: ConversationSourceStore
    source_outcomes: ConversationConsumerOutcomeStore
    memory_outputs: MemoryConversationOutputStore
    behavior_projections: ConversationBehaviorProjectionStore
    behavior_projection_consumer: ConversationBehaviorProjectionConsumer
    source_inspector: ConversationConsumerStateInspector
    source_fence: ConversationConsumerExecutionFence
    source_delivery: ConversationConsumerDelivery
    source_coordinator: ConversationSourceCoordinator
    source_recovery: ConversationSourceRecovery
    journal: ConversationMessageJournal
    retention: ConversationRetentionPlanner
    boundary_scorer: ConversationSemanticBoundaryScorer
    summaries: ConversationSummaryService
    summary_compactor: ConversationSummaryCompactor
    summary_vector_index: PersistentConversationSummaryVectorIndex
    summary_use: SQLiteConversationSummaryUseStore
    summary_expander: ConversationSummaryExpander

    def __post_init__(self) -> None:
        if not isinstance(self.sources, ConversationSourceStore):
            raise TypeError("sources must be ConversationSourceStore")
        if not isinstance(self.source_outcomes, ConversationConsumerOutcomeStore):
            raise TypeError("source_outcomes must be ConversationConsumerOutcomeStore")
        if not isinstance(self.memory_outputs, MemoryConversationOutputStore):
            raise TypeError("memory_outputs must be MemoryConversationOutputStore")
        if not isinstance(self.behavior_projections, ConversationBehaviorProjectionStore):
            raise TypeError("behavior_projections must be ConversationBehaviorProjectionStore")
        if not isinstance(self.behavior_projection_consumer, ConversationBehaviorProjectionConsumer):
            raise TypeError("behavior_projection_consumer must be ConversationBehaviorProjectionConsumer")
        if not isinstance(self.source_inspector, ConversationConsumerStateInspector):
            raise TypeError("source_inspector must be ConversationConsumerStateInspector")
        if not isinstance(self.source_fence, ConversationConsumerExecutionFence):
            raise TypeError("source_fence must be ConversationConsumerExecutionFence")
        if not isinstance(self.source_delivery, ConversationConsumerDelivery):
            raise TypeError("source_delivery must be ConversationConsumerDelivery")
        if not isinstance(self.source_coordinator, ConversationSourceCoordinator):
            raise TypeError("source_coordinator must be ConversationSourceCoordinator")
        if not isinstance(self.source_recovery, ConversationSourceRecovery):
            raise TypeError("source_recovery must be ConversationSourceRecovery")
        if not isinstance(self.journal, ConversationMessageJournal):
            raise TypeError("journal must be ConversationMessageJournal")
        if not isinstance(self.retention, ConversationRetentionPlanner):
            raise TypeError("retention must be ConversationRetentionPlanner")
        if not isinstance(
            self.boundary_scorer,
            ConversationSemanticBoundaryScorer,
        ):
            raise TypeError("boundary_scorer must be ConversationSemanticBoundaryScorer")
        if not isinstance(self.summaries, ConversationSummaryService):
            raise TypeError("summaries must be ConversationSummaryService")
        if not isinstance(self.summary_compactor, ConversationSummaryCompactor):
            raise TypeError("summary_compactor must be ConversationSummaryCompactor")
        if not isinstance(
            self.summary_vector_index,
            PersistentConversationSummaryVectorIndex,
        ):
            raise TypeError(
                "summary_vector_index must be PersistentConversationSummaryVectorIndex"
            )
        if not isinstance(self.summary_use, SQLiteConversationSummaryUseStore):
            raise TypeError("summary_use must be SQLiteConversationSummaryUseStore")
        if not isinstance(self.summary_expander, ConversationSummaryExpander):
            raise TypeError("summary_expander must be ConversationSummaryExpander")
        if self.summaries.store.layout.root != self.journal.layout.root:
            raise ValueError("conversation journal and summaries must share one root")
        if self.summary_compactor.journal is not self.journal:
            raise ValueError("conversation summary compactor must use the shared journal")
        if self.summary_compactor.segment_store is not self.summaries.store:
            raise ValueError("conversation summary compactor must use the shared segment summary store")
        if self.summary_vector_index.compactor is not self.summary_compactor:
            raise ValueError("Conversation must share one Summary compactor and vector index")
        if self.summary_compactor.use_store is not self.summary_use:
            raise ValueError("Conversation must share one Summary use store")
        if self.summary_expander.segment_store is not self.summaries.store:
            raise ValueError("Conversation must share one Summary source expander")
        if not (
            self.sources.root
            == self.source_outcomes.root
            == self.memory_outputs.root
            == self.behavior_projections.root
            == self.journal.layout.root
        ):
            raise ValueError("Conversation Source, Projection and Journal must share one root")
        if self.behavior_projection_consumer.store is not self.behavior_projections:
            raise ValueError("Behavior Projection Consumer must use the shared outbox")
        if (
            self.source_delivery.consumers[ConversationSourceConsumer.BEHAVIOR_PROJECTION]
            is not self.behavior_projection_consumer
        ):
            raise ValueError("Source delivery must use the shared Behavior Projection Consumer")
        if self.source_coordinator.sources is not self.sources:
            raise ValueError("Source Coordinator must use the shared Source Store")
        if self.source_coordinator.delivery is not self.source_delivery:
            raise ValueError("Source Coordinator must use the shared delivery service")
        if self.source_recovery.delivery is not self.source_delivery:
            raise ValueError("Source Recovery must use the shared delivery service")


@dataclass(frozen=True)
class RuntimeMemory:
    """长期记忆检索、编辑与可重建语义层服务。"""

    tree: MemoryTree
    search: SearchService
    editor: MemoryEditor
    semantic_refresher: MemorySemanticRefresher
    vector_index: PersistentMemoryVectorIndex
    lifecycle: MemoryLifecycleManager

    def __post_init__(self) -> None:
        if not isinstance(self.tree, MemoryTree):
            raise TypeError("tree must be MemoryTree")
        if not isinstance(self.search, SearchService):
            raise TypeError("search must be SearchService")
        if not isinstance(self.editor, MemoryEditor):
            raise TypeError("editor must be MemoryEditor")
        if not isinstance(self.semantic_refresher, MemorySemanticRefresher):
            raise TypeError("semantic_refresher must be MemorySemanticRefresher")
        if not isinstance(self.vector_index, PersistentMemoryVectorIndex):
            raise TypeError("vector_index must be PersistentMemoryVectorIndex")
        if not isinstance(self.lifecycle, MemoryLifecycleManager):
            raise TypeError("lifecycle must be MemoryLifecycleManager")
        if self.search.tree is not self.tree:
            raise ValueError("memory search must use the shared memory tree")
        if self.editor.transaction.tree is not self.tree:
            raise ValueError("memory editor must use the shared memory tree")
        if self.semantic_refresher.tree is not self.tree:
            raise ValueError("semantic refresher must use the shared memory tree")
        if self.vector_index.tree is not self.tree:
            raise ValueError("memory vector index must use the shared memory tree")
        if self.lifecycle.tree is not self.tree:
            raise ValueError("memory lifecycle must use the shared memory tree")
        if not isinstance(self.search.semantic_search, MemorySemanticSearchEngine):
            raise TypeError("memory search must use MemorySemanticSearchEngine")
        if self.search.semantic_search.index is not self.vector_index:
            raise ValueError("memory search must use the shared persistent vector index")


@dataclass(frozen=True)
class RuntimeWorkflow:
    """Conversation 到有序记忆提交的耐久工作流入口。"""

    jobs: MemoryJobStore
    receipts: MemoryChangeReceiptStore
    enqueuer: ConversationMemoryEnqueuer
    conversation_consumer: MemoryConversationConsumer
    lifecycle: ConversationLifecycleManager
    runner: MemoryJobRunner
    worker: MemoryWorker
    lifecycle_worker: LifecycleWorker

    def __post_init__(self) -> None:
        if not isinstance(self.jobs, MemoryJobStore):
            raise TypeError("jobs must be MemoryJobStore")
        if not isinstance(self.receipts, MemoryChangeReceiptStore):
            raise TypeError("receipts must be MemoryChangeReceiptStore")
        if not isinstance(self.enqueuer, ConversationMemoryEnqueuer):
            raise TypeError("enqueuer must be ConversationMemoryEnqueuer")
        if not isinstance(self.conversation_consumer, MemoryConversationConsumer):
            raise TypeError("conversation_consumer must be MemoryConversationConsumer")
        if not isinstance(self.lifecycle, ConversationLifecycleManager):
            raise TypeError("lifecycle must be ConversationLifecycleManager")
        if not isinstance(self.runner, MemoryJobRunner):
            raise TypeError("runner must be MemoryJobRunner")
        if not isinstance(self.worker, MemoryWorker):
            raise TypeError("worker must be MemoryWorker")
        if not isinstance(self.lifecycle_worker, LifecycleWorker):
            raise TypeError("lifecycle_worker must be LifecycleWorker")
        if self.enqueuer.jobs is not self.jobs or self.runner.store is not self.jobs:
            raise ValueError("memory workflow must share one job store")
        if self.conversation_consumer.enqueuer is not self.enqueuer:
            raise ValueError("memory workflow must share one Conversation enqueuer")
        if self.lifecycle.jobs is not self.jobs or self.lifecycle.receipts is not self.receipts:
            raise ValueError("conversation lifecycle must share workflow stores")
        if (
            self.lifecycle.summary_vector_index
            is not self.runner.committed_finalizer.summary_vector_index
        ):
            raise ValueError("workflow must share one Summary vector index")
        if self.runner.transaction_recovery.change_receipts is not self.receipts:
            raise ValueError("memory workflow must share one receipt store")
        if self.worker.runner is not self.runner:
            raise ValueError("memory workflow must share one runner with its worker")
        if self.lifecycle_worker.manager is not self.lifecycle:
            raise ValueError("memory workflow must share one lifecycle manager with its worker")
        if self.lifecycle_worker.journal.path_lock is not self.jobs.path_lock:
            raise ValueError("both Runtime workers must share one path lock")


@dataclass(frozen=True)
class RuntimeComponents:
    """Runtime 对外暴露的基础设施与清晰领域边界。"""

    infrastructure: RuntimeInfrastructure
    models: RuntimeModels
    conversation: RuntimeConversation
    memory: RuntimeMemory
    workflow: RuntimeWorkflow
    # 行为管线（观测→融合→归约→行为树）；primary_subject 未配置时为 None（行为侧未启用）。
    behavior: BehaviorRuntimeComponents | None = None
    # 时间预测树夜批；prediction.enabled 为假时为 None。它读行为树，所以行为侧未启用时也必然为 None。
    prediction: PredictionRuntimeComponents | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.infrastructure, RuntimeInfrastructure):
            raise TypeError("infrastructure must be RuntimeInfrastructure")
        if not isinstance(self.models, RuntimeModels):
            raise TypeError("models must be RuntimeModels")
        if not isinstance(self.conversation, RuntimeConversation):
            raise TypeError("conversation must be RuntimeConversation")
        if not isinstance(self.memory, RuntimeMemory):
            raise TypeError("memory must be RuntimeMemory")
        if not isinstance(self.workflow, RuntimeWorkflow):
            raise TypeError("workflow must be RuntimeWorkflow")
        if self.behavior is not None and not isinstance(self.behavior, BehaviorRuntimeComponents):
            raise TypeError("behavior must be BehaviorRuntimeComponents or None")
        if self.prediction is not None:
            if not isinstance(self.prediction, PredictionRuntimeComponents):
                raise TypeError("prediction must be PredictionRuntimeComponents or None")
            if self.behavior is None:
                raise ValueError("prediction rebuilds from the behaviour tree; behaviour must be enabled")
            # 实例同一性：夜批必须读**这个** Runtime 写入的那棵行为树，不是另开一个同路径的实例。
            if self.prediction.rebuilder.behavior_tree is not self.behavior.tree:
                raise ValueError("prediction must rebuild from the assembled behaviour tree")
        if self.workflow.enqueuer.conversations is not self.conversation.journal:
            raise ValueError("workflow enqueuer must use the shared conversation journal")
        if (
            self.conversation.source_delivery.consumers[ConversationSourceConsumer.MEMORY]
            is not self.workflow.conversation_consumer
        ):
            raise ValueError("Source Coordinator must use the shared Memory Conversation Consumer")
        if self.workflow.runner.executor.editor is not self.memory.editor:
            raise ValueError("workflow runner must use the shared memory editor")
        if self.workflow.runner.committed_finalizer.vector_index is not self.memory.vector_index:
            raise ValueError("workflow runner must publish the shared persistent vector index")
        if self.memory.search.conversation_context.journal is not self.conversation.journal:
            raise ValueError("memory search must use the shared Conversation journal")
        if self.memory.search.query_planner.client is not self.models.structured_chat:
            raise ValueError("memory search must use the shared structured chat client")
        if self.memory.search.retrieval_grader.client is not self.models.structured_chat:
            raise ValueError("retrieval grader must use the shared structured chat client")
        if self.memory.search.summary_search is not self.conversation.summary_vector_index:
            raise ValueError("memory search must use the shared Summary fallback index")
        if self.memory.search.semantic_search is not self.memory.editor.extraction_loop.retriever.semantic_search:
            raise ValueError("memory search and editor must share one semantic search engine")
        if self.workflow.jobs.path_lock is not self.infrastructure.path_lock:
            raise ValueError("workflow and Runtime must share one path lock")


__all__ = [
    "RuntimeComponents",
    "RuntimeConversation",
    "RuntimeInfrastructure",
    "RuntimeMemory",
    "RuntimeModels",
    "RuntimeWorkflow",
]
