"""按领域分组暴露 Runtime 已经完成的依赖组装结果。"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import cast

from behavior.claim import (
    ClaimPipelineService,
    ClaimProducerRegistry,
    StructuredSemanticClaimProducer,
)
from behavior.evidence import EvidenceService
from behavior.persistence.sqlite import SQLiteBehaviorEvidenceClaimStore
from behavior.source import SourceRecordService
from foundation.observability import MetricRegistry, Observer
from infrastructure.observability import ManagedObservability
from infrastructure.store.contracts import PathLock
from infrastructure.vector import VectorStoreFactory
from memory.compaction import MemoryLifecycleManager
from memory.conversation import (
    ConversationMessageJournal,
    ConversationRetentionPlanner,
    ConversationSemanticBoundaryScorer,
    ConversationSummaryCompactor,
    ConversationSummaryExpander,
    ConversationSummaryService,
    PersistentConversationSummaryVectorIndex,
    SQLiteConversationSummaryUseStore,
)
from memory.editor import MemoryEditor
from memory.indexing import PersistentMemoryVectorIndex
from memory.retrieval import MemorySemanticSearchEngine, SearchService
from memory.semantic import MemorySemanticRefresher
from memory.tree import MemoryTree
from memory.workflow import (
    ConversationLifecycleManager,
    ConversationMemoryEnqueuer,
    MemoryChangeReceiptStore,
    MemoryJobRunner,
    MemoryJobStore,
)
from ModelClient import (
    ChatClient,
    Embedder,
    ProviderFactory,
    Reranker,
    StructuredChatClient,
)
from Runtime.lifecycle import LifecycleWorker
from Runtime.worker import MemoryWorker


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
class RuntimeBehavior:
    """Evidence & Claim Layer 的单 Store、单模型客户端组合结果。"""

    store: SQLiteBehaviorEvidenceClaimStore
    source_service: SourceRecordService
    evidence_service: EvidenceService
    claim_producers: ClaimProducerRegistry
    claim_pipeline: ClaimPipelineService
    structured_chat: StructuredChatClient

    def __post_init__(self) -> None:
        if not isinstance(self.store, SQLiteBehaviorEvidenceClaimStore):
            raise TypeError("store must be SQLiteBehaviorEvidenceClaimStore")
        if not isinstance(self.source_service, SourceRecordService):
            raise TypeError("source_service must be SourceRecordService")
        if not isinstance(self.evidence_service, EvidenceService):
            raise TypeError("evidence_service must be EvidenceService")
        if not isinstance(self.claim_producers, ClaimProducerRegistry):
            raise TypeError("claim_producers must be ClaimProducerRegistry")
        if not isinstance(self.claim_pipeline, ClaimPipelineService):
            raise TypeError("claim_pipeline must be ClaimPipelineService")
        if not isinstance(self.structured_chat, StructuredChatClient):
            raise TypeError("structured_chat must be StructuredChatClient")
        if self.source_service.store is not self.store:
            raise ValueError("Behavior source service must use the shared Store")
        if self.evidence_service.store is not self.store:
            raise ValueError("Behavior evidence service must use the shared Store")
        if self.claim_pipeline.store is not self.store:
            raise ValueError("Behavior Claim pipeline must use the shared Store")
        if self.claim_pipeline.source_service is not self.source_service:
            raise ValueError("Behavior Claim pipeline must use the shared source service")
        if self.claim_pipeline.evidence_service is not self.evidence_service:
            raise ValueError("Behavior Claim pipeline must use the shared evidence service")
        if self.claim_pipeline.producers is not self.claim_producers:
            raise ValueError("Behavior Claim pipeline must use the shared Producer registry")
        if self.source_service.config != self.store.config.source:
            raise ValueError("Behavior source service must use the Store source configuration")
        if self.evidence_service.config != self.store.config.evidence:
            raise ValueError("Behavior evidence service must use the Store evidence configuration")
        if self.claim_pipeline.config != self.store.config.claim:
            raise ValueError("Behavior Claim pipeline must use the Store Claim configuration")
        for name in self.claim_producers.names():
            producer = self.claim_producers.get(name)
            if isinstance(producer, StructuredSemanticClaimProducer) and producer.client is not self.structured_chat:
                raise ValueError("Behavior model Producer must use the shared StructuredChatClient")


@dataclass(frozen=True)
class RuntimeConversation:
    """Conversation 原文、切段与摘要服务。"""

    journal: ConversationMessageJournal
    retention: ConversationRetentionPlanner
    boundary_scorer: ConversationSemanticBoundaryScorer
    summaries: ConversationSummaryService
    summary_compactor: ConversationSummaryCompactor
    summary_vector_index: PersistentConversationSummaryVectorIndex
    summary_use: SQLiteConversationSummaryUseStore
    summary_expander: ConversationSummaryExpander

    def __post_init__(self) -> None:
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
    behavior: RuntimeBehavior
    workflow: RuntimeWorkflow

    def __post_init__(self) -> None:
        if not isinstance(self.infrastructure, RuntimeInfrastructure):
            raise TypeError("infrastructure must be RuntimeInfrastructure")
        if not isinstance(self.models, RuntimeModels):
            raise TypeError("models must be RuntimeModels")
        if not isinstance(self.conversation, RuntimeConversation):
            raise TypeError("conversation must be RuntimeConversation")
        if not isinstance(self.memory, RuntimeMemory):
            raise TypeError("memory must be RuntimeMemory")
        if not isinstance(self.behavior, RuntimeBehavior):
            raise TypeError("behavior must be RuntimeBehavior")
        if not isinstance(self.workflow, RuntimeWorkflow):
            raise TypeError("workflow must be RuntimeWorkflow")
        if self.workflow.enqueuer.conversations is not self.conversation.journal:
            raise ValueError("workflow enqueuer must use the shared conversation journal")
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
        if self.behavior.structured_chat is not self.models.structured_chat:
            raise ValueError("Behavior must use RuntimeModels.shared structured chat client")


__all__ = [
    "RuntimeComponents",
    "RuntimeConversation",
    "RuntimeBehavior",
    "RuntimeInfrastructure",
    "RuntimeMemory",
    "RuntimeModels",
    "RuntimeWorkflow",
]
