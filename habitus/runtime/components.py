"""按领域分组暴露 Runtime 已经完成的依赖组装结果。"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import cast

from habitus.conversation import (
    BehaviorSessionOutputStore,
    ConversationConsumerDelivery,
    ConversationConsumerExecutionFence,
    ConversationConsumerOutcomeStore,
    ConversationConsumerStateInspector,
    ConversationSourceConsumer,
    ConversationSourceCoordinator,
    ConversationSourceLedger,
    ConversationSourceRecovery,
    ConversationSourceReleaser,
    ConversationSourceStore,
)
from habitus.foundation.observability import Observer
from habitus.infrastructure.observability import ManagedObservability, MetricRegistry
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector import VectorStoreFactory
from habitus.memory.browse import MemoryBrowser
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
from habitus.runtime.behavior import BehaviorBranches
from habitus.runtime.foresight import ForesightRuntimeComponents
from habitus.runtime.foresight_outlet_consumer import OutletObservingConsumer
from habitus.runtime.foresight_worker import ForesightWorker
from habitus.runtime.generation_runner import GenerationRunner
from habitus.runtime.lifecycle import LifecycleWorker
from habitus.runtime.lifecycle_memory import ConversationChainParticipant
from habitus.runtime.model_queue_worker import ModelQueueWorker
from habitus.runtime.resident import ResidentWorker
from habitus.runtime.session_lane import BehaviorSessionConsumer
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
        if self.observer is None:
            object.__setattr__(self, "observer", self.observability)

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
        if self.structured_chat.client is not self.chat:
            raise ValueError("structured_chat must wrap the shared chat client")

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
    """Conversation Source、原文、各消费者的回执、切段与摘要服务。"""

    sources: ConversationSourceStore
    source_outcomes: ConversationConsumerOutcomeStore
    memory_outputs: MemoryConversationOutputStore
    behavior_session_outputs: BehaviorSessionOutputStore
    #: 会话 lane 消费者；预测层开着时外面包着提醒出口那一层（方案 32 第 4.2 条：同一份会话先过出口再进 lane）。
    behavior_session_consumer: BehaviorSessionConsumer | OutletObservingConsumer
    source_inspector: ConversationConsumerStateInspector
    source_fence: ConversationConsumerExecutionFence
    source_delivery: ConversationConsumerDelivery
    source_coordinator: ConversationSourceCoordinator
    source_recovery: ConversationSourceRecovery
    source_ledger: ConversationSourceLedger
    source_releaser: ConversationSourceReleaser
    journal: ConversationMessageJournal
    retention: ConversationRetentionPlanner
    boundary_scorer: ConversationSemanticBoundaryScorer
    summaries: ConversationSummaryService
    summary_compactor: ConversationSummaryCompactor
    summary_vector_index: PersistentConversationSummaryVectorIndex
    summary_use: SQLiteConversationSummaryUseStore
    summary_expander: ConversationSummaryExpander

    def __post_init__(self) -> None:
        if (
            self.source_delivery.ledger is not self.source_ledger
            or self.source_releaser.ledger is not self.source_ledger
        ):
            raise ValueError("conversation source delivery and releaser must share one source ledger")
        if (
            self.source_releaser.sources is not self.sources
            or self.source_releaser.outcomes is not self.source_outcomes
        ):
            raise ValueError("conversation source releaser must share the source and outcome stores")
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
            == self.behavior_session_outputs.root
            == self.journal.layout.root
        ):
            raise ValueError("Conversation Source, consumer outputs and Journal must share one root")
        if self.behavior_session_consumer.store is not self.behavior_session_outputs:
            raise ValueError("Behavior session consumer must use the shared receipt store")
        if (
            self.source_delivery.consumers[ConversationSourceConsumer.BEHAVIOR_SESSION]
            is not self.behavior_session_consumer
        ):
            raise ValueError("Source delivery must use the shared behavior session consumer")
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
    browser: MemoryBrowser

    def __post_init__(self) -> None:
        if self.browser.tree is not self.tree:
            raise ValueError("memory browser must use the shared memory tree")
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
        if self.enqueuer.jobs is not self.jobs or self.runner.store is not self.jobs:
            raise ValueError("memory workflow must share one job store")
        if self.conversation_consumer.enqueuer is not self.enqueuer:
            raise ValueError("memory workflow must share one Conversation enqueuer")
        if self.lifecycle.jobs is not self.jobs or self.lifecycle.receipts is not self.receipts:
            raise ValueError("conversation lifecycle must share workflow stores")
        if self.lifecycle.summary_vector_index is not self.runner.committed_finalizer.summary_vector_index:
            raise ValueError("workflow must share one Summary vector index")
        if self.runner.transaction_recovery.change_receipts is not self.receipts:
            raise ValueError("memory workflow must share one receipt store")
        if self.worker.runner is not self.runner:
            raise ValueError("memory workflow must share one runner with its worker")
        chains = [
            participant
            for participant in self.lifecycle_worker.participants
            if isinstance(participant, ConversationChainParticipant)
        ]
        if len(chains) != 1 or chains[0].manager is not self.lifecycle:
            raise ValueError("memory workflow must share one lifecycle manager with its worker")
        if self.lifecycle_worker.lock_store is not self.jobs.path_lock.lock_store:
            raise ValueError("both Runtime workers must share one lock store")


@dataclass(frozen=True)
class RuntimeComponents:
    """Runtime 对外暴露的基础设施与清晰领域边界。"""

    infrastructure: RuntimeInfrastructure
    models: RuntimeModels
    conversation: RuntimeConversation
    memory: RuntimeMemory
    workflow: RuntimeWorkflow
    # 行为管线的两支（会话、物理各一整套：融合→归约→行为树）；primary_subject 未配置时为 None（行为侧未启用）。
    behavior: BehaviorBranches | None = None
    # 换代（重试收尾 → 结算 → 读序列 → 语义树 → 机会表 → 日子预期 → 完成标记；没有模型调用）的那一眼；语义树没启用时
    # 为 None。它不是常驻 worker：预测 worker 每槽在预测根租约之内先调它。只接会话那一支（见 assembly 的 TODO）。
    generation: GenerationRunner | None = None
    # 只开语义树、没开预测层时每槽看一眼换代的那个 worker（没有预测拍）；开了预测层时它在 ``foresight.worker``，这里是 None。
    generation_slots: ForesightWorker | None = None
    # 模型队列（出错重试、词表、同步词表牵到的重映射、写概念、映射、候选条件与先验的补问）的常驻 worker；随换代一起有、一起没有。
    model_queue: ModelQueueWorker | None = None
    # 预测层的证据装配；foresight.enabled 为假时为 None。它现在只接会话那一支（见 assembly 的 TODO）。
    foresight: ForesightRuntimeComponents | None = None

    def __post_init__(self) -> None:
        if self.generation is not None:
            if self.behavior is None:
                raise ValueError("the generation reads the behaviour tree; behaviour must be enabled")
            # 实例同一性：换代必须读**这个** Runtime 写入的会话那一支行为树——不是另开一个同路径的实例。
            if getattr(self.generation.generation, "behavior_tree", None) is not self.behavior.session.tree:
                raise ValueError("the generation must read the session behaviour tree")
        self._check_model_queue()
        self._check_slot_worker()
        if self.foresight is not None:
            if self.behavior is None or self.generation is None:
                raise ValueError(
                    "foresight reads the semantic layer's generation; behaviour and the generation must be enabled"
                )
            generation = self.generation.generation
            # 接线漏了的话，进程照常启动、health 全绿、结算一代都不会跑，而配置里还开着——无处可查。结算是换代里的一步。
            if getattr(generation, "settlement", None) is not self.foresight.settlement:
                raise ValueError("the settlement stage must be a step of the generation")
            # 机会表那一步漏接：进程照常、健康全绿（一代都没写过不算故障），白天永远停在"还没有机会表"（2026-10-18 四方评审）
            step = getattr(generation, "foresight", None)
            if step is None or getattr(step, "store", None) is not self.foresight.opportunities:
                raise ValueError("the generation must write the opportunity table the foresight layer reads")
            # 结算认"那天有没有这一代"读的必须是换代写的那一份完成标记
            if self.foresight.settlement.finished != generation.marks.finished:
                raise ValueError("the settlement must read the generation's own completion marks")
            self.foresight.assert_attached_to(behavior=self.behavior.session)
            # 出口漏包：提醒记了「待取」、宿主也问了，转写回来却没人认，账上永远没有「说了」——结算全按没说过处理，一处都不报错
            consumer = self.conversation.behavior_session_consumer
            if not isinstance(consumer, OutletObservingConsumer) or consumer.outlet is not self.foresight.outlet:
                raise ValueError("the session lane consumer must be wrapped by the foresight reminder outlet")
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

    @property
    def slot_worker(self) -> ResidentWorker | None:
        """每槽先看一眼换代、再做预测拍的那个常驻 worker（没开预测层时只看换代）；两样都没有是 None。"""

        return self.generation_slots if self.foresight is None else self.foresight.worker

    def _check_slot_worker(self) -> None:
        """换代没有自己的循环：漏接的话进程照常、健康全绿，一代都不会做。驱动它的必须是那一个每槽的 worker。"""

        if self.generation_slots is not None and self.foresight is not None:
            raise ValueError("with the foresight layer on, the slot worker lives in the foresight components")
        if self.generation is None:
            return
        if getattr(self.slot_worker, "generation", None) is not self.generation:
            raise ValueError("the slot worker must drive this generation")

    def _check_model_queue(self) -> None:
        """模型队列随换代一起有、一起没有，做的是换代那棵语义树的活。"""

        if (self.model_queue is None) != (self.generation is None):
            raise ValueError("the model queue and the generation come together")
        if self.model_queue is None or self.generation is None:
            return
        queue, generation = self.model_queue.queue, self.generation.generation
        # 两份存储对不上的话，换代折叠时读不到队列问来的缓存、映射的命中，一处都不报错
        if queue.scene is not getattr(generation, "scene", None):
            raise ValueError("the model queue must work on the generation's scene")
        # 换代问"模型队列是不是在整体退避"读的必须是这个队列自己的状态（裁定 96：是就不判关系、沿用上一代）
        if getattr(generation, "models_down", None) != queue.state.down:
            raise ValueError("the generation must read this model queue's backoff state")
        # 语义树写锁两边必须是同一把：各持各的话，换代读语义树时队列照写，一处都不报错
        if queue.scene_lock is not getattr(generation, "scene_lock", None):
            raise ValueError("the model queue and the generation must share one scene write lock")


__all__ = [
    "RuntimeComponents",
    "RuntimeConversation",
    "RuntimeInfrastructure",
    "RuntimeMemory",
    "RuntimeModels",
    "RuntimeWorkflow",
]
