"""在仓库最外层组装完整记忆主链。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from Config import HabitusConfig
from conversation import (
    ConversationBehaviorProjectionConsumer,
    ConversationBehaviorProjectionStore,
    ConversationBehaviorProjector,
    ConversationConsumerDelivery,
    ConversationConsumerExecutionFence,
    ConversationConsumerOutcomeStore,
    ConversationConsumerStateInspector,
    ConversationSourceConsumer,
    ConversationSourceCoordinator,
    ConversationSourceRecovery,
    ConversationSourceStore,
)
from foundation.observability import CompositeObserver, MetricRegistry, Observer
from infrastructure.observability import ManagedObservability
from infrastructure.store.contracts import PathLock
from infrastructure.store.sqlite import SQLiteLockStore
from infrastructure.vector import VectorStoreFactory, VectorStoreRequirements
from infrastructure.vector.adapters import register_builtin_vector_adapters
from memory.compaction import (
    MemoryFieldCompactor,
    MemoryLifecycleCommitter,
    MemoryLifecycleManager,
    MemoryRecoveryStore,
)
from memory.conversation import (
    ConversationMessageJournal,
    ConversationRangeSummaryGenerator,
    ConversationRangeSummaryStore,
    ConversationRetentionPlanner,
    ConversationSemanticBoundaryScorer,
    ConversationSummaryCompactor,
    ConversationSummaryExpander,
    ConversationSummaryGenerator,
    ConversationSummaryRetirementStore,
    ConversationSummaryService,
    ConversationSummaryStore,
    PersistentConversationSummaryVectorIndex,
    SQLiteConversationSummaryUseStore,
    conversation_summary_embedding_fingerprint,
)
from memory.document import MemoryDocumentCodec
from memory.editor import (
    MemoryCommitTransaction,
    MemoryEditor,
    MemoryExtractionLoop,
    MemoryIdentityPlanner,
    MemoryRelatedRetriever,
    MemoryTransactionJournal,
)
from memory.indexing import PersistentMemoryVectorIndex, memory_embedding_fingerprint
from memory.intention import MemoryIntentionReviewer
from memory.retrieval import (
    ConversationSearchContextReader,
    MemoryContextAssembler,
    MemoryRecallLifecycle,
    MemoryRetrievalGrader,
    MemorySearchQueryPlanner,
    MemorySemanticSearchEngine,
    SearchService,
    SQLiteMemoryRecallLifecycleStore,
)
from memory.schema import MemorySchemaRegistry
from memory.semantic import LLMMemoryOverviewGenerator, MemorySemanticRefresher
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.workflow import (
    ConversationLifecycleManager,
    ConversationMemoryEnqueuer,
    MemoryChangeReceiptStore,
    MemoryConversationConsumer,
    MemoryConversationOutputStore,
    MemoryJobRunner,
    MemoryJobStore,
)
from ModelClient import ProviderFactory, StructuredChatClient
from pre.conversation import ConversationAdapterRegistry
from Runtime.behavior import build_behavior_components
from Runtime.components import (
    RuntimeComponents,
    RuntimeConversation,
    RuntimeInfrastructure,
    RuntimeMemory,
    RuntimeModels,
    RuntimeWorkflow,
)
from Runtime.lifecycle import LifecycleWorker
from Runtime.prediction import build_prediction_components
from Runtime.runtime import Runtime
from Runtime.worker import MemoryWorker


def build_runtime(
    config: HabitusConfig,
    *,
    providers: ProviderFactory | None = None,
    vector_stores: VectorStoreFactory | None = None,
    conversation_adapters: ConversationAdapterRegistry | None = None,
    path_lock: PathLock | None = None,
    observer: Observer | None = None,
) -> Runtime:
    """无存储写入、无模型请求地完成一次显式依赖组装。"""

    if not isinstance(config, HabitusConfig):
        raise TypeError("config must be HabitusConfig")
    if providers is not None and not isinstance(providers, ProviderFactory):
        raise TypeError("providers must be ProviderFactory or None")
    if vector_stores is not None and not isinstance(vector_stores, VectorStoreFactory):
        raise TypeError("vector_stores must be VectorStoreFactory or None")
    if conversation_adapters is not None and not isinstance(
        conversation_adapters, ConversationAdapterRegistry
    ):
        raise TypeError("conversation_adapters must be ConversationAdapterRegistry or None")
    if path_lock is not None and not isinstance(path_lock, PathLock):
        raise TypeError("path_lock must be PathLock or None")
    if observer is not None and not callable(getattr(observer, "record", None)):
        raise TypeError("observer must implement record")

    resolved_providers = providers or _builtin_provider_factory()
    resolved_vector_stores = vector_stores or register_builtin_vector_adapters()
    metrics_config = config.observability.metrics
    observability = MetricRegistry(
        enabled=metrics_config.enabled,
        namespace=metrics_config.namespace,
        max_recent_events=metrics_config.max_recent_events,
        duration_buckets=metrics_config.duration_buckets_seconds,
    )
    managed_observability = ManagedObservability(
        config.observability,
        workflow_root=config.workflow_root,
        tracing_headers=config.credentials.resolve(config.observability.tracing.credential_ref),
    )
    observers: list[Observer] = [observability, managed_observability]
    if observer is not None:
        observers.append(observer)
    operation_observer: Observer = CompositeObserver(*observers)
    resolved_lock = path_lock or PathLock(
        SQLiteLockStore(
            config.workflow_root / "locks.sqlite3",
            config=config.storage.sqlite_lock,
            initialize=False,
        ),
        observer=operation_observer,
    )
    memory_vector_lock = path_lock or PathLock(
        SQLiteLockStore(
            config.workflow_root / "memory_vector_locks.sqlite3",
            config=config.storage.sqlite_lock,
            initialize=False,
        ),
        observer=operation_observer,
    )
    summary_vector_lock = path_lock or PathLock(
        SQLiteLockStore(
            config.workflow_root / "summary_vector_locks.sqlite3",
            config=config.storage.sqlite_lock,
            initialize=False,
        ),
        observer=operation_observer,
    )

    schema_registry = MemorySchemaRegistry.load_default()
    codec = MemoryDocumentCodec(schema_registry)
    model_config = config.models
    conversation_config = config.conversation
    memory_config = config.memory
    workflow_config = config.workflow
    tree = MemoryTree(
        config.memory_root,
        document_codec=codec,
        document_config=memory_config.document,
        tree_config=memory_config.tree,
    )
    snapshot_reader = MemorySnapshotReader(tree, config=memory_config.snapshot)

    embedder = resolved_providers.create_embedder(
        model_config.embedding,
        credentials=_model_credentials(config, model_config.embedding.route.credential_ref),
        observer=operation_observer,
    )
    reranker = (
        resolved_providers.create_reranker(
            model_config.rerank,
            credentials=_model_credentials(config, model_config.rerank.route.credential_ref),
            observer=operation_observer,
        )
        if model_config.rerank is not None
        else None
    )
    vector_store = resolved_vector_stores.create(
        memory_config.vector_store,
        requirements=VectorStoreRequirements(
            dimension=model_config.embedding.dimension,
            max_records=memory_config.vector_index.max_records,
            max_search_hits=memory_config.vector_index.max_search_hits,
            max_record_chars=memory_config.vector_index.max_record_chars,
        ),
        credentials=config.credentials.resolve(memory_config.vector_store.route.credential_ref),
        path_lock=memory_vector_lock,
    )
    vector_index = PersistentMemoryVectorIndex(
        tree,
        embedder,
        vector_store,
        dimension=model_config.embedding.dimension,
        embedding_fingerprint=memory_embedding_fingerprint(
            provider=model_config.embedding.route.provider,
            adapter=model_config.embedding.route.adapter,
            model=model_config.embedding.route.model,
            base_url=model_config.embedding.route.base_url,
            dimension=model_config.embedding.dimension,
            input_mode=model_config.embedding.input_mode,
            extra_body=model_config.embedding.route.extra_body,
            document_parameters=model_config.embedding.document_parameters,
        ),
        config=memory_config.vector_index,
        path_lock=resolved_lock,
    )
    semantic_search = MemorySemanticSearchEngine(
        embedder=embedder,
        index=vector_index,
        reranker=reranker,
        config=memory_config.semantic_search,
        observer=operation_observer,
    )
    retriever = MemoryRelatedRetriever(
        schema_registry=schema_registry,
        snapshot_reader=snapshot_reader,
        semantic_search=semantic_search,
        config=memory_config.retrieval,
    )

    chat = resolved_providers.create_chat_client(
        model_config.chat,
        credentials=_model_credentials(config, model_config.chat.route.credential_ref),
        observer=operation_observer,
    )
    structured_chat = StructuredChatClient(
        chat,
        allow_json_repair=model_config.structured_output.allow_json_repair,
        validation_retries=model_config.structured_output.validation_retries,
    )
    extraction_loop = MemoryExtractionLoop(
        client=structured_chat,
        retriever=retriever,
        config=memory_config.extraction,
    )

    transaction_journal = MemoryTransactionJournal(
        config.transaction_root,
        codec,
        config=memory_config.transaction_journal,
    )
    transaction = MemoryCommitTransaction(
        tree,
        snapshot_reader,
        resolved_lock,
        transaction_journal,
        config=memory_config.commit,
    )
    editor = MemoryEditor(
        extraction_loop=extraction_loop,
        identity_planner=MemoryIdentityPlanner(schema_registry),
        transaction=transaction,
    )

    overview_generator = LLMMemoryOverviewGenerator(
        structured_chat,
        config=memory_config.semantic,
    )
    semantic_refresher = MemorySemanticRefresher(
        tree,
        overview_generator,
        resolved_lock,
        config=memory_config.semantic,
    )

    conversations = ConversationMessageJournal(
        config.conversation_root,
        resolved_lock,
        config=conversation_config.journal,
    )
    retention = ConversationRetentionPlanner(conversation_config.segmentation)
    boundary_embedding_fingerprint = conversation_summary_embedding_fingerprint(
        provider=model_config.embedding.route.provider,
        adapter=model_config.embedding.route.adapter,
        model=model_config.embedding.route.model,
        base_url=model_config.embedding.route.base_url,
        dimension=model_config.embedding.dimension,
        input_mode=model_config.embedding.input_mode,
        extra_body=model_config.embedding.route.extra_body,
        document_parameters=model_config.embedding.document_parameters,
    )
    boundary_scorer = ConversationSemanticBoundaryScorer(
        embedder,
        embedding_fingerprint=boundary_embedding_fingerprint,
        max_unit_chars=model_config.embedding.max_input_chars,
    )
    summary_store = ConversationSummaryStore(
        conversations.layout,
        config=conversation_config.summary,
    )
    summary_generator = ConversationSummaryGenerator(
        structured_chat,
        config=conversation_config.summary,
    )
    summaries = ConversationSummaryService(summary_store, summary_generator)
    range_summary_store = ConversationRangeSummaryStore(
        conversations.layout,
        config=conversation_config.summary,
    )
    range_summary_generator = ConversationRangeSummaryGenerator(
        structured_chat,
        summary_config=conversation_config.summary,
        compaction_config=conversation_config.lifecycle.summary_compaction,
    )
    summary_use = SQLiteConversationSummaryUseStore(
        config.workflow_root / "conversation_summary_use.sqlite3",
        initialize=False,
    )
    summary_compactor = ConversationSummaryCompactor(
        conversations,
        summary_store,
        range_summary_store,
        range_summary_generator,
        use_store=summary_use,
        config=conversation_config.lifecycle.summary_compaction,
    )
    summary_retirements = ConversationSummaryRetirementStore(config.workflow_root)
    summary_expander = ConversationSummaryExpander(
        summary_store,
        range_summary_store,
        max_source_reads=(
            conversation_config.lifecycle.summary_compaction.range_to_archive.max_source_count
            * (
                1
                + conversation_config.lifecycle.summary_compaction.segment_to_range.max_source_count
            )
        ),
    )
    summary_vector_store = resolved_vector_stores.create(
        conversation_config.summary_vector_store,
        requirements=VectorStoreRequirements(
            dimension=model_config.embedding.dimension,
            max_records=conversation_config.summary_vector_index.max_records,
            max_search_hits=conversation_config.summary_vector_index.max_search_hits,
            max_record_chars=conversation_config.summary_vector_index.max_record_chars,
        ),
        credentials=config.credentials.resolve(
            conversation_config.summary_vector_store.route.credential_ref
        ),
        path_lock=summary_vector_lock,
    )
    summary_vector_index = PersistentConversationSummaryVectorIndex(
        conversations,
        summary_compactor,
        embedder,
        summary_vector_store,
        dimension=model_config.embedding.dimension,
        embedding_fingerprint=boundary_embedding_fingerprint,
        reranker=reranker,
        config=conversation_config.summary_vector_index,
        observer=operation_observer,
        retirement_store=summary_retirements,
    )
    search_context_reader = ConversationSearchContextReader(
        conversations,
        summary_compactor,
        config=memory_config.search_service,
        retirement_filter=summary_retirements,
    )
    search_query_planner = MemorySearchQueryPlanner(
        structured_chat,
        config=memory_config.search_service,
    )
    retrieval_grader = MemoryRetrievalGrader(
        structured_chat,
        config=memory_config.search_service,
    )
    recall_lifecycle = MemoryRecallLifecycle(
        SQLiteMemoryRecallLifecycleStore(
            config.workflow_root / "memory_recall_lifecycle.sqlite3",
            config=memory_config.recall_lifecycle,
            initialize=False,
        ),
        config=memory_config.recall_lifecycle,
    )
    field_compactor = MemoryFieldCompactor(
        structured_chat,
        registry=schema_registry,
        config=memory_config.field_compaction,
    )
    recovery_store = MemoryRecoveryStore(tree)
    lifecycle_committer = MemoryLifecycleCommitter(transaction, snapshot_reader)

    async def refresh_lifecycle_derivatives(uris):
        addresses = tuple(uri.to_address() for uri in uris)
        await asyncio.to_thread(semantic_refresher.refresh_for_many, addresses)
        await vector_index.rebuild()

    memory_lifecycle = MemoryLifecycleManager(
        tree,
        snapshot_reader,
        recall_lifecycle,
        field_compactor,
        recovery_store,
        lifecycle_committer,
        config=memory_config.lifecycle_maintenance,
        derived_refresh=refresh_lifecycle_derivatives,
    )
    search_service = SearchService(
        tree=tree,
        snapshot_reader=snapshot_reader,
        semantic_search=semantic_search,
        summary_search=summary_vector_index,
        query_planner=search_query_planner,
        retrieval_grader=retrieval_grader,
        recall_lifecycle=recall_lifecycle,
        conversation_context=search_context_reader,
        assembler=MemoryContextAssembler(config=memory_config.search_service),
        config=memory_config.search_service,
        intention_reviewer=MemoryIntentionReviewer(memory_config.intention_review),
        cold_probe_expander=memory_lifecycle,
        summary_fallback_expander=summary_expander,
        summary_use_recorder=summary_use,
        observer=operation_observer,
    )

    jobs = MemoryJobStore(
        config.workflow_root,
        resolved_lock,
        memory_root=tree.root,
        config=workflow_config.jobs,
    )
    receipts = MemoryChangeReceiptStore(
        config.workflow_root,
        codec,
        config=workflow_config.receipts,
    )
    conversation_lifecycle = ConversationLifecycleManager(
        summary_compactor,
        conversations,
        summary_store,
        range_summary_store,
        summary_vector_index,
        jobs,
        receipts,
        transaction_journal,
        summary_retirements,
        summary_config=conversation_config.lifecycle.summary_compaction,
        workflow_config=workflow_config.lifecycle,
    )
    enqueuer = ConversationMemoryEnqueuer(
        conversations,
        jobs,
        retention_planner=retention,
    )
    source_config = conversation_config.source
    projection_config = conversation_config.behavior_projection
    source_store = ConversationSourceStore(
        config.conversation_root,
        max_files=source_config.max_source_files,
        max_file_bytes=source_config.max_envelope_bytes,
    )
    source_outcomes = ConversationConsumerOutcomeStore(
        config.conversation_root,
        max_file_bytes=source_config.max_outcome_bytes,
    )
    memory_output_store = MemoryConversationOutputStore(
        config.conversation_root,
        max_files_per_source=source_config.max_output_files_per_consumer,
        max_file_bytes=source_config.max_memory_output_bytes,
    )
    behavior_projection_store = ConversationBehaviorProjectionStore(
        config.conversation_root,
        max_files_per_source=source_config.max_output_files_per_consumer,
        max_file_bytes=projection_config.max_projection_output_bytes,
        max_items=projection_config.max_projection_items,
    )
    memory_conversation_consumer = MemoryConversationConsumer(
        enqueuer,
        conversations,
        boundary_scorer,
        memory_output_store,
    )
    behavior_projection_consumer = ConversationBehaviorProjectionConsumer(
        ConversationBehaviorProjector(),
        behavior_projection_store,
    )
    source_inspector = ConversationConsumerStateInspector(source_outcomes)
    source_fence = ConversationConsumerExecutionFence(
        resolved_lock,
        ttl_seconds=source_config.execution_lock_ttl_seconds,
        heartbeat_interval_seconds=source_config.execution_lock_heartbeat_seconds,
        wait_seconds=source_config.execution_lock_wait_seconds,
    )
    source_delivery = ConversationConsumerDelivery(
        source_store,
        source_outcomes,
        source_inspector,
        source_fence,
        {
            ConversationSourceConsumer.MEMORY: memory_conversation_consumer,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION: behavior_projection_consumer,
        },
        observer=operation_observer,
    )
    source_coordinator = ConversationSourceCoordinator(source_store, source_delivery)
    source_recovery = ConversationSourceRecovery(
        source_store,
        source_delivery,
        batch_size=source_config.recovery_batch_size,
    )
    runner = MemoryJobRunner(
        jobs,
        conversations,
        editor,
        semantic_refresher,
        vector_index,
        summaries,
        summary_vector_index,
        receipts,
        recall_lifecycle,
        observer=operation_observer,
    )
    worker = MemoryWorker(
        runner,
        workflow_config.worker,
        observer=operation_observer,
        span_controller=managed_observability,
    )
    lifecycle_worker = LifecycleWorker(
        conversation_lifecycle,
        conversation_config.lifecycle,
        memory_manager=memory_lifecycle,
        observer=operation_observer,
    )

    behavior_components = build_behavior_components(
        config,
        structured_chat=structured_chat,
        lock_store=resolved_lock.lock_store,
        path_lock=resolved_lock,
        observer=operation_observer,
    )
    # behavior 关着而 prediction 开着的组合已经在配置层被硬拒（见 HabitusConfig 的跨域校验），
    # 所以这里 behavior_components 为 None 时 prediction 必然也没开，直接跳过即可。
    prediction_components = (
        None
        if behavior_components is None
        else build_prediction_components(
            config, behavior_tree=behavior_components.tree, observer=operation_observer
        )
    )
    components = RuntimeComponents(
        infrastructure=RuntimeInfrastructure(
            path_lock=resolved_lock,
            vector_stores=resolved_vector_stores,
            observability=observability,
            observer=operation_observer,
            managed_observability=managed_observability,
        ),
        models=RuntimeModels(
            providers=resolved_providers,
            chat=chat,
            structured_chat=structured_chat,
            embedder=embedder,
            reranker=reranker,
        ),
        conversation=RuntimeConversation(
            sources=source_store,
            source_outcomes=source_outcomes,
            memory_outputs=memory_output_store,
            behavior_projections=behavior_projection_store,
            behavior_projection_consumer=behavior_projection_consumer,
            source_inspector=source_inspector,
            source_fence=source_fence,
            source_delivery=source_delivery,
            source_coordinator=source_coordinator,
            source_recovery=source_recovery,
            journal=conversations,
            retention=retention,
            boundary_scorer=boundary_scorer,
            summaries=summaries,
            summary_compactor=summary_compactor,
            summary_vector_index=summary_vector_index,
            summary_use=summary_use,
            summary_expander=summary_expander,
        ),
        memory=RuntimeMemory(
            tree=tree,
            search=search_service,
            editor=editor,
            semantic_refresher=semantic_refresher,
            vector_index=vector_index,
            lifecycle=memory_lifecycle,
        ),
        workflow=RuntimeWorkflow(
            jobs=jobs,
            receipts=receipts,
            enqueuer=enqueuer,
            conversation_consumer=memory_conversation_consumer,
            lifecycle=conversation_lifecycle,
            runner=runner,
            worker=worker,
            lifecycle_worker=lifecycle_worker,
        ),
        behavior=behavior_components,
        prediction=prediction_components,
    )
    return Runtime(config, components, conversation_adapters=conversation_adapters)


def _builtin_provider_factory() -> ProviderFactory:
    """延迟导入可选 HTTP 依赖，并只注册当前内置协议适配器。"""

    from ModelClient.adapters import register_builtin_adapters

    providers = ProviderFactory()
    register_builtin_adapters(providers)
    return providers


def _model_credentials(config: HabitusConfig, reference: str) -> Mapping[str, str]:
    return config.credentials.resolve(reference)


__all__ = ["build_runtime"]
