"""在仓库最外层组装完整记忆主链。"""

from __future__ import annotations

from collections.abc import Mapping

from Config import M2BOSConfig
from infrastructure.store.contracts import PathLock
from infrastructure.store.sqlite import SQLiteLockStore
from infrastructure.vector import VectorStoreFactory, VectorStoreRequirements
from infrastructure.vector.adapters import register_builtin_vector_adapters
from memory.conversation import (
    ConversationMessageJournal,
    ConversationRangeSummaryGenerator,
    ConversationRangeSummaryStore,
    ConversationRetentionPlanner,
    ConversationSummaryCompactor,
    ConversationSummaryGenerator,
    ConversationSummaryService,
    ConversationSummaryStore,
    PersistentConversationSummaryVectorIndex,
    conversation_summary_embedding_fingerprint,
)
from memory.document import MemoryDocumentCodec
from memory.editor import (
    MemoryCommitTransaction,
    MemoryEditor,
    MemoryExtractionLoop,
    MemoryRelatedRetriever,
    MemoryTransactionJournal,
)
from memory.indexing import PersistentMemoryVectorIndex, memory_embedding_fingerprint
from memory.intention import MemoryIntentionReviewer
from memory.retrieval import (
    ConversationSearchContextReader,
    MemoryContextAssembler,
    MemoryRetrievalGrader,
    MemorySearchQueryPlanner,
    MemorySemanticSearchEngine,
    SearchService,
)
from memory.schema import MemorySchemaRegistry
from memory.semantic import LLMMemoryOverviewGenerator, MemorySemanticRefresher
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.workflow import (
    ConversationLifecycleManager,
    ConversationMemoryEnqueuer,
    MemoryChangeReceiptStore,
    MemoryJobRunner,
    MemoryJobStore,
)
from ModelClient import ProviderFactory, StructuredChatClient
from Runtime.components import (
    RuntimeComponents,
    RuntimeConversation,
    RuntimeInfrastructure,
    RuntimeMemory,
    RuntimeModels,
    RuntimeWorkflow,
)
from Runtime.lifecycle import LifecycleWorker
from Runtime.runtime import Runtime
from Runtime.worker import MemoryWorker


def build_runtime(
    config: M2BOSConfig,
    *,
    providers: ProviderFactory | None = None,
    vector_stores: VectorStoreFactory | None = None,
    path_lock: PathLock | None = None,
    environ: Mapping[str, str] | None = None,
) -> Runtime:
    """无存储写入、无模型请求地完成一次显式依赖组装。"""

    if not isinstance(config, M2BOSConfig):
        raise TypeError("config must be M2BOSConfig")
    if providers is not None and not isinstance(providers, ProviderFactory):
        raise TypeError("providers must be ProviderFactory or None")
    if vector_stores is not None and not isinstance(vector_stores, VectorStoreFactory):
        raise TypeError("vector_stores must be VectorStoreFactory or None")
    if path_lock is not None and not isinstance(path_lock, PathLock):
        raise TypeError("path_lock must be PathLock or None")
    if environ is not None and not isinstance(environ, Mapping):
        raise TypeError("environ must be a string mapping or None")

    resolved_providers = providers or _builtin_provider_factory()
    resolved_vector_stores = vector_stores or register_builtin_vector_adapters()
    resolved_lock = path_lock or PathLock(
        SQLiteLockStore(
            config.workflow_root / "locks.sqlite3",
            config=config.storage.sqlite_lock,
            initialize=False,
        )
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
        environ=environ,
    )
    reranker = (
        resolved_providers.create_reranker(model_config.rerank, environ=environ)
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
        environ=environ,
    )
    vector_index = PersistentMemoryVectorIndex(
        tree,
        embedder,
        vector_store,
        dimension=model_config.embedding.dimension,
        embedding_fingerprint=memory_embedding_fingerprint(
            provider=model_config.embedding.route.provider,
            model=model_config.embedding.route.model,
            dimension=model_config.embedding.dimension,
            input_mode=model_config.embedding.input_mode,
            document_parameters=model_config.embedding.document_parameters,
        ),
        config=memory_config.vector_index,
    )
    semantic_search = MemorySemanticSearchEngine(
        embedder=embedder,
        index=vector_index,
        reranker=reranker,
        config=memory_config.semantic_search,
    )
    retriever = MemoryRelatedRetriever(
        schema_registry=schema_registry,
        snapshot_reader=snapshot_reader,
        semantic_search=semantic_search,
        config=memory_config.retrieval,
    )

    chat = resolved_providers.create_chat_client(model_config.chat, environ=environ)
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
    summary_compactor = ConversationSummaryCompactor(
        conversations,
        summary_store,
        range_summary_store,
        range_summary_generator,
        config=conversation_config.lifecycle.summary_compaction,
    )
    summary_vector_store = resolved_vector_stores.create(
        conversation_config.summary_vector_store,
        requirements=VectorStoreRequirements(
            dimension=model_config.embedding.dimension,
            max_records=conversation_config.summary_vector_index.max_records,
            max_search_hits=conversation_config.summary_vector_index.max_search_hits,
            max_record_chars=conversation_config.summary_vector_index.max_record_chars,
        ),
        environ=environ,
    )
    summary_vector_index = PersistentConversationSummaryVectorIndex(
        conversations,
        summary_compactor,
        embedder,
        summary_vector_store,
        dimension=model_config.embedding.dimension,
        embedding_fingerprint=conversation_summary_embedding_fingerprint(
            provider=model_config.embedding.route.provider,
            model=model_config.embedding.route.model,
            dimension=model_config.embedding.dimension,
            input_mode=model_config.embedding.input_mode,
            document_parameters=model_config.embedding.document_parameters,
        ),
        reranker=reranker,
        config=conversation_config.summary_vector_index,
    )
    search_context_reader = ConversationSearchContextReader(
        conversations,
        summary_compactor,
        config=memory_config.search_service,
    )
    search_query_planner = MemorySearchQueryPlanner(
        structured_chat,
        config=memory_config.search_service,
    )
    retrieval_grader = MemoryRetrievalGrader(
        structured_chat,
        config=memory_config.search_service,
    )
    search_service = SearchService(
        tree=tree,
        snapshot_reader=snapshot_reader,
        semantic_search=semantic_search,
        summary_search=summary_vector_index,
        query_planner=search_query_planner,
        retrieval_grader=retrieval_grader,
        conversation_context=search_context_reader,
        assembler=MemoryContextAssembler(config=memory_config.search_service),
        config=memory_config.search_service,
        intention_reviewer=MemoryIntentionReviewer(memory_config.intention_review),
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
        summary_config=conversation_config.lifecycle.summary_compaction,
        workflow_config=workflow_config.lifecycle,
    )
    enqueuer = ConversationMemoryEnqueuer(
        conversations,
        jobs,
        retention_planner=retention,
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
    )
    worker = MemoryWorker(runner, workflow_config.worker)
    lifecycle_worker = LifecycleWorker(
        conversation_lifecycle,
        conversation_config.lifecycle,
    )

    components = RuntimeComponents(
        infrastructure=RuntimeInfrastructure(
            path_lock=resolved_lock,
            vector_stores=resolved_vector_stores,
        ),
        models=RuntimeModels(
            providers=resolved_providers,
            chat=chat,
            structured_chat=structured_chat,
            embedder=embedder,
            reranker=reranker,
        ),
        conversation=RuntimeConversation(
            journal=conversations,
            retention=retention,
            summaries=summaries,
            summary_compactor=summary_compactor,
            summary_vector_index=summary_vector_index,
        ),
        memory=RuntimeMemory(
            tree=tree,
            search=search_service,
            editor=editor,
            semantic_refresher=semantic_refresher,
            vector_index=vector_index,
        ),
        workflow=RuntimeWorkflow(
            jobs=jobs,
            receipts=receipts,
            enqueuer=enqueuer,
            lifecycle=conversation_lifecycle,
            runner=runner,
            worker=worker,
            lifecycle_worker=lifecycle_worker,
        ),
    )
    return Runtime(config, components)


def _builtin_provider_factory() -> ProviderFactory:
    """延迟导入可选 HTTP 依赖，并只注册当前内置协议适配器。"""

    from ModelClient.adapters import register_builtin_adapters

    providers = ProviderFactory()
    register_builtin_adapters(providers)
    return providers


__all__ = ["build_runtime"]
