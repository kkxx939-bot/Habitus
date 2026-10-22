"""在仓库最外层组装完整记忆主链。

## TODO(PLUGIN-NO-SQLITE-001)：插件化交付下三处 SQLite 文件的去向（用户 2026-09-11 要求登记；怎么处理、怎么改逻辑由他决定）

**背景与裁定**：项目以插件形式交付（``plugins/habitus-memory*``：宿主 hooks 是短命 Node 进程，背后是本地
Python 运行时）。用户裁定：插件不该在本地建数据库表；各层权威数据只能是文件树（行为树/语义树/预测层账本/
记忆树，按天目录 + 原子替换 + 回读核对），"表"只允许是读时在内存里投影出来的形状。JS 侧
``plugins/memory-plugin-shared/lib``（state-store / atomic-file / operation-log）已经按这条走原子 JSON/JSONL。

**现状**：Python 侧仍有三处 SQLite，文件全在 ``config.workflow_root`` 下，都由本模块装配：

1. ``memory_recall_lifecycle.sqlite3`` —— ``memory/retrieval/lifecycle_store.py`` 的
   ``SQLiteMemoryRecallLifecycleStore``，表 ``memory_recall_lifecycle``，主键 uri。一行 = 一篇 L2 记忆的
   使用/冷热/退休事实（useful_recall_count、last_useful_recall_at、cold2_probe_count、compacted_at、
   retire_candidate_at、retired_at、version）。写入方是 ``memory/compaction/lifecycle.py``
   （record_use / mark_retire_candidate / mark_retired），检索排名经 ``read_many`` 读。
   ``config.memory.recall_lifecycle.enabled`` 可整体关掉。
2. ``conversation_summary_use.sqlite3`` —— ``memory/conversation/access.py`` 的
   ``SQLiteConversationSummaryUseStore``，表 ``conversation_summary_use``，主键 identity
   （started_on / conversation_id / stage / summary_id）。一行 = 一份 Summary 的实际使用回执与
   退休候选/退休中状态。写入方是 Summary 压缩器与检索服务的 use recorder；Summary 文件仍是内容真相源。
3. ``locks.sqlite3`` / ``memory_vector_locks.sqlite3`` / ``summary_vector_locks.sqlite3`` ——
   ``infrastructure/store/sqlite/lock_store.py`` 的 ``SQLiteLockStore``，表 ``locks``
   （lock_key、token、expires_at、owner、created_at、fence）。这是跨进程租约锁：**行为归约 runner、
   行为写入器、情景刷新器、记忆提交事务全靠它**。``infrastructure/store/locks/process_local.py`` 是
   进程内实现，不跨进程。

**性质区分**：1、2 是派生状态（从使用回执可重建，丢了只是冷热判断退回默认）；3 是协调机制
（不存数据，但跨进程 fence 必须正确）。改造时两类分开定。

**候选方案（供决定，未裁定）**：

- A. 1、2 改成与 ``scene/refresh/progress.py`` 同形态的原子 JSON 文件：按 uri / identity 分片成小文件
  （每篇一份），version 的 CAS 用"读 → 比 version → 原子替换"实现，写入前持 PathLock。调用契约
  （Protocol / 构造器注入）不变，只换 store 实现与本模块的装配；要补损坏降级与并发写测试。
- B. 3 改成文件锁：``O_EXCL`` 创建租约文件（token、expires_at、fence 写在文件里），过期抢占；fence 单调性
  靠一个 counter 文件原子替换。acquire / renew / fenced / release 语义不变，但崩溃后过期抢占与时钟
  偏差要专门测。或者保留 SQLite 只作锁（它不存数据），由用户裁定"锁文件算不算建表"。
- C. 全部保留，只把三个 .sqlite3 移到插件的状态目录并声明"可删除重建"。改动最小，但与裁定相悖。

**具体场景**：hook 每次触发起一个短命进程写回执；两个 hook 并发写同一篇记忆的使用计数（方案 A 靠
PathLock + version CAS）；用户把数据目录放在 iCloud/Dropbox（SQLite 单文件被部分同步易损坏，小文件安全）。

**影响面**：不动记忆树/行为树/情景树的 schema 与写入逻辑；只换三个存储实现与本模块的装配；
``tests/unit`` 里对应 store 的测试与 ``tests/architecture`` 的边界登记要跟着改。
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable, Mapping
from datetime import timedelta

from habitus.behavior.kinds.reader import VocabularyReader
from habitus.config import HabitusConfig
from habitus.conversation import (
    AddressWatermarks,
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
    PendingSources,
    SourceIndex,
)
from habitus.conversation.source.salvage import SourceSalvage
from habitus.foundation.observability import CompositeObserver, MetricRegistry, Observer
from habitus.infrastructure.lifecycle import CapacitySnapshotStore, LifecycleParticipant
from habitus.infrastructure.observability import ManagedObservability
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.sqlite import SQLiteLockStore
from habitus.infrastructure.vector import VectorStoreFactory, VectorStoreRequirements
from habitus.infrastructure.vector.adapters import register_builtin_vector_adapters
from habitus.memory.browse import MemoryBrowser
from habitus.memory.compaction import (
    MemoryFieldCompactor,
    MemoryLifecycleCommitter,
    MemoryLifecycleManager,
    MemoryRecoveryStore,
)
from habitus.memory.conversation import (
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
from habitus.memory.document import MemoryDocumentCodec
from habitus.memory.editor import (
    MemoryCommitTransaction,
    MemoryEditor,
    MemoryExtractionLoop,
    MemoryIdentityPlanner,
    MemoryRelatedRetriever,
    MemoryTransactionJournal,
)
from habitus.memory.indexing import PersistentMemoryVectorIndex, memory_embedding_fingerprint
from habitus.memory.intention import MemoryIntentionReviewer
from habitus.memory.retrieval import (
    ConversationSearchContextReader,
    MemoryContextAssembler,
    MemoryRecallLifecycle,
    MemoryRetrievalGrader,
    MemorySearchQueryPlanner,
    MemorySemanticSearchEngine,
    SearchService,
    SQLiteMemoryRecallLifecycleStore,
)
from habitus.memory.schema import MemorySchemaRegistry
from habitus.memory.semantic import LLMMemoryOverviewGenerator, MemorySemanticRefresher
from habitus.memory.snapshot import MemorySnapshotReader
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
from habitus.model_client import ProviderFactory, StructuredChatClient
from habitus.pre.conversation import ConversationAdapterRegistry
from habitus.runtime.behavior import build_behavior_components, rereader_of, retry_stage_of
from habitus.runtime.components import (
    RuntimeComponents,
    RuntimeConversation,
    RuntimeInfrastructure,
    RuntimeMemory,
    RuntimeModels,
    RuntimeWorkflow,
)
from habitus.runtime.conditions import local_calendar
from habitus.runtime.evidence import evidence_of
from habitus.runtime.foresight import (
    build_foresight_components,
    build_foresight_tick,
    build_foresight_worker,
    build_outlet_consumer,
    build_reminded_lookup,
    build_reminder_responses,
)
from habitus.runtime.foresight_outlet_consumer import OutletObservingConsumer
from habitus.runtime.lifecycle import LifecycleWorker
from habitus.runtime.lifecycle_behavior import BehaviorParticipant
from habitus.runtime.lifecycle_foresight import ForesightParticipant
from habitus.runtime.lifecycle_housekeeping import LifecycleHousekeepingParticipant
from habitus.runtime.lifecycle_memory import ConversationChainParticipant, MemoryL2Participant, conversation_lease_key
from habitus.runtime.lifecycle_retry import retry_ledger
from habitus.runtime.lifecycle_scene import SceneParticipant, scene_vocabulary_versions
from habitus.runtime.lifecycle_sources import ConversationSourceParticipant, ConversationSourceRecoveryParticipant
from habitus.runtime.runtime import Runtime
from habitus.runtime.scene import build_nightly, build_nightly_worker, build_scene_nightly
from habitus.runtime.session_lane import BehaviorSessionConsumer, answered_question_tools, session_salvager
from habitus.runtime.unsealed import UnsealedFromJudgements
from habitus.runtime.worker import MemoryWorker


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
    if conversation_adapters is not None and not isinstance(conversation_adapters, ConversationAdapterRegistry):
        raise TypeError("conversation_adapters must be ConversationAdapterRegistry or None")
    if path_lock is not None and not isinstance(path_lock, PathLock):
        raise TypeError("path_lock must be PathLock or None")
    if observer is not None and not callable(getattr(observer, "record", None)):
        raise TypeError("observer must implement record")

    # 会话协议的适配器也声明各宿主的提问工具与宿主名：出口、会话 lane、出口那层消费者都从这一份取
    adapters = conversation_adapters if conversation_adapters is not None else ConversationAdapterRegistry.with_builtins()
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
            * (1 + conversation_config.lifecycle.summary_compaction.segment_to_range.max_source_count)
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
        credentials=config.credentials.resolve(conversation_config.summary_vector_store.route.credential_ref),
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
    source_ledger = ConversationSourceLedger(
        SourceIndex(
            config.conversation_root,
            max_addresses=source_config.max_source_addresses,
            max_markers_per_address=source_config.max_markers_per_address,
        ),
        PendingSources(config.conversation_root, max_entries=source_config.max_pending_sources),
        AddressWatermarks(config.conversation_root),
    )
    source_store = ConversationSourceStore(
        config.conversation_root,
        max_file_bytes=source_config.max_envelope_bytes,
        ledger=source_ledger,
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
    behavior_session_outputs = BehaviorSessionOutputStore(
        config.conversation_root,
        max_files_per_source=source_config.max_output_files_per_consumer,
        max_file_bytes=source_config.max_behavior_session_output_bytes,
    )
    memory_conversation_consumer = MemoryConversationConsumer(
        enqueuer,
        conversations,
        boundary_scorer,
        memory_output_store,
    )

    # 行为组件要先于会话源的交付组装：会话 lane 是会话源的一个消费者，写的是行为侧的判断存储。
    def behavior_settled(source_id: str) -> bool:
        """这份会话源的 Behavior 到终态了没有：Outcome 只在终态才建；整份已释放（信封都删了）也算。"""

        if source_outcomes.read(source_id, ConversationSourceConsumer.BEHAVIOR_SESSION) is not None:
            return True
        return source_store.read(source_id) is None

    behavior_components = build_behavior_components(
        config,
        structured_chat=structured_chat,
        lock_store=resolved_lock.lock_store,
        path_lock=resolved_lock,
        observer=operation_observer,
        span_controller=managed_observability,
        source_settled=behavior_settled,
    )
    # behavior 关着而 scene / foresight 开着的组合已经在配置层被硬拒（见 HabitusConfig 的跨域校验），
    # 所以这里 behavior_components 为 None 时整晚那一趟与预测层必然都没开，直接跳过即可。
    # 夜批顺序：归约把一天定稿 → 整晚那一趟（结算 → 语义树 → 机会表；预测树删了，裁定 59）。
    # 预测层站在语义树之上，所以排在后面组装；未启用时 build 自己返回 None。
    # TODO(BHV-SESSION-LANE-001)：预测层按不按 lane 分还没定（用户 2026-10-11：「这个我还没有考虑好，因为预测层的
    # 逻辑还没有写好」）。现状：整晚那一趟与预测层只接会话那一支（行为树、词表、定稿日、未封口的判断都取自会话一支），
    # 物理那一支没有语义树夜批、没有预测层、也不结算。定了之后要么再起一份接物理一支，要么改成同时读两支。
    session_behavior = None if behavior_components is None else behavior_components.session
    # 当地日历只建一份：语义树与预测层读同一份，夜批开头重读放假数据时两边一起换（2026-10-18 四方评审）
    calendar = None if session_behavior is None else local_calendar(config)
    # 读账的两个值（裁定 49）：α 借关系检验前向验证的显著性；遗忘半衰期是几次机会（裁定 57，config.foresight 的临时值）
    evidence = None if session_behavior is None or not config.foresight.enabled else evidence_of(config)
    foresight_components = (
        None
        if session_behavior is None or evidence is None
        else build_foresight_components(
            config,
            behavior_tree=session_behavior.tree,
            calendar=calendar,
            # 预期上的类编号到结算时可能已被拆改：按"现在对应哪些编号"结算（同一份词表）
            current_classes=VocabularyReader(session_behavior.kind_store).current_ids,
            # 词表迁移做到一半时不结算（有未完成的迁移计划就是在迁移）
            vocabulary_migrating=lambda: session_behavior.kind_store.read_migration() is not None,
            # "那天定稿了"只有归约说了算（链都落树、封口视界已过那天的本地结束），结算按它来。
            closed_days=session_behavior.reduction_runner.closed_days,
            evidence=evidence,
            observer=operation_observer,
            # 出口认提问时拿类名配题目（词表给人看的名字；编号仍是身份）
            kind_label=_kind_label(VocabularyReader(session_behavior.kind_store)),
            question_tools=adapters.question_tools(),
        )
    )
    if foresight_components is not None and session_behavior is not None and evidence is not None:
        # 白天那一拍的常驻 worker：未封口的判断经那座桥（判断存储 + 消费账本 + 归约的白天归类）读，
        # 词表版本记在预期上。
        unsealed = UnsealedFromJudgements(
            session_behavior.judgements,
            session_behavior.reduction_runner.ledger,
            session_behavior.reduction_runner.kinds,
        )
        kind_store = session_behavior.kind_store
        tick = build_foresight_tick(
            foresight_components,
            config=config,
            unsealed=unsealed,
            evidence=evidence,
            structured_chat=structured_chat,
            vocabulary_version=lambda: kind_store.read().version,
            vocabulary_migrating=lambda: kind_store.read_migration() is not None,
            observer=operation_observer,
            reread=rereader_of(session_behavior),
        )
        foresight_components = dataclasses.replace(
            foresight_components, worker=build_foresight_worker(tick, config=config, observer=operation_observer)
        )
        # 第 0 步：归约封口时查预测层账本盖 reminded（归约先装、预测层后装，所以在这里接端口）
        session_behavior.reduction_runner.attach_reminder_lookup(build_reminded_lookup(foresight_components))
        # 说那一半（方案 30 第五节）：会话 lane 每轮带上开着的提醒、把模型判出的回应写回账本
        assert behavior_components is not None
        behavior_components.session_lane.attach_reminders(
            build_reminder_responses(foresight_components, config=config, observer=operation_observer)
        )
    # 行为侧没启用时消费者仍要注册（交付层要求名单里每一项都有实现），它把每份源落成"跳过"。
    behavior_session_consumer: BehaviorSessionConsumer | OutletObservingConsumer = BehaviorSessionConsumer(
        None if behavior_components is None else behavior_components.session_lane,
        behavior_session_outputs,
        question_tools=answered_question_tools(adapters),
    )
    if foresight_components is not None and isinstance(behavior_session_consumer, BehaviorSessionConsumer):
        # 提醒出口（方案 32 第 4.2 条）：同一份会话先过出口（认出 agent 的提问、写「说了」与固定选项的回话），再进会话 lane
        behavior_session_consumer = build_outlet_consumer(
            foresight_components, behavior_session_consumer, observer=operation_observer, host_of=adapters.host
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
            ConversationSourceConsumer.BEHAVIOR_SESSION: behavior_session_consumer,
        },
        observer=operation_observer,
    )
    source_coordinator = ConversationSourceCoordinator(source_store, source_delivery)
    # 行为侧的账坏了的会话源：逐轮补救（没写过的轮落「没读懂」、进每晚重试的堆，裁定 67 第 5 条、68 第 3 条）
    source_salvage = SourceSalvage(
        config.conversation_root,
        salvager=session_salvager(
            None if behavior_components is None else behavior_components.session_lane,
            question_tools=answered_question_tools(adapters),
        ),
    )
    source_recovery = ConversationSourceRecovery(
        source_store,
        source_delivery,
        batch_size=source_config.recovery_batch_size,
        retries=retry_ledger(
            config.conversation_root, owner="conversation-source-recovery", config=config.lifecycle.retry
        ),
        salvage=source_salvage,
        orphan_grace=timedelta(seconds=source_config.orphan_grace_seconds),
    )
    source_releaser = ConversationSourceReleaser(
        source_store,
        source_outcomes,
        source_ledger,
        {
            ConversationSourceConsumer.MEMORY: memory_output_store,
            ConversationSourceConsumer.BEHAVIOR_SESSION: behavior_session_outputs,
        },
        resolved_lock.lock_store,
        lock_ttl_seconds=source_config.execution_lock_ttl_seconds,
        salvage=source_salvage,
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
    lifecycle_lock_store = conversation_lifecycle.journal.path_lock.lock_store
    participants: list[LifecycleParticipant] = [
        ConversationChainParticipant(
            conversation_lifecycle,
            conversation_config.lifecycle,
            lock_store=lifecycle_lock_store,
            lease_ttl_seconds=config.lifecycle.worker.lease_ttl_seconds,
        ),
        MemoryL2Participant(memory_lifecycle),
        ConversationSourceParticipant(source_releaser, batch_addresses=source_config.release_batch_addresses),
        ConversationSourceRecoveryParticipant(source_recovery, source_coordinator, wake_memory=worker.wake),
    ]

    nightly_worker = None
    scene = None
    if session_behavior is not None:
        # 整晚那一趟（语义树第 11 步）：结算 → 行为树读一次 → 语义树 → 机会表 → 找原因，自己一个常驻节拍。
        # 语义树没启用时没有整晚（预测层在配置层就要求语义树开着，结算也就不会落空）。
        scene = build_scene_nightly(
            config,
            behavior_tree=session_behavior.tree,
            calendar=calendar,
            kind_store=session_behavior.kind_store,
            structured_chat=structured_chat,
        )
        if scene is not None:
            nightly_worker = build_nightly_worker(
                config,
                build_nightly(
                    config,
                    behavior_tree=session_behavior.tree,
                    scene=scene,
                    kind_store=session_behavior.kind_store,
                    closed_days=session_behavior.reduction_runner.closed_days,
                    foresight=foresight_components,
                    dirty_days=session_behavior.reduction_runner.dirty_days,
                    retry=retry_stage_of(() if behavior_components is None else behavior_components),
                ),
                observer=operation_observer,
            )
    # 行为侧、语义树、预测层的释放各一个参与者（26 号清单 6.14），挂进同一个生命周期 Worker
    scene_versions = None
    if scene is not None:
        scene_versions = scene_vocabulary_versions(scene.relations, scene.concepts)
        participants.append(SceneParticipant(scene.relations))
    if behavior_components is not None:
        participants.append(
            BehaviorParticipant(
                behavior_components,
                backfill_days=config.lifecycle.physical_max_backfill_days,
                derived_versions=scene_versions,
            )
        )
    if foresight_components is not None:
        participants.append(
            ForesightParticipant(
                foresight_components.ledger,
                foresight_components.history.store,
                lane="session",
                feedback_kept=config.lifecycle.feedback_kept,
                window=evidence_of(config).window,
            )
        )
    # 生命周期骨架自己的状态（重试账、每晚重试的堆）按统一寿命清、报容量（L10.10；裁定 67）
    branches = () if behavior_components is None else tuple(behavior_components)
    participants.append(
        LifecycleHousekeepingParticipant(
            (
                source_recovery.retries,
                *(item for branch in branches for item in branch.retry_ledgers),
                *(() if scene is None else (scene.mapping_retries,)),
            ),
            keep_days=config.lifecycle.retry.given_up_days,
            backlogs=tuple(branch.retry_backlog for branch in branches if branch.retry_backlog is not None),
        )
    )
    lifecycle_worker = LifecycleWorker(
        tuple(participants),
        config.lifecycle.worker,
        lock_store=lifecycle_lock_store,
        lease_key=conversation_lease_key(conversation_lifecycle.journal),
        capacity=CapacitySnapshotStore(jobs.root),
        observer=operation_observer,
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
            behavior_session_outputs=behavior_session_outputs,
            behavior_session_consumer=behavior_session_consumer,
            source_inspector=source_inspector,
            source_fence=source_fence,
            source_delivery=source_delivery,
            source_coordinator=source_coordinator,
            source_recovery=source_recovery,
            source_ledger=source_ledger,
            source_releaser=source_releaser,
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
            browser=MemoryBrowser(tree),
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
        nightly=nightly_worker,
        foresight=foresight_components,
    )
    return Runtime(config, components, conversation_adapters=adapters)


def _kind_label(reader: VocabularyReader) -> Callable[[str], str | None]:
    """类编号 → 词表里给人看的类名（没有这个编号、不是编号的给 None）。"""

    def label(token: str) -> str | None:
        try:
            found = reader.describe(token)
        except Exception:  # noqa: BLE001 - 类名只是出口配题目的辅助，词表读不出来不拦取
            return None
        return found.name if found is not None else None

    return label


def _builtin_provider_factory() -> ProviderFactory:
    """延迟导入可选 HTTP 依赖，并只注册当前内置协议适配器。"""

    from habitus.model_client.adapters import register_builtin_adapters

    providers = ProviderFactory()
    register_builtin_adapters(providers)
    return providers


def _model_credentials(config: HabitusConfig, reference: str) -> Mapping[str, str]:
    return config.credentials.resolve(reference)


__all__ = ["build_runtime"]
