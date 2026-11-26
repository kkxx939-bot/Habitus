"""组合根里长期记忆一侧的组装：记忆树、编辑器、语义刷新、冷热维护，以及助手检索。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from habitus.config import HabitusConfig
from habitus.foundation.observability import Observer
from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.browse import MemoryBrowser
from habitus.memory.compaction import (
    MemoryFieldCompactor,
    MemoryLifecycleCommitter,
    MemoryLifecycleManager,
    MemoryLifecycleOperationStore,
    MemoryLifecycleScanStore,
    MemoryRecoveryStore,
)
from habitus.memory.document import MemoryDocumentCodec
from habitus.memory.editor import (
    MemoryCommitWriter,
    MemoryEditor,
    MemoryExtractionLoop,
    MemoryIdentityPlanner,
    MemoryRelatedRetriever,
)
from habitus.memory.recall import MemoryRecallLifecycle, SQLiteMemoryRecallLifecycleStore
from habitus.memory.retrieval import MemoryCatalogReader, NoBehaviorRecords, SearchService
from habitus.memory.schema import MemorySchemaRegistry
from habitus.memory.semantic import MemorySemanticRefresher
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.model_client import StructuredChatClient
from habitus.runtime.behavior import BehaviorBranches
from habitus.runtime.behavior_records import BehaviorTreeRecords
from habitus.runtime.components import RuntimeMemory
from habitus.runtime.conversation_assembly import ConversationParts


@dataclass(frozen=True)
class MemoryParts:
    """长期记忆一侧建好的部件；检索要等行为侧建好之后再装（``build_memory_search``）。"""

    schema_registry: MemorySchemaRegistry
    tree: MemoryTree
    snapshot_reader: MemorySnapshotReader
    recall_lifecycle: MemoryRecallLifecycle
    editor: MemoryEditor
    semantic_refresher: MemorySemanticRefresher
    lifecycle: MemoryLifecycleManager


def build_memory(config: HabitusConfig, path_lock: PathLock, structured_chat: StructuredChatClient) -> MemoryParts:
    memory_config = config.memory
    schema_registry = MemorySchemaRegistry.load_default()
    tree = MemoryTree(
        config.tree_root,
        document_codec=MemoryDocumentCodec(schema_registry),
        document_config=memory_config.document,
        tree_config=memory_config.tree,
    )
    snapshot_reader = MemorySnapshotReader(tree, config=memory_config.snapshot)
    recall_lifecycle = MemoryRecallLifecycle(  # 提取时搜内容要剔退休卡片，所以在检索器之前建
        SQLiteMemoryRecallLifecycleStore(
            config.workflow_root / "memory_recall_lifecycle.sqlite3",
            config=memory_config.recall_lifecycle,
            initialize=False,
        ),
        config=memory_config.recall_lifecycle,
    )
    writer = MemoryCommitWriter(tree, snapshot_reader, path_lock, config=memory_config.commit)
    semantic_refresher = MemorySemanticRefresher(tree, schema_registry, path_lock, config=memory_config.semantic)

    async def refresh_lifecycle_derivatives(uris):
        addresses = tuple(uri.to_address() for uri in uris)
        await asyncio.to_thread(semantic_refresher.refresh_for_many, addresses)

    lifecycle = MemoryLifecycleManager(
        tree,
        snapshot_reader,
        recall_lifecycle,
        MemoryFieldCompactor(structured_chat, registry=schema_registry, config=memory_config.field_compaction),
        MemoryRecoveryStore(tree, config.workflow_root),
        MemoryLifecycleCommitter(writer, snapshot_reader),
        MemoryLifecycleOperationStore(tree, path_lock, config.workflow_root),
        MemoryLifecycleScanStore(tree, path_lock, config.workflow_root),
        config=memory_config.lifecycle_maintenance,
        derived_refresh=refresh_lifecycle_derivatives,
    )
    return MemoryParts(
        schema_registry=schema_registry,
        tree=tree,
        snapshot_reader=snapshot_reader,
        recall_lifecycle=recall_lifecycle,
        editor=_editor(config, schema_registry, snapshot_reader, recall_lifecycle, writer, structured_chat),
        semantic_refresher=semantic_refresher,
        lifecycle=lifecycle,
    )


def _editor(
    config: HabitusConfig,
    schema_registry: MemorySchemaRegistry,
    snapshot_reader: MemorySnapshotReader,
    recall_lifecycle: MemoryRecallLifecycle,
    writer: MemoryCommitWriter,
    structured_chat: StructuredChatClient,
) -> MemoryEditor:
    memory_config = config.memory
    retriever = MemoryRelatedRetriever(
        schema_registry=schema_registry,
        snapshot_reader=snapshot_reader,
        config=memory_config.editor_retrieval,
        recall_lifecycle=recall_lifecycle,
    )
    extraction_loop = MemoryExtractionLoop(
        client=structured_chat,
        retriever=retriever,
        zone=config.locale.zone(),
        config=memory_config.extraction,
    )
    return MemoryEditor(
        extraction_loop=extraction_loop,
        identity_planner=MemoryIdentityPlanner(schema_registry),
        writer=writer,
    )


def build_memory_search(
    config: HabitusConfig,
    memory: MemoryParts,
    conversation: ConversationParts,
    behavior: BehaviorBranches | None,
    structured_chat: StructuredChatClient,
    observer: Observer,
) -> RuntimeMemory:
    """检索在行为组件之后建：它要读两支行为树里每一轮的记录（行为侧没开就没有这一处来源）。"""

    memory_config = config.memory
    search = SearchService(
        catalog=MemoryCatalogReader(
            memory.tree,
            memory.schema_registry,
            memory.recall_lifecycle,
            max_catalog_chars=memory_config.search_service.max_catalog_chars,
            overview_max_bytes=memory_config.semantic.max_overview_chars * 4 + 4,
            max_unlisted_scan=memory_config.search_service.max_unlisted_scan,
        ),
        snapshot_reader=memory.snapshot_reader,
        client=structured_chat,
        recall_lifecycle=memory.recall_lifecycle,
        conversation_context=conversation.search_context_reader,
        summaries=conversation.summary_reader,
        records=(
            NoBehaviorRecords() if behavior is None else BehaviorTreeRecords(tuple(branch.tree for branch in behavior))
        ),
        zone=config.locale.zone(),
        config=memory_config.search_service,
        cold_probe_expander=memory.lifecycle,
        summary_use_recorder=conversation.summary_use,
        summary_expander=conversation.summary_expander,
        observer=observer,
    )
    return RuntimeMemory(
        tree=memory.tree,
        search=search,
        editor=memory.editor,
        semantic_refresher=memory.semantic_refresher,
        lifecycle=memory.lifecycle,
        browser=MemoryBrowser(memory.tree),
    )


__all__ = ["MemoryParts", "build_memory", "build_memory_search"]
