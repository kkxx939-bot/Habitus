"""组装真实 Chat、Embedding、记忆检索和候选解析入口。"""

from __future__ import annotations

from infrastructure.editor.snapshot import SnapshotReadConfig
from memory.editor.extraction.config import MemoryExtractionConfig
from memory.editor.extraction.loop import MemoryExtractionLoop
from memory.editor.reader import MemorySnapshotReader
from memory.editor.retrieval import (
    MemoryRelatedRetriever,
    MemoryRetrievalConfig,
    MemorySemanticSearchConfig,
    MemoryVectorIndexConfig,
    build_memory_semantic_search,
)
from memory.schema import MemorySchemaRegistry
from memory.tree import MemoryTree
from ModelClient import (
    ChatModelConfig,
    EmbeddingModelConfig,
    ProviderFactory,
    RerankModelConfig,
    StructuredChatClient,
)


def build_memory_extraction_loop(
    tree: MemoryTree,
    *,
    providers: ProviderFactory,
    chat: ChatModelConfig,
    embedding: EmbeddingModelConfig,
    rerank: RerankModelConfig | None = None,
    snapshot_config: SnapshotReadConfig | None = None,
    retrieval_config: MemoryRetrievalConfig | None = None,
    index_config: MemoryVectorIndexConfig | None = None,
    search_config: MemorySemanticSearchConfig | None = None,
    extraction_config: MemoryExtractionConfig | None = None,
    allow_json_repair: bool = True,
    validation_retries: int = 1,
) -> MemoryExtractionLoop:
    """使用统一 Provider 配置构造可直接处理 ConversationSegment 的完整入口。"""

    if not isinstance(tree, MemoryTree):
        raise TypeError("tree must be a MemoryTree")
    if not isinstance(providers, ProviderFactory):
        raise TypeError("providers must be a ProviderFactory")
    if not isinstance(chat, ChatModelConfig):
        raise TypeError("chat must be a ChatModelConfig")
    if not isinstance(embedding, EmbeddingModelConfig):
        raise TypeError("embedding must be an EmbeddingModelConfig")
    semantic_search = build_memory_semantic_search(
        tree,
        providers=providers,
        embedding=embedding,
        rerank=rerank,
        index_config=index_config,
        search_config=search_config,
    )
    reader = MemorySnapshotReader(tree, config=snapshot_config)
    retriever = MemoryRelatedRetriever(
        schema_registry=MemorySchemaRegistry.load_default(),
        snapshot_reader=reader,
        semantic_search=semantic_search,
        config=retrieval_config,
    )
    structured_client = StructuredChatClient(
        providers.create_chat_client(chat),
        allow_json_repair=allow_json_repair,
        validation_retries=validation_retries,
    )
    return MemoryExtractionLoop(
        client=structured_client,
        retriever=retriever,
        config=extraction_config,
    )


__all__ = ["build_memory_extraction_loop"]
