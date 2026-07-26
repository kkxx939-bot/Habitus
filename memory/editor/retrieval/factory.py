"""把模型配置、记忆树索引和分阶段搜索组装为一个领域入口。"""

from __future__ import annotations

from memory.editor.retrieval.index import (
    MemoryTreeVectorIndex,
    MemoryVectorIndexConfig,
)
from memory.editor.retrieval.search import (
    MemorySemanticSearchConfig,
    MemorySemanticSearchEngine,
)
from memory.tree import MemoryTree
from ModelClient import (
    EmbeddingModelConfig,
    ProviderFactory,
    RerankModelConfig,
)


def build_memory_semantic_search(
    tree: MemoryTree,
    *,
    providers: ProviderFactory,
    embedding: EmbeddingModelConfig,
    rerank: RerankModelConfig | None = None,
    index_config: MemoryVectorIndexConfig | None = None,
    search_config: MemorySemanticSearchConfig | None = None,
) -> MemorySemanticSearchEngine:
    """构造使用同一 Embedder 的 query 生成器和 L2/L0/L1 索引。"""

    if not isinstance(tree, MemoryTree):
        raise TypeError("tree must be a MemoryTree")
    if not isinstance(providers, ProviderFactory):
        raise TypeError("providers must be a ProviderFactory")
    if not isinstance(embedding, EmbeddingModelConfig):
        raise TypeError("embedding must be an EmbeddingModelConfig")
    if rerank is not None and not isinstance(rerank, RerankModelConfig):
        raise TypeError("rerank must be a RerankModelConfig")
    embedder = providers.create_embedder(embedding)
    reranker = providers.create_reranker(rerank) if rerank is not None else None
    index = MemoryTreeVectorIndex(tree, embedder, config=index_config)
    return MemorySemanticSearchEngine(
        embedder=embedder,
        index=index,
        reranker=reranker,
        config=search_config,
    )


__all__ = ["build_memory_semantic_search"]
