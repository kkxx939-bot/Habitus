"""记忆与摘要两个向量 Collection 的统一构造（组合根的一小块）。"""

from __future__ import annotations

from habitus.config import HabitusConfig
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector import VectorStore, VectorStoreConfig, VectorStoreFactory, VectorStoreRequirements
from habitus.memory.conversation import ConversationSummaryVectorIndexConfig
from habitus.memory.indexing import MemoryVectorIndexConfig


def build_vector_store(
    factory: VectorStoreFactory,
    config: HabitusConfig,
    store: VectorStoreConfig,
    capacity: MemoryVectorIndexConfig | ConversationSummaryVectorIndexConfig,
    path_lock: PathLock,
) -> VectorStore:
    """记忆与摘要两个 Collection 同一套构造；本地落盘型后端的文件都放在存储根下 ``vector/``，各 Collection 一个。"""

    return factory.create(
        store,
        requirements=VectorStoreRequirements(
            dimension=config.models.embedding.dimension,
            max_records=capacity.max_records,
            max_search_hits=capacity.max_search_hits,
            max_record_chars=capacity.max_record_chars,
        ),
        credentials=config.credentials.resolve(store.route.credential_ref),
        path_lock=path_lock,
        data_root=config.storage_root / "vector",
    )


__all__ = ["build_vector_store"]
