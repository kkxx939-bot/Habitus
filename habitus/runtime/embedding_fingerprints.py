"""两个向量索引的 embedding 指纹由同一份 Embedding 配置算出（W7）。"""

from __future__ import annotations

from habitus.memory.conversation.indexing import conversation_summary_embedding_fingerprint
from habitus.memory.indexing import memory_embedding_fingerprint
from habitus.model_client import EmbeddingModelConfig


def embedding_fingerprints(embedding: EmbeddingModelConfig) -> tuple[str, str]:
    """记忆向量索引与会话摘要/切段打分的两个 embedding 指纹。

    只放会改变文档向量的项：``document_prefix`` 进（改了要全量重建），``query_prefix`` 只影响查询、不进，
    改它不让已有索引失效。
    """

    route = embedding.route
    memory = memory_embedding_fingerprint(
        provider=route.provider,
        adapter=route.adapter,
        model=route.model,
        base_url=route.base_url,
        dimension=embedding.dimension,
        document_prefix=embedding.document_prefix,
        extra_body=route.extra_body,
        document_parameters=embedding.document_parameters,
    )
    summary = conversation_summary_embedding_fingerprint(
        provider=route.provider,
        adapter=route.adapter,
        model=route.model,
        base_url=route.base_url,
        dimension=embedding.dimension,
        document_prefix=embedding.document_prefix,
        extra_body=route.extra_body,
        document_parameters=embedding.document_parameters,
    )
    return memory, summary


__all__ = ["embedding_fingerprints"]
