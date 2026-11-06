"""两个向量索引的 embedding 指纹：哪些配置项会让已有索引全量重建（W7）。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from habitus.model_client import EmbeddingModelConfig, ProviderConfig
from habitus.runtime.embedding_fingerprints import embedding_fingerprints


def _config(**prefixes: str) -> EmbeddingModelConfig:
    route = ProviderConfig(
        provider="custom",
        adapter="openai_compatible_embedding",
        model="embedding-model",
        base_url="http://127.0.0.1:11434/v1",
    )
    return EmbeddingModelConfig(route, dimension=4, **prefixes)


def test_document_prefix_changes_both_fingerprints() -> None:
    """文档前缀改变文档向量本身：记忆与会话摘要两个索引都要重建。"""

    before = embedding_fingerprints(_config())
    after = embedding_fingerprints(_config(document_prefix="passage: "))
    assert before[0] != after[0]
    assert before[1] != after[1]


@pytest.mark.parametrize("query_prefix", ["query: ", "Instruct: find the memory\nQuery: "])
def test_query_prefix_keeps_both_fingerprints(query_prefix: str) -> None:
    """查询前缀只影响查询向量，已有索引照样能用：改它不能触发全量重建。"""

    assert embedding_fingerprints(_config()) == embedding_fingerprints(_config(query_prefix=query_prefix))


def test_query_parameters_keep_and_document_parameters_change_the_fingerprints() -> None:
    base = _config()
    assert embedding_fingerprints(base) == embedding_fingerprints(replace(base, query_parameters={"x": 1}))
    assert embedding_fingerprints(base) != embedding_fingerprints(replace(base, document_parameters={"x": 1}))
