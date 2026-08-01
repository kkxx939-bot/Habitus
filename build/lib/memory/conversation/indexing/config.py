"""Conversation Summary 远程向量索引的容量与检索配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationSummaryVectorIndexConfig:
    """约束可重建 Summary 索引的发布、候选召回和可选重排。"""

    max_records: int = 100_000
    max_records_per_conversation: int = 30_000
    max_search_hits: int = 10_000
    max_record_chars: int = 16_000
    embedding_batch_size: int = 128
    stale_retries: int = 2
    candidate_multiplier: int = 4
    min_vector_candidates: int = 20
    max_rerank_candidates: int = 50
    max_rerank_document_chars: int = 12_000
    vector_score_threshold: float = -1.0
    rerank_score_threshold: float = 0.0

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("max_records", self.max_records, 1, 100_000_000),
            (
                "max_records_per_conversation",
                self.max_records_per_conversation,
                1,
                1_000_000,
            ),
            ("max_search_hits", self.max_search_hits, 1, 1_000_000),
            ("max_record_chars", self.max_record_chars, 1, 1_000_000),
            ("embedding_batch_size", self.embedding_batch_size, 1, 4_096),
            ("stale_retries", self.stale_retries, 0, 10),
            ("candidate_multiplier", self.candidate_multiplier, 1, 20),
            ("min_vector_candidates", self.min_vector_candidates, 1, 10_000),
            ("max_rerank_candidates", self.max_rerank_candidates, 1, 10_000),
            (
                "max_rerank_document_chars",
                self.max_rerank_document_chars,
                1,
                1_000_000,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"summary vector index {name} must be between {minimum} and {maximum}")
        if self.max_records_per_conversation > self.max_records:
            raise ValueError("summary records per Conversation cannot exceed total records")
        if self.max_rerank_candidates > self.max_search_hits:
            raise ValueError("summary rerank candidates cannot exceed search hits")
        if (
            isinstance(self.vector_score_threshold, bool)
            or not isinstance(self.vector_score_threshold, int | float)
            or not math.isfinite(float(self.vector_score_threshold))
            or not -1.0 <= float(self.vector_score_threshold) <= 1.0
        ):
            raise ValueError("summary vector index vector_score_threshold must be between -1 and 1")
        if isinstance(self.rerank_score_threshold, bool) or not isinstance(
            self.rerank_score_threshold,
            int | float,
        ):
            raise TypeError("summary vector index rerank_score_threshold must be numeric")
        if not math.isfinite(float(self.rerank_score_threshold)):
            raise ValueError("summary vector index rerank_score_threshold must be finite")


__all__ = ["ConversationSummaryVectorIndexConfig"]
