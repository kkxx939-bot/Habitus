"""Conversation 到记忆候选解析阶段的显式资源边界。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.memory.editor.page_id import EXISTING_PAGE_ID_MAX


@dataclass(frozen=True)
class MemoryExtractionConfig:
    """限制受控 ReAct、关系扩展和候选生成的资源消耗。"""

    max_retrieval_iterations: int = 3
    additional_search_limit: int = 5
    max_query_chars: int = 5_000
    max_old_memory_items: int = 64
    max_old_memory_bytes: int = 4_000_000
    max_old_memory_tokens: int = 28_000
    max_context_chars: int = 1_000_000
    max_input_tokens: int = 56_000
    max_observation_chars: int = 12_000
    max_relation_neighbors_per_seed: int = 8
    max_relation_neighbors_total: int = 32
    grader_max_output_tokens: int = 1_000
    candidate_max_output_tokens: int = 8_000

    def __post_init__(self) -> None:
        bounded = {
            "max_retrieval_iterations": (self.max_retrieval_iterations, 1, 8),
            "additional_search_limit": (self.additional_search_limit, 1, 50),
            "max_query_chars": (self.max_query_chars, 1, 20_000),
            "max_old_memory_items": (
                self.max_old_memory_items,
                1,
                EXISTING_PAGE_ID_MAX,
            ),
            "max_old_memory_bytes": (self.max_old_memory_bytes, 1, 32_000_000),
            "max_old_memory_tokens": (self.max_old_memory_tokens, 1, 1_000_000),
            "max_context_chars": (self.max_context_chars, 1, 4_000_000),
            "max_input_tokens": (self.max_input_tokens, 1_024, 4_000_000),
            "max_observation_chars": (self.max_observation_chars, 1, 100_000),
            "max_relation_neighbors_per_seed": (
                self.max_relation_neighbors_per_seed,
                1,
                EXISTING_PAGE_ID_MAX,
            ),
            "max_relation_neighbors_total": (
                self.max_relation_neighbors_total,
                1,
                EXISTING_PAGE_ID_MAX,
            ),
            "grader_max_output_tokens": (self.grader_max_output_tokens, 1, 32_000),
            "candidate_max_output_tokens": (
                self.candidate_max_output_tokens,
                1,
                64_000,
            ),
        }
        for name, (value, minimum, maximum) in bounded.items():
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if self.max_relation_neighbors_per_seed > self.max_relation_neighbors_total:
            raise ValueError("max_relation_neighbors_per_seed cannot exceed max_relation_neighbors_total")


__all__ = ["MemoryExtractionConfig"]
