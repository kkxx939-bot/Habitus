"""Conversation 到记忆候选解析阶段的显式资源边界。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.memory.editor.page_id import EXISTING_PAGE_ID_MAX


@dataclass(frozen=True)
class MemoryExtractionConfig:
    """限制受控 ReAct、关系扩展和候选生成的资源消耗。

    ``max_retrieval_iterations`` 默认 3 按逻辑下限定：找到再读最多要"列目录或搜内容一次、读一次、
    结束"三步（名字清单已在第一轮输入里，常态是读一次就结束；搜到的地址直接可读，不必先列目录）。
    ``max_listing_tokens`` 是名字清单在输入里占的固定预算。``max_search_results`` 默认与 ``max_reads_per_action``
    相同：搜一次给出的卡片，下一轮一次就能全部读进来。
    """

    max_retrieval_iterations: int = 3
    max_reads_per_action: int = 8
    max_search_results: int = 8
    max_listing_tokens: int = 8_000
    max_old_memory_items: int = 64
    max_old_memory_bytes: int = 4_000_000
    max_old_memory_tokens: int = 20_000
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
            "max_reads_per_action": (self.max_reads_per_action, 1, EXISTING_PAGE_ID_MAX),
            "max_search_results": (self.max_search_results, 1, EXISTING_PAGE_ID_MAX),
            "max_listing_tokens": (self.max_listing_tokens, 1, 1_000_000),
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
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if self.max_relation_neighbors_per_seed > self.max_relation_neighbors_total:
            raise ValueError("max_relation_neighbors_per_seed cannot exceed max_relation_neighbors_total")


__all__ = ["MemoryExtractionConfig"]
