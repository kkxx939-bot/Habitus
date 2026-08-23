"""行为语义 L2 文档的统一物理边界。"""

from __future__ import annotations

from dataclasses import dataclass


class BehaviorDocumentLimitError(ValueError):
    """行为 L2 文档超过显式容量边界。"""


@dataclass(frozen=True)
class BehaviorDocumentConfig:
    max_markdown_body_chars: int = 24_000
    max_encoded_bytes: int = 512_000
    max_relations_per_document: int = 512

    def __post_init__(self) -> None:
        for name, value in {
            "max_markdown_body_chars": self.max_markdown_body_chars,
            "max_encoded_bytes": self.max_encoded_bytes,
            "max_relations_per_document": self.max_relations_per_document,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def validate_body(self, markdown_body: str) -> None:
        if not isinstance(markdown_body, str):
            raise TypeError("behavior Markdown body must be a string")
        if len(markdown_body) > self.max_markdown_body_chars:
            raise BehaviorDocumentLimitError("behavior Markdown body exceeds its configured limit")

    def validate_encoded(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("encoded behavior document must be bytes")
        if len(payload) > self.max_encoded_bytes:
            raise BehaviorDocumentLimitError("encoded behavior document exceeds its configured limit")

    def validate_semantic_layer(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("encoded behavior semantic layer must be bytes")
        if len(payload) > self.max_encoded_bytes:
            raise BehaviorDocumentLimitError("behavior semantic layer exceeds its configured byte limit")

    def validate_relations(self, *, links: int) -> None:
        if isinstance(links, bool) or not isinstance(links, int) or links < 0:
            raise ValueError("behavior document links count must be a non-negative integer")
        if links > self.max_relations_per_document:
            raise BehaviorDocumentLimitError("behavior document links exceeds its configured relation limit")


__all__ = ["BehaviorDocumentConfig", "BehaviorDocumentLimitError"]
