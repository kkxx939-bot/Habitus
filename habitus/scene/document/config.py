"""情景 L2 文档的统一物理边界。"""

from __future__ import annotations

from dataclasses import dataclass


class SceneDocumentLimitError(ValueError):
    """情景 L2 文档超过显式容量边界。"""


@dataclass(frozen=True)
class SceneDocumentConfig:
    max_markdown_body_chars: int = 24_000
    max_encoded_bytes: int = 512_000
    max_relations_per_document: int = 64
    max_members_per_document: int = 1_024

    def __post_init__(self) -> None:
        for name in ("max_markdown_body_chars", "max_encoded_bytes", "max_relations_per_document", "max_members_per_document"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def validate_body(self, markdown_body: str) -> None:
        if not isinstance(markdown_body, str):
            raise TypeError("scene Markdown body must be a string")
        if len(markdown_body) > self.max_markdown_body_chars:
            raise SceneDocumentLimitError("scene Markdown body exceeds its configured limit")

    def validate_encoded(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("encoded scene document must be bytes")
        if len(payload) > self.max_encoded_bytes:
            raise SceneDocumentLimitError("encoded scene document exceeds its configured limit")

    def validate_relations(self, *, links: int, members: int) -> None:
        for name, value in (("links", links), ("members", members)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"scene document {name} count must be a non-negative integer")
        if links > self.max_relations_per_document:
            raise SceneDocumentLimitError("scene document links exceed its configured limit")
        if members > self.max_members_per_document:
            raise SceneDocumentLimitError("scene document members exceed its configured limit")


__all__ = ["SceneDocumentConfig", "SceneDocumentLimitError"]
