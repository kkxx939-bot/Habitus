"""预测样本文档与目录语义层的统一物理边界。"""

from __future__ import annotations

from dataclasses import dataclass


class PredictionDocumentLimitError(ValueError):
    """预测样本文档超过显式容量边界。"""


@dataclass(frozen=True)
class PredictionDocumentConfig:
    max_markdown_body_chars: int = 32_000
    max_encoded_bytes: int = 768_000

    def __post_init__(self) -> None:
        for name, value in {
            "max_markdown_body_chars": self.max_markdown_body_chars,
            "max_encoded_bytes": self.max_encoded_bytes,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def validate_body(self, markdown_body: str) -> None:
        if not isinstance(markdown_body, str):
            raise TypeError("prediction Markdown body must be a string")
        if len(markdown_body) > self.max_markdown_body_chars:
            raise PredictionDocumentLimitError("prediction Markdown body exceeds its configured limit")

    def validate_encoded(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("encoded prediction document must be bytes")
        if len(payload) > self.max_encoded_bytes:
            raise PredictionDocumentLimitError("encoded prediction document exceeds its configured limit")

    def validate_semantic_layer(self, payload: bytes) -> None:
        self.validate_encoded(payload)


__all__ = ["PredictionDocumentConfig", "PredictionDocumentLimitError"]
