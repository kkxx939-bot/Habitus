"""完整、不可变且可重建的预测样本文档。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from foundation.integrity import immutable_snapshot
from prediction.model import PredictionAddress, PredictionKind


@dataclass(frozen=True)
class PredictionDocument:
    """已经严格 Schema 校验的不可变预测样本。"""

    kind: PredictionKind
    address: PredictionAddress
    fields: Mapping[str, Any]
    markdown_body: str

    def __post_init__(self) -> None:
        kind = PredictionKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.address, PredictionAddress) or self.address.kind is not kind:
            raise ValueError("prediction document address does not match its kind")
        if not isinstance(self.fields, Mapping) or any(not isinstance(name, str) for name in self.fields):
            raise TypeError("prediction document fields must be a mapping with string keys")
        object.__setattr__(self, "fields", immutable_snapshot(self.fields))
        if not isinstance(self.markdown_body, str) or not self.markdown_body.strip():
            raise ValueError("prediction document Markdown body must be non-empty")
        if not self.markdown_body.endswith("\n"):
            raise ValueError("prediction document Markdown body must end with a newline")


__all__ = ["PredictionDocument"]
