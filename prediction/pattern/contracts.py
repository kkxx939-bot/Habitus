"""Pattern 编排只依赖的窄存储合同。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from prediction.model import PredictionPatternAddress
from prediction.pattern.document import PredictionPatternDocument, PredictionPatternDocumentCodec


@runtime_checkable
class PredictionPatternRepository(Protocol):
    """Pattern 发布与查询所需的最小不可变文档仓库。"""

    root: Path

    @property
    def pattern_codec(self) -> PredictionPatternDocumentCodec: ...

    def create_pattern(self, document: PredictionPatternDocument) -> PredictionPatternDocument: ...

    def read_pattern(self, address: PredictionPatternAddress) -> PredictionPatternDocument: ...


__all__ = ["PredictionPatternRepository"]
