"""预测样本文档公开入口。"""

from prediction.document.codec import PredictionDocumentCodec, PredictionDocumentIntegrityError
from prediction.document.config import PredictionDocumentConfig, PredictionDocumentLimitError
from prediction.document.model import PredictionDocument

__all__ = [
    "PredictionDocument",
    "PredictionDocumentCodec",
    "PredictionDocumentConfig",
    "PredictionDocumentIntegrityError",
    "PredictionDocumentLimitError",
]
