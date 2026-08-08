"""行为语义 L2 文档的公开入口。"""

from behavior.document.codec import BehaviorDocumentCodec, BehaviorDocumentIntegrityError
from behavior.document.config import BehaviorDocumentConfig, BehaviorDocumentLimitError
from behavior.document.link import BehaviorLinkType, BehaviorStoredLink
from behavior.document.model import BehaviorDocument, BehaviorDocumentMetadata

__all__ = [
    "BehaviorDocument",
    "BehaviorDocumentCodec",
    "BehaviorDocumentConfig",
    "BehaviorDocumentIntegrityError",
    "BehaviorDocumentLimitError",
    "BehaviorDocumentMetadata",
    "BehaviorLinkType",
    "BehaviorStoredLink",
]
