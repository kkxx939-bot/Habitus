"""后端无关的远程向量索引发布层。"""

from infrastructure.vector.model import VectorPublicationSnapshot
from infrastructure.vector.publication.store import PublishedVectorStore

__all__ = ["PublishedVectorStore", "VectorPublicationSnapshot"]
