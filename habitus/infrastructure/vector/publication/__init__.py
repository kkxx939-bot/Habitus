"""后端无关的远程向量索引发布层。"""

from habitus.infrastructure.vector.model import VectorPublicationSnapshot
from habitus.infrastructure.vector.publication.store import PublishedVectorStore

__all__ = ["PublishedVectorStore", "VectorPublicationSnapshot"]
