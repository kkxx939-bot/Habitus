"""持久化向量存储公共协议和显式 Adapter 工厂。"""

from habitus.infrastructure.vector.config import (
    VectorStoreConfig,
    VectorStoreRequirements,
    VectorStoreRouteConfig,
)
from habitus.infrastructure.vector.contracts import RawVectorBackend, VectorStore
from habitus.infrastructure.vector.factory import (
    VectorBackendBuilder,
    VectorStoreBuildContext,
    VectorStoreFactory,
)
from habitus.infrastructure.vector.model import (
    VectorPublicationSnapshot,
    VectorScalar,
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreError,
    VectorStoreFilter,
    VectorStoreIntegrityError,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreState,
    VectorStoreUnsupportedTopologyError,
    VectorValue,
)
from habitus.infrastructure.vector.publication import PublishedVectorStore

__all__ = [
    "VectorScalar",
    "VectorValue",
    "PublishedVectorStore",
    "RawVectorBackend",
    "VectorBackendBuilder",
    "VectorPublicationSnapshot",
    "VectorStore",
    "VectorStoreBuildContext",
    "VectorStoreBusyError",
    "VectorStoreConfig",
    "VectorStoreRequirements",
    "VectorStoreRouteConfig",
    "VectorStoreConflictError",
    "VectorStoreError",
    "VectorStoreFactory",
    "VectorStoreFilter",
    "VectorStoreIntegrityError",
    "VectorStoreMatch",
    "VectorStoreRecord",
    "VectorStoreState",
    "VectorStoreUnsupportedTopologyError",
]
